import ast
import json
import math
from pathlib import Path

import pandas as pd
import pytest
from sacrebleu.metrics import BLEU

from evaluation.compare_systems import bootstrap_bleu, load_predictions, SYSTEMS
from evaluation.prepare_human_comparison import agreement, prepare, read_csv, summarize


ROOT = Path(__file__).resolve().parents[1]


def toy_frame():
    return pd.DataFrame({
        "id": [f"item{i}" for i in range(20)],
        "text_az": [f"source text {i}" for i in range(20)],
        "gold_ar": ["this is the reference passage"] * 20,
        **{column: ["this is the reference passage"] * 20 for column in SYSTEMS.values()},
        "reached_eos": [True] * 10 + [False] * 10,
    })


def test_alignment_is_by_id_and_rejects_changed_sources(tmp_path):
    frame = toy_frame()
    baseline_path, mt5_path = tmp_path / "baseline.csv", tmp_path / "mt5.csv"
    frame.drop(columns=["pred_mt5", "reached_eos"]).to_csv(baseline_path, index=False)
    mt5 = frame[["id", "text_az", "gold_ar", "pred_mt5", "reached_eos"]].iloc[::-1]
    mt5.to_csv(mt5_path, index=False)
    joined = load_predictions(baseline_path, mt5_path, expected_rows=20)
    assert joined["id"].tolist() == frame["id"].tolist()
    mt5.loc[mt5.index[0], "text_az"] = "changed source"
    mt5.to_csv(mt5_path, index=False)
    with pytest.raises(ValueError, match="Mismatched text_az"):
        load_predictions(baseline_path, mt5_path, expected_rows=20)


def test_bootstrap_matches_public_bleu_and_is_paired():
    frame = toy_frame().iloc[:3]
    results = bootstrap_bleu(frame, samples=100, seed=42)
    expected = BLEU(tokenize="intl").corpus_score(frame["pred_mle"].tolist(),
                                                 [frame["gold_ar"].tolist()]).score
    for system in results["systems"].values():
        assert system["BLEU"] == pytest.approx(expected)
        assert system["BLEU_95_percentile_CI"] == pytest.approx([expected, expected])
    assert results["focal_comparison"]["difference_95_percentile_CI"] == [0.0, 0.0]
    assert results["focal_comparison"]["sacrebleu_paired_bootstrap_p"] == 1.0


def test_annotation_samples_are_blinded_blank_and_disjoint(tmp_path):
    manifest = prepare(toy_frame(), tmp_path, sample_size=6, failure_size=2)
    a, b = [read_csv(tmp_path / f"mt5_mle_annotator_{name}.csv") for name in ["A", "B"]]
    key = read_csv(tmp_path / "PRIVATE_blinding_key.csv")
    assert a.equals(b) and len(a) == 8
    assert a[["judgment_X", "judgment_Y", "preferred_candidate"]].eq("").all().all()
    assert "stratum" not in a and "source_id" not in a and "reached_eos" not in a
    assert key["source_id"].is_unique
    assert key["stratum"].value_counts().to_dict() == {"random": 6, "additional_no_eos": 2}
    assert manifest["status"] == "awaiting_human_labels"
    with pytest.raises(FileExistsError):
        prepare(toy_frame(), tmp_path, sample_size=6, failure_size=2)


def test_human_summary_does_not_invent_labels_or_consensus(tmp_path):
    prepare(toy_frame(), tmp_path, sample_size=6, failure_size=2)
    template = read_csv(tmp_path / "mt5_mle_annotator_A.csv")
    key = read_csv(tmp_path / "PRIVATE_blinding_key.csv")
    with pytest.raises(ValueError, match="Complete judgment"):
        summarize(template.copy(), template.copy(), template, key)
    completed = template.copy()
    completed["judgment_X"] = "acceptable"
    completed["judgment_Y"] = "acceptable"
    completed["preferred_candidate"] = "tie"
    results = summarize(completed.copy(), completed.copy(), template, key)
    assert results["status"] == "independent_labels_completed_not_adjudicated"
    assert results["strata"]["random"]["passages"] == 6
    assert results["strata"]["additional_no_eos"]["passages"] == 2
    assert results["strata"]["random"]["per_model"]["MLE + regex"]["agreement"]["cohen_kappa"] is None
    assert results["strata"]["random"]["preference_agreement"]["observed_agreement_percent"] == 100
    assert results["strata"]["random"]["preference_agreement"]["cohen_kappa"] is None
    completed.at[0, "Candidate X"] = "changed output"
    with pytest.raises(ValueError, match="text changed"):
        summarize(completed, completed, template, key)


def test_preferences_are_unblinded_before_computing_kappa(tmp_path):
    prepare(toy_frame(), tmp_path, sample_size=6, failure_size=2)
    template = read_csv(tmp_path / "mt5_mle_annotator_A.csv")
    key = read_csv(tmp_path / "PRIVATE_blinding_key.csv")
    completed = template.copy()
    completed["judgment_X"] = "inaccurate"
    completed["judgment_Y"] = "inaccurate"
    completed["preferred_candidate"] = [
        "X" if model == "MLE + regex" else "Y" for model in key["model_X"]
    ]
    result = summarize(completed, completed.copy(), template, key)["strata"]["random"]
    assert result["preferences_by_annotator"][0]["MLE + regex"] == 6
    assert result["preference_agreement"]["cohen_kappa"] is None
    assert sum(row["count"] for row in result["preference_confusion"]) == 6


def test_agreement_handles_identical_and_variable_labels():
    assert agreement(["valid", "valid"], ["valid", "valid"])["cohen_kappa"] is None
    assert agreement(["valid", "invalid"], ["valid", "invalid"])["cohen_kappa"] == 1


def test_notebook_kappa_is_undefined_for_all_positive_labels():
    notebook = json.loads((ROOT / "notebooks/02_baseline_training_and_evaluation.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def cohen_kappa(" in "".join(cell["source"]))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "cohen_kappa")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-kappa", "exec"), namespace)
    kappa = namespace["cohen_kappa"]
    assert math.isnan(kappa(["1"] * 300, ["1"] * 300))
    assert kappa(["1", "0"], ["1", "0"]) == 1
    with pytest.raises(ValueError):
        kappa([], [])


def test_paper_has_supported_statistical_claims_and_no_unused_tool():
    tex = (ROOT / "paper/icalp2026_paper.tex").read_text(encoding="utf-8")
    bibliography = (ROOT / "paper/custom.bib").read_text(encoding="utf-8")
    stats = json.loads((ROOT / "reproducibility/paired_bootstrap_bleu.json").read_text(encoding="utf-8"))
    contrast = stats["focal_comparison"]
    assert f"p={contrast['sacrebleu_paired_bootstrap_p']:.3f}" in tex
    assert "CAMeL" not in tex and "obeid-etal-2020-camel" not in bibliography
    assert "private GitHub repository" in tex
    assert "fig:ngram" in tex and "tab:ngram" not in tex
    assert (ROOT / "paper/figures/ngram_precision.png").exists()


def test_exploratory_human_comparison_is_preserved_and_not_a_paper_result():
    summary = json.loads((ROOT / "reproducibility/mt5_mle_human_summary.json").read_text(encoding="utf-8"))
    tex = (ROOT / "paper/icalp2026_paper.tex").read_text(encoding="utf-8")
    assert summary["protocol"]["independent_and_blinded"] is True
    assert summary["validation"]["original_text_unchanged"] is True
    assert summary["status"] == "independent_labels_completed_not_adjudicated"
    assert summary["pending_acceptability_adjudications"] == 2
    assert summary["manuscript_reporting"]["status"] == "archived_exploratory_not_in_main_results"
    assert summary["manuscript_reporting"]["original_labels_preserved"] is True
    for stratum, count in [("random", 50), ("additional_no_eos", 10)]:
        result = summary["strata"][stratum]
        assert result["passages"] == count
        for preferences in result["preferences_by_annotator"]:
            assert sum(preferences.values()) == count
        for model in result["per_model"].values():
            assert sum(model["annotator_A"].values()) == count
            assert sum(model["annotator_B"].values()) == count
        left, right = [], []
        for entry in result["preference_confusion"]:
            left.extend([entry["annotator_A"]] * entry["count"])
            right.extend([entry["annotator_B"]] * entry["count"])
        assert agreement(left, right) == result["preference_agreement"]
    assert "tab:human-comparison" not in tex
    assert "\\subsection{Blinded mT5--MLE Output Comparison}" not in tex
    assert "An exploratory blinded mT5--MLE output review was completed" in tex
    assert "no human-preference advantage is claimed" in tex
    assert "no mT5 output review was conducted" not in tex
