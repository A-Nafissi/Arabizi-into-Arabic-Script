"""Paired passage-level BLEU uncertainty for the four frozen systems."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU
from sacrebleu.significance import PairedTest

from evaluation.metrics import clean


SYSTEMS = {
    "Rule mapper": "pred_rule_based",
    "MLE + regex": "pred_mle",
    "Word LSTM": "pred_seq2seq",
    "mT5-small": "pred_mt5",
}


def load_predictions(baseline_path, mt5_path, expected_rows=805):
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig", dtype=str,
                           keep_default_na=False)
    mt5 = pd.read_csv(mt5_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"id", "text_az", "gold_ar"}
    for frame, columns in [(baseline, required | set(list(SYSTEMS.values())[:3])),
                           (mt5, required | {"pred_mt5", "reached_eos"})]:
        if not columns <= set(frame.columns):
            raise ValueError(f"Missing columns: {sorted(columns - set(frame.columns))}")
        if not frame["id"].is_unique or frame["id"].eq("").any():
            raise ValueError("Prediction IDs must be unique and nonempty.")
    if len(baseline) != expected_rows or set(baseline["id"]) != set(mt5["id"]):
        raise ValueError("Prediction files must contain the same complete test IDs.")
    mt5 = mt5.set_index("id").loc[baseline["id"]].reset_index()
    for column in ["text_az", "gold_ar"]:
        if baseline[column].map(clean).tolist() != mt5[column].map(clean).tolist():
            raise ValueError(f"Mismatched {column} for a test ID.")
    result = baseline.copy()
    result["pred_mt5"] = mt5["pred_mt5"].tolist()
    result["reached_eos"] = mt5["reached_eos"].str.lower().map(
        {"true": True, "false": False})
    if result["reached_eos"].isna().any() or result["gold_ar"].map(clean).eq("").any():
        raise ValueError("Invalid EOS flag or empty reference.")
    return result


def percentile_interval(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def bootstrap_bleu(frame, samples=10000, seed=42):
    if samples < 100 or len(frame) < 2:
        raise ValueError("Use at least 100 resamples and two passages.")
    references = frame["gold_ar"].map(clean).tolist()
    metric = BLEU(tokenize="intl", smooth_method="exp", effective_order=False)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame), size=(samples, len(frame)))
    scores = {}
    systems = {}
    # Cache sufficient statistics, then resample complete passages with shared indices.
    for name, column in SYSTEMS.items():
        hypotheses = frame[column].map(clean).tolist()
        stats = np.asarray(metric._extract_corpus_statistics(hypotheses, [references]),
                           dtype=np.int64)
        full = metric._compute_score_from_stats(stats.sum(axis=0).tolist())
        scores[name] = np.array([
            metric._compute_score_from_stats(stats[index].sum(axis=0).tolist()).score
            for index in indices
        ])
        systems[name] = {
            "BLEU": full.score,
            "BLEU_95_percentile_CI": percentile_interval(scores[name]),
            "precisions": full.precisions,
        }
        print(f"Bootstrap complete: {name}", flush=True)

    previous_seed = os.environ.get("SACREBLEU_SEED")
    os.environ["SACREBLEU_SEED"] = str(seed)
    try:
        test = PairedTest(
            [("MLE + regex", frame["pred_mle"].map(clean).tolist()),
             ("mT5-small", frame["pred_mt5"].map(clean).tolist())],
            {"BLEU": BLEU(tokenize="intl", smooth_method="exp", effective_order=False)},
            [references], test_type="bs", n_samples=samples, n_jobs=1,
        )
        signatures, results = test()
    finally:
        if previous_seed is None:
            os.environ.pop("SACREBLEU_SEED", None)
        else:
            os.environ["SACREBLEU_SEED"] = previous_seed
    difference = scores["mT5-small"] - scores["MLE + regex"]
    observed_difference = systems["mT5-small"]["BLEU"] - systems["MLE + regex"]["BLEU"]
    # A zero observed contrast cannot establish superiority; guard the degenerate test.
    p_value = 1.0 if observed_difference == 0 else results["BLEU"][1].p_value
    return {
        "rows": len(frame), "resampling_unit": "passage", "samples": samples,
        "seed": seed, "sacrebleu_version": sacrebleu.__version__,
        "systems": systems,
        "focal_comparison": {
            "contrast": "mT5-small minus MLE + regex",
            "BLEU_difference": observed_difference,
            "difference_95_percentile_CI": percentile_interval(difference),
            "sacrebleu_paired_bootstrap_p": p_value,
            "zero_contrast_policy": "A zero observed BLEU difference is reported with p = 1.",
            "test_signature": str(signatures["BLEU"]),
        },
        "interpretation": [
            "Post-hoc focal BLEU comparison; all test passages and decoding failures retained.",
            "Percentile intervals use paired sampling; p-value uses SacreBLEU's paired bootstrap test.",
            "Conditional on the saved models and an independent-passage resampling assumption.",
            "Does not measure training-seed variability or resolve unverified source-document grouping.",
        ],
    }


def plot_precisions(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = [('#555555', 'o', '--'), ('#0072B2', 's', '-'),
              ('#009E73', '^', ':'), ('#CC79A7', 'D', '-.')]
    fig, ax = plt.subplots(figsize=(6.1, 3.35), constrained_layout=True)
    for (name, values), (color, marker, line) in zip(results["systems"].items(), styles):
        ax.plot([1, 2, 3, 4], values["precisions"], label=name, color=color,
                marker=marker, linestyle=line, linewidth=1.7, markersize=5)
    ax.set(xlabel="N-gram order", ylabel="Modified precision (%)", ylim=(0, 100))
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--mt5-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = load_predictions(args.baseline_predictions, args.mt5_predictions)
    results = bootstrap_bleu(frame, args.samples, args.seed)
    results["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [("baseline_predictions", args.baseline_predictions),
                           ("mt5_predictions", args.mt5_predictions)]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figure:
        plot_precisions(results, args.figure)
    print(json.dumps(results["focal_comparison"], indent=2))


if __name__ == "__main__":
    main()
