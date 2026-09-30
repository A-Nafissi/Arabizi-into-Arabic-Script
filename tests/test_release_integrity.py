import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_split_ids_and_optional_local_files_match_manifest():
    manifest = json.loads((ROOT / "reproducibility/run_manifest.json").read_text(encoding="utf-8"))
    split_ids = (ROOT / "data/split_ids.csv").read_text(encoding="utf-8").splitlines()[1:]
    counts = {
        split: sum(line.startswith(f"{split},") for line in split_ids)
        for split in ["train", "val", "test"]
    }
    assert counts == {"train": 6436, "val": 804, "test": 805}
    for split, expected in manifest["protocol"]["split_sha256"].items():
        path = ROOT / "data" / f"toy_{split}.csv"
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_final_human_counts_match_camera_ready_table():
    summary = json.loads(
        (ROOT / "reproducibility/human_annotation_summary.json").read_text(encoding="utf-8")
    )["lstm_error_review"]
    assert sum(
        summary[key]
        for key in ["true_model_error", "acceptable_variant_or_not_error", "uncertain"]
    ) == summary["sample_size"] == 50
    tex = (ROOT / "paper/icalp2026_paper.tex").read_text(encoding="utf-8")
    assert f"True model error & {summary['true_model_error']}" in tex
    assert f"Acceptable variant / not a model error & {summary['acceptable_variant_or_not_error']}" in tex
    assert f"Uncertain & {summary['uncertain']}" in tex
