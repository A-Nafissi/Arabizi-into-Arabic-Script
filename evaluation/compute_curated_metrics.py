"""Score all four systems on the independently reviewed 300-row subset."""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import score_corpus


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-predictions", type=Path, required=True)
    parser.add_argument("--mt5-predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/curated_four_system_metrics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    curated = pd.read_csv(args.curated_predictions, encoding="utf-8-sig", keep_default_na=False)
    mt5 = pd.read_csv(args.mt5_predictions, encoding="utf-8-sig", keep_default_na=False)
    if not curated["id"].is_unique or not mt5["id"].is_unique:
        raise ValueError("Duplicate IDs detected.")
    merged = curated.merge(mt5[["id", "pred_mt5"]], on="id", how="left", validate="one_to_one")
    if len(merged) != 300 or merged["pred_mt5"].map(str).str.len().eq(0).any():
        raise ValueError("The curated evaluation requires 300 complete matched rows.")

    systems = {
        "Rule-Based": "pred_rule_based",
        "Statistical MLE": "pred_mle",
        "Neural Seq2Seq": "pred_seq2seq",
        "Pretrained mT5-small": "pred_mt5",
    }
    results = {
        model: score_corpus(merged[column], merged["gold_ar"])
        for model, column in systems.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
