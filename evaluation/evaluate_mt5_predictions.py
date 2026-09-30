"""Validate and score frozen mT5 blind-test predictions."""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import clean, score_corpus


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path)
    parser.add_argument("--test", type=Path, default=ROOT / "data/toy_test.csv")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=ROOT / "reproducibility/verified_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/mt5_evaluation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(args.test, encoding="utf-8-sig", keep_default_na=False)
    required = {"id", "text_az", "text_ar", "gold_ar", "pred_mt5", "reached_eos"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if len(predictions) != len(test) or len(test) != 805:
        raise ValueError("Predictions must contain the 805 frozen test rows.")
    if not predictions["id"].is_unique or not test["id"].is_unique:
        raise ValueError("Duplicate IDs detected.")
    for column in ["id", "text_az", "text_ar"]:
        if predictions[column].tolist() != test[column].tolist():
            raise ValueError(f"Frozen test mismatch in {column}.")
    if predictions["gold_ar"].map(clean).tolist() != test["text_ar"].map(clean).tolist():
        raise ValueError("Prediction references do not match the frozen test targets.")

    reached = predictions["reached_eos"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if reached.isna().any():
        raise ValueError("reached_eos must contain only true/false values.")
    predictions["reached_eos"] = reached

    groups = [
        ("All blind-test rows", predictions),
        ("Reached EOS (diagnostic)", predictions.loc[reached]),
        ("No EOS (diagnostic)", predictions.loc[~reached]),
    ]
    metric_rows = []
    for label, frame in groups:
        metrics = score_corpus(frame["pred_mt5"], frame["gold_ar"])
        metric_rows.append({"subset": label, **metrics})

    summary = None
    if args.generation_summary:
        summary = json.loads(args.generation_summary.read_text(encoding="utf-8"))
        if summary.get("rows") != 805:
            raise ValueError("Generation summary row count does not equal 805.")
        if summary.get("outputs_without_eos") != int((~reached).sum()):
            raise ValueError("Generation summary EOS count does not match predictions.")

    comparison = []
    if args.baseline_metrics.exists():
        baseline = pd.read_csv(args.baseline_metrics)
        fields = ["Model", "BLEU", "chrF", "CER_percent", "WER_percent", "position_wise_exact_percent"]
        comparison = baseline[fields].to_dict(orient="records")
    comparison.append({"Model": "Pretrained mT5-small", **{
        key: metric_rows[0][key]
        for key in ["BLEU", "chrF", "CER_percent", "WER_percent", "position_wise_exact_percent"]
    }})

    result = {
        "validation": {
            "rows": len(predictions),
            "unique_ids": int(predictions["id"].nunique()),
            "source_rows_match_frozen_test": True,
            "gold_rows_match_frozen_test": True,
            "empty_predictions": int(predictions["pred_mt5"].map(clean).eq("").sum()),
            "outputs_without_eos": int((~reached).sum()),
            "generation_summary": summary,
        },
        "metrics": metric_rows,
        "comparison": comparison,
        "interpretation_guardrails": [
            "All 805 frozen test rows are included in the primary metrics.",
            "EOS subsets are diagnostic only and do not replace the primary scores.",
            "The generation limit was frozen before test evaluation.",
            "The pretrained full-passage model is not an architecture-only comparison.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
