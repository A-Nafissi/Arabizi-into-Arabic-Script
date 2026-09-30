# Reproducibility records

This directory contains compact, public records supporting the camera-ready
paper. Raw model checkpoints, private annotation workbooks, cached pretrained
assets, and machine-specific paths are intentionally excluded.

| File | Purpose |
|---|---|
| `run_manifest.json` | Frozen mT5 protocol and split hashes |
| `run_record.json` | Sanitized successful mT5 training record |
| `mt5_evaluation.json` | Full-test mT5 validation and metrics |
| `verified_metrics.csv` | Recomputed rule, MLE, and LSTM metrics |
| `curated_four_system_metrics.json` | Four-system results on 300 reviewed references |
| `training_alignment_agreement.json` | Two-annotator alignment audit summary |
| `human_annotation_summary.json` | Aggregate human-review outcomes |
| `consistency_checks.json` | Manuscript table consistency checks |

The selected mT5 weights are not stored in Git. The exact original training
script was not preserved; `pretrained_baseline/run_baseline.py` is the maintained
resumable runner and differs from the successful run in evaluation/checkpoint
frequency. This limitation is also stated in the paper.

To score a returned mT5 prediction file:

```bash
python -m evaluation.evaluate_mt5_predictions \
  --predictions path/to/test_predictions_mt5.csv \
  --generation-summary path/to/generation_summary.json
```

To recompute the 300-row curated-subset table, supply the reviewed four-system
CSV and mT5 predictions explicitly:

```bash
python -m evaluation.compute_curated_metrics \
  --curated-predictions path/to/curated_predictions.csv \
  --mt5-predictions path/to/test_predictions_mt5.csv
```
