# Reproducibility records

This directory contains compact, shareable records supporting the camera-ready
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
| `mt5_mle_human_summary.json` | Archived exploratory output review; not a camera-ready results table |
| `consistency_checks.json` | Manuscript table consistency checks |
| `paired_bootstrap_bleu.json` | Passage-level BLEU intervals and focal mT5/MLE paired test |

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

To recompute paired BLEU uncertainty and the n-gram figure from authorized local
prediction exports:

```bash
python -m evaluation.compare_systems \
  --baseline-predictions path/to/blind_test_predictions_all_models.csv \
  --mt5-predictions path/to/test_predictions_mt5.csv \
  --output reproducibility/paired_bootstrap_bleu.json \
  --figure paper/figures/ngram_precision.png \
  --samples 10000 --seed 42
```

The saved result uses SacreBLEU 2.6.0, case-sensitive international tokenization,
and exponential smoothing. All 805 passages and the 78 no-EOS outputs are retained.
The interval for the paired mT5-minus-MLE difference crosses zero, and the focal
paired-bootstrap p-value is 0.074. Passage resampling does not quantify variation
between training seeds or resolve the unverified upstream document groups.

The exploratory mT5/MLE human comparison was completed with independent, blinded
ratings on 50 random passages plus 10 additional no-EOS diagnostic cases. Its
dedicated subsection and table were removed from the camera-ready manuscript;
the limitations disclose the nearly constant acceptability labels, differing
preferences, and unresolved judgments. These findings are not used to validate
the automatic ranking. This aggregate record is preserved rather than erased.
Random-sample
preference agreement is 48% (kappa 0.134); reviewer preferences are not collapsed
into a consensus winner. Two mT5 acceptability ratings remain unresolved and are
preserved as disagreements. The returned-label hashes and unchanged-text checks
are recorded in `mt5_mle_human_summary.json`, without redistributing passage text.
The rubric remains in `evaluation/HUMAN_COMPARISON.md`. Further adjudication is
not required for the current manuscript; document any later adjudication explicitly.
Repository access is by request while the repository remains private.

Recovered preprocessing and split evidence is in `../results/split_verification.json`; full protocol details are in `PROTOCOL.md`. The sanitized original Colab notebook is in `../notebooks/archive/`, with model sections excluded from the focused preprocessing notebook.
