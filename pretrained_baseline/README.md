# Pretrained full-passage mT5 baseline

`run_baseline.py` implements the full-passage `google/mt5-small` baseline used in
the camera-ready paper. The model revision and split hashes are pinned in
`protocol.json`.

## Environment

Install a CUDA-compatible PyTorch build for the selected GPU, then install:

```bash
pip install -r requirements-mt5.txt
```

The completed run used an NVIDIA L4 with 22 GiB, PyTorch 2.11.0+cu128,
Transformers 5.8.1, Accelerate 1.13.0, and SentencePiece 0.2.1. Full training is
disabled when CUDA is unavailable or the GPU has less than 12 GiB.

## Commands

Run a short feasibility pilot:

```bash
python pretrained_baseline/run_baseline.py pilot --output runs/mt5-pilot
```

Run full training, writing resumable checkpoints:

```bash
python pretrained_baseline/run_baseline.py train --output runs/mt5-full
```

Resume an interrupted run:

```bash
python pretrained_baseline/run_baseline.py train --output runs/mt5-full --resume
```

Generate blind-test predictions from the selected model:

```bash
python pretrained_baseline/run_baseline.py predict --output runs/mt5-test --checkpoint runs/mt5-full/selected_model
```

All commands default to the repository's local `data/` directory. The gated CSV
exports are not distributed; place the three authorized files there as described
in `data/README.md`, or pass their directory with `--data-dir`. Generated
weights, caches, checkpoints, and logs are intentionally excluded from Git.

## Interpretation

This is a practical pretrained baseline, not an architecture-controlled
comparison. It uses external pretraining, full passages, subword tokenization,
and all 6,436 training passages. The MLE and word LSTM use only the 692 passages
eligible for positional word alignment. The preserved successful run evaluated
at epoch boundaries; the maintained resumable runner checkpoints more frequently,
as disclosed in the paper.

Run the local CPU-safe utility tests with:

```bash
python -m unittest pretrained_baseline.test_baseline
```

The final run metadata and aggregate scores are stored in `reproducibility/`.
