# Breaking the Script Barrier

**Automatic De-Romanization of Maghribi Arabizi to Arabic Script in Social Media**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY-NC 4.0](https://img.shields.io/badge/data-CC%20BY--NC%204.0-orange.svg)](DATA_LICENSE.md)
[![Paper: PDF](https://img.shields.io/badge/paper-camera--ready-red.svg)](<Breaking the Script Barrier Automatic De_Romanization of Maghribi Arabizi to Arabic Script in Social Media.pdf>)

This repository accompanies a paper accepted for camera-ready at the **9th
International Conference on Arabic Language Processing (ICALP 2026)**. The revised
manuscript has been submitted; final confirmation is pending. The included PDF is
the author's camera-ready manuscript, not the final Springer version of record.

The research code and aggregate evidence are released under the licensing terms
described below. Code access does not grant access to the upstream gated corpus
or permit redistribution of passage text. Questions and reproduction inquiries:
**abdessamadnfs@gmail.com**.

The project evaluates automatic de-romanization of Moroccan Arabizi into Arabic
script. The benchmark uses a frozen export derived from the UBC-NLP
[NileChat Arabizi-Morocco](https://huggingface.co/datasets/UBC-NLP/nilechat-arabizi-mor)
resource. Its Arabizi side was generated from Arabic-script Moroccan material
with Command R+; the benchmark does not represent naturally authored Arabizi.

## Systems

Four systems are compared on 8,045 parallel passages: 6,436 training, 804
validation, and 805 held-out test pairs.

1. **Ordered regex mapper** using hand-authored numeral, digraph, consonant, and vowel rules.
2. **Maximum-likelihood word mapper with regex fallback** for out-of-vocabulary tokens.
3. **Word-level character LSTM encoder-decoder** trained on positionally aligned word pairs.
4. **Full-passage `google/mt5-small`** fine-tuned on all training passages.

MLE and the LSTM use the 692 training passages whose Arabizi and Arabic sides
have equal whitespace-token counts. mT5 uses all 6,436 training passages and
external multilingual pretraining, so it is a practical stronger baseline rather
than an architecture-controlled comparison.

## Main results

All systems use the same frozen 805-passage test set. Lower CER and WER are
better; higher BLEU, chrF, and PosEM are better.

| System | BLEU | chrF | CER (%) | WER (%) | PosEM (%) |
|---|---:|---:|---:|---:|---:|
| Rule mapper | 7.24 | 46.73 | 24.50 | 72.50 | 8.69 |
| MLE + regex | 35.44 | **69.60** | **15.17** | **42.48** | 16.76 |
| Word LSTM | 9.62 | 51.98 | 44.94 | 69.00 | 9.42 |
| mT5-small | **36.75** | 68.86 | 42.86 | 64.03 | **21.54** |

The ranking is metric-dependent. mT5 has the highest BLEU and longer n-gram
precision, while MLE has the strongest chrF, CER, and WER. Seventy-eight mT5
outputs did not emit EOS before the frozen generation limit and are retained in
the primary scores.

The post-hoc paired passage bootstrap uses 10,000 resamples and seed 42. The
mT5-minus-MLE BLEU difference is 1.31, with a 95% percentile interval of
[-0.55, 3.24] and SacreBLEU paired-bootstrap p = 0.074. This does not establish
a BLEU advantage at the nominal 0.05 level or demonstrate equivalence. Intervals
condition on the saved models and assume independent passages; source-document
grouping and training-seed variation are not accounted for.

## Human evaluation

- Two native Moroccan Arabic speakers independently accepted all 300 sampled test references. Cohen's kappa is undefined because every label was positive.
- Two independent annotators agreed that 75 of 100 sampled positional training pairs were aligned and 25 were misaligned, giving kappa 1.00.
- The final review of 50 apparent LSTM errors contains 42 true model errors, 5 acceptable variants or non-errors, and 3 uncertain cases.

Only aggregate, anonymized annotation evidence is released in
`reproducibility/`. Raw workbooks and reviewer-identifying information are not
included.

An exploratory blinded mT5/MLE output review was completed, but its dedicated
results subsection and table are excluded from the camera-ready manuscript.
The whole-passage criterion yielded nearly constant inaccuracy labels, reviewers
differed in their preferences, and two acceptability judgments remain unresolved.
The manuscript discloses this limitation and does not claim human-preference
superiority for either model. No automatic scores or earlier audits were changed.

For an auditable record, the aggregate results remain in
`reproducibility/mt5_mle_human_summary.json`; the rubric and validation commands
remain in `evaluation/HUMAN_COMPARISON.md`. Random and additional no-EOS groups
are kept separate, and original labels are not overwritten. Raw packages and
returned passage text stay local. Further adjudication is not required for the
current manuscript scope.

## Repository layout

```text
.
|-- notebooks/                       # Numbered preprocessing, baseline, and mT5 notebooks
|   `-- archive/                     # Sanitized recovered historical Colab notebook
|-- results/                         # Verification records and result index
|-- data/                            # Split IDs, expected hashes, and local-data instructions
|-- pretrained_baseline/             # Resumable full-passage mT5 runner
|-- evaluation/                      # Portable metric and scoring commands
|-- reproducibility/                 # Sanitized run records and aggregate evidence
|-- paper/                           # Camera-ready LaTeX source and bibliography
|-- tests/                           # Metric regression tests
|-- requirements.txt                 # Notebook and evaluation dependencies
|-- requirements-mt5.txt             # mT5-specific dependencies
|-- DATA_LICENSE.md                  # Dataset attribution and reuse terms
|-- CITATION.cff                     # Machine-readable project citation
`-- Breaking the Script Barrier...pdf
```

## Installation

Create an isolated Python environment and install the notebook/evaluation stack:

```bash
pip install -r requirements.txt
```

For mT5, first install the CUDA-compatible PyTorch build recommended for the GPU
runtime, then install:

```bash
pip install -r requirements-mt5.txt
```

The successful mT5 run used an NVIDIA L4 with 22 GiB. See
`pretrained_baseline/README.md` for training, resumption, and prediction commands.

## Preprocessing and split provenance

The recovered `notebooks/01_preprocessing_and_split.ipynb` loads an existing cleaned corpus, applies URL/repetition/punctuation cleaning and script/length filters, then performs two scikit-learn random splits with seed 42 (80/10/10). Reconstruction reproduced all archived partitions exactly in IDs, text, and row order. The clean corpus has no exact duplicate sources, targets, or pairs; the original deduplication operation, upstream subset selection, and dataset revision remain undocumented. Shared identifier prefixes leave source-document independence unverified, and the full history of test-informed rule development is unavailable.

For an authorized clean corpus, place `clean_parallel_corpus.csv` in `data/` and run the focused preprocessing notebook. Existing matching exports are preserved byte-for-byte; differing existing exports cause an error. Verify with `python -m evaluation.verify_preprocessing --corpus data/clean_parallel_corpus.csv --splits data`.

Run notebooks from the repository root. Baseline and mT5 notebooks load frozen partitions directly. The mT5 notebook is a new wrapper around the maintained runner, not the original training notebook. See `reproducibility/PROTOCOL.md` for settings and `results/split_verification.json` for reconstruction evidence.

## Reproduction

1. Obtain the gated source data, reconstruct or receive the authorized frozen exports, and place them locally as described in `data/README.md`.
2. Verify their hashes against `data/README.md` or `reproducibility/run_manifest.json`.
3. Set the working directory to the repository root, then open `notebooks/02_baseline_training_and_evaluation.ipynb` and execute the cells sequentially for the rule, MLE, and LSTM systems.
4. Run `python -m unittest pretrained_baseline.test_baseline` for mT5 utility checks.
5. Run `pytest` for metric regression tests.
6. Fine-tune or resume mT5 using `pretrained_baseline/run_baseline.py`.
7. Score returned predictions using `python -m evaluation.evaluate_mt5_predictions --predictions PATH`.
8. Run `python -m evaluation.compare_systems` with both local prediction files to regenerate paired BLEU uncertainty and the n-gram chart. See `reproducibility/README.md` for the full command.

The selected mT5 weights and training checkpoints are not stored in Git. The
exact original mT5 training script was not preserved; the released resumable
runner differs from the successful run in evaluation/checkpoint frequency. The
paper and run records state this limitation explicitly.

## Data and licensing

The code is MIT-licensed. The dataset is separately licensed under CC BY-NC 4.0
and is subject to the upstream Hugging Face access conditions. The full frozen
CSV exports are not distributed through Git. `data/split_ids.csv` preserves the
8,045 split assignments without reproducing the passage text, and
`DATA_LICENSE.md` documents provenance and reuse conditions. The historical
local filenames `toy_*.csv` are retained for compatibility; they refer to the
complete experimental splits, not toy examples.

## Citation

```bibtex
@inproceedings{nafissi2026breaking,
  title     = {Breaking the Script Barrier: Automatic De-Romanization of Maghribi Arabizi to Arabic Script in Social Media},
  author    = {Nafissi, Abdessamad},
  booktitle = {Proceedings of the 9th International Conference on Arabic Language Processing},
  year      = {2026}
}
```

The source dataset requests citation of:

```bibtex
@inproceedings{el-mekki-etal-2025-nilechat,
  title     = {{N}ile{C}hat: Towards Linguistically Diverse and Culturally Aware {LLM}s for Local Communities},
  author    = {El Mekki, Abdellah and Atou, Houdaifa and Nacar, Omer and Shehata, Shady and Abdul-Mageed, Muhammad},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  pages     = {10967--10991},
  doi       = {10.18653/v1/2025.emnlp-main.556}
}
```
