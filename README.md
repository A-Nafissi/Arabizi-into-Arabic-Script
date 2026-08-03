# Breaking the Script Barrier: Automatic Transliteration of Maghribi Arabizi into Arabic Script

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Domain: NLP](https://img.shields.io/badge/Domain-NLP%20%2F%20Transliteration-green.svg)](#)
[![Paper: PDF](https://img.shields.io/badge/Paper-Download%20PDF-red.svg)](<Breaking the Script Barrier Automatic De_Romanization of Maghribi Arabizi to Arabic Script in Social Media.pdf>)

This research project investigates automatic de-romanization (transliteration) of **Maghribi Arabizi** (Moroccan Darija written in Latin script with numerical representations for Arabic phonemes) into standard Arabic script.

> 📄 **Read the Research Paper:** [Breaking the Script Barrier Automatic De_Romanization of Maghribi Arabizi to Arabic Script in Social Media.pdf](<Breaking the Script Barrier Automatic De_Romanization of Maghribi Arabizi to Arabic Script in Social Media.pdf>)

---

## 📌 Project Overview

Arabizi is widely used across social media and messaging platforms in the Maghrib region. Due to non-standardized orthography, phonetic variation, and heavy use of prefix clitics, processing Arabizi presents significant NLP challenges for low-resource languages.

This repository evaluates and benchmarks three distinct modeling paradigms on a parallel corpus of **8,045 parallel sentences** split into Training ($80\%$), Validation ($10\%$), and a Blind Test set ($10\%$):

1. **Rule-Based Baseline (Regex):** Hand-crafted character and n-gram transformation rules matching Arabizi numerals and digraphs to Arabic phonemes.
2. **Statistical Maximum Likelihood Estimation (MLE) + Regex Fallback:** A word-alignment dictionary trained strictly on the training set, utilizing the rule-based regex model as a fallback for out-of-vocabulary (OOV) tokens.
3. **Neural Character-Level Seq2Seq:** A TensorFlow/Keras encoder-decoder model trained character-by-character to learn end-to-end transliteration patterns.

---

## 📊 Experimental Benchmarks & Results

All models were evaluated on the unseen **blind test set (805 samples)** using corpus-level **SacreBLEU** (`tokenize="intl"`), $1$-to-$4$-gram precisions, and **Exact Word Match Accuracy**.

| Model Architecture | SacreBLEU | 1-gram Precision | 2-gram Precision | 3-gram Precision | 4-gram Precision | Exact Word Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based (Regex)** | **7.19** | 35.23% | 12.51% | 4.41% | 1.58% | 8.56% |
| **Statistical MLE (+ Regex Fallback)** | **35.37** | **67.76%** | **43.72%** | **29.05%** | **19.42%** | **16.75%** |
| **Neural Seq2Seq (Char-level)** | **7.25** | 38.91% | 12.91% | 4.46% | 1.55% | 8.11% |

### Key Findings
* **Statistical MLE Dominance in Low-Resource Regimes:** Statistical word-level alignment with a rule-based fallback heavily outperforms vanilla Neural Seq2Seq ($35.37$ vs. $7.25$ BLEU).
* **Data Scale Sensitivity:** 6,436 training pairs provide sufficient statistical coverage for direct word-level mapping, but are insufficient for un-annotated neural character-level seq2seq models to generalize without attention mechanisms or pre-trained embeddings.
* **Leakage Verification:** The statistical dictionary was validated against a strict leakage audit (`mle_tokens - train_tokens = ∅`), confirming no evaluation bleed.

---

## 🔍 Qualitative Error Taxonomy

Error analysis on blind predictions revealed three main categories of system failure:

1. **Phonetic Ambiguity & Orthographic Variation:** Multiple Latin character combinations mapped to single Arabic letters (e.g., `ee` / `i` / `y` $\rightarrow$ `ي`).
2. **Clitic Attachment & Agglutination:** Arabizi prefix clitics (`l-` article, `f-` preposition, `k-` continuous aspect) merged into single tokens (e.g., `flmal3b` $\rightarrow$ `فالملعب`).
3. **Numerical Transliteration Ambiguity:** Disambiguating numerals used phonetically (`7` = `ح`, `3` = `ع`) versus numeric dates/quantities (e.g., `2022`).

---

## 📁 Repository Structure

```text
Arabizi-into-Arabic-Script/
├── README.md                              # Project documentation
├── LICENSE                                # MIT Open-Source License
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git exclusion settings
├── Breaking the Script Barrier...pdf      # Full paper (PDF)
├── Maghribi_NLP_success.ipynb             # Full experiment notebook
└── data/                                  # Parallel corpus splits
    ├── toy_train.csv                      # Training dataset split (6,436 pairs)
    ├── toy_val.csv                        # Validation dataset split (804 pairs)
    └── toy_test.csv                       # Blind test dataset split (805 pairs)
```

---

## 🚀 Getting Started

### Prerequisites

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the Experiments

1. Open `Maghribi_NLP_success.ipynb` in Jupyter Notebook or VS Code.
2. Execute cells sequentially to load the dataset from `data/`, train the MLE dictionary, build the Neural Seq2Seq model, and run the SacreBLEU evaluation suite.

---

## 📄 Citation

If you find this work or dataset helpful for your research, please cite:

```bibtex
@article{nafissi2026arabizi,
  title={Breaking the Script Barrier: Automatic Transliteration of Maghribi Arabizi into Arabic Script in Social Media},
  author={Nafissi, A.},
  journal={Local Publication},
  year={2026}
}
```
