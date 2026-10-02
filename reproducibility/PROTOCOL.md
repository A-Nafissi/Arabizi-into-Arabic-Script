# Data, training, and evaluation protocol

## Preprocessing and splitting

The recovered Colab notebook loads an existing 8,045-row clean corpus. It removes URLs, compresses character runs of length >=3 to two, and replaces ! ? , - and newlines with spaces. It excludes Arabizi containing U+0600–U+06FF characters, empty sides, and source/target whitespace-word ratios outside [0.5, 2]. Saved outputs show no rows removed by those filters. No original corpus-deduplication code is preserved; current checks find no exact duplicate sources, targets, or pairs.

Two successive shuffled scikit-learn train_test_split calls use random_state=42: test_size=0.2 for training versus temporary, then test_size=0.5 for validation versus test. The resulting counts are 6,436/804/805. Reconstruction matches all frozen IDs, texts, and row order. Byte hashes are recorded separately because CSV serialization differs. Original upstream retrieval, source revision, subset selection, and source-document grouping remain unresolved.

## Baselines

The ordered regex mapper is hand-authored. MLE counts positional token pairs only in equal-word-count training passages (692 of 6,436); its unknown words use regex fallback. The MLE audit verifies dictionary keys occur in the training vocabulary; it is not a general leakage certificate.

The character word LSTM derives word pairs and vocabularies from training data, with word-length limits of 32 and duplicate word-pair removal. Validation word pairs supply validation loss. The maintained experiment notebook uses a 128-unit encoder and decoder, batch size 128, RMSprop, at most 50 epochs, validation-loss checkpoint selection and early stopping with patience 5. The recovered Colab archive contains superseded experiments and must not be used to infer these final settings.

## mT5 and test evaluation

mT5-small uses all 6,436 training passages, external pretraining, three epochs, learning rate 0.001, and seed 42; the model revision and frozen split byte hashes are pinned in protocol.json/run_manifest.json. Saved run records are authoritative. The maintained resumable runner differs from the successful run in evaluation/checkpoint frequency. Test generation uses a 2,048-new-token cap; all 805 outputs, including 78 missing-EOS cases, are retained.

BLEU, chrF, CER, WER and position-wise token agreement are computed using the documented normalization and implementations in evaluation/. Exact cross-split IDs, source strings, targets, and pairs were screened, before/after experiment normalization; the separate trigram similarity check is heuristic. Shared identifier prefixes leave source-document independence unverified. The complete history of test-informed rule development is not recoverable; reconstruction does not prove an untouched test-development history.

## Human annotation

The 300-reference audit has 100% observed agreement and undefined Cohen's kappa because both reviewers assigned the positive label throughout. The separate 100-pair training-alignment audit has 75 aligned/25 misaligned labels and kappa 1.00. Preserve these distinct interpretations and original labels. Raw annotations are not distributed.
