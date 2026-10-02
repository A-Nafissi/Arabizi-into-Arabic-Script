# Local frozen experimental splits

The full passage text is derived from the gated UBC-NLP
[`nilechat-arabizi-mor`](https://huggingface.co/datasets/UBC-NLP/nilechat-arabizi-mor)
resource and is **not distributed in this Git repository**. Obtain the source
through the upstream access process and comply with its CC BY-NC 4.0 terms.

`split_ids.csv` publishes only the 8,045 identifiers assigned to the train,
validation, and test splits. Reconstructing the exact exports also depends on the
upstream rows and revision; the original upstream subset-construction procedure
was not preserved. Authorized local copies must use these historical filenames:

| Local file | Rows | SHA-256 |
|---|---:|---|
| `toy_train.csv` | 6,436 | `c29fb3212cc586695b936ef8b6584b187353a9a981000d843958ef47e22c351f` |
| `toy_val.csv` | 804 | `5e4dd8aec65faba7835a7fa794e9d1cac3b652b76caa999f8e663b50990e268f` |
| `toy_test.csv` | 805 | `fb9c010ee6ce4340485149a7d61e33bb6510d06d5658e45751b2919e3625baad` |

Place the three files in this directory before running the notebook or mT5
pipeline. They are ignored by Git. Verify them with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 data/toy_train.csv, data/toy_val.csv, data/toy_test.csv
```

The `toy_` prefix is historical. These are the complete experimental exports,
not small demonstration datasets.

## Recovered split procedure

The recovered notebook performs `train_test_split(corpus, test_size=0.2, random_state=42)`, followed by `train_test_split(temporary, test_size=0.5, random_state=42)`. Reconstruction from the authorized clean corpus reproduces the archived splits in content and order. The ZIP CSV byte hashes differ from the frozen files, but decoded IDs and texts match exactly; preserve frozen byte hashes for the mT5 runner. Keep `clean_parallel_corpus.csv` locally here to run preprocessing. Do not overwrite frozen exports merely to normalize CSV encoding or line endings. Original upstream selection/revision and deduplication code remain unavailable.
