# Blinded mT5 and MLE output review

This protocol is separate from the completed 300-reference, 100-alignment-pair,
and 50-LSTM-error audits. Preparing these files does not constitute human evaluation.

Two native Moroccan Arabic speakers independently review the same 50 randomly
sampled test passages, each with two full candidate outputs. Ten additional
no-EOS mT5 passages outside that random sample form a diagnostic group. Model
names, EOS status, source IDs, and group membership are concealed from annotators.
Candidate order is randomized per passage and fixed between the two annotators.

## Instructions for annotators

Work independently, without viewing the other annotator's labels or model names.
Keep all supplied text unchanged. Read the full Arabizi passage and use the Arabic
reference as context, not as the only permissible spelling. Do not require MSA
translation or penalize a valid Darija spelling solely for differing from the reference.

Fill `judgment_X` and `judgment_Y` using:

- `acceptable`: preserves the dialectal content without a substantive meaning,
  lexical, omission, insertion, or repetition error. Minor punctuation differences
  and valid spelling variants alone do not make an output inaccurate.
- `inaccurate`: has a substantive de-romanization error. State the main issue and
  cite an example in the corresponding notes column.
- `uncertain`: source noise, unclear alignment, or insufficient context prevents
  a defensible judgment. Explain why in the notes.

Fill `preferred_candidate` using `X`, `Y`, `tie`, or `uncertain`. Choose the more
faithful de-romanization, not simply the closest spelling to the reference.
Preference and acceptability are distinct: both outputs can be inaccurate while
one is preferable. For a tie or uncertain preference, explain briefly in either notes field.

Do not truncate long outputs: omission and repetition are part of the evaluation.
Keep annotator identities out of the exported files.

## Preparation and return

```bash
python -m evaluation.prepare_human_comparison prepare \
  --baseline-predictions path/to/blind_test_predictions_all_models.csv \
  --mt5-predictions path/to/test_predictions_mt5.csv \
  --output-dir private_annotations/mt5_mle
```

Give each annotator only their CSV and these instructions. Keep the original
blank template unchanged locally. Do not share `PRIVATE_blinding_key.csv` or the
sampling manifest with annotators before labels are returned. Do not commit raw
passage text, returned annotation files, or the blinding key to Git.

```bash
python -m evaluation.prepare_human_comparison summarize \
  --annotator-a path/to/annotator_A_completed.csv \
  --annotator-b path/to/annotator_B_completed.csv \
  --template private_annotations/mt5_mle/mt5_mle_annotator_A.csv \
  --key private_annotations/mt5_mle/PRIVATE_blinding_key.csv \
  --output reproducibility/mt5_mle_human_summary.json
```

The summarizer verifies unchanged texts, complete valid labels, and matching IDs.
It reports each annotator's judgments and preferences separately, with per-model
observed agreement and kappa (undefined when chance agreement is 1). It never
silently creates consensus labels. Adjudicate disagreements explicitly before
reporting a single final count. Keep the random sample and additional no-EOS
diagnostic sample separate; pooling them would bias a population accuracy estimate.
These judgments do not change BLEU or establish transfer to naturally authored Arabizi.
