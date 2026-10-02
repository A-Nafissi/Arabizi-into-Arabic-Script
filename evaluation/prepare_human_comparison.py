"""Prepare and summarize an independently labeled, model-blinded mT5/MLE review."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_systems import load_predictions


CONTENT = ["review_id", "Arabizi", "Gold Arabic", "Candidate X", "Candidate Y"]
LABELS = ["judgment_X", "judgment_Y", "preferred_candidate", "notes_X", "notes_Y"]
VALID_LABELS = {"acceptable", "inaccurate", "uncertain"}
PREFERENCES = {"X", "Y", "tie", "uncertain"}


def prepare(frame, output, sample_size=50, failure_size=10, seed=42):
    if not 1 <= sample_size <= len(frame):
        raise ValueError("Random sample size is outside the available corpus.")
    rng = np.random.default_rng(seed)
    primary = rng.choice(len(frame), size=sample_size, replace=False)
    failures = np.array([i for i in range(len(frame))
                         if not frame.iloc[i]["reached_eos"] and i not in set(primary)])
    if not 0 <= failure_size <= len(failures):
        raise ValueError("Not enough additional no-EOS passages for this diagnostic sample.")
    diagnostic = rng.choice(failures, size=failure_size, replace=False)
    selected = [(int(i), "random") for i in primary]
    selected += [(int(i), "additional_no_eos") for i in diagnostic]
    rng.shuffle(selected)
    blinded, keys = [], []
    for number, (index, stratum) in enumerate(selected, start=1):
        row = frame.iloc[index]
        x_model, y_model = ("MLE + regex", "mT5-small") if rng.integers(2) == 0 else (
            "mT5-small", "MLE + regex")
        column = {"MLE + regex": "pred_mle", "mT5-small": "pred_mt5"}
        review_id = f"P{number:03d}"
        blinded.append({"review_id": review_id, "Arabizi": row["text_az"],
                        "Gold Arabic": row["gold_ar"],
                        "Candidate X": row[column[x_model]],
                        "Candidate Y": row[column[y_model]],
                        **dict.fromkeys(LABELS, "")})
        keys.append({"review_id": review_id, "source_id": row["id"],
                     "stratum": stratum, "model_X": x_model, "model_Y": y_model})
    output.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(blinded)
    for annotator in ["A", "B"]:
        destination = output / f"mt5_mle_annotator_{annotator}.csv"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite annotation file: {destination}")
    for annotator in ["A", "B"]:
        template.to_csv(output / f"mt5_mle_annotator_{annotator}.csv", index=False,
                        encoding="utf-8-sig")
    pd.DataFrame(keys).to_csv(output / "PRIVATE_blinding_key.csv", index=False)
    manifest = {
        "status": "awaiting_human_labels", "seed": seed,
        "random_passages": sample_size, "additional_no_eos_passages": failure_size,
        "total_passages": len(template), "candidates_per_passage": 2,
        "template_sha256": hashlib.sha256(
            (output / "mt5_mle_annotator_A.csv").read_bytes()).hexdigest(),
        "primary_sampling": "Uniform without replacement from all frozen test passages.",
        "diagnostic_sampling": "Uniform from no-EOS passages not already in the random sample.",
        "diagnostic_guardrail": "Report separately; do not pool with the random sample for accuracy.",
        "full_outputs": True,
    }
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def agreement(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Agreement needs equal, nonempty label sequences.")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(left.count(label) * right.count(label) / len(left) ** 2
                   for label in set(left) | set(right))
    return {"items": len(left), "observed_agreement_percent": 100 * observed,
            "cohen_kappa": None if expected == 1 else (observed - expected) / (1 - expected)}


def summarize(annotator_a, annotator_b, template, key):
    frames = []
    for completed in [annotator_a, annotator_b]:
        if set(completed.columns) != set(template.columns):
            raise ValueError("Returned annotation columns do not match the template.")
        if not completed["review_id"].is_unique or set(completed["review_id"]) != set(template["review_id"]):
            raise ValueError("Returned review IDs do not match the template.")
        completed = completed.set_index("review_id").loc[template["review_id"]].reset_index()
        if not completed[CONTENT].equals(template[CONTENT].reset_index(drop=True)):
            raise ValueError("Source, reference, or candidate text changed during annotation.")
        for field in ["judgment_X", "judgment_Y"]:
            completed[field] = completed[field].str.strip().str.lower()
            if not completed[field].isin(VALID_LABELS).all():
                raise ValueError(f"Complete {field} using acceptable, inaccurate, or uncertain.")
        completed["preferred_candidate"] = completed["preferred_candidate"].str.strip().replace(
            {"x": "X", "y": "Y"})
        if not completed["preferred_candidate"].isin(PREFERENCES).all():
            raise ValueError("Complete preferred_candidate using X, Y, tie, or uncertain.")
        frames.append(completed)
    if not key["review_id"].is_unique or set(key["review_id"]) != set(template["review_id"]):
        raise ValueError("Blinding key IDs do not match the template.")
    key = key.set_index("review_id").loc[template["review_id"]].reset_index()
    results = {"status": "independent_labels_completed_not_adjudicated", "strata": {}}
    for stratum in key["stratum"].unique():
        selected = key.index[key["stratum"].eq(stratum)]
        per_model = {}
        for model in ["MLE + regex", "mT5-small"]:
            labels = [[], []]
            for index in selected:
                candidate = "X" if key.at[index, "model_X"] == model else "Y"
                for annotator, frame in enumerate(frames):
                    labels[annotator].append(frame.at[index, f"judgment_{candidate}"])
            per_model[model] = {
                "annotator_A": {label: labels[0].count(label) for label in sorted(VALID_LABELS)},
                "annotator_B": {label: labels[1].count(label) for label in sorted(VALID_LABELS)},
                "agreement": agreement(*labels),
            }
        preferences = []
        preference_labels = []
        for frame in frames:
            counts = dict.fromkeys(["MLE + regex", "mT5-small", "tie", "uncertain"], 0)
            model_choices = []
            for index in selected:
                choice = frame.at[index, "preferred_candidate"]
                model_choice = key.at[index, f"model_{choice}"] if choice in {"X", "Y"} else choice
                counts[model_choice] += 1
                model_choices.append(model_choice)
            preferences.append(counts)
            preference_labels.append(model_choices)
        results["strata"][stratum] = {"passages": len(selected), "per_model": per_model,
                                      "preferences_by_annotator": preferences,
                                      "preference_agreement": agreement(*preference_labels),
                                      "preference_confusion": [
                                          {"annotator_A": left, "annotator_B": right,
                                           "count": sum(a == left and b == right for a, b in
                                                        zip(*preference_labels))}
                                          for left in preferences[0] for right in preferences[1]
                                      ]}
    results["guardrails"] = [
        "No automatic consensus or relabeling; disagreements require explicit adjudication.",
        "Report the additional no-EOS diagnostic sample separately from the random sample.",
        "Judgments concern generated-corpus outputs, not naturally authored Arabizi transfer.",
    ]
    return results


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-predictions", type=Path, required=True)
    prepare_parser.add_argument("--mt5-predictions", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--sample-size", type=int, default=50)
    prepare_parser.add_argument("--failure-size", type=int, default=10)
    prepare_parser.add_argument("--seed", type=int, default=42)
    score_parser = subparsers.add_parser("summarize")
    for name in ["annotator-a", "annotator-b", "template", "key", "output"]:
        score_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(load_predictions(args.baseline_predictions, args.mt5_predictions),
                         args.output_dir, args.sample_size, args.failure_size, args.seed)
    else:
        result = summarize(read_csv(args.annotator_a), read_csv(args.annotator_b),
                           read_csv(args.template), read_csv(args.key))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
