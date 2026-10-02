"""Verify recovered preprocessing against authorized local frozen exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def verify(corpus_path, splits_dir):
    corpus = pd.read_csv(corpus_path)
    columns = ['id', 'text_az', 'text_ar']
    train, temporary = train_test_split(corpus, test_size=0.2, random_state=42)
    val, test = train_test_split(temporary, test_size=0.5, random_state=42)
    manifest = json.loads((ROOT / 'reproducibility/run_manifest.json').read_text(encoding='utf-8'))
    assignments = {}
    with (ROOT / 'data/split_ids.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            assignments.setdefault(row['split'], []).append(row['id'])
    result = {'verification_date': '2026-10-02', 'method': 'Two sklearn train_test_split calls; test_size 0.2 then 0.5, random_state 42 for both',
              'corpus_rows': len(corpus), 'exact_duplicate_sources': int(corpus.duplicated('text_az').sum()),
              'exact_duplicate_targets': int(corpus.duplicated('text_ar').sum()),
              'exact_duplicate_pairs': int(corpus.duplicated(['text_az', 'text_ar']).sum()), 'splits': {}}
    for name, frame in [('train', train), ('val', val), ('test', test)]:
        path = splits_dir / f'toy_{name}.csv'
        frozen = pd.read_csv(path)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        result['splits'][name] = {'rows': len(frame),
            'matches_frozen_content_and_order': frame[columns].reset_index(drop=True).equals(frozen[columns].reset_index(drop=True)),
            'matches_published_id_membership': set(frame['id'].astype(str)) == set(assignments[name]),
            'csv_sha256': actual_hash, 'matches_frozen_byte_hash': actual_hash == manifest['protocol']['split_sha256'][name]}
    result['limitations'] = ['No original upstream retrieval, source revision, subset-selection or corpus-deduplication operation preserved.', 'Row splitting does not establish source-document independence or an untouched rule-development history.', 'CSV byte serialization is separate from decoded content equality.']
    result['passed'] = all(s['matches_frozen_content_and_order'] and s['matches_published_id_membership'] for s in result['splits'].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/local_split_verification.json')
    args = parser.parse_args()
    result = verify(args.corpus, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Reconstructed split content or assignments differ from frozen records.')


if __name__ == '__main__':
    main()
