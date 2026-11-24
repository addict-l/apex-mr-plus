"""Reclassify immutable retained failures; this is not a motion execution."""
import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simbridge.backend import _failure_classification
from simbridge.compiler import CompileFailure
from tools.motion_v135_control import BASE
from tools.overnight_batch import atomic_json, read_json


def main():
    rows = []
    for case in read_json(BASE/'m1/baseline_cases.json'):
        for old in case['failures']:
            failure = CompileFailure(old.get('original_status') or old['status'],
                old['detail'], old.get('failed_segment'), old.get('failure_evidence'))
            status, evidence = _failure_classification(failure)
            rows.append({'case': case['id'], 'record': old['index'], 'action': old['action'],
                'segment': old.get('failed_segment'), 'old_status': old['status'],
                'new_status': status.value, 'old_classification': (old.get('classification') or {}).get('kind'),
                'new_classification': evidence['kind'], 'scope': evidence['failure_scope'],
                'source': case['source'], 'evidence': evidence})
    output = BASE/'m2'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'retained_failure_classification.json', {'scope':
        'Classification-only replay of retained records; no new path or execution success is claimed',
        'records': rows, 'new_kinds': dict(Counter(row['new_classification'] for row in rows))})
    columns = [key for key in rows[0] if key != 'evidence']
    with (output/'retained_failure_classification.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(dict(Counter(row['new_classification'] for row in rows)))


if __name__ == '__main__':
    main()
