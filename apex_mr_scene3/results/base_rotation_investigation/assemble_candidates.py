"""Freeze selected native-search paths as inputs, never as execution evidence."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apex.local_routes import _sha, _value_hash, WINDOWS
from apex.local_route_chain import replace_window

LAB = Path(__file__).resolve().parent
BASE = ROOT/'backups/base_rotation_baseline_20261104_01/results/apex'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pick-file', type=Path, default=LAB/'support_pick_side_candidates.json')
    parser.add_argument('--pick-index', type=int, default=0)
    parser.add_argument('--return-file', type=Path, default=LAB/'support_return_assembled_candidates.json')
    parser.add_argument('--return-index', type=int, default=0)
    args = parser.parse_args()
    pick = read(args.pick_file)['candidates'][args.pick_index]
    assert pick['accepted'] is True
    returns = read(args.return_file)['candidates']
    assert returns, 'No complete native return candidate exists'
    back = returns[args.return_index]
    assert back.get('accepted_native_reference', back.get('accepted')) is True
    gate_file = LAB/'gate_short_lift_candidates.json'
    gate = next(row for row in read(gate_file) if abs(row['height_m']-.1) < 1e-9)
    assert gate['lift_dense_valid'] is True and gate['reconnect']['ok'] is True
    reference = read(BASE/'alloc_and_reference.json')
    originals = {t['id']: t for t in reference['tasks']}
    all_paths = {
        'support_pick': pick['replacements'],
        'support_return': back['replacements'],
        'gate_lift': {'pick_gate_lift': gate['lift_path'], 'close_gate_approach': gate['reconnect']['path']},
    }
    methods = {'support_route_front': 'overhead_waypoints', 'pick_gate_lift': 'cartesian_continuation',
               'return_support_retreat': 'cartesian_continuation', 'pick_gate_approach': 'overhead_waypoints'}
    windows = []
    tasks = deepcopy(reference['tasks'])
    checks = []
    for name, (first, last) in WINDOWS.items():
        paths = {ident: path for ident, path in all_paths[name].items() if originals[ident]['kind'] == 'transit'}
        tasks, check = replace_window(tasks, first, last, paths)
        checks.append(check)
        windows.append({'name': name, 'paths': paths,
                        'methods': {ident: methods.get(ident, 'joint_reconnect') for ident in paths}})
    physical = {'scenes/10_Scene3_Dynamic.ttt': _sha(ROOT/'scenes/10_Scene3_Dynamic.ttt'),
                'apex/motion_backend.lua': _sha(ROOT/'apex/motion_backend.lua')}
    physical.update({p.relative_to(ROOT).as_posix(): _sha(p) for p in (ROOT/'config').glob('*.json')})
    asset = {'schema_version': 1, 'baseline_run_id': '20261104T163855Z-0804c434',
             'reference_signature': _value_hash(reference['tasks']), 'physical_input_hashes': physical,
             'scope': 'Frozen geometric candidates. Every changed task requires current native validation and full execution.',
             'search_evidence_hashes': {str(p.relative_to(ROOT)): _sha(p) for p in (args.pick_file, args.return_file, gate_file)},
             'windows': windows}
    (ROOT/'apex/local_route_candidates.json').write_text(json.dumps(asset, indent=2, allow_nan=False), encoding='utf-8')
    (LAB/'reference_tasks_candidate.json').write_text(json.dumps(tasks, indent=2, allow_nan=False), encoding='utf-8')
    (LAB/'candidate_structure.json').write_text(json.dumps(checks, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'frozen_windows': checks, 'tasks': len(tasks), 'actual_execution_pending': True}, indent=2))


if __name__ == '__main__':
    main()
