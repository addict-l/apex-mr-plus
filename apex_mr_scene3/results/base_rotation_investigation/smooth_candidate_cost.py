"""Independent native C2 / timing quick-check for local route candidates.

Only writes a report beside this script. This is a fixed-reference-world
candidate check, not an asynchronous TPG, dynamics or delivery certificate.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from apex.program import parameterize
from apex.smoothing import smooth_cartesian_macros
from apex.tpg import build_tpg


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def fingerprint():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT/'apex').glob('*') if path.suffix in ('.py', '.lua')}


def check_task(planner, cfg, record, path, method):
    task = deepcopy(record['task'])
    task['q_path'] = deepcopy(path)
    task['dependencies'] = []
    task['reference_order'] = 0
    task['metadata']['local_route_candidate'] = {
        'window': 'gate_lift', 'method': method,
        'certificate_scope': 'native dense fixed-reference-state',
        'require_profile_validation': True,
    }
    local_cfg = deepcopy(cfg)
    for arm in ('A', 'B'):
        local_cfg['arms'][arm]['home_q'] = deepcopy(record['before']['q'][arm])
    local_cfg['arms'][task['arm']]['home_q'] = deepcopy(path[0])
    local_cfg['initial_payloads'] = deepcopy(record['before']['payloads'])
    # Independently recheck the unsmoothed candidate too: a sharp reconnect
    # may correctly be rejected by C2 without invoking its geometry checker.
    other = 'A' if task['arm'] == 'B' else 'B'
    options = deepcopy(task['metadata'].get('options', {}))
    options['contact_exclusions'] = deepcopy(task['metadata']['contact_exclusions'])
    options['contact_tolerance'] = task['metadata'].get('contact_tolerance', .002)
    dense = [path[0]]
    for left, right in zip(path, path[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(right)-left)/.003)))
        dense.extend((np.asarray(left)+(np.asarray(right)-left)*index/count).tolist()
                     for index in range(1, count+1))
    reference_proof = {'accepted': True, 'step_rad': .003, 'sample_count': len(dense),
                       'samples_checked': 0, 'first_failure': None,
                       'scope': 'entire candidate polyline in fixed task reference context'}
    for at in range(0, len(dense), 128):
        states = [{task['arm']: q, other: record['before']['q'][other],
                   'payloads': deepcopy(record['before']['payloads']),
                   'placed': deepcopy(task['metadata']['placed']), 'options': deepcopy(options)}
                  for q in dense[at:at+128]]
        answers = planner.sample_state_pairs(states)
        if len(answers) != len(states):
            raise RuntimeError('Native candidate validation returned incomplete sample batch')
        for index, answer in enumerate(answers):
            reference_proof['samples_checked'] += 1
            if answer.get('ok') is not True:
                reference_proof['accepted'] = False
                reference_proof['first_failure'] = {'sample': at+index, 'evidence': answer}
                break
        if not reference_proof['accepted']:
            break
    reference_proof['start_fk'] = planner.fk(task['arm'], path[0])
    reference_proof['end_fk'] = planner.fk(task['arm'], path[-1])
    original = build_tpg([task], max_joint_span=.24)
    parameterize(original)
    graph = build_tpg([task], max_joint_span=.24)
    started = time.perf_counter()
    smoothing = smooth_cartesian_macros(graph, planner, local_cfg)
    parameterize(graph)
    difference = np.diff(np.asarray(path), axis=0)
    return {
        'task_id': task['id'], 'candidate_waypoints': len(path),
        'joint_path_length_rad': float(np.linalg.norm(difference, axis=1).sum()),
        'joint_total_travel_rad': np.abs(difference).sum(axis=0).tolist(),
        'parameterized_original_motion_s': sum(n['duration'] for n in original['nodes']),
        'parameterized_candidate_motion_s': sum(n['duration'] for n in graph['nodes']),
        'candidate_node_count': len(graph['nodes']),
        'pass_through_boundaries': graph['time_parameterization']['pass_through_boundaries'],
        'timing_scope': 'isolated original task; no physical settling, schedule or dynamics',
        'start_q': path[0], 'end_q': path[-1],
        'native_validation_and_timing_wall_s': time.perf_counter()-started,
        'candidate_polyline_validation': reference_proof,
        'smoothing': smoothing,
    }


def run(source, output):
    from src.sim_connection import Simulator
    from apex.scene import resolve
    from apex.motion import MotionPlanner

    candidates = read(source)
    cases = read(LAB/'accepted_cases.json')
    records = cases['all_reference_tasks']
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    report = {
        'scope': 'native stopped scene; fixed reference other-arm and payload world',
        'candidate_only': True, 'actual_pending': True,
        'full_asynchronous_oracle_pending': True, 'rollout_pending': True,
        'source_run_id': cases['run_id'], 'candidate_file': str(Path(source).resolve()),
        'candidate_sha256': source_hash, 'source_sha256_before': fingerprint(),
        'scene_path': cases['source_scene'], 'port': 23070, 'candidates': [],
    }
    owner = Simulator(LAB, port=23070, timeout=30.)
    started = time.perf_counter()
    try:
        owner.start(headless=True)
        client, sim = owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg = resolve(sim)
        with MotionPlanner(sim, client, cfg) as planner:
            for candidate in candidates:
                parts = []
                parts.append(check_task(planner, cfg, records['pick_gate_lift'],
                                        candidate['lift_path'], 'cartesian_continuation'))
                reconnect = candidate.get('reconnect', {})
                if reconnect.get('ok') is True and reconnect.get('path'):
                    parts.append(check_task(planner, cfg, records['close_gate_approach'],
                                            reconnect['path'], 'joint_reconnect'))
                row = {'height_m': candidate['height_m'], 'parts': parts,
                       'combined_parameterized_candidate_motion_s': sum(
                           p['parameterized_candidate_motion_s'] for p in parts),
                       'combined_parameterized_original_motion_s': sum(
                           p['parameterized_original_motion_s'] for p in parts),
                       'chain_boundary_max_joint_error_rad': float(np.max(abs(
                           np.asarray(parts[0]['end_q'])-parts[1]['start_q']))) if len(parts)==2 else None,
                       'reconnect_present': len(parts)==2}
                report['candidates'].append(row)
                write(output, report)
                print(json.dumps({'height_m': row['height_m'],
                                  'motion_s': row['combined_parameterized_candidate_motion_s'],
                                  'accepted_C2': [p['smoothing']['accepted_local_route_tasks'] for p in parts]}),
                      flush=True)
        report['source_sha256_after'] = fingerprint()
        report['source_changed_during_check'] = report['source_sha256_before'] != report['source_sha256_after']
        report['candidate_file_changed_during_check'] = source_hash != hashlib.sha256(Path(source).read_bytes()).hexdigest()
        report['wall_time_s'] = time.perf_counter()-started
        report['completed'] = True
        write(output, report)
    finally:
        owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=LAB/'gate_short_lift_candidates.json')
    parser.add_argument('--output', type=Path, default=LAB/'gate_candidate_profile_cost.json')
    args = parser.parse_args()
    run(args.input, args.output)
