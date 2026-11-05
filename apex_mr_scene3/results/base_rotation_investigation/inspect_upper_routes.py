"""Export frozen baseline cases and optionally probe IK in isolated port 23070.

Results are research candidates, not replacements for the accepted scene.
No production source, input result, or scene is saved by this script.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent
BASELINE = ROOT/'backups/base_rotation_baseline_20261104_01'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def export_cases():
    source = BASELINE/'results/apex'
    tasks = read(source/'reference_tasks.json')
    cfg = read(source/'source_signature.json')['resolved_scene']
    execution = read(source/'execution.json')
    q = {side: deepcopy(cfg['arms'][side]['home_q']) for side in ('A', 'B')}
    held = {'A': None, 'B': None}
    placed = deepcopy(cfg['initial'])
    articulation = deepcopy(cfg.get('initial_articulations', {'SupportLift': 0.}))
    records = {}

    def snapshot():
        return deepcopy({'q': q, 'payloads': held, 'placed': placed,
                         'articulations': articulation})

    for task in tasks:
        before = snapshot()
        assert np.max(abs(np.asarray(task['q_path'][0])-q[task['arm']])) < 1e-8
        assert (task.get('payload') or None) == (held[task['arm']] or None)
        q[task['arm']] = deepcopy(task['q_path'][-1])
        effects = task['metadata'].get('effects', {})
        for part, pose in effects.get('placed', {}).items():
            if pose is None:
                placed.pop(part, None)
            else:
                placed[part] = deepcopy(pose)
        for side, payload in effects.get('payloads', {}).items():
            held[side] = deepcopy(payload) if payload else None
        articulation.update(effects.get('articulations', {}))
        event_rows = [e for e in execution['events'] if e.get('task_id') == task['id']]
        starts = [e['time'] for e in event_rows if e['kind'] == 'node_started']
        ends = [e['time'] for e in event_rows if e['kind'] == 'node_completed']
        records[task['id']] = {'task': deepcopy(task), 'before': before, 'after': snapshot(),
                               'actual_start_s': min(starts) if starts else None,
                               'actual_end_s': max(ends) if ends else None}
    windows = {}
    for name, first, last in [
        ('initial_support', 'pick_support_approach', 'support_high_approach'),
        ('support_return', 'lift_extracted_support', 'return_support_retreat_contact_clear'),
        ('gate_pick', 'pick_gate_approach', 'close_gate_approach'),
    ]:
        names = list(records)
        selected = names[names.index(first):names.index(last)+1]
        windows[name] = {'before': records[first]['before'], 'after': records[last]['after'],
                         'task_ids': selected, 'tasks': [records[key] for key in selected]}
    out = {'status': 'frozen_accepted_baseline_inputs_not_candidate_success',
           'run_id': execution['run_id'], 'actual_makespan_s': execution['makespan'],
           'source_directory': str(source),
           'reference_sha256': hashlib.sha256((source/'reference_tasks.json').read_bytes()).hexdigest(),
           'windows': windows,
           'context_rule': 'Use payload transforms in preference to stale stored source poses of held parts. Preserve exact contacts and articulation state from each task.',
           'source_scene': str(BASELINE/'scenes/10_Scene3_Dynamic.ttt'),
           'all_reference_tasks': records}
    write(LAB/'accepted_cases.json', out)
    return out


def run_probe(cases):
    from src.sim_connection import Simulator
    from apex.scene import resolve
    from apex.motion import MotionPlanner

    report = {'status': 'endpoint_IK_and_geometry_probe_only', 'collision_free_path_proved': False,
              'source_run_id': cases['run_id'], 'port': 23070, 'probes': [],
              'backend_sha256': hashlib.sha256((ROOT/'apex/motion_backend.lua').read_bytes()).hexdigest()}
    existing_file = LAB/'upper_endpoint_probe.json'
    if existing_file.exists():
        prior = read(existing_file)
        if prior.get('backend_sha256') == report['backend_sha256']:
            report['probes'] = prior.get('probes', [])
    owner = Simulator(LAB, port=23070, timeout=30.)
    start = time.perf_counter()
    try:
        owner.start(headless=True)
        client, sim = owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg = resolve(sim)
        report['loaded_scene_sha256'] = hashlib.sha256(Path(cases['source_scene']).read_bytes()).hexdigest()
        with MotionPlanner(sim, client, cfg) as planner:
            requests = []
            all_tasks = cases['all_reference_tasks']
            for label in ('pick_support_approach', 'pick_support_contact', 'pick_support_lift',
                          'return_support_approach'):
                record = all_tasks[label]
                fk = planner.fk('B', record['task']['q_path'][-1])
                assert fk['ok']
                requests.append((label, record, fk['pose']))
            support_quat = Rotation.from_euler('y', -math.pi/2).as_quat().tolist()
            for phase, anchor in [('initial', 'support_route_front'), ('return', 'return_support_approach')]:
                record = all_tasks[anchor]
                for height in (1.32, 1.38, 1.44):
                    for x,y in ((.512, .43), (.45, .13), (.40, -.20)):
                        requests.append((f'{phase}_high_{x}_{y}_{height}', record, [x,y,height]+support_quat))
            for label, record, tcp in requests:
                if any(p['label'] == label for p in report['probes']):
                    continue
                task, state = record['task'], record['before']
                options = deepcopy(task['metadata'].get('options', {}))
                options.update(other_q=state['q']['A'], other_payload=state['payloads']['A'] or {},
                               contact_exclusions=task['metadata']['contact_exclusions'], contact_tolerance=.002,
                               articulations=state['articulations'], max_solutions=16, ik_time=.25)
                seeds = [('current', state['q']['B']), ('accepted_endpoint', task['q_path'][-1]),
                         ('free', cfg['free_q']['B'])]
                answers = []
                for seed_name, seed in seeds:
                    result = planner.solve_ik('B', tcp[:3], tcp[3:], task.get('payload'),
                                              task['metadata']['placed'], {**options, 'seed': seed})
                    candidates = []
                    solutions = result.get('solutions', [])
                    # Empty Lua tables may arrive as {}, including a valid
                    # negative IK result. Preserve that failure as evidence.
                    for qgoal in (solutions if isinstance(solutions, list) else [])[:6]:
                        answer = planner.fk('B', qgoal)
                        actual = answer['pose']
                        error = float(np.linalg.norm(np.asarray(actual[:3])-tcp[:3]))
                        angle = 2*math.acos(min(1., abs(float(np.dot(actual[3:],tcp[3:])))))
                        candidates.append({'q': qgoal, 'position_error_m': error, 'orientation_error_rad': angle,
                                           'raw_joint_delta': (np.asarray(qgoal)-state['q']['B']).tolist(),
                                           'raw_base_delta_deg': math.degrees(qgoal[0]-state['q']['B'][0])})
                    answers.append({'seed': seed_name, 'ok': result.get('ok', False),
                                    'problem': result.get('problem'), 'stats': result.get('stats'),
                                    'solutions': candidates})
                row = {'label': label, 'tcp_pose': tcp, 'reference_task': task['id'],
                       'other_q': options['other_q'], 'payloads': state['payloads'],
                       'articulations': options['articulations'], 'answers': answers,
                       'reachable_sampled_solution_found': any(a['ok'] for a in answers)}
                report['probes'].append(row)
                write(LAB/'upper_endpoint_probe.json', report)
                print(json.dumps({'probe': label, 'found': row['reachable_sampled_solution_found'],
                                  'solutions': sum(len(a['solutions']) for a in answers)}), flush=True)
        report['wall_time_s'] = time.perf_counter()-start
        report['status'] = 'completed_endpoint_IK_and_geometry_probe_only'
        write(LAB/'upper_endpoint_probe.json', report)
    finally:
        owner.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    cases = export_cases()
    print(json.dumps({'exported': str(LAB/'accepted_cases.json'), 'run_id': cases['run_id']}), flush=True)
    if args.probe:
        run_probe(cases)
