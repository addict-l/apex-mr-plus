"""Independent complete support-return candidate search, isolated native port.

Writes only this experiment's candidate evidence. No simulation execution,
scene save, production mutation, or asynchronous safety claim is performed.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
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
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'apex').glob('*') if p.suffix in ('.py', '.lua')}


def shifted(path):
    out = deepcopy(path)
    for q in out:
        q[0] -= 2*math.pi
    return out


def options(record):
    md, state = record['task']['metadata'], record['before']
    opts = deepcopy(md.get('options', {}))
    opts.update(other_q=state['q']['A'], other_payload=state['payloads']['A'] or {},
                contact_exclusions=md['contact_exclusions'],
                contact_tolerance=md.get('contact_tolerance', .002),
                max_joint_step=.003, max_rrt_candidates=2, shortcut_budget_s=.35)
    return opts


def plan(planner, record, start, end):
    task = record['task']
    return planner.plan('B', start, end, task.get('payload'), task['metadata']['placed'],
                        solve_time=1.2, options=options(record))


def check_path(planner, record, path):
    dense = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(np.asarray(b)-a))/.003))
        dense.extend((np.asarray(a)+(np.asarray(b)-a)*i/n).tolist() for i in range(1, n+1))
    proof = {'ok': True, 'samples': len(dense), 'checked': 0, 'step_rad': .003}
    for at in range(0, len(dense), 128):
        batch = [{'B': q, 'A': record['before']['q']['A'],
                  'payloads': deepcopy(record['before']['payloads']),
                  'placed': deepcopy(record['task']['metadata']['placed']),
                  'options': options(record)} for q in dense[at:at+128]]
        answers = planner.validate_samples(batch)
        assert len(answers) == len(batch)
        for k, answer in enumerate(answers):
            proof['checked'] += 1
            if answer.get('ok') is not True:
                proof.update(ok=False, first_failure={'sample': at+k, 'answer': answer})
                return proof
    return proof


def travel(path):
    return np.abs(np.diff(np.asarray(path), axis=0)).sum(axis=0).tolist()


def run():
    from src.sim_connection import Simulator
    from apex.scene import resolve
    from apex.motion import MotionPlanner

    cases = read(LAB/'accepted_cases.json')
    records = cases['all_reference_tasks']
    ids = ['return_support_approach', 'return_support_contact', 'return_support',
           'return_support_retreat', 'return_support_retreat_contact_clear', 'pick_gate_approach']
    before = deepcopy(records[ids[0]]['before'])
    report = {'status': 'candidate_search_running', 'source_run_id': cases['run_id'],
              'scope': 'native stopped scene; complete local reference chain; fixed other arm',
              'actual_pending': True, 'asynchronous_oracle_pending': True,
              'source_sha256_before': hashes(), 'task_ids': ids,
              'before': before, 'source_scene': cases['source_scene'],
              'start_anchor': records[ids[0]]['task']['q_path'][0],
              'end_anchor': records[ids[-1]]['task']['q_path'][-1],
              'attempts': [], 'candidates': []}
    output = LAB/'support_return_candidates.json'
    write(output, report)
    owner = Simulator(LAB, port=23070, timeout=30.)
    started = time.perf_counter()
    try:
        owner.start(headless=True)
        client, sim = owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg = resolve(sim)
        for side in ('A', 'B'):
            cfg['arms'][side]['home_q'] = deepcopy(before['q'][side])
        cfg['initial_payloads'] = deepcopy(before['payloads'])
        with MotionPlanner(sim, client, cfg) as planner:
            fixed = {key: shifted(records[key]['task']['q_path']) for key in ids[1:-1]}
            fixed_checks = {key: check_path(planner, records[key], path) for key, path in fixed.items()}
            report['fixed_rebinding_checks'] = fixed_checks
            if not all(v['ok'] for v in fixed_checks.values()):
                raise RuntimeError('Shifted contact/release/retreat reference context failed')
            outgoing = plan(planner, records[ids[-1]], fixed[ids[-2]][-1], report['end_anchor'])
            report['outgoing_to_gate'] = outgoing
            if not outgoing.get('ok'):
                raise RuntimeError('No fixed-anchor outgoing route found in bounded search')
            park = fixed['return_support_contact'][0]
            start = report['start_anchor']
            tcp = planner.fk('B', park)['pose']
            proposed = [('direct_negative_winding', None)]
            for z in (1.32, 1.38):
                for x in (.30, .35):
                    proposed.append((f'upper_x{x}_y0.13_z{z}', [x, .13, z]+tcp[3:]))
            for label, pose in proposed:
                qs = [None]
                attempt = {'label': label, 'upper_pose': pose, 'solutions': []}
                report['attempts'].append(attempt)
                if pose is not None:
                    answers = []
                    for seed in (start, park):
                        opts = options(records[ids[0]])
                        opts.update(seed=seed, ik_time=.25, max_solutions=8)
                        answer = planner.solve_ik('B', pose[:3], pose[3:],
                            records[ids[0]]['task']['payload'], records[ids[0]]['task']['metadata']['placed'], opts)
                        answers.append(answer)
                    attempt['ik_answers'] = answers
                    qs = []
                    for answer in answers:
                        for raw in answer.get('solutions', []):
                            q = (np.asarray(start)+np.arctan2(np.sin(np.asarray(raw)-start),
                                                           np.cos(np.asarray(raw)-start))).tolist()
                            if q[0] > start[0]+.15 or q[0] < park[0]-.15:
                                continue
                            if not any(np.linalg.norm(np.asarray(q)-old)<1e-5 for old in qs):
                                qs.append(q)
                    qs.sort(key=lambda q: np.linalg.norm(np.asarray(q)-start)+np.linalg.norm(np.asarray(q)-park))
                    qs = qs[:3]
                for index, via in enumerate(qs):
                    if via is None:
                        pieces = [plan(planner, records[ids[0]], start, park)]
                    else:
                        pieces = [plan(planner, records[ids[0]], start, via)]
                        if pieces[0].get('ok'):
                            pieces.append(plan(planner, records[ids[0]], via, park))
                    item = {'index': index, 'via_q': via, 'plans': pieces, 'accepted': False}
                    attempt['solutions'].append(item)
                    write(output, report)
                    if not all(p.get('ok') for p in pieces) or (via is not None and len(pieces)!=2):
                        continue
                    approach = deepcopy(pieces[0]['path'])
                    for p in pieces[1:]:
                        approach.extend(deepcopy(p['path'][1:]))
                    replacements = {ids[0]: approach, **fixed, ids[-1]: outgoing['path']}
                    travel_by_task = {key: travel(path) for key, path in replacements.items()}
                    if any(value[0] > math.pi+1e-7 for value in travel_by_task.values()):
                        item['rejected_reason'] = 'some_task_base_absolute_travel_exceeds_180_degrees'
                        continue
                    paths_ok = {key: check_path(planner, records[key], path)
                                for key, path in replacements.items()}
                    item['candidate_checks'] = paths_ok
                    if not all(value['ok'] for value in paths_ok.values()):
                        continue
                    tasks = []
                    for key in ids:
                        task = deepcopy(records[key]['task'])
                        task['q_path'] = replacements[key]
                        task['dependencies'] = [d for d in task['dependencies'] if d in ids]
                        if task['kind'] == 'transit':
                            task['metadata']['local_route_candidate'] = {
                                'window': 'support_return',
                                'method': ('overhead_waypoints' if via is not None else 'joint_reconnect')
                                           if key == ids[0] else 'joint_reconnect',
                                'certificate_scope': 'native dense fixed-reference-state',
                                'require_profile_validation': True}
                        tasks.append(task)
                    graph = build_tpg(tasks, max_joint_span=.24)
                    smooth = smooth_cartesian_macros(graph, planner, cfg)
                    parameterize(graph)
                    chain_error = max(float(np.max(abs(np.asarray(a['q_path'][-1])-b['q_path'][0])))
                                      for a,b in zip(tasks,tasks[1:]))
                    candidate = {'label': label, 'solution_index': index, 'upper_pose': pose,
                                 'via_q': via, 'replacements': replacements, 'tasks': tasks,
                                 'before': before, 'fixed_anchor_error_rad': 0.,
                                 'chain_max_joint_error_rad': chain_error,
                                 'joint_travel_by_task_rad': travel_by_task,
                                 'total_base_travel_degrees': math.degrees(sum(v[0] for v in travel_by_task.values())),
                                 'parameterized_local_motion_and_dwell_s': sum(n['duration'] for n in graph['nodes']),
                                 'parameterization': graph['time_parameterization'], 'smoothing': smooth,
                                 'geometry_checks': paths_ok, 'accepted_native_reference': chain_error < 1e-8,
                                 'actual_pending': True, 'full_TPG_certificate_pending': True}
                    item['accepted'] = candidate['accepted_native_reference']
                    report['candidates'].append(candidate)
                    write(output, report)
                    print(json.dumps({'candidate': label, 'branch': index,
                        'base_travel_degrees': candidate['total_base_travel_degrees'],
                        'profile_cost_s': candidate['parameterized_local_motion_and_dwell_s'],
                        'chain_error': chain_error}), flush=True)
                write(output, report)
            report['status'] = 'completed_bounded_candidate_search'
            report['source_sha256_after'] = hashes()
            report['source_changed_during_search'] = report['source_sha256_before'] != report['source_sha256_after']
            report['wall_time_s'] = time.perf_counter()-started
            report['best_by_parameterized_time'] = min(
                report['candidates'], key=lambda c:c['parameterized_local_motion_and_dwell_s'])['label'] if report['candidates'] else None
            write(output, report)
    except Exception as exc:
        report['status'] = 'failed_bounded_candidate_search'
        report['error'] = repr(exc)
        write(output, report)
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    run()
