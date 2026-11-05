"""Refine the empty-arm departure after a verified negative-winding return."""
from copy import deepcopy
from pathlib import Path
import json
import math
import time
import numpy as np

from support_return_probe import ROOT, LAB, read, write, hashes, shifted, options, check_path
from apex.tpg import build_tpg
from apex.smoothing import smooth_cartesian_macros
from apex.program import parameterize
from apex.motion import MotionPlanner
from apex.scene import resolve
from src.sim_connection import Simulator


def run():
    cases = read(LAB/'accepted_cases.json')
    records = cases['all_reference_tasks']
    ids = ['return_support_approach', 'return_support_contact', 'return_support',
           'return_support_retreat', 'return_support_retreat_contact_clear', 'pick_gate_approach']
    first = read(LAB/'support_return_candidates.json')
    if first.get('status') == 'refined_complete' and not first.get('candidates'):
        write(LAB/'support_return_vertical_departure_failures.json', first)
    if first.get('status') != 'refined_complete':
        write(LAB/'support_return_initial_search.json', first)
    initial = read(LAB/'support_return_initial_search.json')
    direct = next(a for a in initial['attempts'] if a['label']=='direct_negative_winding')['solutions'][0]['plans'][0]
    assert direct['ok']
    fixed = {key: shifted(records[key]['task']['q_path']) for key in ids[1:-1]}
    approach = direct['path']
    start, goal = fixed[ids[-2]][-1], records[ids[-1]]['task']['q_path'][-1]
    rec = records[ids[-1]]
    report = {'status': 'refining_empty_departure', 'source_run_id': cases['run_id'],
              'candidate_only': True, 'actual_pending': True, 'full_TPG_certificate_pending': True,
              'source_sha256_before': hashes(), 'before': records[ids[0]]['before'],
              'task_ids': ids, 'attempts': [], 'candidates': []}
    out = LAB/'support_return_candidates.json'
    write(out, report)
    owner = Simulator(LAB, port=23070, timeout=30.)
    began = time.perf_counter()
    try:
        owner.start(headless=True)
        client, sim = owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg = resolve(sim)
        for arm in ('A','B'):
            cfg['arms'][arm]['home_q'] = records[ids[0]]['before']['q'][arm]
        cfg['initial_payloads'] = records[ids[0]]['before']['payloads']
        with MotionPlanner(sim, client, cfg) as planner:
            start_tcp = planner.fk('B', start)['pose']
            print(json.dumps({'started': True, 'start_tcp': start_tcp}), flush=True)
            def planned(a, b, margin=.25):
                opts = options(rec)
                opts.update(joint_bound_margin=margin, max_rrt_candidates=2)
                return planner.plan('B', a, b, None, rec['task']['metadata']['placed'],
                                    solve_time=1.2, options=opts)

            for offset_x, offset_y, height in ((.08,0,1.035),(.12,0,1.035),(.08,0,1.20),
                                               (.12,0,1.20),(0,-.10,1.035),(0,-.10,1.20)):
                pose = [start_tcp[0]+offset_x,start_tcp[1]+offset_y,height,*start_tcp[3:]]
                opts = options(rec)
                opts.update(seed=start, multiple=False, max_solutions=1)
                ik = planner.solve_ik('B', pose[:3], pose[3:], None, rec['task']['metadata']['placed'], opts)
                attempt = {'height_m': height, 'offset_xy_m':[offset_x,offset_y],
                           'tcp': pose, 'ik': ik, 'plans': [], 'accepted': False}
                report['attempts'].append(attempt)
                write(out, report)
                if not ik.get('ok') or not ik.get('solutions'):
                    continue
                raw = ik['solutions'][0]
                qhigh = (np.asarray(start)+np.arctan2(np.sin(np.asarray(raw)-start),np.cos(np.asarray(raw)-start))).tolist()
                lift = planned(start, qhigh)
                attempt['plans'].append(lift)
                write(out, report)
                if not lift.get('ok'):
                    continue
                for margin in (.15,.4):
                    reconnect = planned(qhigh, goal, margin)
                    attempt['plans'].append(reconnect)
                    write(out, report)
                    if not reconnect.get('ok'):
                        continue
                    outgoing = lift['path']+reconnect['path'][1:]
                    base_travel = sum(abs(b[0]-a[0]) for a,b in zip(outgoing,outgoing[1:]))
                    if base_travel > math.pi+1e-7:
                        attempt['rejected_base_travel_degrees'] = math.degrees(base_travel)
                        continue
                    replacements = {ids[0]: approach, **fixed, ids[-1]: outgoing}
                    checks = {key: check_path(planner, records[key], path) for key,path in replacements.items()}
                    attempt['candidate_checks'] = checks
                    if not all(v['ok'] for v in checks.values()):
                        continue
                    tasks = []
                    for key in ids:
                        task = deepcopy(records[key]['task'])
                        task['q_path'] = replacements[key]
                        task['dependencies'] = [d for d in task['dependencies'] if d in ids]
                        if task['kind']=='transit':
                            task['metadata']['local_route_candidate'] = {
                                'window': 'support_return', 'method': 'overhead_waypoints' if key==ids[-1] else 'joint_reconnect',
                                'certificate_scope': 'native dense fixed-reference-state', 'require_profile_validation': True}
                        tasks.append(task)
                    graph = build_tpg(tasks, max_joint_span=.24)
                    smoothing = smooth_cartesian_macros(graph, planner, cfg)
                    parameterize(graph)
                    travels = {key: np.abs(np.diff(np.asarray(path),axis=0)).sum(axis=0).tolist()
                               for key,path in replacements.items()}
                    candidate = {'label': f'negative_winding_return_empty_exit_{offset_x}_{offset_y}_{height}_margin_{margin}',
                        'replacements': replacements, 'tasks': tasks, 'before': report['before'],
                        'geometry_checks': checks, 'smoothing': smoothing,
                        'joint_travel_by_task_rad': travels,
                        'total_base_travel_degrees': math.degrees(sum(v[0] for v in travels.values())),
                        'parameterized_local_motion_and_dwell_s': sum(n['duration'] for n in graph['nodes']),
                        'parameterization': graph['time_parameterization'],
                        'chain_max_joint_error_rad': max(float(np.max(abs(np.asarray(a['q_path'][-1])-b['q_path'][0])))
                                                        for a,b in zip(tasks,tasks[1:])),
                        'start_anchor': records[ids[0]]['task']['q_path'][0], 'end_anchor': goal,
                        'accepted_native_reference': True, 'actual_pending': True,
                        'full_TPG_certificate_pending': True}
                    report['candidates'].append(candidate)
                    attempt['accepted'] = True
                    write(out, report)
                    print(json.dumps({'accepted': candidate['label'],'seconds':candidate['parameterized_local_motion_and_dwell_s'],
                                      'base_degrees':candidate['total_base_travel_degrees']}),flush=True)
                    break
            report.update(status='refined_complete', wall_time_s=time.perf_counter()-began,
                          source_sha256_after=hashes())
            report['source_changed_during_search'] = report['source_sha256_before'] != report['source_sha256_after']
            write(out, report)
    finally:
        owner.close()


if __name__ == '__main__':
    run()
