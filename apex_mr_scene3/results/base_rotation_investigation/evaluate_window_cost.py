"""Native local-window C2 validation and real production timing comparison."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
from apex.tpg import build_tpg
from apex.smoothing import smooth_cartesian_macros
from apex.program import parameterize
from apex.scene import resolve
from apex.motion import MotionPlanner
from src.sim_connection import Simulator


def run(source, output, window):
    data=json.loads(source.read_text(encoding='utf-8'))
    cases=json.loads((LAB/'accepted_cases.json').read_text(encoding='utf-8'))
    baseline=json.loads((ROOT/'backups/base_rotation_baseline_20261104_01/results/apex/tpg.json').read_text())
    report={'candidate_file':str(source),'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_run_id':cases['run_id'],'scope':'isolated window production parameterization, excludes servo settling and asynchronous schedule',
            'actual_pending':True,'full_TPG_certificate_pending':True,'candidates':[]}
    owner=Simulator(LAB,port=23070,timeout=30.)
    began=time.perf_counter()
    try:
        owner.start(headless=True)
        client,sim=owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg=resolve(sim)
        with MotionPlanner(sim,client,cfg) as planner:
            for candidate in data['candidates']:
                ids=list(candidate['replacements'])
                records=candidate.get('context_records',cases['all_reference_tasks'])
                before=records[ids[0]]['before']
                cfg['initial_payloads']=deepcopy(before['payloads'])
                for arm in ('A','B'): cfg['arms'][arm]['home_q']=deepcopy(before['q'][arm])
                tasks=[]
                for key in ids:
                    task=deepcopy(records[key]['task'])
                    task['q_path']=candidate['replacements'][key]
                    task['dependencies']=[d for d in task['dependencies'] if d in ids]
                    if task['kind']=='transit':
                        task['metadata']['local_route_candidate']={
                            'window':window,'method':'overhead_waypoints',
                            'certificate_scope':'native dense fixed-reference-state','require_profile_validation':True}
                    tasks.append(task)
                graph=build_tpg(tasks,max_joint_span=.24)
                plain=deepcopy(graph)
                parameterize(plain)
                smooth=smooth_cartesian_macros(graph,planner,cfg)
                parameterize(graph)
                bytask={key:sum(n['duration']for n in graph['nodes']if n['task_id']==key)for key in ids}
                old={key:sum(n['duration']for n in baseline['nodes']if n['task_id']==key)for key in ids}
                item={'label':candidate['label'],'task_ids':ids,
                      'candidate_duration_by_task_s':bytask,'accepted_baseline_duration_by_task_s':old,
                      'candidate_window_motion_and_dwell_s':sum(bytask.values()),
                      'accepted_baseline_window_motion_and_dwell_s':sum(old.values()),
                      'candidate_without_C2_motion_and_dwell_s':sum(n['duration']for n in plain['nodes']),
                      'smoothing':smooth,'parameterization':graph['time_parameterization'],
                      'original_candidate_native_checks':'recorded by source producer; this quick-check independently validates every C2 candidate',
                      'actual_pending':True}
                report['candidates'].append(item)
                output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                print(json.dumps({'label':item['label'],'candidate_s':item['candidate_window_motion_and_dwell_s'],
                                  'old_s':item['accepted_baseline_window_motion_and_dwell_s'],
                                  'accepted_C2':smooth['accepted_local_route_tasks']}),flush=True)
        report.update(completed=True,wall_time_s=time.perf_counter()-began)
        report['candidate_file_changed_during_check']=report['candidate_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest()
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    finally:
        owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=LAB/'support_pick_approach_candidates.json')
    p.add_argument('--output',type=Path,default=LAB/'support_pick_profile_cost.json')
    p.add_argument('--window',default='support_pick')
    a=p.parse_args()
    run(a.input,a.output,a.window)
