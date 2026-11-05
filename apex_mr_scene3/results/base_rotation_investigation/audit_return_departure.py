"""Independently check and time contact-preserving split of real departure path."""
from copy import deepcopy
import math
import time
import numpy as np
from support_return_probe import ROOT, LAB, read, write, hashes, check_path, shifted
from apex.motion import MotionPlanner
from apex.scene import resolve
from apex.tpg import build_tpg
from apex.smoothing import smooth_cartesian_macros
from apex.program import parameterize
from src.sim_connection import Simulator


def run():
    cases=read(LAB/'accepted_cases.json');records=cases['all_reference_tasks']
    low=read(LAB/'support_return_low_side_aux.json')
    raw=next(a for a in low['attempts']if a['label']=='side_to_gate'and a['margin']==.4)['result']['path']
    side=deepcopy(read(LAB/'support_pick_side_candidates.json')['candidates'][0]['replacements']['pick_support_contact'][::-1])
    for q in side:q[5]-=2*math.pi
    side[0]=shifted(records['return_support']['task']['q_path'])[-1]
    assert np.max(abs(np.asarray(side[-1])-raw[0]))<1e-5
    side[-1]=deepcopy(raw[0])
    ids=['return_support_retreat','return_support_retreat_contact_clear','pick_gate_approach']
    out=LAB/'support_return_departure_audit.json'
    report={'status':'running','source_run_id':cases['run_id'],'source_sha256_before':hashes(),
            'scope':'independent native fixed-reference chain; no actual or asynchronous acceptance',
            'actual_pending':True,'full_TPG_certificate_pending':True,'candidates':[]}
    owner=Simulator(LAB,port=23070,timeout=30.)
    started=time.perf_counter()
    try:
        owner.start(headless=True);client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
        cfg['initial_payloads']=deepcopy(records[ids[0]]['before']['payloads'])
        for arm in ('A','B'):cfg['arms'][arm]['home_q']=deepcopy(records[ids[0]]['before']['q'][arm])
        cfg['arms']['B']['home_q']=side[0]
        with MotionPlanner(sim,client,cfg)as planner:
            sideproof=check_path(planner,records[ids[0]],side)
            departureproof=check_path(planner,records[ids[-1]],raw)
            report.update(side_contact_path_check=sideproof,entire_departure_without_contact_exclusions=departureproof)
            assert sideproof['ok']and departureproof['ok']
            for boundary in (-3.60,-3.56,-2.8641197021):
                index=next(i for i in range(1,len(raw))if raw[i][0]>=boundary)
                left,right=np.asarray(raw[index-1]),np.asarray(raw[index])
                q=(left+(right-left)*(boundary-left[0])/(right[0]-left[0])).tolist()
                retreat=side+deepcopy(raw[1:index])+[q]
                gate=[q]+deepcopy(raw[index:])
                replacement={ids[0]:retreat,ids[1]:[q],ids[2]:gate}
                tasks=[]
                for key in ids:
                    task=deepcopy(records[key]['task']);task['q_path']=replacement[key]
                    task['dependencies']=[d for d in task['dependencies']if d in ids]
                    if task['kind']=='transit':task['metadata']['local_route_candidate']={
                        'window':'support_return','method':'joint_reconnect',
                        'certificate_scope':'native dense fixed-reference-state','require_profile_validation':True}
                    tasks.append(task)
                graph=build_tpg(tasks,max_joint_span=.24);smoothing=smooth_cartesian_macros(graph,planner,cfg);parameterize(graph)
                bytask={key:sum(n['duration']for n in graph['nodes']if n['task_id']==key)for key in ids}
                travels={key:np.abs(np.diff(np.asarray(path),axis=0)).sum(axis=0).tolist()for key,path in replacement.items()}
                # The new clear point must also pass the stronger no-contact world.
                boundaryproof=check_path(planner,records[ids[-1]],[q])
                candidate={'label':f'clear_at_q1_{boundary}','boundary_q':q,'boundary_fk':planner.fk('B',q),
                           'replacements':replacement,'context_records':{k:records[k]for k in ids},
                           'tasks':tasks,'boundary_no_contact_check':boundaryproof,'smoothing':smoothing,
                           'joint_travel_rad':travels,'base_travel_degrees':{k:math.degrees(v[0])for k,v in travels.items()},
                           'total_base_travel_degrees':math.degrees(sum(v[0]for v in travels.values())),
                           'duration_by_task_s':bytask,'total_motion_and_dwell_s':sum(bytask.values()),
                           'retreat_velocity_limit_preserved_rad_s':tasks[0]['metadata']['max_velocity_rad_s'],
                           'retreat_acceleration_limit_preserved_rad_s2':tasks[0]['metadata']['max_acceleration_rad_s2'],
                           'accepted_native_reference':boundaryproof['ok'],'actual_pending':True,
                           'full_TPG_certificate_pending':True}
                report['candidates'].append(candidate);write(out,report)
                print(candidate['label'],candidate['total_motion_and_dwell_s'],candidate['base_travel_degrees'],flush=True)
        report.update(status='completed',wall_time_s=time.perf_counter()-started,source_sha256_after=hashes())
        write(out,report)
    finally:owner.close()


if __name__=='__main__':run()
