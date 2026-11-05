"""Empty-arm upper route after the original return retreat, independent check."""
from copy import deepcopy
import math
import time
import numpy as np
from support_return_probe import ROOT, LAB, read, write, shifted, options, check_path, hashes
from apex.motion import MotionPlanner
from apex.scene import resolve
from apex.tpg import build_tpg
from apex.smoothing import smooth_cartesian_macros
from apex.program import parameterize
from src.sim_connection import Simulator


def run():
    cases=read(LAB/'accepted_cases.json')
    records=cases['all_reference_tasks']
    ids=['return_support_approach','return_support_contact','return_support',
         'return_support_retreat','return_support_retreat_contact_clear','pick_gate_approach']
    initial=read(LAB/'support_return_initial_search.json')
    approach=next(a for a in initial['attempts']if a['label']=='direct_negative_winding')['solutions'][0]['plans'][0]['path']
    prior=read(LAB/'support_return_candidates.json')
    horizontal=next(a for a in prior['attempts']if a.get('offset_xy_m')==[.08,0] and a['height_m']==1.035)['plans'][0]
    assert horizontal['ok']
    fixed={key:shifted(records[key]['task']['q_path'])for key in ids[1:-1]}
    rec=records[ids[-1]]
    start=horizontal['path'][-1]
    goal=rec['task']['q_path'][-1]
    out=LAB/'support_return_high_departure_candidates.json'
    report={'status':'running','scope':'fixed-reference native upper-route candidate, no actual execution',
            'source_run_id':cases['run_id'],'source_sha256_before':hashes(),
            'attempts':[],'candidates':[],'actual_pending':True,'full_TPG_certificate_pending':True}
    write(out,report)
    owner=Simulator(LAB,port=23070,timeout=30.)
    began=time.perf_counter()
    try:
        owner.start(headless=True)
        client,sim=owner.connect()
        sim.loadScene(cases['source_scene'])
        cfg=resolve(sim)
        for arm in ('A','B'):cfg['arms'][arm]['home_q']=records[ids[0]]['before']['q'][arm]
        cfg['initial_payloads']=records[ids[0]]['before']['payloads']
        with MotionPlanner(sim,client,cfg)as planner:
            parkquat=planner.fk('B',start)['pose'][3:]
            gate=planner.fk('B',goal)['pose']
            def solutions(pose,seed):
                opts=options(rec);opts.update(seed=seed,max_solutions=8,ik_time=.25)
                result=planner.solve_ik('B',pose[:3],pose[3:],None,rec['task']['metadata']['placed'],opts)
                qs=[]
                for raw in result.get('solutions',[]):
                    q=(np.asarray(seed)+np.arctan2(np.sin(np.asarray(raw)-seed),np.cos(np.asarray(raw)-seed))).tolist()
                    if not fixed[ids[-2]][0][0]-.05 <= q[0] <= goal[0]+.05:continue
                    if not any(np.linalg.norm(np.asarray(q)-old)<1e-5 for old in qs):qs.append(q)
                qs.sort(key=lambda q:np.linalg.norm(np.asarray(q)-seed))
                return qs[:2],result
            mids,midanswer=solutions([.30,.13,1.32]+parkquat,start)
            report['midpoint_IK']=midanswer
            write(out,report)
            for height in (1.15,1.20,1.26):
                targets,answer=solutions(gate[:2]+[height]+gate[3:],goal)
                for im,mid in enumerate(mids):
                    for ig,target in enumerate(targets):
                        item={'height_m':height,'mid_index':im,'gate_index':ig,'gate_ik':answer,'plans':[]}
                        report['attempts'].append(item)
                        anchors=[start,mid,target,goal]
                        whole=deepcopy(horizontal['path'])
                        for a,b in zip(anchors,anchors[1:]):
                            opts=options(rec);opts.update(joint_bound_margin=.25,max_rrt_candidates=1,shortcut_budget_s=.2)
                            part=planner.plan('B',a,b,None,rec['task']['metadata']['placed'],solve_time=.8,options=opts)
                            item['plans'].append(part)
                            write(out,report)
                            if not part.get('ok'):break
                            whole.extend(part['path'][1:])
                        if len(item['plans'])!=3 or not all(p.get('ok')for p in item['plans']):continue
                        angle=sum(abs(b[0]-a[0])for a,b in zip(whole,whole[1:]))
                        item['base_travel_degrees']=math.degrees(angle)
                        if angle>math.pi+1e-7:continue
                        paths={ids[0]:approach,**fixed,ids[-1]:whole}
                        checks={key:check_path(planner,records[key],path)for key,path in paths.items()}
                        item['checks']=checks
                        if not all(p['ok']for p in checks.values()):continue
                        tasks=[]
                        for key in ids:
                            task=deepcopy(records[key]['task']);task['q_path']=paths[key]
                            task['dependencies']=[d for d in task['dependencies']if d in ids]
                            if task['kind']=='transit':task['metadata']['local_route_candidate']={
                                'window':'support_return','method':'overhead_waypoints',
                                'certificate_scope':'native dense fixed-reference-state','require_profile_validation':True}
                            tasks.append(task)
                        graph=build_tpg(tasks,max_joint_span=.24)
                        smooth=smooth_cartesian_macros(graph,planner,cfg)
                        parameterize(graph)
                        candidate={'label':f'old_retreat_upper_gate{height}_m{im}_g{ig}',
                                   'replacements':paths,'tasks':tasks,'checks':checks,
                                   'context_records':{k:records[k]for k in ids},'smoothing':smooth,
                                   'parameterized_local_motion_and_dwell_s':sum(n['duration']for n in graph['nodes']),
                                   'base_departure_travel_degrees':math.degrees(angle),
                                   'accepted':True,'actual_pending':True,'full_TPG_certificate_pending':True}
                        report['candidates'].append(candidate)
                        write(out,report)
                        print(candidate['label'],candidate['parameterized_local_motion_and_dwell_s'],flush=True)
            report.update(status='completed_bounded_search',wall_time_s=time.perf_counter()-began,source_sha256_after=hashes())
            write(out,report)
    finally:owner.close()


if __name__=='__main__':run()
