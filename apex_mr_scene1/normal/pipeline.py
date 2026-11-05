"""Normal baseline: ILP -> sequential tasks -> RRTConnect -> TPG -> dynamics."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from .scene import resolve
from .compile import compile_reference
from .motion import MotionPlanner
from .tpg import build_tpg,simulate_tpg
from .program import parameterize
from .plots import report_figures
from .runtime import install_controller,execute,capture

OUT=ROOT/'results'/'normal'

def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def signature(cfg):
    values={'parts':cfg['parts'],'home':{s:a['home_q'] for s,a in cfg['arms'].items()},'physics':cfg['physics']}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replan',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--reuse-reference',action='store_true')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--reuse-tpg',action='store_true')
    parser.add_argument('--span',type=float,default=.22)
    args=parser.parse_args()
    with Simulator(ROOT,port=23043,timeout=45) as owner:
        client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
        sig=signature(cfg)
        rigid=[]
        for h in sim.getObjectsInTree(cfg['root'],sim.object_shape_type,0):
            if sim.getObjectInt32Param(h,sim.shapeintparam_static)==0:
                inertia,frame=sim.getShapeInertia(h)
                rigid.append({'path':sim.getObjectAlias(h,1),'mass_kg':sim.getShapeMass(h),
                              'inertia_kg_m2':inertia,'inertia_frame':frame})
        save('rigid_bodies.json',rigid)
        save('problem.json',{'arms':['A','B'],'parts':cfg['parts'],'physics':cfg['physics']})
        print('STAGE: geometry-qualified ILP and sequential RRTConnect reference',flush=True)
        if args.reuse_reference and (OUT/'alloc_and_reference.json').exists() and not args.replan:
            reference=json.loads((OUT/'alloc_and_reference.json').read_text(encoding='utf-8'))
            if reference.get('scene_signature') not in [None,sig]:raise RuntimeError('Cached reference belongs to a different scene')
        else:
            reference=compile_reference(sim,client,cfg,OUT)
        reference['scene_signature']=sig;save('alloc_and_reference.json',reference)
        tasks=reference['tasks']
        reference_signature=hashlib.sha256(json.dumps(tasks,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        # Reference trajectories start at these exact, geometry-qualified homes.
        homes=reference.get('problem',{}).get('initial_joint_configurations',{})
        for side,q in homes.items():
            cfg['arms'][side]['home_q']=q
            for j,v in zip(cfg['arms'][side]['joints'],q):sim.setJointPosition(j,v);sim.setJointTargetPosition(j,v)
        sequential={'arms':{a:[t for t in tasks if t['arm']==a] for a in ['A','B']},
                    'given_assembly_order':[p['id'] for p in cfg['parts']],
                    'allocation':reference['allocation'],'scope':'global serial feasible reference; TPG later relaxes order'}
        save('sequential_task_graph.json',sequential)
        print('STAGE: sampled geometry conflicts and multi-modal TPG',flush=True)
        with MotionPlanner(sim,client,cfg) as planner:
            from .geometry import SampledConflictOracle,verify_rollout
            if args.reuse_tpg and (OUT/'tpg.json').exists() and not args.replan:
                tpg=json.loads((OUT/'tpg.json').read_text(encoding='utf-8'))
                if tpg.get('scene_signature')!=sig:raise RuntimeError('Cached TPG scene mismatch')
                if tpg.get('reference_signature')!=reference_signature:
                    raise RuntimeError('Cached TPG reference paths changed; rerun without --reuse-tpg')
            else:
                from .fast_interarm import FastInterarmSampler
                with FastInterarmSampler(sim,client,cfg) as fast:
                    oracle=SampledConflictOracle(fast,cfg,OUT,step=.07)
                    tpg=build_tpg(tasks,oracle,max_joint_span=args.span)
                    tpg['interarm_collision_model']=fast.info
                tpg['scene_signature']=sig
                tpg['reference_signature']=reference_signature
                parameterize(tpg)
                save('tpg.json',tpg)
            print(json.dumps({'tpg_nodes':len(tpg['nodes']),'tpg_edges':len(tpg['edges']),
                              'nominal_makespan':tpg['nominal_makespan']}),flush=True)
            validation=verify_rollout(tpg,planner,cfg,step_s=.06,outdir=OUT)
            save('nominal_geometry_validation.json',validation)
            if not validation.get('ok',validation.get('valid',False)):
                raise RuntimeError('Nominal TPG/waiting geometry failed; see nominal_geometry_validation.json')
            tpg['certificate']['waiting_endpoints_verified']=True
            tpg['certificate']['contact_skill_height_adjustment_bound_m']=.002
        save('tpg.json',tpg)
        scheduler=install_controller(sim,cfg,tpg)
        save('pipeline_manifest.json',{'scene_signature':sig,'source_scene':'10_Normal_Assembly.ttt',
            'reference_signature':reference_signature,
            'source_scene_sha256':hashlib.sha256((ROOT/'scenes'/'10_Normal_Assembly.ttt').read_bytes()).hexdigest(),
            'controller_sha256':hashlib.sha256((ROOT/'normal'/'controller.lua').read_bytes()).hexdigest(),
            'planned_scene':'11_Normal_APEX_MR_Planned.ttt','stages':['HiGHS ILP','sequential skill graph','simOMPL RRTConnect','multimodal TPG','central completion-driven dynamic dispatcher'],
            'physics':cfg['physics'],'grasp_and_snap_idealized':True,'autonomous_failure_recovery':False})
        report_figures(tasks,tpg,None,OUT)
        if args.plan_only:
            print('Planning completed; dynamics not executed (--plan-only).',flush=True);return
        print('STAGE: actual rigid-body execution via dynamic joint servos',flush=True)
        execution=execute(sim,client,cfg,OUT,video=not args.no_video)
        report_figures(tasks,tpg,execution,OUT)
        summary={'success':execution['success'],'parts':len(cfg['parts']),
                 'allocation':[{k:c.get(k) for k in ['part','arm','grasp','support_arm']} for c in reference['allocation']['choices']],
                 'rrt_reference_paths':len(tasks),'rrt_planning_requests':reference['motion_stats']['path_count'],
                 'tpg':tpg['counts'],
                 'actual_makespan_s':execution['makespan'],'max_goal_error_m':max(execution['goal_position_errors_m'].values()),
                 'peak_tracking_rad':execution['peak_tracking_rad'],'peak_support_normal_N':execution['peak_support_normal_N'],
                 'failure':execution.get('failure')}
        save('summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        if not execution['success']:raise RuntimeError('Normal execution failed validation; see execution.json')
        from .validate_result import main as audit_actual_events
        audit_actual_events()
        from .plot_dynamics import plot_dynamics
        plot_dynamics(OUT/'execution.json',OUT)
        shutil.copy2(ROOT/'scenes'/'11_Normal_APEX_MR_Planned.ttt',ROOT/'APEX_MR_Plus.ttt')
        save('delivery.json',{'main_scene':str((ROOT/'APEX_MR_Plus.ttt').resolve()),
            'main_scene_sha256':hashlib.sha256((ROOT/'APEX_MR_Plus.ttt').read_bytes()).hexdigest(),
            'actual_execution_audited':True,'normal_tasks':len(tasks),
            'scene_contains_initial_state_and_embedded_TPG':True})

if __name__=='__main__':main()
