"""Repair only an invalid support contact patch in an existing reference.

The first 46 tasks are immutable. Main allocation/grasp bindings already used
by that prefix remain fixed; ILP reselects among geometry-qualified residual
support modes. No scene, real pose, or existing reference path is teleported.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from .scene import resolve
from .motion import MotionPlanner
from .allocation import solve_allocation
from .compile import support_contact_geometry, trajectory_duration


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def dense(path,step=.005):
    out=[path[0]]
    for a,b in zip(path,path[1:]):
        n=max(1,math.ceil(np.linalg.norm(np.asarray(b)-a)/step))
        out.extend([[x+(y-x)*i/n for x,y in zip(a,b)] for i in range(1,n+1)])
    return out


def repair():
    output=ROOT/'results'/'normal'
    staging=output/'support_repair_staging'
    data=json.loads((output/'alloc_and_reference.json').read_text(encoding='utf-8'))
    original=deepcopy(data['tasks']);tasks=deepcopy(original)
    assert len(tasks)==53 and tasks[46]['kind']=='transit' and tasks[47]['kind']=='support_start'
    original_prefix=hashlib.sha256(json.dumps(original[:46],sort_keys=True).encode()).hexdigest()
    problem=deepcopy(data['problem'])
    previous={c['part']:c for c in data['allocation']['choices']}
    old_choice=previous['cap'];candidate=problem['candidates'][old_choice['candidate_index']]
    old_support=deepcopy(candidate['support_candidates'][old_choice['support_candidate_index']])
    helper=old_support['arm'];installer=old_choice['arm'];support_part=old_support['support_part']
    cap_payload=tasks[45]['payload'];placed=deepcopy(tasks[46]['metadata']['placed'])
    home=problem['initial_joint_configurations'];q={a:list(home[a]) for a in ['A','B']}
    for task in tasks[:46]:q[task['arm']]=list(task['q_path'][-1])
    repaired=deepcopy(old_support)
    repaired['tcp_position'][1]=.160
    report={'repair':'positive XY support contact patch','immutable_prefix_tasks':46,
            'prefix_sha256':original_prefix,'old_support':old_support,'motion_calls':[]}
    with Simulator(ROOT,port=23044,timeout=35) as owner:
        client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
        parts={p['id']:p for p in cfg['parts']}
        pad=cfg['arms'][helper]['support_pad']
        pad_size=[sim.getObjectFloatParam(pad,getattr(sim,'objfloatparam_objbbox_max_'+a))-
                  sim.getObjectFloatParam(pad,getattr(sim,'objfloatparam_objbbox_min_'+a)) for a in ['x','y','z']]
        report['old_contact_geometry']=support_contact_geometry(old_support['tcp_position'],old_support['quaternion'],parts[support_part],pad_size)
        with MotionPlanner(sim,client,cfg) as planner:
            options={'seed':old_support['q_contact'],'multiple':False,'other_q':q[installer],
                     'other_payload':cap_payload,'contact_exclusions':repaired['contact_exclusions'],
                     'contact_tolerance':.003}
            ik=planner.solve_ik(helper,repaired['tcp_position'],repaired['quaternion'],placed=placed,options=options)
            assert ik.get('ok'),ik
            repaired['q_contact']=ik['q']
            actual=planner.fk(helper,ik['q'])['pose']
            repaired['contact_geometry']=support_contact_geometry(actual[:3],actual[3:],parts[support_part],pad_size)
            assert repaired['contact_geometry']['qualified'],repaired['contact_geometry']
            # Exact cuboid contact fixture in this isolated stopped instance.
            before_q={a:[sim.getJointPosition(j) for j in cfg['arms'][a]['joints']] for a in ['A','B']}
            before_part=sim.getObjectPose(cfg['blocks'][support_part],-1)
            for j,v in zip(cfg['arms'][helper]['joints'],ik['q']):sim.setJointPosition(j,v)
            sim.setObjectPose(cfg['blocks'][support_part],parts[support_part]['goal']+[0,0,0,1],-1)
            hit,pair=sim.checkCollision(pad,cfg['blocks'][support_part])
            report['fixture']={'actual_tcp_pose':actual,'geometric_pad_beam_contact':bool(hit),'pair':pair}
            assert hit>0,report['fixture']
            for a,values in before_q.items():
                for j,v in zip(cfg['arms'][a]['joints'],values):sim.setJointPosition(j,v)
            sim.setObjectPose(cfg['blocks'][support_part],before_part,-1)
            repaired['feasible']=True
            repaired['cost']=2*float(np.linalg.norm(np.asarray(home[helper])-ik['q']))/.6+1.2
            # Preserve completed/retained main bindings; filter old support
            # frames by genuine positive patch geometry before residual ILP.
            for index,c in enumerate(problem['candidates']):
                c['feasible']=index==previous[c['part']]['candidate_index']
                for support in c.get('support_candidates',[]):
                    target=parts[support.get('support_part',support_part)]
                    certificate=support_contact_geometry(support['tcp_position'],support['quaternion'],target,pad_size)
                    support['contact_geometry']=certificate
                    support['feasible']=bool(support.get('feasible') and certificate['qualified'])
            candidate['support_candidates'].append(repaired)
            problem['fixed_prefix_choices']={p:{'arm':c['arm'],'grasp':c['grasp']} for p,c in previous.items()}
            allocation=solve_allocation(problem)
            assert allocation.get('solved'),allocation
            choice=next(c for c in allocation['choices'] if c['part']=='cap')
            selected=problem['candidates'][choice['candidate_index']]['support_candidates'][choice['support_candidate_index']]
            assert selected['contact_geometry']['qualified']
            assert selected['quaternion']==old_support['quaternion'] and selected['q_precontact']==old_support['q_precontact'], 'Residual solver selected another branch; explicit replanning required'
            report['new_choice']=choice;report['new_support']=selected
            save(staging/'contact_fixture.json',report)
            print('CONTACT_FIXTURE',json.dumps(report['fixture']),flush=True)
            print('CONTACT_GEOMETRY',json.dumps(selected['contact_geometry']),flush=True)

            # Exact unchanged A approach/withdraw paths retain their genuine
            # RRT certificates. All affected contact/installer paths are new
            # RRTConnect queries in the updated opposite-arm configuration.
            for i in range(46,53):
                task=tasks[i];a=task['arm'];other='B' if a=='A' else 'A'
                if i==46:goal=selected['q_precontact']
                elif i==47:goal=selected['q_contact']
                elif i==51:goal=selected['q_precontact']
                else:goal=original[i]['q_path'][-1]
                mode_placed=task['metadata']['placed'];payload=task.get('payload')
                payloads={installer:cap_payload} if i<=48 else {}
                contacts=task['metadata'].get('contact_exclusions',[])
                reuse=i in [46,52] and np.max(np.abs(np.asarray(task['q_path'][0])-q[a]))<1e-9 and np.max(np.abs(np.asarray(task['q_path'][-1])-goal))<1e-9
                if reuse:
                    path=task['q_path']
                    answers=planner.sample_state_pairs([{'qA':v if a=='A' else q['A'],
                        'qB':v if a=='B' else q['B'],'payloads':payloads,'placed':mode_placed,
                        'options':{'contact_exclusions':contacts,'contact_tolerance':.003}} for v in dense(path)])
                    assert all(x.get('ok') for x in answers),(task['id'],next((x for x in answers if not x.get('ok')),None))
                    stats={'reused_original_RRT_path':True,'validated_states':len(answers)}
                else:
                    distance=float(np.linalg.norm(np.asarray(q[a])-goal))
                    opts={'joint_bound_margin':.18 if distance<.7 else .55,'contact_tolerance':.003,
                          'other_payload':cap_payload if other==installer and i<48 else {}}
                    print('REPAIR_RRT',task['id'],distance,flush=True)
                    answer=planner.plan(a,q[a],goal,payload=payload,placed=mode_placed,other_q=q[other],
                        contact_exclusions=contacts,solve_time=.8,options=opts)
                    if not answer.get('ok'):
                        opts['joint_bound_margin']=1.1
                        answer=planner.plan(a,q[a],goal,payload=payload,placed=mode_placed,other_q=q[other],
                            contact_exclusions=contacts,solve_time=2.,options=opts)
                    assert answer.get('ok'),(task['id'],answer)
                    path=answer['path'];stats=answer['stats']
                task['q_path']=deepcopy(path)
                task['duration']=trajectory_duration(path,task['metadata'].get('dwell_s',0.))
                if task['kind']=='support_start':
                    task['metadata']['contact_position']=selected['tcp_position']
                    task['metadata']['precontact_position']=selected['precontact_position']
                    task['metadata']['contact_geometry']=selected['contact_geometry']
                q[a]=list(path[-1]);report['motion_calls'].append({'task':task['id'],'stats':stats})
                print('REPAIRED',task['id'],len(path),flush=True)

            # Whole reference mode-aware, dense geometry validation, not only
            # the replaced tail endpoints. No original prefix task is altered.
            assert tasks[:46]==original[:46]
            qs=deepcopy(home);held={};samples=[];continuity=[]
            for task in tasks:
                a=task['arm'];continuity.append(max(abs(x-y) for x,y in zip(qs[a],task['q_path'][0])))
                for v in dense(task['q_path']):
                    state=dict(qs);state[a]=v;bindings=dict(held)
                    if task.get('payload'):bindings[a]=task['payload']
                    samples.append({'qA':state['A'],'qB':state['B'],'payloads':bindings,
                        'placed':task['metadata']['placed'],'options':{'contact_exclusions':task['metadata'].get('contact_exclusions',[]),'contact_tolerance':.003},'_task':task['id']})
                qs[a]=task['q_path'][-1]
                if task['metadata'].get('action')=='grasp':held[a]=task['metadata']['payload_after']
                elif task['metadata'].get('action')=='release':held.pop(a,None)
            violations=[]
            for start in range(0,len(samples),250):
                batch=samples[start:start+250];answers=planner.sample_state_pairs(batch)
                violations.extend({'sample':start+i,'task':batch[i]['_task'],'problem':x.get('problem')} for i,x in enumerate(answers) if not x.get('ok'))
            validation={'ok':not violations and max(continuity)<1e-5,'sample_count':len(samples),
                'max_joint_sampling_step_rad':.005,'violations':violations,'max_task_start_error_rad':max(continuity),
                'collision_geometry_policy':planner.info['collision_geometry_policy'],
                'scene_sha256':hashlib.sha256((ROOT/'scenes'/'10_Normal_Assembly.ttt').read_bytes()).hexdigest()}
            assert validation['ok'],validation
            data.update(allocation=allocation,problem=problem,tasks=tasks)
            data['motion_stats']['support_contact_repair']=report
            for name,value in [('alloc_and_reference.json',data),('tasks.json',tasks),('reference_tasks.json',tasks),
                               ('allocation.json',allocation),('allocation_problem.json',problem),
                               ('candidate_qualification.json',problem['candidates']),('motion_stats.json',data['motion_stats']),
                               ('reference_validation.json',validation),('support_contact_repair.json',report)]:
                save(staging/name,value)
            # Publish only after all replacement paths and full validation pass.
            for file in staging.glob('*.json'):
                if file.name!='contact_fixture.json':(output/file.name).write_bytes(file.read_bytes())
            print('SUPPORT_REPAIR_COMPLETE',json.dumps({'tasks':len(tasks),'validated_states':len(samples),
                'prefix_unchanged':True,'support_position':selected['tcp_position'],'overlap_m':selected['contact_geometry']['projected_overlap_m']}),flush=True)
            return data


if __name__=='__main__':repair()
