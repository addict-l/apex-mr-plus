"""Replace lateral preloaded support insertion by lowered staged contact.

Only support_start/support_end task paths change. Outside lowering and
under-beam transit have no shoe/beam contact exemption. The final engage and
initial disengage use a narrow, verified vertical TCP motion.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from .scene import resolve
from .motion import MotionPlanner
from .compile import trajectory_duration


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def dense(path,step=.005):
    result=[path[0]]
    for a,b in zip(path,path[1:]):
        n=max(1,math.ceil(np.linalg.norm(np.asarray(b)-a)/step))
        result.extend([[x+(y-x)*i/n for x,y in zip(a,b)] for i in range(1,n+1)])
    return result


def main():
    output=ROOT/'results'/'normal';staging=output/'contact_skill_staging'
    original=json.loads((output/'alloc_and_reference.json').read_text(encoding='utf-8'))
    data=deepcopy(original);tasks=data['tasks']
    start=next(t for t in tasks if t['kind']=='support_start')
    end=next(t for t in tasks if t['kind']=='support_end')
    helper=start['arm'];other='B' if helper=='A' else 'A';target=start['metadata']['support_part']
    home=data['problem']['initial_joint_configurations']
    q=deepcopy(home);held={}
    for t in tasks[:tasks.index(start)]:
        q[t['arm']]=t['q_path'][-1]
        if t['metadata'].get('action')=='grasp':held[t['arm']]=t['metadata']['payload_after']
        elif t['metadata'].get('action')=='release':held.pop(t['arm'],None)
    pre_q=list(start['q_path'][0]);contact_q=list(start['q_path'][-1])
    contact=start['metadata']['contact_position'];pre=start['metadata']['precontact_position']
    contact_pairs=deepcopy(start['metadata']['contact_exclusions'])
    mode_placed=deepcopy(start['metadata']['placed'])
    report={'changed_task_ids':[start['id'],end['id']],'unchanged_task_count':len(tasks)-2,
            'max_velocity_rad_s':.025,'max_acceleration_rad_s2':.12,'rrt_stages':[]}
    with Simulator(ROOT,port=23044,timeout=35) as owner:
        client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
        parts={p['id']:p for p in cfg['parts']};bottom=parts[target]['goal'][2]-parts[target]['size'][2]/2
        with MotionPlanner(sim,client,cfg) as planner:
            rotation=planner.fk(helper,contact_q)['pose'][3:]
            def ik(position,seed):
                answer=planner.solve_ik(helper,position,rotation,placed=mode_placed,
                    options={'seed':seed,'multiple':False,'other_q':q[other],
                             'other_payload':held.get(other,{}),'contact_exclusions':[],
                             'contact_tolerance':.003})
                return answer
            for clearance in [.005,.003]:
                outside=[pre[0],pre[1],bottom-clearance]
                inside=[contact[0],contact[1],bottom-clearance]
                lo=ik(outside,pre_q);li=ik(inside,contact_q)
                print('LOWERED_IK',clearance,lo.get('ok'),li.get('ok'),lo.get('problem'),li.get('problem'),flush=True)
                if lo.get('ok') and li.get('ok'):break
            else:raise RuntimeError('Neither 5mm nor 3mm under-beam clearance has valid IK')
            low_out=lo['q'];low_in=li['q']
            report.update(clearance_m=clearance,outside_low=outside,inside_low=inside)

            def plan_stage(name,a,b,placed,other_q,other_payload,allow_contact=False,vertical=False):
                contacts=contact_pairs if allow_contact else []
                delta=float(np.linalg.norm(np.asarray(b)-a))
                opts={'joint_bound_margin':.06 if delta<.08 else .16,'contact_tolerance':.003,
                      'other_payload':other_payload or {},'max_joint_step':.015}
                answer=planner.plan(helper,a,b,placed=placed,other_q=other_q,
                    contact_exclusions=contacts,solve_time=.6,options=opts)
                if not answer.get('ok'):
                    opts['joint_bound_margin']=.40
                    answer=planner.plan(helper,a,b,placed=placed,other_q=other_q,
                        contact_exclusions=contacts,solve_time=1.4,options=opts)
                assert answer.get('ok'),(name,answer)
                path=answer['path']
                evidence={'name':name,'contact_allowed':allow_contact,'stats':answer['stats']}
                if vertical:
                    poses=[planner.fk(helper,v)['pose'] for v in dense(path,.003)]
                    xy=max(float(np.linalg.norm(np.asarray(p[:2])-contact[:2])) for p in poses)
                    evidence['max_xy_axis_deviation_m']=xy
                    assert xy<.0003,(name,'not a vertical engagement path',xy)
                report['rrt_stages'].append(evidence)
                print('STAGED_RRT',name,len(path),answer['stats'].get('wall_time_s'),flush=True)
                return path

            enter=[
                ('outside_lower',plan_stage('outside_lower',pre_q,low_out,mode_placed,q[other],held.get(other))),
                ('under_beam_translate',plan_stage('under_beam_translate',low_out,low_in,mode_placed,q[other],held.get(other))),
                ('vertical_engage',plan_stage('vertical_engage',low_in,contact_q,mode_placed,q[other],held.get(other),True,True)),
            ]
            end_placed=deepcopy(end['metadata']['placed'])
            # At the support_end reference point the installer has returned HOME.
            leave=[
                ('vertical_disengage',plan_stage('vertical_disengage',contact_q,low_in,end_placed,home[other],None,True,True)),
                ('outside_translate',plan_stage('outside_translate',low_in,low_out,end_placed,home[other],None)),
                ('outside_raise',plan_stage('outside_raise',low_out,pre_q,end_placed,home[other],None)),
            ]
            for task,stages in [(start,enter),(end,leave)]:
                path=[];markers=[]
                for name,segment in stages:
                    first=max(0,len(path)-1)
                    path.extend(deepcopy(segment if not path else segment[1:]))
                    markers.append({'name':name,'q_start_index':first,'q_end_index':len(path)-1,
                                    'contact_allowed':name in ['vertical_engage','vertical_disengage']})
                task['q_path']=path
                task['metadata'].update(max_velocity_rad_s=.025,max_acceleration_rad_s2=.12,
                    contact_stages=markers,under_beam_clearance_m=clearance,
                    contact_skill='lower_outside_then_translate_under_then_vertical_engage')
                task['duration']=trajectory_duration(path,task['metadata'].get('dwell_s',.20))
            unchanged=[a==b for a,b in zip(original['tasks'],tasks)
                       if a['id'] not in [start['id'],end['id']]]
            assert all(unchanged) and len(unchanged)==len(tasks)-2
            fixture={'partial_reference_validation':True,'tasks':tasks,
                     'changed_skills':[start,end],'report':report,'initial_joint_configurations':home}
            save(staging/'staged_skills.json',fixture)
            print('STAGED_SKILLS_READY',str(staging/'staged_skills.json'),flush=True)

            qs=deepcopy(home);held={};samples=[];continuity=[]
            for t in tasks:
                a=t['arm'];continuity.append(max(abs(x-y) for x,y in zip(qs[a],t['q_path'][0])))
                for v in dense(t['q_path']):
                    state=dict(qs);state[a]=v;bindings=dict(held)
                    if t.get('payload'):bindings[a]=t['payload']
                    samples.append({'qA':state['A'],'qB':state['B'],'payloads':bindings,
                        'placed':t['metadata']['placed'],'options':{'contact_exclusions':t['metadata'].get('contact_exclusions',[]),'contact_tolerance':.003},'_task':t['id']})
                qs[a]=t['q_path'][-1]
                if t['metadata'].get('action')=='grasp':held[a]=t['metadata']['payload_after']
                elif t['metadata'].get('action')=='release':held.pop(a,None)
            violations=[]
            for at in range(0,len(samples),250):
                batch=samples[at:at+250];answers=planner.sample_state_pairs(batch)
                violations.extend({'sample':at+i,'task':batch[i]['_task'],'problem':v.get('problem')} for i,v in enumerate(answers) if not v.get('ok'))
            validation={'ok':not violations and max(continuity)<1e-5,'sample_count':len(samples),
                'max_joint_sampling_step_rad':.005,'violations':violations,
                'max_task_start_error_rad':max(continuity),'collision_geometry_policy':planner.info['collision_geometry_policy']}
            assert validation['ok'],validation
            data['motion_stats']['staged_contact_skill_repair']=report
            save(staging/'alloc_and_reference.json',data)
            save(staging/'reference_validation.json',validation)
            for name,value in [('alloc_and_reference.json',data),('tasks.json',tasks),('reference_tasks.json',tasks),
                               ('motion_stats.json',data['motion_stats']),('reference_validation.json',validation),
                               ('staged_contact_skill_repair.json',report)]:save(output/name,value)
            print('STAGED_CONTACT_REPAIR_COMPLETE',json.dumps({'samples':len(samples),'clearance_m':clearance,'changed_tasks':report['changed_task_ids']}),flush=True)


if __name__=='__main__':main()
