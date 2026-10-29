"""Exercise the fresh compiler's staged support against the real scene.

This isolated check writes only diagnostic results. It never replaces the
published reference, TPG, or shared scene.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from .scene import resolve
from .motion import MotionPlanner
from .compile import _Compiler


def main():
    data=json.loads((ROOT/'results/normal/alloc_and_reference.json').read_text(encoding='utf-8'))
    choice=next(c for c in data['allocation']['choices'] if c['support_arm'])
    support=deepcopy(data['problem']['candidates'][choice['candidate_index']]['support_candidates'][choice['support_candidate_index']])
    tasks=data['tasks'];start=next(t for t in tasks if t['kind']=='support_start')
    end=next(t for t in tasks if t['kind']=='support_end')
    helper=support['arm'];other=choice['arm'];home=data['problem']['initial_joint_configurations']
    q=deepcopy(home);payloads={}
    for task in tasks[:tasks.index(start)]:
        q[task['arm']]=task['q_path'][-1]
        if task['metadata'].get('action')=='grasp':payloads[task['arm']]=task['metadata']['payload_after']
        if task['metadata'].get('action')=='release':payloads.pop(task['arm'],None)
    evidence={'ok':False,'samples':0,'violations':[],'skills':{}}
    with Simulator(ROOT,port=23044,timeout=35) as owner:
        client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes/10_Normal_Assembly.ttt'))
        cfg=resolve(sim)
        with MotionPlanner(sim,client,cfg) as planner:
            compiler=_Compiler(sim,planner,cfg)
            for reverse,task,other_q,bindings in [(False,start,q[other],payloads),(True,end,home[other],{})]:
                path,metadata=compiler.staged_support_path(choice,support,task['metadata']['placed'],other_q,
                    other_payload=bindings.get(other),reverse=reverse)
                assert np.max(np.abs(np.asarray(path[0])-task['q_path'][0]))<1e-9
                assert np.max(np.abs(np.asarray(path[-1])-task['q_path'][-1]))<1e-9
                samples=[]
                for stage in metadata['contact_stages']:
                    first,last=stage['q_start_index'],stage['q_end_index']
                    for a,b in zip(path[first:last],path[first+1:last+1]):
                        count=max(1,math.ceil(np.linalg.norm(np.asarray(b)-a)/.005))
                        for i in range(count+1):
                            v=[x+(y-x)*i/count for x,y in zip(a,b)]
                            state={helper:v,other:other_q}
                            samples.append({'qA':state['A'],'qB':state['B'],'payloads':bindings,
                                'placed':task['metadata']['placed'],'options':{'contact_exclusions':support['contact_exclusions'] if stage['contact_allowed'] else [],'contact_tolerance':.003}})
                for at in range(0,len(samples),250):
                    answers=planner.sample_state_pairs(samples[at:at+250])
                    evidence['violations'].extend({'skill':task['id'],'sample':at+i,'problem':a.get('problem')} for i,a in enumerate(answers) if not a.get('ok'))
                evidence['samples']+=len(samples)
                evidence['skills'][task['id']]={'q_path':path,'metadata':metadata}
            evidence['ok']=not evidence['violations']
            evidence['calls']=compiler.calls
            assert evidence['ok'],evidence['violations']
    target=ROOT/'results/normal/contact_skill_staging/fresh_compiler_validation.json'
    target.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FRESH_COMPILER_STAGED_VALIDATION',json.dumps({'ok':evidence['ok'],'samples':evidence['samples'],'output':str(target)}),flush=True)


if __name__=='__main__':main()
