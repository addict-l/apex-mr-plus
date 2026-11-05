from pathlib import Path
import sys,json,math
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];LAB=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(LAB)]
from support_return_probe import options,check_path,travel
from src.sim_connection import Simulator
from apex.scene import resolve
from apex.motion import MotionPlanner
from apex.program import parameterize
from apex.tpg import build_tpg

def run():
 cases=json.loads((LAB/'accepted_cases.json').read_text());r=cases['all_reference_tasks'];old=json.loads((LAB/'support_pick_candidates.json').read_text());accepted=json.loads((LAB/'support_pick_approach_candidates.json').read_text())['candidates'][0]
 rec=r['pick_support_approach'];start=rec['task']['q_path'][0];park=accepted['replacements']['pick_support_approach'][-1];lift=accepted['replacements']['pick_support_lift'][-1]
 high=next(x['result']['path'][-1] for x in old['attempts'] if x['label']=='payload_up')
 out={'status':'running','attempts':[],'candidates':[]}
 def save(): (LAB/'support_pick_overhead_candidates.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,left,right,margin):
    opt=options(rec);opt.update(max_joint_step=.012,max_rrt_candidates=2,joint_bound_margin=margin)
    ans=p.plan('B',left,right,None,rec['task']['metadata']['placed'],solve_time=3,options=opt)
    base=travel(ans['path'])[0]*180/math.pi if ans.get('ok') else None
    out['attempts'].append({'label':label,'margin':margin,'result':ans,'base_deg':base});save();print(label,margin,ans.get('ok'),base,flush=True)
    return ans
   down=plan('high_to_park',high,park,.2)
   if not down.get('ok'):raise RuntimeError('High park descent failed')
   for margin in (.15,.4,.8):
    first=plan('home_to_high_park',start,high,margin)
    if not first.get('ok'):continue
    path=first['path']+down['path'][1:]
    if travel(path)[0]>math.pi:continue
    check=check_path(p,rec,path)
    c=deepcopy(accepted);c['replacements']['pick_support_approach']=path;c['checks']['pick_support_approach']=check;c['travel_rad']['pick_support_approach']=travel(path);c['base_travel_deg']=sum(v[0] for v in c['travel_rad'].values())*180/math.pi;c['accepted']=check['ok'];c['label']='upper_park_'+str(margin)
    task=deepcopy(rec['task']);task['q_path']=path;task['dependencies']=[]
    tpg=build_tpg([task],max_joint_span=.24);parameterize(tpg);c['approach_rest_to_rest_motion_s']=sum(n['duration'] for n in tpg['nodes']);c['approach_L2']=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    out['candidates'].append(c);save();print('candidate',c['approach_rest_to_rest_motion_s'],c['approach_L2'],flush=True)
   out['status']='completed';save()
if __name__=='__main__':run()
