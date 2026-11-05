from pathlib import Path
import json,sys,math
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];LAB=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(LAB)]
from support_return_probe import options,check_path,travel
from src.sim_connection import Simulator
from apex.scene import resolve
from apex.motion import MotionPlanner
from apex.program import parameterize
from apex.tpg import build_tpg
r=json.loads((LAB/'accepted_cases.json').read_text());rec=r['all_reference_tasks']['return_support_approach'];old=json.loads((LAB/'support_return_incoming_upper.json').read_text());prefix=next(x['result']['path'] for x in old['attempts'] if x['label']=='start_to_mid' and x['result'].get('ok'));goal=old['candidates'][0]['path'][-1];out={'attempts':[],'candidates':[]}
def save():(LAB/'support_return_incoming_shortcut.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
with Simulator(LAB,port=23068,timeout=45) as owner:
 client,sim=owner.connect();sim.loadScene(r['source_scene']);cfg=resolve(sim)
 with MotionPlanner(sim,client,cfg) as p:
  for margin in (.15,.4):
   opt=options(rec);opt.update(max_joint_step=.012,joint_bound_margin=margin,max_rrt_candidates=2)
   ans=p.plan('B',prefix[-1],goal,rec['task']['payload'],rec['task']['metadata']['placed'],solve_time=3,options=opt);out['attempts'].append({'margin':margin,'result':ans});save();print('mid_direct',margin,ans.get('ok'),flush=True)
   if not ans.get('ok'):continue
   path=prefix+ans['path'][1:];check=check_path(p,rec,path);t=deepcopy(rec['task']);t['q_path']=path;t['dependencies']=[];g=build_tpg([t],max_joint_span=.24);parameterize(g)
   c={'accepted':check['ok'] and travel(path)[0]<=math.pi,'path':path,'check':check,'base_travel_deg':travel(path)[0]*180/math.pi,'duration_s':sum(n['duration'] for n in g['nodes']),'L2':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())};out['candidates'].append(c);save();print('candidate',c['accepted'],c['duration_s'],c['base_travel_deg'],flush=True)
   if c['accepted']:break
out['status']='completed';save()
