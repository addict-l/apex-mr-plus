from pathlib import Path
import sys,json,math
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];LAB=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(LAB)]
from support_return_probe import options,check_path,shifted,travel
from src.sim_connection import Simulator
from apex.scene import resolve
from apex.motion import MotionPlanner
from apex.program import parameterize
from apex.tpg import build_tpg

def run():
 cases=json.loads((LAB/'accepted_cases.json').read_text());r=cases['all_reference_tasks'];pick=json.loads((LAB/'support_pick_candidates.json').read_text());a={x['label']:x for x in pick['attempts']};rec=r['return_support_approach'];start=rec['task']['q_path'][0];end=shifted(rec['task']['q_path'])[-1]
 def ws(q):q=deepcopy(q);q[5]-=2*math.pi;return q
 mid=ws(a['upper_park_to_mid']['result']['path'][-1]);highpark=ws(a['payload_up']['result']['path'][-1])
 out={'status':'running','attempts':[],'candidates':[]}
 def save(): (LAB/'support_return_incoming_upper.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,left,right,margin=.15):
    opt=options(rec);opt.update(max_joint_step=.012,max_rrt_candidates=2,joint_bound_margin=margin)
    ans=p.plan('B',left,right,rec['task']['payload'],rec['task']['metadata']['placed'],solve_time=3,options=opt)
    base=travel(ans['path'])[0]*180/math.pi if ans.get('ok') else None
    out['attempts'].append({'label':label,'margin':margin,'q_start':left,'q_end':right,'result':ans,'base_deg':base});save();print(label,ans.get('ok'),base,flush=True);return ans
   def ik(label,pose,seed):
    opt=options(rec);opt.update(seed=seed,max_solutions=4,ik_time=.3)
    ans=p.solve_ik('B',pose[:3],pose[3:],rec['task']['payload'],rec['task']['metadata']['placed'],opt);out['attempts'].append({'label':label,'pose':pose,'result':ans});save();print(label,ans.get('ok'),flush=True);return ans.get('solutions',[])
   middle=plan('mid_to_park_high',mid,highpark);down=plan('park_high_down',highpark,end)
   if not middle.get('ok') or not down.get('ok'):raise RuntimeError('Known physical upper segment failed in final world')
   tails=middle['path']+down['path'][1:];paths=[]
   for margin in (.15,.4):
    first=plan('start_to_mid',start,mid,margin)
    if first.get('ok'):
     path=first['path']+tails[1:]
     if travel(path)[0]<=math.pi:paths.append(('direct_mid_'+str(margin),path));break
   if not paths:
    quat=p.fk('B',start)['pose'][3:]
    for xyz in ([.20,-.20,1.32],[.30,-.20,1.32],[.112,-.20,1.32],[.25,-.08,1.32],[.30,-.20,1.38]):
     for high in ik('front_high_'+str(xyz),xyz+quat,start)[:2]:
      if not -2.1<high[0]<-.7:continue
      up=plan('start_to_front_high',start,high)
      if not up.get('ok'):continue
      across=plan('front_high_to_mid',high,mid)
      if not across.get('ok'):continue
      path=up['path']+across['path'][1:]+tails[1:]
      if travel(path)[0]<=math.pi:paths.append(('via_'+str(xyz),path));break
     if paths:break
   for label,path in paths:
    check=check_path(p,rec,path);task=deepcopy(rec['task']);task['q_path']=path;task['dependencies']=[];g=build_tpg([task],max_joint_span=.24);parameterize(g)
    c={'label':label,'accepted':check['ok'],'path':path,'check':check,'base_travel_deg':travel(path)[0]*180/math.pi,'duration_s':sum(n['duration'] for n in g['nodes']),'L2':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())};out['candidates'].append(c);save();print('candidate',c['duration_s'],c['base_travel_deg'],c['L2'],flush=True)
   out['status']='completed';save()
if __name__=='__main__':run()
