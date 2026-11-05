from pathlib import Path
import sys,json,math
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];LAB=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(LAB)]
from support_return_probe import options,check_path,shifted,travel,hashes
from src.sim_connection import Simulator
from apex.scene import resolve
from apex.motion import MotionPlanner

def run():
 cases=json.loads((LAB/'accepted_cases.json').read_text(encoding='utf-8'));r=cases['all_reference_tasks'];old=json.loads((LAB/'support_pick_candidates.json').read_text());out={'status':'running','source_run_id':cases['run_id'],'attempts':[],'candidates':[]}
 def save(): (LAB/'support_pick_approach_candidates.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 ids=['pick_support_approach','pick_support_contact','pick_support','pick_support_lift','pick_support_lift_contact_clear','support_route_front','support_high_approach'];fixed={k:shifted(r[k]['task']['q_path']) for k in ids[1:5]};start=r[ids[0]]['task']['q_path'][0];park=fixed[ids[1]][0]
 a={x['label']:x for x in old['attempts']};transfer=a['payload_up']['result']['path']+a['upper_park_to_mid']['result']['path'][1:]+a['mid_to_original_front']['result']['path'][1:]
 save()
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,left,right,margin):
    opt=options(r[ids[0]]);opt.update(max_joint_step=.012,max_rrt_candidates=3,joint_bound_margin=margin)
    ans=p.plan('B',left,right,None,r[ids[0]]['task']['metadata']['placed'],solve_time=5.,options=opt)
    base=travel(ans['path'])[0]*180/math.pi if ans.get('ok') else None
    out['attempts'].append({'label':label,'margin':margin,'q_start':left,'q_end':right,'result':ans,'base_deg':base});save();print(label,margin,ans.get('ok'),base,ans.get('problem'),flush=True)
    return ans
   candidates=[]
   for margin in (.05,.15,.4):
    ans=plan('direct_limited',start,park,margin)
    if ans.get('ok') and travel(ans['path'])[0]<math.pi:candidates.append(('limited_'+str(margin),ans['path']));break
   quat=p.fk('B',park)['pose'][3:]
   if not candidates:
    for xyz in ([.512,.43,1.32],[.512,.43,1.38],[.30,.13,1.32],[.35,.13,1.32]):
     opt=options(r[ids[0]]);opt.update(seed=park,max_solutions=6,ik_time=.5)
     ik=p.solve_ik('B',xyz,quat,None,r[ids[0]]['task']['metadata']['placed'],opt);out['attempts'].append({'label':'ik','xyz':xyz,'result':ik});save()
     for qmid in ik.get('solutions',[])[:3]:
      if not park[0]-.01<=qmid[0]<=start[0]+.01:continue
      for margin in (.15,.5):
       first=plan('via_'+str(xyz),start,qmid,margin)
       if not first.get('ok') or travel(first['path'])[0]>math.pi:continue
       second=plan('from_via_to_park',qmid,park,margin)
       if not second.get('ok'):continue
       path=first['path']+second['path'][1:]
       if travel(path)[0]<math.pi:candidates.append(('via_'+str(xyz),path));break
      if candidates:break
     if candidates:break
   for label,path in candidates:
    repl={ids[0]:path,**fixed,ids[5]:transfer,ids[6]:r[ids[6]]['task']['q_path']}
    checks={k:check_path(p,r[k],v) for k,v in repl.items()};tr={k:travel(v) for k,v in repl.items()}
    out['candidates'].append({'label':label,'accepted':all(v['ok'] for v in checks.values()) and all(v[0]<=math.pi+1e-8 for v in tr.values()),'replacements':repl,'checks':checks,'travel_rad':tr,'base_travel_deg':sum(v[0] for v in tr.values())*180/math.pi,'context_records':{k:r[k] for k in ids}});save()
   out['status']='completed';save()
if __name__=='__main__':run()
