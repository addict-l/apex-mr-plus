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
 cases=json.loads((LAB/'accepted_cases.json').read_text());r=cases['all_reference_tasks'];accepted=json.loads((LAB/'support_pick_approach_candidates.json').read_text())['candidates'][0]
 rec=r['pick_support_approach'];contactrec=r['pick_support_contact'];start=rec['task']['q_path'][0];contact=accepted['replacements']['pick_support_contact'][-1]
 out={'status':'running','attempts':[],'candidates':[]}
 def save(): (LAB/'support_pick_side_candidates.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,record,left,right,margin):
    opt=options(record);opt.update(max_joint_step=.012,max_rrt_candidates=2,joint_bound_margin=margin)
    ans=p.plan('B',left,right,None,record['task']['metadata']['placed'],solve_time=3,options=opt)
    base=travel(ans['path'])[0]*180/math.pi if ans.get('ok') else None
    out['attempts'].append({'label':label,'margin':margin,'result':ans,'base_deg':base});save();print(label,margin,ans.get('ok'),base,flush=True)
    return ans
   tcp=p.fk('B',contact)['pose'];print('contact',tcp,flush=True)
   for dx in (.13,.09,.18):
    pose=[tcp[0]+dx,tcp[1],tcp[2]]+tcp[3:];opt=options(rec);opt.update(seed=contact,max_solutions=5,ik_time=.5)
    ik=p.solve_ik('B',pose[:3],pose[3:],None,rec['task']['metadata']['placed'],opt);out['attempts'].append({'label':'side_ik','pose':pose,'result':ik});save();print('sideik',dx,ik.get('ok'),flush=True)
    for side in ik.get('solutions',[])[:2]:
     if not -4<side[0]<-1.3:continue
     insert=plan('side_contact',contactrec,side,contact,.15)
     if not insert.get('ok'):continue
     approach=None
     for margin in (.15,.4):
      a=plan('home_side_'+str(dx),rec,start,side,margin)
      if a.get('ok') and travel(a['path'])[0]+travel(insert['path'])[0]<math.pi:approach=a['path'];break
     if approach is None:
      highpose=pose[:];highpose[2]=1.32;opt.update(seed=side,max_solutions=3)
      ikhigh=p.solve_ik('B',highpose[:3],highpose[3:],None,rec['task']['metadata']['placed'],opt);out['attempts'].append({'label':'high_side_ik','pose':highpose,'result':ikhigh});save()
      for high in ikhigh.get('solutions',[])[:2]:
       down=plan('side_down',rec,high,side,.15)
       if not down.get('ok'):continue
       a=plan('home_high_side',rec,start,high,.4)
       if a.get('ok') and travel(a['path'])[0]+travel(down['path'])[0]+travel(insert['path'])[0]<math.pi:approach=a['path']+down['path'][1:];break
     if approach is None:continue
     c=deepcopy(accepted);c['replacements']['pick_support_approach']=approach;c['replacements']['pick_support_contact']=insert['path'];c['label']='side_approach_'+str(dx)
     for k in ('pick_support_approach','pick_support_contact'):c['checks'][k]=check_path(p,r[k],c['replacements'][k]);c['travel_rad'][k]=travel(c['replacements'][k])
     c['base_travel_deg']=sum(v[0] for v in c['travel_rad'].values())*180/math.pi;c['accepted']=all(v['ok'] for v in c['checks'].values());c['timing']={}
     for k,path in c['replacements'].items():
      task=deepcopy(r[k]['task']);task['q_path']=path;task['dependencies']=[]
      tpg=build_tpg([task],max_joint_span=.24);parameterize(tpg);c['timing'][k]=sum(n['duration'] for n in tpg['nodes'])
     out['candidates'].append(c);save();print('candidate',c['label'],sum(c['timing'].values()),c['base_travel_deg'],flush=True)
     if sum(c['timing'].values())<50:break
    if out['candidates'] and min(sum(c['timing'].values()) for c in out['candidates'])<50:break
   out['status']='completed';save()
if __name__=='__main__':run()
