"""Native bounded support pickup route experiment. No dynamics/promotions."""
from pathlib import Path
import sys,json,math,time
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; LAB=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(LAB)]
from support_return_probe import options,check_path,shifted,travel,hashes
from src.sim_connection import Simulator
from apex.scene import resolve
from apex.motion import MotionPlanner

def write(v):
 (LAB/'support_pick_candidates.json').write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')

def run():
 cases=json.loads((LAB/'accepted_cases.json').read_text(encoding='utf-8')); records=cases['all_reference_tasks']
 ids=['pick_support_approach','pick_support_contact','pick_support','pick_support_lift','pick_support_lift_contact_clear','support_route_front','support_high_approach']
 r={k:records[k] for k in ids}; out={'status':'running','scope':'native stopped geometry; fixed A holding bridge; actual and asynchronous pending','source_run_id':cases['run_id'],'source_scene':cases['source_scene'],'source_sha256_before':hashes(),'attempts':[],'candidates':[]};write(out)
 start=r[ids[0]]['task']['q_path'][0]; fixed={k:shifted(r[k]['task']['q_path']) for k in ids[1:5]}; park=fixed[ids[1]][0]; lift=fixed[ids[3]][-1]; routeend=r[ids[5]]['task']['q_path'][-1]; end=r[ids[-1]]['task']['q_path'][-1]
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,rec,a,b):
    opt=options(rec); opt.update(max_joint_step=.012,max_rrt_candidates=2,shortcut_budget_s=.35)
    ans=p.plan('B',a,b,rec['task'].get('payload'),rec['task']['metadata']['placed'],solve_time=3,options=opt)
    out['attempts'].append({'label':label,'q_start':a,'q_end':b,'result':ans});write(out)
    print(label,ans.get('ok'),ans.get('problem'),len(ans.get('path',[])),flush=True)
    return ans
   def ik(label,rec,pose,seed):
    opt=options(rec);opt.update(seed=seed,max_solutions=4,ik_time=.3)
    ans=p.solve_ik('B',pose[:3],pose[3:],rec['task'].get('payload'),rec['task']['metadata']['placed'],opt)
    out['attempts'].append({'label':label,'tcp_pose':pose,'ik_result':ans});write(out)
    print(label,ans.get('ok'),[np.round(q,3).tolist() for q in ans.get('solutions',[])],flush=True)
    return ans.get('solutions',[])
   out['fixed_checks']={k:check_path(p,r[k],path) for k,path in fixed.items()};write(out)
   if not all(v['ok'] for v in out['fixed_checks'].values()): raise RuntimeError('Fixed contact shift invalid')
   approach=plan('approach_negative_direct',r[ids[0]],start,park)
   quat=p.fk('B',park)['pose'][3:]
   if not approach.get('ok'):
    for z in (1.32,1.38):
     for qhigh in ik('park_upper_'+str(z),r[ids[0]],[.512,.43,z]+quat,park)[:2]:
      first=plan('approach_to_high',r[ids[0]],start,qhigh)
      if not first.get('ok'):continue
      second=plan('approach_high_to_park',r[ids[0]],qhigh,park)
      if second.get('ok'):
       approach={'ok':True,'path':first['path']+second['path'][1:]};break
     if approach.get('ok'):break
   if not approach.get('ok'):out['status']='no_approach';write(out);return
   direct=plan('route_negative_to_original_front',r[ids[5]],lift,routeend)
   transfers=[]
   if direct.get('ok') and travel(direct['path'])[0]<=math.pi:transfers.append(('direct',direct['path']))
   if not transfers:
    for z in (1.32,1.38):
     qpark=ik('payload_upper_park_'+str(z),r[ids[5]],[.512,.43,z]+quat,lift)
     for qp in qpark[:2]:
      up=plan('payload_up',r[ids[5]],lift,qp)
      if not up.get('ok'):continue
      # First try fixed original front; safe native planner can use upper endpoint as escape.
      across=plan('upper_park_to_original_front',r[ids[5]],qp,routeend)
      if across.get('ok'):
       path=up['path']+across['path'][1:]
       if travel(path)[0]<=math.pi:transfers.append(('high_park_'+str(z),path));break
      for x in (.30,.35,.25):
       mids=ik('upper_mid_'+str(x)+'_'+str(z),r[ids[5]],[x,.13,z]+quat,qp)
       for mid in mids[:2]:
        if not -3.65<=mid[0]<=-.40:continue
        a=plan('upper_park_to_mid',r[ids[5]],qp,mid)
        if not a.get('ok'):continue
        b=plan('mid_to_original_front',r[ids[5]],mid,routeend)
        if b.get('ok'):
         path=up['path']+a['path'][1:]+b['path'][1:]
         if travel(path)[0]<=math.pi:transfers.append(('upper_mid_'+str(x)+'_'+str(z),path));break
       if transfers:break
      if transfers:break
     if transfers:break
   for label,path in transfers:
    repl={ids[0]:approach['path'],**fixed,ids[5]:path,ids[6]:r[ids[6]]['task']['q_path']}
    checks={k:check_path(p,r[k],v) for k,v in repl.items()}
    tr={k:travel(v) for k,v in repl.items()}
    accepted=all(v['ok'] for v in checks.values()) and all(v[0]<=math.pi+1e-8 for v in tr.values())
    out['candidates'].append({'label':label,'accepted':accepted,'replacements':repl,'checks':checks,'travel_rad':tr,'base_travel_deg':sum(v[0] for v in tr.values())*180/math.pi,'start_anchor':start,'end_anchor':end,'context_records':r})
    write(out)
   out['status']='completed';out['source_sha256_after']=hashes();write(out)
if __name__=='__main__':run()
