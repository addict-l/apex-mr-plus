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
 cases=json.loads((LAB/'accepted_cases.json').read_text());r=cases['all_reference_tasks'];pick=json.loads((LAB/'support_pick_side_candidates.json').read_text())['candidates'][0]
 retreat=deepcopy(list(reversed(pick['replacements']['pick_support_contact'])))
 for q in retreat:q[5]-=2*math.pi
 release=shifted(r['return_support']['task']['q_path'])[-1];print('initialdelta',np.asarray(retreat[0])-release,flush=True);retreat[0]=release
 side=retreat[-1];rec=r['pick_gate_approach'];end=rec['task']['q_path'][-1]
 out={'status':'running','source_run_id':cases['run_id'],'attempts':[],'candidates':[]}
 def save(): (LAB/'support_return_low_side_aux.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 with Simulator(LAB,port=23068,timeout=45) as owner:
  client,sim=owner.connect();sim.loadScene(cases['source_scene']);cfg=resolve(sim)
  with MotionPlanner(sim,client,cfg) as p:
   def plan(label,record,left,right,margin):
    opt=options(record);opt.update(max_joint_step=.012,max_rrt_candidates=2,joint_bound_margin=margin)
    ans=p.plan('B',left,right,record['task'].get('payload'),record['task']['metadata']['placed'],solve_time=3,options=opt)
    base=travel(ans['path'])[0]*180/math.pi if ans.get('ok') else None
    out['attempts'].append({'label':label,'margin':margin,'result':ans,'base_deg':base});save();print(label,margin,ans.get('ok'),base,flush=True)
    return ans
   def ik(label,pose,seed):
    opt=options(rec);opt.update(seed=seed,max_solutions=5,ik_time=.3)
    ans=p.solve_ik('B',pose[:3],pose[3:],None,rec['task']['metadata']['placed'],opt);out['attempts'].append({'label':label,'pose':pose,'result':ans});save();print(label,ans.get('ok'),flush=True)
    return ans.get('solutions',[])
   proof=check_path(p,r['return_support_retreat'],retreat);out['retreat_check']=proof;save()
   if not proof['ok']:raise RuntimeError('Low side retreat invalid')
   paths=[]
   for margin in (.05,.15,.4,.8):
    a=plan('side_to_gate',rec,side,end,margin)
    if a.get('ok') and travel(a['path'])[0]<=math.pi:paths.append(('direct_side_'+str(margin),a['path']));break
   sidetcp=p.fk('B',side)['pose'];gatetcp=p.fk('B',end)['pose']
   if not paths:
    for z in (1.32,1.20,1.38):
     highside=ik('side_high_'+str(z),sidetcp[:2]+[z]+sidetcp[3:],side)
     for hs in highside[:2]:
      if not -3.7<=hs[0]<=-.3:continue
      up=plan('side_up',rec,side,hs,.15)
      if not up.get('ok'):continue
      direct=plan('high_side_to_gate',rec,hs,end,.15)
      if direct.get('ok'):
       path=up['path']+direct['path'][1:]
       if travel(path)[0]<=math.pi:paths.append(('high_side_'+str(z),path));break
      highgate=ik('gate_high_'+str(z),gatetcp[:2]+[z]+gatetcp[3:],end)
      for hg in highgate[:2]:
       if not -3.7<=hg[0]<=-.3:continue
       across=plan('side_high_to_gate_high',rec,hs,hg,.15)
       if not across.get('ok'):continue
       down=plan('gate_down',rec,hg,end,.15)
       if down.get('ok'):
        path=up['path']+across['path'][1:]+down['path'][1:]
        if travel(path)[0]<=math.pi:paths.append(('overhead_'+str(z),path));break
      if paths:break
     if paths:break
   incoming=plan('return_incoming_negative',r['return_support_approach'],r['return_support_approach']['task']['q_path'][0],shifted(r['return_support_contact']['task']['q_path'])[0],.15)
   if not incoming.get('ok'):raise RuntimeError('Return incoming invalid')
   for label,path in paths:
    repl={'return_support_approach':incoming['path'],'return_support_contact':shifted(r['return_support_contact']['task']['q_path']),'return_support':shifted(r['return_support']['task']['q_path']),'return_support_retreat':retreat,'return_support_retreat_contact_clear':[side,side],'pick_gate_approach':path}
    checks={k:check_path(p,r[k],v) for k,v in repl.items()};tr={k:travel(v) for k,v in repl.items()};c={'label':label,'accepted':all(v['ok'] for v in checks.values()) and all(v[0]<=math.pi+1e-8 for v in tr.values()),'replacements':repl,'checks':checks,'travel_rad':tr,'base_travel_deg':sum(v[0] for v in tr.values())*180/math.pi,'timing':{},'context_records':{k:r[k] for k in repl}}
    for k,v in repl.items():
     task=deepcopy(r[k]['task']);task['q_path']=v;task['dependencies']=[];tpg=build_tpg([task],max_joint_span=.24);parameterize(tpg);c['timing'][k]=sum(n['duration'] for n in tpg['nodes'])
    out['candidates'].append(c);save();print('candidate',sum(c['timing'].values()),c['base_travel_deg'],c['accepted'],flush=True)
   out['status']='completed';save()
if __name__=='__main__':run()
