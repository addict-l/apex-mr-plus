"""One-part actual-dynamics integration test from computed RRT paths."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from .scene import resolve
from .tpg import build_tpg
from .program import parameterize
from .runtime import install_controller,execute
with Simulator(ROOT,port=23043,timeout=45) as owner:
    client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
    partial=json.loads((ROOT/'results'/'normal'/'fast_reference'/'progress_reference.json').read_text(encoding='utf-8'))
    tasks=partial['tasks'];first=tasks[0]['part'];tasks=[t for t in tasks if t['part']==first]
    cfg['parts']=[p for p in cfg['parts'] if p['id']==first]
    tpg=parameterize(build_tpg(tasks,max_joint_span=.35))
    install_controller(sim,cfg,tpg,'config/_prefix_smoke.ttt')
    result=execute(sim,client,cfg,ROOT/'results'/'normal'/'prefix_smoke',video=False)
    print(json.dumps({'partial':True,'part':first,'success':result['success'],'failure':result.get('failure'),
                      'goal_errors':result['goal_position_errors_m'],'time':result['time']}),flush=True)
