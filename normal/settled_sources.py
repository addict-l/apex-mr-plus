from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve,PARTS
with Simulator(ROOT,port=23043,timeout=40) as owner:
    client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
    for part in cfg['parts']:
        initial=next(p['source'] for p in PARTS if p['id']==part['id'])
        part['source']=list(initial);cfg['initial'][part['id']]=list(initial)
        sim.setObjectPosition(cfg['blocks'][part['id']],-1,initial)
    sim.writeCustomTableData(cfg['root'],'NormalAssembly',cfg)
    sim.saveScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'))
    print('Source heights aligned to their actual tray/buffer surfaces.',flush=True)
