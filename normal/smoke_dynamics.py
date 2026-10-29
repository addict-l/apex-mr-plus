"""Check actual dynamic enablement and hold tracking, rather than flags alone."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve
with Simulator(ROOT,port=23043,timeout=40) as owner:
    client,sim=owner.connect();sim.loadScene(str(ROOT/'scenes'/'10_Normal_Assembly.ttt'));cfg=resolve(sim)
    sim.setStepping(True);sim.startSimulation()
    for _ in range(50):sim.step()
    data={'time':sim.getSimulationTime(),'arms':{},'parts':{}}
    for side,a in cfg['arms'].items():
        q=[sim.getJointPosition(j) for j in a['joints']]
        data['arms'][side]={'joint_dynamic':[sim.isDynamicallyEnabled(j) for j in a['joints']],
            'max_hold_error_rad':max(abs(x-y) for x,y in zip(q,a['home_q'])),
            'tool_sensor_dynamic':sim.isDynamicallyEnabled(a['sensor']),
            'pad_sensor_dynamic':sim.isDynamicallyEnabled(a['support_sensor']),
            'pad_dynamic':sim.isDynamicallyEnabled(a['support_pad']),
            'tool_force':sim.readForceSensor(a['sensor']),'pad_force':sim.readForceSensor(a['support_sensor'])}
    for name,h in cfg['blocks'].items():
        data['parts'][name]={'dynamic':sim.isDynamicallyEnabled(h),'position':sim.getObjectPosition(h,-1),'initial':cfg['initial'][name]}
    sim.stopSimulation()
    (ROOT/'results'/'normal'/'dynamics_smoke.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2),flush=True)
