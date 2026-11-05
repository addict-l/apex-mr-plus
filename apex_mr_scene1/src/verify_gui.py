"""Smoke-check project GUI controls and capture its actual application window."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import time
from sim_connection import Simulator

ROOT=Path(__file__).resolve().parents[1]
session=json.loads((ROOT/'logs'/'gui_session.json').read_text(encoding='utf-8'))
connection=Simulator(ROOT,port=session['port'],timeout=25)
client,sim=connection.connect()
script=sim.getObject('/APEX_MR_Plus/SceneControls')
states=[]
for mode in [1,2,0]:
    accepted=sim.callScriptFunction('setPreset',script,mode)
    state=sim.callScriptFunction('getPreviewState',script)
    assert accepted and state['preset']==mode and not state['planner_connected']
    states.append(state)
sim.setStepping(True)
sim.startSimulation()
for _ in range(6): sim.step()
elapsed=sim.getSimulationTime()
sim.stopSimulation()
for _ in range(20):
    if sim.getSimulationState()==sim.simulation_stopped: break
    sim.step()
assert sim.getSimulationState()==sim.simulation_stopped
sim.setStepping(False)
sim.callScriptFunction('setPreset',script,0)
sim.callScriptFunction('overview',script)
report={'scene':session['scene'],'pid':session['pid'],'presets':states,'play_stop_smoke_s':elapsed,'stopped':True}
(ROOT/'config'/'gui_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
connection._disconnect(client)

print(json.dumps(report,ensure_ascii=False),flush=True)
