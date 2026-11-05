"""Update embedded UI source in all project scenes using the owned GUI."""
from pathlib import Path
from sim_connection import Simulator

ROOT=Path(__file__).resolve().parents[1]
link=Simulator(ROOT,port=23041,timeout=25)
client,sim=link.connect()
source=(ROOT/'src'/'scene_ui.lua').read_text(encoding='utf-8')
for mode,relative in [(1,'scenes/02_Capability_Change.ttt'),(2,'scenes/03_Temporary_Support.ttt'),(0,'scenes/01_Workcell_Overview.ttt')]:
    sim.loadScene(str(ROOT/relative))
    old=sim.getObject('/APEX_MR_Plus/SceneControls')
    sim.removeObjects([old])
    root=sim.getObject('/APEX_MR_Plus')
    script=sim.createScript(sim.scripttype_customization,source,0,'lua')
    sim.setObjectAlias(script,'SceneControls')
    sim.setObjectParent(script,root,True)
    sim.initScript(script)
    assert sim.callScriptFunction('setPreset',script,mode)
    sim.saveScene(str(ROOT/relative))
sim.saveScene(str(ROOT/'APEX_MR_Plus.ttt'))
link._disconnect(client)
print('Updated all embedded control panels; GUI left open on the main overview.')
