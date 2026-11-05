"""Regenerate portable scenes/previews from the geometry+IK build checkpoint."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from sim_connection import Simulator
from label_mesh import replace_labels

ROOT=Path(__file__).resolve().parents[1]

with Simulator(ROOT,port=23040,timeout=40) as owner:
    client,sim=owner.connect()
    sim.loadScene(str(ROOT/'config'/'_build_checkpoint.ttt'))
    root=sim.getObject('/APEX_MR_Plus')
    replace_labels(sim,root,ROOT/'assets')
    cfg=sim.readCustomTableData(root,'APEX_MR_Plus')
    cfg['prop_support']=[.08,.22,.76]
    sim.writeCustomTableData(root,'APEX_MR_Plus',cfg)
    script=sim.createScript(sim.scripttype_customization,(ROOT/'src'/'scene_ui.lua').read_text(encoding='utf-8'),0,'lua')
    sim.setObjectAlias(script,'SceneControls')
    sim.setObjectParent(script,root,True)
    sim.initScript(script)
    sim.callScriptFunction('initialize',script)
    cameras={name:sim.getObject('/APEX_MR_Plus/'+name+'RGB') for name in ['Overview','Detail']}
    for mode,name in [(1,'02_Capability_Change'),(2,'03_Temporary_Support'),(0,'01_Workcell_Overview')]:
        sim.callScriptFunction('setPreset',script,mode)
        sim.saveScene(str(ROOT/'scenes'/(name+'.ttt')))
        sim.handleVisionSensor(cameras['Overview'])
        raw,res=sim.getVisionSensorImg(cameras['Overview'])
        Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()).save(ROOT/'previews'/(name+'.png'))
        if mode==1:
            sim.handleVisionSensor(cameras['Detail'])
            raw,res=sim.getVisionSensorImg(cameras['Detail'])
            Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()).save(ROOT/'previews'/'04_Blocked_Closeup.png')
    sim.saveScene(str(ROOT/'APEX_MR_Plus.ttt'))
    excluded=['arms','root','connector','prop','bridge','default_camera','cameras']
    manifest={k:v for k,v in cfg.items() if k not in excluded}
    manifest.update(scene=str(ROOT/'APEX_MR_Plus.ttt'),simulator_version=sim.getInt32Param(sim.intparam_program_version))
    (ROOT/'config'/'scene_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'saved':manifest['scene'],'bridge_bottom':cfg['bridge_bottom'],'ik_residual_m':cfg['ik_residual_m']}),flush=True)
