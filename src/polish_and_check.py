"""Verify saved preview scenes and improve their embedded label mapping."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
from sim_connection import Simulator
from build_scene import look_at

ROOT=Path(__file__).resolve().parents[1]

with Simulator(ROOT, port=23040, timeout=40) as owner:
    client,sim=owner.connect()
    report=[]
    for mode,file in [(0,'scenes/01_Workcell_Overview.ttt'),(1,'scenes/02_Capability_Change.ttt'),(2,'scenes/03_Temporary_Support.ttt')]:
        sim.loadScene(str(ROOT/file))
        root=sim.getObject('/APEX_MR_Plus')
        objects={sim.getObjectAlias(h):h for h in sim.getObjectsInTree(root,sim.handle_all,0)}
        script=objects['SceneControls']
        sim.callScriptFunction('setPreset',script,mode)
        # More frontal view keeps both arms and the central structure visible.
        camera=objects['OverviewRGB']
        pose=look_at([1.10,-2.45,2.30],[0,.02,.99])
        sim.setObjectPose(camera,pose,-1)
        sim.setObjectPose(sim.getObject('/DefaultCamera'),pose,-1)
        sim.saveScene(str(ROOT/file))
        sim.handleVisionSensor(camera)
        raw,res=sim.getVisionSensorImg(camera)
        Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()).save(ROOT/'previews'/(Path(file).stem+'.png'))
        if mode==1:
            sim.handleVisionSensor(objects['DetailRGB'])
            raw,res=sim.getVisionSensorImg(objects['DetailRGB'])
            Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()).save(ROOT/'previews'/'04_Blocked_Closeup.png')
        collections={}
        for side in ['A','B']:
            c=sim.createCollection(0)
            sim.addItemToCollection(c,sim.handle_tree,objects[side+'_UR5'],0)
            collections[side]=c
        castle=sim.createCollection(0)
        sim.addItemToCollection(castle,sim.handle_tree,objects['CastleStructure'],0)
        pairs=[('A_table',collections['A'],objects['Workbench']),('B_table',collections['B'],objects['Workbench']),
               ('A_B',collections['A'],collections['B']),('A_castle',collections['A'],castle),('B_castle',collections['B'],castle)]
        checks={}
        for name,a,b in pairs:
            hit,handles=sim.checkCollision(a,b)
            checks[name]={'collision':bool(hit),'pair':[sim.getObjectAlias(h) for h in handles] if hit else []}
        sim.destroyCollection(castle)
        for c in collections.values(): sim.destroyCollection(c)
        current_legal=True
        for i in range(1,7):
            j=objects[f'A_joint_{i}']; cyclic,interval=sim.getJointInterval(j); q=sim.getJointPosition(j)
            current_legal &= bool(cyclic or interval[0]-1e-6 <= q <= sum(interval)+1e-6)
        report.append({'preset':mode,'collisions':checks,'A_current_joint_state_legal':current_legal})
        if mode==0:
            sim.saveScene(str(ROOT/'APEX_MR_Plus.ttt'))
    (ROOT/'config'/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
