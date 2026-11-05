"""Reload the saved design scene and check its panel and all five presets."""
import argparse
import hashlib
import json
from pathlib import Path
from sim_connection import Simulator

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gui',action='store_true',help='Also validate the native simUI panel')
    parser.add_argument('--scene',type=Path,help='Static preview scene; relative paths resolve against the scene3 directory')
    args=parser.parse_args()
    scene=args.scene
    if scene is None:
        scene=ROOT/'scenes/00_Scene3_Design_Preview.ttt'
        if not scene.is_file():scene=ROOT/'backups/design_preview/apex_mr_scene3.ttt'
    elif not scene.is_absolute():scene=ROOT/scene
    if not scene.is_file():raise FileNotFoundError('Build_Scene3_Preview.cmd must create the preview first: '+str(scene))
    results=[]
    with_owner=Simulator(ROOT,port=23055,timeout=40)
    try:
        with_owner.start(headless=not args.gui)
        _,sim=with_owner.connect()
        sim.loadScene(str(scene))
        root=sim.getObject('/APEX_MR_Scene3')
        script=sim.getObject('/APEX_MR_Scene3/SceneControls')
        sim.callScriptFunction('initialize',script)
        def obj(alias): return sim.getObject('/APEX_MR_Scene3/'+alias)
        walls=[h for h in sim.getObjectsInTree(root,sim.object_shape_type,0)
               if sim.getObjectAlias(h).startswith('Pocket')]
        robot_cols=[]
        for side in ('A','B'):
            collection=sim.createCollection(0)
            sim.addItemToCollection(collection,sim.handle_tree,obj(side+'_UR5'),0)
            robot_cols.append(collection)
        tool_col=sim.createCollection(0)
        sim.addItemToCollection(tool_col,sim.handle_tree,obj('TemporarySupport'),0)
        for mode in range(5):
            assert sim.callScriptFunction('setPreset',script,mode)
            state=sim.callScriptFunction('getPreviewState',script)
            assert state['preset']==mode
            assert state['dynamics_validated'] is False
            if args.gui: assert state['ui_created'], 'Native panel missing'
            cfg=sim.readCustomTableData(root,'Scene3')
            expected=cfg['states'][mode]
            for alias,desired in expected['objects'].items():
                handle=sim.getObject('/APEX_MR_Scene3/'+alias)
                actual=sim.getObjectPose(handle,-1)
                assert max(abs(a-b) for a,b in zip(actual[:3],desired[:3]))<1e-6, alias
                assert abs(abs(sum(a*b for a,b in zip(actual[3:],desired[3:])))-1)<1e-6, alias
            for side,col in zip(('A','B'),robot_cols):
                collision=sim.checkCollision(col,tool_col)
                assert not collision[0], f'Arm {side} intersects support tool in mode {mode}: {collision}'
            if mode in (3,4):
                for wall in walls:
                    assert not sim.checkCollision(obj('Gate/Gate_Body'),wall)[0], 'Closed gate intersects pocket wall'
            if mode == 0:
                for name in ('K1','K2','Gate'):
                    body=obj(name+'/'+name+'_Body')
                    bottom=sim.getObjectPosition(body,-1)[2]+sim.getObjectFloatParam(body,sim.objfloatparam_objbbox_min_z)
                    assert abs(bottom-.766)<1e-6, 'Source object not resting on supply mat: '+name
                foot=obj('TemporarySupport/BallastFoot')
                foot_bottom=sim.getObjectPosition(foot,-1)[2]+sim.getObjectFloatParam(foot,sim.objfloatparam_objbbox_min_z)
                assert abs(foot_bottom-.760)<1e-6, 'Parked support not on table'
            results.append(state)
        # Returning to the starting state must reset all moved objects.
        assert sim.callScriptFunction('setPreset',script,0)
        output={'kind':'saved_scene_and_UI_validation','gui_checked':args.gui,'states':results,
                'geometry_checked':['source_parts_rest_on_supply_mats','parked_support_on_table',
                                    'closed_gate_clear_of_walls','arms_clear_of_support_in_all_presets'],
                'scene':str(scene),'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
                'scope':'Saved design presets and panel only; not motion execution or dynamics.'}
        (ROOT/'results'/('gui_validation.json' if args.gui else 'scene_validation.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'validated_presets':len(results),'gui_checked':args.gui}),flush=True)
    finally:
        with_owner.close()

if __name__=='__main__': main()
