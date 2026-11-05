"""Repair native-UR5 mounts and force-sensor attachment, then run 50 frames.

This is deliberately limited to the shared normal scene. It preserves source
coordinates, endpoint configurations and all part bodies. No runtime pose
commands are used in the dynamic hold test.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from sim_connection import Simulator
from normal.scene import resolve
from normal.probe_dynamics import ancestry, run


def main(metadata_only=False):
    scene=ROOT/'scenes'/'10_Normal_Assembly.ttt'
    report={'scene':str(scene),'input_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
            'fixes':[],'scene_saved':False}
    with Simulator(ROOT,port=23045,timeout=40) as owner:
        client,sim=owner.connect(); sim.loadScene(str(scene)); cfg=resolve(sim)
        if metadata_only:
            cfg['obstacles']=[sim.getObject('/APEX_MR_Plus/'+name)
                              for name in ['CastleBaseplate','A_SupplyTray','B_SupplyTray']]
            sim.writeCustomTableData(cfg['root'],'NormalAssembly',cfg)
            sim.saveScene(str(scene))
            output=ROOT/'results'/'normal'/'dynamics_fix.json'
            previous=json.loads(output.read_text(encoding='utf-8'))
            previous['obstacle_paths']=[sim.getObjectAlias(h,1) for h in cfg['obstacles']]
            previous['output_sha256']=hashlib.sha256(scene.read_bytes()).hexdigest()
            output.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'metadata_saved':True,'obstacle_paths':previous['obstacle_paths']},ensure_ascii=False),flush=True)
            return
        initial_sources={p['id']:list(p['source']) for p in cfg['parts']}
        report['source_positions_preserved']=initial_sources
        # Removal of an ordinary dummy does not recursively delete its old
        # children. Preserve all current robot and part trees by live handles,
        # then remove only named legacy geometry/obsolete robot branches.
        protected={cfg['root'],cfg['table'],cfg['camera'],*cfg['obstacles']}
        for arm in cfg['arms'].values():
            protected.update(sim.getObjectsInTree(arm['model'],sim.handle_all,0))
        for body in cfg['blocks'].values():
            protected.update(sim.getObjectsInTree(body,sim.handle_all,0))
        all_objects=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
        remove={}
        preview_prefixes=('A_SupplyBrick_','B_SupplyBrick_','A_SupplyTray_',
                          'B_SupplyTray_','LeftTower_','RightTower_',
                          'Left_Crenel','Right_Crenel','Left_Window','Right_Window',
                          'Left_Tower','Right_Tower','BridgeModule','RedConnector',
                          'ConnectorGoalOutline','ConnectorInstallTarget',
                          'TemporaryPier','PierBase','PierStem','PierTopContact',
                          'FaultIndicator')
        for h in all_objects:
            if h in protected:
                continue
            alias=sim.getObjectAlias(h)
            reason=None
            if alias.startswith(preview_prefixes):
                reason='legacy_preview_geometry'
            elif alias=='CastleBaseplate':
                reason='duplicate_baseplate_not_used_by_cfg'
            elif alias.startswith(('A_joint_','B_joint_','A_TCP','B_TCP',
                                   'A_Gripper','B_Gripper','A_Finger','B_Finger',
                                   'A_Palm','B_Palm','A_SupportShoe','B_SupportShoe')):
                reason='obsolete_robot_tree_outside_current_cfg'
            if reason:
                for child in sim.getObjectsInTree(h,sim.handle_all,0):
                    if child not in protected:
                        remove[child]=reason
        report['removed_legacy_objects']=[
            {'alias':sim.getObjectAlias(h),'path':sim.getObjectAlias(h,1),
             'type':sim.getObjectType(h),'reason':reason} for h,reason in remove.items()]
        if remove:
            sim.removeObjects(list(remove))
        cfg['obstacles']=[sim.getObject('/APEX_MR_Plus/'+name)
                          for name in ['CastleBaseplate','A_SupplyTray','B_SupplyTray']]
        report['obstacle_paths']=[sim.getObjectAlias(h,1) for h in cfg['obstacles']]
        proposal_path=ROOT/'results'/'normal'/'motion_home_proposal.json'
        proposed=json.loads(proposal_path.read_text(encoding='utf-8'))
        report['home_proposal']=proposed
        for side,arm in cfg['arms'].items():
            q=proposed[side]
            if len(q)!=6 or not all(math.isfinite(v) for v in q):
                raise ValueError(f'Invalid home proposal for {side}')
            arm['home_q']=list(q)
            for joint,value in zip(arm['joints'],q):
                sim.setJointPosition(joint,value)
                sim.setJointTargetPosition(joint,value)
        for side,arm in cfg['arms'].items():
            base=arm['model']; sensor=arm['sensor']
            old_parent=sim.getObjectParent(sensor)
            sensor_pose=sim.getObjectPose(sensor,-1)
            was_static=sim.getObjectInt32Param(base,sim.shapeintparam_static)
            sim.setObjectInt32Param(base,sim.shapeintparam_static,1)
            parent=sim.getObjectParent(arm['connection'])
            while parent>=0:
                if (sim.getObjectType(parent)==sim.object_shape_type
                    and sim.getObjectInt32Param(parent,sim.shapeintparam_static)==0):
                    break
                parent=sim.getObjectParent(parent)
            if parent<0:
                raise RuntimeError(f'{side}: no dynamic shape above the native connection')
            sim.setObjectParent(sensor,parent,True)
            after_pose=sim.getObjectPose(sensor,-1)
            if math.dist(sensor_pose[:3],after_pose[:3])>1e-9:
                raise RuntimeError('Tool sensor world position changed during reparent')
            report['fixes'].append({'arm':side,'base_static_before':bool(was_static),
                'base_static_after':True,'old_tool_parent':sim.getObjectAlias(old_parent),
                'new_tool_parent':sim.getObjectAlias(parent),
                'tool_world_translation_change_m':math.dist(sensor_pose[:3],after_pose[:3]),
                'tool_ancestors':ancestry(sim,sensor)})
        # Run all source/static collision checks inside Lua to avoid thousands
        # of remote round trips. Merely inspect; no legacy shapes are removed.
        code='''sim=require 'sim'
function sysCall_init() end
function initialContacts(blocks)
 local out={}
 local shapes=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
 for name,b in pairs(blocks) do
  for _,h in ipairs(shapes) do
   if h~=b and sim.getObjectInt32Param(h,sim.shapeintparam_static)~=0 and sim.getObjectInt32Param(h,sim.shapeintparam_respondable)~=0 then
    if sim.checkCollision(b,h)>0 then
     out[#out+1]={part=name,static_shape=sim.getObjectAlias(h,1),alias=sim.getObjectAlias(h)}
    end
   end
  end
 end
 return out
end'''
        inspector=sim.createScript(sim.scripttype_customization,code,0,'lua')
        sim.setObjectAlias(inspector,'TemporaryDynamicContactInspector'); sim.initScript(inspector)
        report['initial_static_collisions']=sim.callScriptFunction('initialContacts',inspector,cfg['blocks'])
        sim.removeObjects([inspector])
        cfg['physics']['base_mount']='UR5 model-root shape is static and anchored to the worktable'
        cfg['physics']['tool_force_chain']='last dynamic UR5 link -> ToolForceSensor -> dynamic Palm; native empty connection sensor is bypassed'
        sim.writeCustomTableData(cfg['root'],'NormalAssembly',cfg)
        sim.saveScene(str(scene)); report['scene_saved']=True
        initial_arms={side:{'base':sim.getObjectPose(a['model'],-1),'tcp':sim.getObjectPose(a['tip'],-1)}
                      for side,a in cfg['arms'].items()}
        samples=run(sim,cfg,50); report['samples']=samples
        final=samples[-1]
        report['summary']={
            'all_12_joints_dynamic':all(all(a['joint_dynamic']) for a in final['arms'].values()),
            'all_tool_sensors_dynamic':all(a['tool_sensor_dynamic'] for a in final['arms'].values()),
            'all_pad_sensors_dynamic':all(a.get('pad_sensor_dynamic',False) for a in final['arms'].values()),
            'all_pads_dynamic':all(a.get('pad_dynamic',False) for a in final['arms'].values()),
            'all_parts_dynamic':all(p['dynamic'] for p in final['parts'].values()),
            'all_force_data_valid':all(a['tool_sensor']['valid'] and a['pad_sensor']['valid'] for a in final['arms'].values()),
            'base_translation_drift_m':{side:math.dist(a['base_world_pose'][:3],initial_arms[side]['base'][:3])
                                        for side,a in final['arms'].items()},
            'tcp_translation_drift_m':{side:math.dist(a['tcp_world_pose'][:3],initial_arms[side]['tcp'][:3])
                                       for side,a in final['arms'].items()},
            'max_joint_target_error_rad':max(a['max_error_to_target_rad'] for a in final['arms'].values()),
            'max_source_part_displacement_m':max(math.dist(p['position_m'],p['initial_m']) for p in final['parts'].values()),
            'initial_static_collision_count':len(report['initial_static_collisions']),
            'removed_legacy_object_count':len(report['removed_legacy_objects']),
        }
        report['output_sha256']=hashlib.sha256(scene.read_bytes()).hexdigest()
        report['status']='passed' if (all(report['summary'][k] for k in [
            'all_12_joints_dynamic','all_tool_sensors_dynamic','all_pad_sensors_dynamic',
            'all_pads_dynamic','all_parts_dynamic','all_force_data_valid'])
            and max(report['summary']['base_translation_drift_m'].values())<1e-9
            and max(report['summary']['tcp_translation_drift_m'].values())<.002
            and report['summary']['max_source_part_displacement_m']<.005
            and not report['initial_static_collisions']) else 'needs_attention'
    output=ROOT/'results'/'normal'/'dynamics_fix.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],**report['summary'],
                      'initial_static_collisions':report['initial_static_collisions']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata-only',action='store_true')
    main(parser.parse_args().metadata_only)
