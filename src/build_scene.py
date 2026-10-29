"""Build the APEX MR Plus workcell in CoppeliaSim (SI units).

This is an environment/IK preview, not an autonomous recovery planner.
Every saved state is explicitly identified as a preset scene snapshot.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from sim_connection import Simulator
from label_mesh import make_label

ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = .76
ORANGE = [1., .34, .055]
BLUE = [.025, .40, .86]
CREAM = [.88, .77, .55]
RED = [.84, .085, .09]
GREEN = [.10, .61, .31]
YELLOW = [1., .75, .05]


def look_at(eye, target):
    eye, target = np.array(eye), np.array(target)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return eye.tolist() + Rotation.from_matrix(np.column_stack([x, y, z])).as_quat().tolist()


def build(sim, client, install):
    sim.closeScene()
    sim.setFloatParam(sim.floatparam_simulation_time_step, .05)
    sim.setBoolParam(sim.boolparam_realtime_simulation, True)
    root = sim.createDummy(.001)
    sim.setObjectAlias(root, 'APEX_MR_Plus')
    sim.setObjectInt32Param(root, sim.objintparam_visibility_layer, 0)
    sim.setArrayParam(sim.arrayparam_background_color1, [.88, .92, .96])
    sim.setArrayParam(sim.arrayparam_background_color2, [.96, .97, .99])
    sim.setArrayParam(sim.arrayparam_ambient_light, [.40, .40, .40])
    count = [0]

    def dummy(name, pos=(0, 0, 0), parent=root):
        h = sim.createDummy(.002)
        sim.setObjectAlias(h, name)
        sim.setObjectParent(h, parent, True)
        sim.setObjectPosition(h, -1, list(pos))
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        return h

    def shape(name, size, pos, color, parent=root, kind=None, collision=True):
        count[0] += 1
        h = sim.createPrimitiveShape(kind or sim.primitiveshape_cuboid, list(size), 3)
        sim.setObjectAlias(h, name)
        sim.setObjectParent(h, parent, True)
        sim.setObjectPosition(h, -1, list(pos))
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setShapeColor(h, None, sim.colorcomponent_specular, [.18, .18, .18])
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, int(collision))
        prop = sim.objectspecialproperty_renderable
        if collision:
            prop |= sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
        sim.setObjectSpecialProperty(h, prop)
        return h

    def label(name, file, pos, size, vertical=False, parent=root):
        return make_label(sim, parent, name, ROOT/'assets'/file, pos, size, vertical)

    def brick(name, pos, nx=2, ny=2, color=CREAM, height=.032, pitch=.026, parent=root):
        group = dummy(name, pos, parent)
        width, depth = nx * pitch - .001, ny * pitch - .001
        shape(name + '_Body', [width, depth, height], pos, color, group)
        for i in range(nx):
            for j in range(ny):
                sp = [pos[0] + (i - (nx - 1) / 2) * pitch,
                      pos[1] + (j - (ny - 1) / 2) * pitch,
                      pos[2] + height / 2 + .002]
                shape(f'{name}_Stud_{i}_{j}', [.014, .014, .004], sp, color, group,
                      sim.primitiveshape_cylinder, False)
        return group

    def tray(name, center, color):
        group = dummy(name, center)
        x, y, z = center
        shape(name + '_Floor', [.30, .19, .008], [x, y, z], [.22, .27, .34], group)
        for dx in [-.151, .151]:
            shape(name + '_Side', [.008, .20, .018], [x + dx, y, z + .009], color, group)
        for dy in [-.099, .099]:
            shape(name + '_Edge', [.31, .008, .018], [x, y + dy, z + .009], color, group)
        return group

    # Workbench and clear colored work zones.
    shape('Workbench', [1.90, 1.27, .065], [0, 0, TABLE_Z - .0325], [.72, .77, .81])
    shape('WorkbenchFrontRail', [1.90, .025, .075], [0, -.635, TABLE_Z - .045], [.12, .17, .24])
    for x in [-.80, .80]:
        for y in [-.49, .49]:
            shape('WorkbenchLeg', [.065, .065, .67], [x, y, .36], [.19, .23, .29])
    for side, x, color in [('A', -.70, ORANGE), ('B', .70, BLUE)]:
        shape(side + '_MountingPlate', [.23, .24, .022], [x, .13, TABLE_Z + .011], [.20, .24, .29])
        for dx in [-.09, .09]:
            for dy in [-.09, .09]:
                shape(side + '_MountBolt', [.011, .011, .008], [x + dx, .13 + dy, TABLE_Z + .026],
                      [.42, .45, .49], kind=sim.primitiveshape_cylinder, collision=False)
        label(side + '_RobotLabel', 'label_' + side + '.png', [x, -.065, TABLE_Z + .002], [.31, .0775])

    # Castle: visible modular bricks, two square towers with crenellations.
    assembly = dummy('CastleStructure')
    shape('CastleBaseplate', [.67, .31, .012], [0, .22, TABLE_Z + .006], [.24, .32, .40], assembly)
    tower_x = [-.225, .225]
    base = TABLE_Z + .012
    for side, x in zip(['Left', 'Right'], tower_x):
        layers = 8 if side == 'Left' else 6
        for layer in range(layers):
            brick(f'{side}Tower_Layer{layer + 1}', [x, .22, base + .016 + layer * .032],
                  5, 5, CREAM if layer != 2 else [.74, .57, .36], parent=assembly)
        # Small window facade, only visual; not a hollow structural cut-out.
        shape(side + '_WindowInset', [.023, .002, .047], [x, .153, base + .095], [.09, .15, .21], assembly, collision=False)
        for dx in [-.041, .041]:
            for dy in [-.055, .055]:
                brick(side + '_Crenel', [x + dx, .22 + dy, base + layers*.032 + .018], 1, 1, CREAM, .028, parent=assembly)
    bridge_bottom = base + .260
    bridge = brick('BridgeModule', [0, .22, bridge_bottom + .016], 16, 3, GREEN, .032, parent=root)
    # Target dummy marks the connector across the bridge/tower boundary.
    target = dummy('ConnectorInstallTarget', [.203, .22, bridge_bottom + .054])
    target_shape = shape('ConnectorGoalOutline', [.055, .060, .003], [.203, .22, bridge_bottom + .038], [.96, .30, .29], collision=False)
    sim.setShapeColor(target_shape, None, sim.colorcomponent_transparency, [.65])
    label('CastleLabel', 'label_goal.png', [0, .005, TABLE_Z + .002], [.41, .1025])

    # Left and right material trays, with loose visible parts.
    for side, x, color in [('A', -.48, ORANGE), ('B', .48, BLUE)]:
        t = tray(side + '_SupplyTray', [x, -.335, TABLE_Z + .009], color)
        label(side + '_TrayLabel', 'label_supply_' + side + '.png', [x, -.465, TABLE_Z + .002], [.27, .0675])
        for i in range(3):
            for j in range(2):
                c = [RED, CREAM, YELLOW][i] if side == 'A' else [CREAM, GREEN, [.18, .50, .87]][i]
                brick(f'{side}_SupplyBrick_{i}_{j}', [x + (i - 1) * .078, -.335 + (j - .5) * .077, TABLE_Z + .032],
                      2, 2, c, parent=t)
    connector = brick('RedConnector', [-.35, -.18, TABLE_Z + .030], 2, 2, RED)

    # Shared buffer: independent visible landing pad, two sample placement frames.
    buffer = dummy('SharedBuffer')
    shape('BufferBase', [.25, .17, .026], [0, -.35, TABLE_Z + .013], [.44, .29, .69], buffer)
    shape('BufferLandingSurface', [.23, .15, .003], [0, -.35, TABLE_Z + .028], [.77, .69, .91], buffer)
    for x in [-.055, .055]:
        pad = shape('BufferPoseMarker', [.060, .060, .001], [x, -.35, TABLE_Z + .030], [.89, .85, .97], buffer, collision=False)
    label('BufferLabel', 'label_buffer.png', [0, -.49, TABLE_Z + .002], [.28, .07])

    # Preassembled side-insertable temporary support, stored away from bridge.
    prop_park = [-.20, -.115, TABLE_Z]
    pier = dummy('TemporaryPier', prop_park)
    pier_h = bridge_bottom - TABLE_Z
    shape('PierBase', [.09, .09, .012], [-.20, -.115, TABLE_Z + .006], YELLOW, pier)
    shape('PierStem', [.050, .050, pier_h - .024], [-.20, -.115, TABLE_Z + pier_h / 2], YELLOW, pier)
    shape('PierTopContact', [.09, .09, .012], [-.20, -.115, bridge_bottom - .006], YELLOW, pier)
    label('PierLabel', 'label_prop.png', [-.20, -.215, TABLE_Z + .002], [.20, .05])

    # Thin labels are embedded textures: scene remains portable without PNG files.
    shape('Backboard', [1.12, .025, .215], [0, .59, 1.36], [.075, .12, .20], collision=False)
    label('Title', 'title.png', [0, .575, 1.37], [1.08, .2025], vertical=True)
    label('Legend', 'legend.png', [0, -.651, .66], [1.08, .2025], vertical=True)
    fault_label = label('FaultIndicator', 'label_fault.png', [-.63, .08, 1.35], [.28, .07], vertical=True)
    sim.setObjectInt32Param(fault_label, sim.objintparam_visibility_layer, 0)

    ik = client.require('simIK')
    arms = {}
    for side, x, yaw, color in [('A', -.70, 0., ORANGE), ('B', .70, math.pi, BLUE)]:
        model = sim.loadModel(str(install / 'models/robots/non-mobile/UR5.ttm'))
        sim.setObjectAlias(model, side + '_UR5')
        scripts = sim.getObjectsInTree(model, sim.object_script_type, 0)
        if scripts:
            sim.removeObjects(scripts)
        sim.setObjectParent(model, root, True)
        sim.setObjectPose(model, [x, .13, TABLE_Z + .023] + Rotation.from_euler('z', yaw).as_quat().tolist(), -1)
        joints = sim.getObjectsInTree(model, sim.object_joint_type, 0)
        for n, j in enumerate(joints, 1):
            sim.setObjectAlias(j, f'{side}_joint_{n}')
            sim.setJointMode(j, sim.jointmode_kinematic, 0)
            sim.setJointInterval(j, True, [-math.pi, 2*math.pi])
        shapes = sim.getObjectsInTree(model, sim.object_shape_type, 0)
        for h in shapes:
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
            # Keep dark housings; recolor the original pale blue UR5 shells.
            _, original = sim.getShapeColor(h, None, sim.colorcomponent_ambient_diffuse)
            if original and max(original) > .35:
                sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        connection = sim.getObject(':/connection', {'proxy': model})
        palm = shape(side + '_GripperPalm', [.050, .084, .023], [0, 0, 0], [.16, .20, .25], connection, collision=False)
        sim.setObjectPose(palm, [0, 0, .041, 0, 0, 0, 1], connection)
        for direction in [-1, 1]:
            finger = shape(side + '_Finger_' + str(direction), [.018, .009, .060], [0, 0, 0], [.54, .59, .65], connection, collision=False)
            sim.setObjectPose(finger, [0, direction * .032, .082, 0, 0, 0, 1], connection)
        tip = dummy(side + '_TCP', parent=connection)
        sim.setObjectPose(tip, [0, 0, .112, 0, 0, 0, 1], connection)
        target_ik = dummy(side + '_IK_Target')
        env = ik.createEnvironment()
        group = ik.createGroup(env)
        ik.setGroupCalculation(env, group, ik.method_damped_least_squares, .05, 150)
        _, mapping, _ = ik.addElementFromScene(env, group, model, tip, target_ik, ik.constraint_pose)
        arms[side] = dict(model=model, joints=joints, tip=tip, target=target_ik,
                          connection=connection, env=env, group=group, mapping=mapping)

    robot_collections = {}
    for side, arm in arms.items():
        c = sim.createCollection(0)
        sim.addItemToCollection(c, sim.handle_tree, arm['model'], 0)
        robot_collections[side] = c
    castle_collection = sim.createCollection(0)
    sim.addItemToCollection(castle_collection, sim.handle_tree, assembly, 0)
    bench = sim.getObject('/APEX_MR_Plus/Workbench')
    validator = sim.createScript(sim.scripttype_customization, '''sim=require 'sim'
function sysCall_init() end
function firstClear(configs,joints,robot,bench,castle)
  for i,q in ipairs(configs) do
    for n,j in ipairs(joints) do sim.setJointPosition(j,q[n]) end
    if sim.checkCollision(robot,bench)==0 and sim.checkCollision(robot,castle)==0 then return q,i end
  end
  return {},0
end''', 0, 'lua')
    sim.setObjectAlias(validator, 'TemporaryEndpointValidator')
    sim.initScript(validator)

    def pose_is_clear(side):
        c = robot_collections[side]
        return not sim.checkCollision(c, bench)[0] and not sim.checkCollision(c, castle_collection)[0]

    def solve(side, xyz, quat, seed=None):
        arm = arms[side]
        seed = seed or [0., -1.15, 1.7, -2.1, -1.57, 0.]
        sim.setObjectPose(arm['target'], xyz + quat, -1)
        for j, q in zip(arm['joints'], seed):
            sim.setJointPosition(j, q)
        ik.handleGroup(arm['env'], arm['group'], {'syncWorlds': True, 'allowError': False})
        err = np.linalg.norm(np.array(sim.getObjectPosition(arm['tip'], -1)) - xyz)
        if err > .001 or not pose_is_clear(side):
            configs = ik.findConfigs(arm['env'], arm['group'], [arm['mapping'][j] for j in arm['joints']],
                                     {'maxTime': 1.0, 'findMultiple': True, 'findAlt': False})
            if not configs:
                raise RuntimeError(f'No {side} IK for {xyz}, residual={err:.5f}')
            configs = sorted(configs, key=lambda qs: sum((v-s)**2 for v, s in zip(qs, seed)))
            q, accepted_index = sim.callScriptFunction('firstClear', validator, configs, arm['joints'], robot_collections[side], bench, castle_collection)
            print(json.dumps({'candidates': len(configs), 'accepted_index': accepted_index, 'arm': side}), flush=True)
            if not q:
                raise RuntimeError(f'{side}: {len(configs)} IK candidates but none clear of table/towers for {xyz}')
        q = [sim.getJointPosition(j) for j in arm['joints']]
        err = float(np.linalg.norm(np.array(sim.getObjectPosition(arm['tip'], -1)) - xyz))
        print(json.dumps({'pose': side, 'xyz': xyz, 'residual_m': err}), flush=True)
        if err > .001:
            raise RuntimeError(f'IK residual too high: {err}')
        return q, err

    poses, errors = {}, {}
    for side, x, down in [('A', -.36, [1., 0., 0., 0.]), ('B', .36, [0., 1., 0., 0.])]:
        poses[side + '_home'], errors[side + '_home'] = solve(side, [x, -.03, 1.19], down)
    poses['A_hold'], errors['A_hold'] = solve('A', [-.285, -.12, 1.015], [1., 0., 0., 0.], poses['A_home'])
    poses['B_support'], errors['B_support'] = solve('B', [.06, .17, bridge_bottom - .004],
        Rotation.from_euler('x', -math.pi/2).as_quat().tolist(), poses['B_home'])
    poses['B_install'], errors['B_install'] = solve('B', [.203, .22, bridge_bottom + .095], [0., 1., 0., 0.], poses['B_home'])
    for side, arm in arms.items():
        ik.eraseEnvironment(arm['env'])
        arm.pop('env'); arm.pop('group'); arm.pop('mapping')
    sim.destroyCollection(castle_collection)
    for c in robot_collections.values():
        sim.destroyCollection(c)
    sim.removeObjects([validator])

    # A visible contact shoe only in the support preset, not a physical clamp.
    shoe = shape('B_SupportShoe', [.055, .08, .008], [0, 0, 0], [.20, .24, .28], arms['B']['connection'], collision=False)
    sim.setObjectPose(shoe, [0, 0, .112] + Rotation.from_euler('x', math.pi/2).as_quat().tolist(), arms['B']['connection'])
    sim.setObjectInt32Param(shoe, sim.objintparam_visibility_layer, 0)
    cameras = {}
    for name, eye, aim in [('Overview', [1.10, -2.45, 2.30], [0, .02, .99]),
                           ('Detail', [.70, -1.25, 1.72], [0, .12, 1.02])]:
        h = sim.createVisionSensor(1+2+4+128, [1600, 1100, 0, 0],
                                    [.025, 10., math.radians(48), .01, 0., 0., .91, .94, .97, 0., 0.])
        sim.setObjectAlias(h, name + 'RGB')
        sim.setObjectParent(h, root, True)
        sim.setObjectPose(h, look_at(eye, aim), -1)
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        cameras[name] = h
    default_camera = sim.getObject('/DefaultCamera')
    sim.setObjectPose(default_camera, look_at([1.10, -2.45, 2.30], [0, .02, .99]), -1)
    sim.adjustView(0, default_camera, 0)

    cfg = dict(version=1, stage='environment_preview', table_z=TABLE_Z, poses=poses,
               ik_residual_m=errors, arm_joint_aliases={s: [f'{s}_joint_{i}' for i in range(1, 7)] for s in arms},
               root=root, arms=arms, connector=connector, prop=pier, bridge=bridge,
               connector_initial=[-.35, -.18, TABLE_Z + .030],
               connector_buffer=[-.055, -.35, TABLE_Z + .048],
               prop_park=prop_park, prop_support=[.08, .22, TABLE_Z],
               bridge_bottom=bridge_bottom, default_camera=default_camera, cameras=cameras,
               preset=0,
               limitations=['Kinematic environment preview, not autonomous repair planning.',
                            'Static reference bridge/towers; no LEGO snap contact or support-force dynamics.',
                            'Range-limit preset preserves current configuration; full task infeasibility is not certified.',
                            'IK endpoints checked; collision-free connecting trajectories are not yet planned.'])
    sim.writeCustomTableData(root, 'APEX_MR_Plus', cfg)
    sim.saveScene(str(ROOT / 'config' / '_build_checkpoint.ttt'))
    script = sim.createScript(sim.scripttype_customization, (ROOT / 'src' / 'scene_ui.lua').read_text(encoding='utf-8'), 0, 'lua')
    sim.setObjectAlias(script, 'SceneControls')
    sim.setObjectParent(script, root, True)
    # Explicitly init after config has been stored.
    sim.initScript(script)
    sim.callScriptFunction('initialize', script)
    output_scenes = []
    for preset, name in [(1, '02_Capability_Change.ttt'), (2, '03_Temporary_Support.ttt'), (0, '01_Workcell_Overview.ttt')]:
        sim.callScriptFunction('setPreset', script, preset)
        sim.saveScene(str(ROOT / 'scenes' / name))
        output_scenes.append(name)
        sim.handleVisionSensor(cameras['Overview'])
        raw, res = sim.getVisionSensorImg(cameras['Overview'])
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(res[1], res[0], 3)[::-1].copy()
        Image.fromarray(rgb).save(ROOT / 'previews' / (Path(name).stem + '.png'))
        if preset == 1:
            sim.handleVisionSensor(cameras['Detail'])
            raw, res = sim.getVisionSensorImg(cameras['Detail'])
            Image.fromarray(np.frombuffer(raw, np.uint8).reshape(res[1], res[0], 3)[::-1].copy()).save(ROOT / 'previews' / '04_Blocked_Closeup.png')
    sim.saveScene(str(ROOT / 'APEX_MR_Plus.ttt'))
    manifest = {k: v for k, v in cfg.items() if k not in ['arms', 'root', 'connector', 'prop', 'bridge', 'default_camera', 'cameras']}
    manifest.update(scene=str(ROOT / 'APEX_MR_Plus.ttt'), snapshots=output_scenes,
                    shape_count=count[0], simulator_version=sim.getInt32Param(sim.intparam_program_version))
    (ROOT / 'config' / 'scene_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'saved': manifest['scene'], 'shapes': count[0], 'ik_residual_m': errors}), flush=True)


if __name__ == '__main__':
    with Simulator(ROOT, port=23040, timeout=40) as owner:
        client, sim = owner.connect()
        build(sim, client, owner.install)
