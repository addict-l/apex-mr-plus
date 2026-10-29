"""Independent dynamic-chain smoke probe; never saves or edits a scene file."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from sim_connection import Simulator
from normal.scene import resolve


def angle_error(actual, target):
    return math.atan2(math.sin(actual-target), math.cos(actual-target))


def sensor_reading(sim, handle):
    result, force, torque = sim.readForceSensor(handle)
    return {'status': result, 'valid': bool(result & 1),
            'force_local_N': force, 'torque_local_Nm': torque}


def ancestry(sim, handle):
    result = []
    h = handle
    while h >= 0:
        item = {'alias': sim.getObjectAlias(h), 'type': sim.getObjectType(h)}
        if sim.getObjectType(h) == sim.object_shape_type:
            item.update(static=bool(sim.getObjectInt32Param(h, sim.shapeintparam_static)),
                        mass_kg=sim.getShapeMass(h),
                        respondable=bool(sim.getObjectInt32Param(h, sim.shapeintparam_respondable)))
        result.append(item)
        h = sim.getObjectParent(h)
    return result


def resolve_optional(sim, arm, key, alias):
    if key not in arm:
        try:
            arm[key] = sim.getObject(':/'+alias, {'proxy': arm['model']})
        except Exception:
            return None
    return arm[key]


def sample(sim, cfg):
    output = {'time_s': sim.getSimulationTime(), 'arms': {}, 'parts': {}}
    for side, arm in cfg['arms'].items():
        joints = arm['joints']
        q = [sim.getJointPosition(j) for j in joints]
        targets = [sim.getJointTargetPosition(j) for j in joints]
        target_errors = [angle_error(a, b) for a, b in zip(q, targets)]
        home_errors = [angle_error(a, b) for a, b in zip(q, arm['home_q'])]
        record = {
            'joint_dynamic': [bool(sim.isDynamicallyEnabled(j)) for j in joints],
            'q_rad': q, 'target_q_rad': targets,
            'error_to_target_rad': target_errors,
            'max_error_to_target_rad': max(abs(e) for e in target_errors),
            'max_error_to_home_rad': max(abs(e) for e in home_errors),
            'motor_torque_Nm': [sim.getJointForce(j) for j in joints],
            'palm_dynamic': bool(sim.isDynamicallyEnabled(arm['palm'])),
            'tool_sensor_dynamic': bool(sim.isDynamicallyEnabled(arm['sensor'])),
            'tool_sensor': sensor_reading(sim, arm['sensor']),
            'tcp_world_pose': sim.getObjectPose(arm['tip'], -1),
            'base_world_pose': sim.getObjectPose(arm['model'], -1),
            'base_static_flag': bool(sim.getObjectInt32Param(arm['model'], sim.shapeintparam_static)),
        }
        pad_ft = resolve_optional(sim, arm, 'support_sensor', side+'_PadForceSensor')
        pad = resolve_optional(sim, arm, 'support_pad', side+'_SupportShoe')
        record.update(pad_sensor_present=pad_ft is not None, pad_present=pad is not None)
        if pad_ft is not None:
            record.update(pad_sensor_dynamic=bool(sim.isDynamicallyEnabled(pad_ft)),
                          pad_sensor=sensor_reading(sim, pad_ft))
        if pad is not None:
            record.update(pad_dynamic=bool(sim.isDynamicallyEnabled(pad)),
                          pad_world_pose=sim.getObjectPose(pad, -1),
                          pad_respondable=bool(sim.getObjectInt32Param(pad, sim.shapeintparam_respondable)))
        output['arms'][side] = record
    by_part = {p['id']: p for p in cfg['parts']}
    for name, h in cfg['blocks'].items():
        current = sim.getObjectPosition(h, -1)
        initial = cfg['initial'][name]
        output['parts'][name] = {
            'dynamic': bool(sim.isDynamicallyEnabled(h)),
            'static_flag': bool(sim.getObjectInt32Param(h, sim.shapeintparam_static)),
            'position_m': current, 'initial_m': initial,
            'delta_from_initial_m': [a-b for a,b in zip(current, initial)],
            'mass_kg': sim.getShapeMass(h), 'expected_mass_kg': by_part[name]['mass'],
        }
    return output


def stop(sim):
    if sim.getSimulationState() == sim.simulation_stopped:
        return
    sim.stopSimulation()
    for _ in range(30):
        if sim.getSimulationState() == sim.simulation_stopped:
            return
        sim.step()
    raise RuntimeError('Simulator did not stop after the smoke probe')


def run(sim, cfg, frames=50):
    sim.setStepping(True)
    sim.startSimulation()
    samples = []
    for frame in range(1, frames+1):
        sim.step()
        if frame in {1, 10, 25, frames}:
            samples.append(sample(sim, cfg))
    stop(sim)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=23045)
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--diagnose-base', action='store_true',
                        help='also anchor model bases in a private reloaded scene for diagnosis')
    args = parser.parse_args()
    scene = ROOT/'scenes'/'10_Normal_Assembly.ttt'
    output = ROOT/'results'/'normal'/'dynamics_probe.json'
    metadata = {'scene': str(scene), 'scene_size_bytes': scene.stat().st_size,
                'scene_mtime_ns': scene.stat().st_mtime_ns,
                'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
                'port': args.port, 'frames': args.frames, 'scene_saved': False}
    try:
        with Simulator(ROOT, port=args.port, timeout=40) as owner:
            client, sim = owner.connect()
            sim.loadScene(str(scene))
            cfg = resolve(sim)
            chains = {side: ancestry(sim, arm['sensor']) for side, arm in cfg['arms'].items()}
            before = {name: sim.getObjectPose(h, -1) for name, h in cfg['blocks'].items()}
            initial_arms = {side: {'base_world_pose': sim.getObjectPose(arm['model'], -1),
                                   'tcp_world_pose': sim.getObjectPose(arm['tip'], -1)}
                            for side, arm in cfg['arms'].items()}
            baseline = run(sim, cfg, args.frames)
            data = {**metadata, 'tool_sensor_ancestors': chains,
                    'initial_part_poses': before, 'initial_arm_poses': initial_arms,
                    'baseline': baseline, 'diagnostic_reparent': []}
            failed = [side for side, arm in baseline[-1]['arms'].items()
                      if not arm['tool_sensor_dynamic']]
            for side in failed:
                arm = cfg['arms'][side]
                parent = sim.getObjectParent(arm['connection'])
                while parent >= 0:
                    if (sim.getObjectType(parent) == sim.object_shape_type
                            and not sim.getObjectInt32Param(parent, sim.shapeintparam_static)):
                        break
                    parent = sim.getObjectParent(parent)
                if parent >= 0:
                    sim.setObjectParent(arm['sensor'], parent, True)
                    sim.resetDynamicObject(arm['palm'])
                    data['diagnostic_reparent'].append({'arm': side, 'new_parent': sim.getObjectAlias(parent)})
            if data['diagnostic_reparent']:
                data['after_reparent'] = run(sim, cfg, args.frames)
            if args.diagnose_base:
                if scene.stat().st_mtime_ns != metadata['scene_mtime_ns']:
                    data['anchored_diagnostic_skipped'] = 'Scene file changed during baseline; rerun on the new scene.'
                else:
                    sim.loadScene(str(scene))
                    diag_cfg = resolve(sim)
                    for side, arm in diag_cfg['arms'].items():
                        sim.setObjectInt32Param(arm['model'], sim.shapeintparam_static, 1)
                        parent = sim.getObjectParent(arm['connection'])
                        while parent >= 0:
                            if (sim.getObjectType(parent) == sim.object_shape_type
                                    and not sim.getObjectInt32Param(parent, sim.shapeintparam_static)):
                                break
                            parent = sim.getObjectParent(parent)
                        if parent >= 0:
                            sim.setObjectParent(arm['sensor'], parent, True)
                    data['anchored_diagnostic'] = run(sim, diag_cfg, args.frames)
            last = baseline[-1]
            data['summary'] = {
                'all_12_joints_dynamic': all(all(a['joint_dynamic']) for a in last['arms'].values()),
                'all_tool_sensors_dynamic': all(a['tool_sensor_dynamic'] for a in last['arms'].values()),
                'all_pad_sensors_dynamic': all(a.get('pad_sensor_dynamic', False) for a in last['arms'].values()),
                'all_pads_dynamic': all(a.get('pad_dynamic', False) for a in last['arms'].values()),
                'all_parts_dynamic': all(p['dynamic'] for p in last['parts'].values()),
                'max_target_error_rad': max(a['max_error_to_target_rad'] for a in last['arms'].values()),
                'base_displacement_m': {side: math.dist(a['base_world_pose'][:3], initial_arms[side]['base_world_pose'][:3])
                                        for side,a in last['arms'].items()},
                'tcp_displacement_m': {side: math.dist(a['tcp_world_pose'][:3], initial_arms[side]['tcp_world_pose'][:3])
                                       for side,a in last['arms'].items()},
                'max_source_settling_displacement_m': max(math.dist(p['position_m'], p['initial_m']) for p in last['parts'].values()),
                'force_valid': {side: {'tool': a['tool_sensor']['valid'], 'pad': a.get('pad_sensor', {}).get('valid', False)}
                                for side, a in last['arms'].items()},
            }
            output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(data['summary'], ensure_ascii=False, indent=2), flush=True)
    except Exception:
        output.write_text(json.dumps({**metadata, 'error': traceback.format_exc()}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
