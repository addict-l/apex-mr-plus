"""Install and run the scene3 completion-driven TPG on actual rigid bodies."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = '/APEX_MR_Scene3'


def install_controller(sim, cfg, tpg, scene_name='scenes/31_Scene3_APEX_Planned.ttt'):
    """Embed the graph and controller without starting or claiming execution."""
    root = cfg['root']
    existing = sim.getObject(SCENE_ROOT + '/CentralTPGScheduler', {'noError': True})
    if existing != -1:
        sim.removeObjects([existing])
    cfg['program'] = tpg
    sim.writeCustomTableData(root, 'Scene3APEX', cfg)
    source = Path(__file__).with_name('controller.lua').read_text(encoding='utf-8')
    script = sim.createScript(sim.scripttype_simulation, source, 0, 'lua')
    sim.setObjectAlias(script, 'CentralTPGScheduler')
    sim.setObjectParent(script, root, True)
    output = ROOT / scene_name
    output.parent.mkdir(parents=True, exist_ok=True)
    sim.saveScene(str(output))
    return script


def capture(sim, camera, path, caption=None):
    sim.handleVisionSensor(camera)
    raw, resolution = sim.getVisionSensorImg(camera)
    rgb = np.frombuffer(raw, np.uint8).reshape(resolution[1], resolution[0], 3)[::-1].copy()
    image = Image.fromarray(rgb)
    if caption:
        draw = ImageDraw.Draw(image)
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
        font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
        draw.rectangle([0, 0, image.width, 46], fill=(20, 34, 52))
        draw.text((16, 8), caption, font=font, fill='white')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return np.asarray(image)


def _goal_poses(cfg, graph):
    explicit = graph.get('validation_requirements', {}).get('goal_poses')
    if explicit:
        return explicit
    if cfg.get('goal_poses'):
        return cfg['goal_poses']
    goals = {}
    for part in cfg['parts']:
        if part['id'] == 'Bridge' or 'goal' not in part:
            continue
        pose = list(part['goal'])
        if len(pose) == 3:
            pose.extend(part.get('goal_quaternion', [0, 0, 0, 1]))
        goals[part['id']] = pose
    return goals


def execute(sim, client, cfg, outdir, timeout_sim_s=1800, timeout_wall_s=1200, video=True):
    """Return measured execution, including an independent event/load audit.

    A timed-out or failed run still writes its measured partial result when
    the controller remains accessible. Simulator state is never teleported.
    """
    from .validation import validate_execution

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    scheduler = sim.getObject(SCENE_ROOT + '/CentralTPGScheduler')
    camera = cfg.get('camera')
    if camera is None:
        camera = sim.getObject(SCENE_ROOT + '/OverviewRGB')
    graph = cfg['program']
    image = capture(sim, camera, output / 'initial_dynamic_scene.png',
                    'Scene 3：ILP → RRT-Connect → TPG → 实际动力学')
    writer = None
    if video:
        import cv2
        writer = cv2.VideoWriter(str(output / 'scene3_execution.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), 8,
                                 (image.shape[1], image.shape[0]))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError('Could not create execution video')
    start = time.monotonic()
    last_status, frames = -10., 0
    captured_support = captured_permanent = False
    timeout_reason = None
    sim.setStepping(False)
    sim.startSimulation()
    try:
        while True:
            time.sleep(.15)
            state = sim.callScriptFunction('getState', scheduler)
            t = state['time']
            if t - last_status >= 5 or state.get('failure') or state.get('done'):
                print(json.dumps(state, ensure_ascii=False), flush=True)
                last_status = t
            if not captured_support and state.get('support_load_verified'):
                capture(sim, camera, output / 'support_load_transfer.png', f'实际支撑反力已建立 | t={t:.2f}s')
                captured_support = True
            if not captured_permanent and state.get('permanent_verified'):
                capture(sim, camera, output / 'permanent_support_verified.png', f'永久承载实测验证 | t={t:.2f}s')
                captured_permanent = True
            if writer is not None:
                import cv2
                frame = capture(sim, camera, output / 'latest_execution.png',
                                f"Scene 3 实际执行 | t={t:.2f}s | 节点 {state['completed']}/{state['total']}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames += 1
            if state.get('done') or state.get('failure'):
                break
            if t > timeout_sim_s or time.monotonic() - start > timeout_wall_s:
                timeout_reason = 'Actual execution exceeded simulation/wall time budget'
                break
        result = sim.callScriptFunction('getResult', scheduler)
        result['wall_time_s'] = time.monotonic() - start
        result['video_frames'] = frames
        if timeout_reason:
            result['failure'] = timeout_reason
        goals = _goal_poses(cfg, graph)
        errors, angles = {}, {}
        for part, desired in goals.items():
            measured = result.get('part_poses', {}).get(part)
            if measured is None:
                continue
            errors[part] = float(np.linalg.norm(np.asarray(measured[:3]) - np.asarray(desired[:3])))
            if len(desired) >= 7:
                qa, qb = np.asarray(measured[3:7]), np.asarray(desired[3:7])
                denominator = float(np.linalg.norm(qa) * np.linalg.norm(qb))
                angles[part] = 2 * math.acos(min(1., abs(float(qa @ qb)) / denominator)) if denominator else math.pi
        result['goal_position_errors_m'] = errors
        result['goal_orientation_errors_rad'] = angles
        # Lua represents empty hands with false because nil fields disappear
        # during message encoding. JSON uses null for the same empty state.
        for side in ('A', 'B'):
            state_held = result.setdefault('final_state', {}).setdefault('held', {})
            if state_held.get(side) is False:
                state_held[side] = None
        result['success'] = bool(result.get('done') and not result.get('failure'))
        requirements = {'support_part_id': 'TemporarySupport', 'goal_poses': goals,
                        **cfg.get('validation_requirements', {})}
        audit = validate_execution(graph, result, requirements)
        # Independent audit is part of actual execution acceptance.
        audit_ok = audit.get('ok', audit.get('valid', audit.get('success', False)))
        result['independent_validation_passed'] = bool(audit_ok)
        result['success'] = bool(result['success'] and audit_ok)
        (output / 'execution_validation.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (output / 'execution.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        capture(sim, camera, output / ('completed_dynamic_scene.png' if result['success'] else 'failed_dynamic_scene.png'),
                'Scene 3 实际动力学完成并通过独立验收' if result['success'] else 'Scene 3 实际执行诊断：未通过验收')
        return result
    finally:
        if writer is not None:
            writer.release()
        sim.stopSimulation()
