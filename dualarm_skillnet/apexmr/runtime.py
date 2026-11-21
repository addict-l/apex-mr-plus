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
    if cfg.get('profile') == 'skillnet_v1_3':
        source += '\n' + (ROOT / 'simbridge' / 'controller_ext.lua').read_text(encoding='utf-8')
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
    if cfg.get('profile') == 'skillnet_v1_3':
        return _execute_skillnet(sim, client, cfg, outdir, timeout_sim_s,
                                 timeout_wall_s, video)
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


def _execute_skillnet(sim, client, cfg, outdir, timeout_sim_s, timeout_wall_s, video):
    """Run the generic profile and preserve measured failures and timeouts."""
    from .validation import validate_execution
    from simbridge.errors import is_transport_failure

    if not math.isfinite(timeout_wall_s) or timeout_wall_s <= 0:
        raise ValueError('仿真墙钟超时必须为有限正数')
    if video:
        raise ValueError('本环境的 v1.3 录像后端未启用；请使用默认关闭录像的模式')
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    scheduler = sim.getObject(cfg.get('root_path', SCENE_ROOT)+'/CentralTPGScheduler')
    camera = cfg.get('camera')
    if camera is not None:
        capture(sim, camera, output/'initial_dynamic_scene.png', '双臂技能网络：实际动力学执行')
    start, last_status = time.monotonic(), -10.
    failure, timed_out, result = None, False, {}
    transport_errors = []

    def record_transport(exc, stage):
        if not is_transport_failure(exc):
            return False
        cause = exc.__cause__ or exc.__context__
        transport_errors.append({'stage': stage, 'type': type(exc).__name__,
                                 'module': type(exc).__module__, 'message': str(exc),
                                 'cause': None if cause is None else f'{type(cause).__name__}: {cause}'})
        return True

    def save_result():
        result['remote_interface_failure'] = bool(transport_errors)
        if transport_errors:
            result['remote_interface_errors'] = list(transport_errors)
        result['success'] = bool(result.get('done') and not result.get('failure') and not transport_errors)
        audit = validate_execution(cfg['program'], result, cfg['validation_requirements'])
        result['independent_validation_passed'] = bool(audit['success'])
        result['success'] = bool(result['success'] and audit['success'])
        for filename, data in [('execution.json', result), ('execution_validation.json', audit)]:
            (output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    sim.setStepping(False)
    sim.startSimulation()
    try:
        try:
            while True:
                state = sim.callScriptFunction('getState', scheduler)
                stamp = float(state.get('time', 0.))
                if stamp-last_status >= 5 or state.get('done') or state.get('failure'):
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    last_status = stamp
                if state.get('done') or state.get('failure'):
                    break
                if stamp > timeout_sim_s or time.monotonic()-start > timeout_wall_s:
                    failure, timed_out = '实际仿真超过墙钟或仿真时间预算', True
                    break
                time.sleep(.15)
        except Exception as exc:
            failure = f'实际仿真轮询失败：{type(exc).__name__}: {exc}'
            record_transport(exc, 'poll')
        if not transport_errors:
            try:
                result = sim.callScriptFunction('getResult', scheduler)
            except Exception as exc:
                failure = failure or f'执行证据读取失败：{type(exc).__name__}: {exc}'
                record_transport(exc, 'result')
                result = {}
        result['wall_time_s'] = time.monotonic()-start
        result['video_frames'] = 0
        result['timeout'] = timed_out
        if failure:
            result['failure'] = failure
        # Empty Lua tables have no array/map distinction over the remote API.
        for key in ('events', 'trace', 'collision_failures', 'tracking_failures', 'task_schedule'):
            if result.get(key) == {}:
                result[key] = []
        held = result.setdefault('final_state', {}).setdefault('held', {})
        for arm in ('A', 'B'):
            if held.get(arm) is False:
                held[arm] = None
        save_result()
        if camera is not None and not transport_errors:
            name = 'completed_dynamic_scene.png' if result['success'] else 'failed_dynamic_scene.png'
            capture(sim, camera, output/name, '实际执行通过验收' if result['success'] else '实际执行未通过验收')
        return result
    finally:
        try:
            sim.stopSimulation()
            stop_deadline = time.monotonic()+5.
            while sim.getSimulationState() != sim.simulation_stopped and time.monotonic() < stop_deadline:
                time.sleep(.05)
        except Exception as exc:
            if not record_transport(exc, 'stop') and not transport_errors:
                raise
            result['failure'] = result.get('failure') or f'停止仿真失败：{type(exc).__name__}: {exc}'
            result['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        if transport_errors:
            # A cleanup outage must invalidate a previously complete result,
            # and an earlier outage must not disappear behind a return value.
            save_result()
