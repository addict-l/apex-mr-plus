"""Scene 3: qualified ILP -> RRT-Connect -> multimodal TPG -> actual dynamics.

Only an independently audited actual execution promotes the planned initial
scene to the main .ttt file. A plan-only run never creates execution evidence.
All cache keys include complete source-scene bytes and implementation hashes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'apex'
SOURCE = ROOT/'scenes'/'10_Scene3_Dynamic.ttt'
PLANNED = ROOT/'scenes'/'11_Scene3_APEX_Planned.ttt'
MAIN = ROOT/'apex_mr_scene3.ttt'


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def source_fingerprint(root=ROOT, source=SOURCE):
    """Hash scene bytes, implementation, and declarative configuration inputs."""
    root, source = Path(root), Path(source)
    files = [source]
    for directory in ('apex', 'src'):
        base = root/directory
        if base.exists():
            files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py', '.lua'))
    if (root/'config').exists():
        files.extend((root/'config').glob('*.json'))
    if (root/'requirements.txt').exists():
        files.append(root/'requirements.txt')
    hashes = {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in sorted(set(files))}
    return {'files': hashes, 'sha256': value_hash(hashes)}


def _environment():
    versions = {}
    for package in ('numpy', 'scipy', 'pyzmq', 'matplotlib'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'packages': versions}


def _cache(path, signature, enabled, **extra):
    if not enabled:
        return None, 'reuse_not_requested_or_replan_forced'
    if not path.exists():
        return None, 'cache_missing'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        return None, 'cache_unreadable: '+str(error)
    if data.get('scene_signature') != signature:
        return None, 'source_code_configuration_or_environment_signature_changed'
    for key, expected in extra.items():
        if data.get(key) != expected:
            return None, key+'_changed'
    return data, 'exact_signature_match'


def _archive_previous_execution(outdir):
    """Keep prior actual evidence coherent when a new run has not executed yet."""
    runtime_names = {'execution.json', 'execution_validation.json', 'summary.json', 'delivery.json',
                     'initial_dynamic_scene.png', 'completed_dynamic_scene.png', 'failed_dynamic_scene.png',
                     'support_load_transfer.png', 'permanent_support_verified.png', 'latest_execution.png',
                     'scene3_execution.mp4'}
    existing = [p for p in outdir.iterdir() if p.is_file() and
                (p.name in runtime_names or p.name.startswith(('scene3_execution_', 'scene3_actual_dynamics',
                                                              'scene3_dynamics_diagnostic')))]
    if not existing:
        return None
    target = (outdir/'history'/('previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])).resolve()
    if not target.is_relative_to(outdir.resolve()):
        raise RuntimeError('History target escaped results directory')
    target.mkdir(parents=True)
    for name in ('pipeline_manifest.json', 'tpg.json', 'alloc_and_reference.json',
                 'nominal_geometry_validation.json', 'rigid_bodies.json', 'problem.json'):
        if (outdir/name).is_file():
            shutil.copy2(outdir/name, target/name)
    for source in existing:
        if not source.resolve().is_relative_to(outdir.resolve()):
            raise RuntimeError('Evidence source escaped results directory')
        shutil.move(str(source), str(target/source.name))
    return str(target)


class RunManifest:
    def __init__(self, outdir, args):
        self.outdir = outdir
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        self.data = {'schema_version': 1, 'run_id': self.run_id, 'status': 'running',
                     'started_at_utc': datetime.now(timezone.utc).isoformat(),
                     'arguments': vars(args), 'stages': [], 'execution_performed': False,
                     'main_scene_promoted': False, 'continuous_collision_guarantee': False,
                     'grasp_and_connection_model': 'idealized rigid grasp and permanent connection; measured rigid-body execution required'}
        self.write()

    def write(self):
        save_json(self.outdir/'pipeline_manifest.json', self.data)

    @contextmanager
    def stage(self, name):
        record = {'name': name, 'status': 'running'}
        self.data['stages'].append(record); self.write()
        print('STAGE: '+name, flush=True)
        started = time.monotonic()
        try:
            yield record
        except BaseException as error:
            record.update(status='failed', wall_time_s=time.monotonic()-started,
                          error_type=type(error).__name__, error=str(error))
            self.write()
            raise
        else:
            record.update(status='passed', wall_time_s=time.monotonic()-started)
            self.write()


def _goals(cfg):
    if cfg.get('goal_poses'):
        return cfg['goal_poses']
    result = {}
    for part in cfg['parts']:
        if 'goal' not in part:
            continue
        goal = list(part['goal'])
        if len(goal) == 3:
            goal += part.get('goal_quaternion', [0., 0., 0., 1.])
        result[part['id']] = goal
    return result


def run(args):
    # Import the simulation modules only after parsing CLI arguments so help
    # and provenance utilities remain usable without a running simulator.
    sys.path.insert(0, str(ROOT/'src'))
    from sim_connection import Simulator
    from .scene import build, resolve
    from .compile import compile_reference
    from .motion import MotionPlanner
    from .fast_interarm import FastInterarmSampler
    from .geometry import SampledConflictOracle, verify_rollout
    from .tpg import build_tpg
    from .program import parameterize
    from .runtime import install_controller, execute
    from .validation import validate_execution
    from .plots import render_figures, plot_execution

    OUT.mkdir(parents=True, exist_ok=True)
    previous = _archive_previous_execution(OUT)
    manifest = RunManifest(OUT, args)
    manifest.data['previous_execution_archived_to'] = previous
    summary = {'run_id': manifest.run_id, 'success': False, 'planning_success': False,
               'execution_performed': False, 'main_scene_promoted': False}
    try:
        with Simulator(ROOT, port=args.port, timeout=45) as owner:
            client, sim = owner.connect()
            with manifest.stage('dynamic_source_scene') as stage:
                if args.rebuild_scene or not SOURCE.exists():
                    build(sim, client, owner.install)
                    SOURCE.parent.mkdir(parents=True, exist_ok=True)
                    sim.saveScene(str(SOURCE))
                    stage['rebuilt'] = True
                else:
                    sim.loadScene(str(SOURCE)); stage['rebuilt'] = False
                cfg = resolve(sim)
                frozen_inputs = source_fingerprint()
                environment = _environment()
                try:
                    environment['coppeliasim_program_version'] = sim.getInt32Param(sim.intparam_program_version)
                except Exception:
                    environment['coppeliasim_program_version'] = 'unavailable'
                settings = {'joint_conflict_step_rad': args.step, 'node_span_rad': args.span,
                            'rollout_step_s': args.rollout_step, 'velocity_rad_s': args.vmax,
                            'acceleration_rad_s2': args.amax}
                signature_data = {'inputs': frozen_inputs, 'resolved_scene': cfg,
                                  'settings': settings, 'environment': environment}
                signature = value_hash(signature_data)
                manifest.data.update(scene_signature=signature, source_scene=str(SOURCE),
                                     planned_scene=str(PLANNED), input_fingerprint=frozen_inputs,
                                     environment=environment, planning_settings=settings)
                save_json(OUT/'source_signature.json', signature_data)
                rigid = []
                for handle in sim.getObjectsInTree(cfg['root'], sim.object_shape_type, 0):
                    if sim.getObjectInt32Param(handle, sim.shapeintparam_static) == 0:
                        inertia, frame = sim.getShapeInertia(handle)
                        rigid.append({'path': sim.getObjectAlias(handle, 1), 'mass_kg': sim.getShapeMass(handle),
                                      'inertia_kg_m2': inertia, 'inertia_frame': frame})
                save_json(OUT/'rigid_bodies.json', rigid)
                save_json(OUT/'problem.json', {'parts': cfg['parts'], 'physics': cfg.get('physics', {}),
                                              'scene_signature': signature})
                stage['dynamic_bodies'] = len(rigid)

            allow_reuse = not args.replan and not args.rebuild_scene
            with manifest.stage('geometry_qualified_ILP_and_RRTConnect_reference') as stage:
                reference, reason = _cache(OUT/'alloc_and_reference.json', signature,
                                           args.reuse_reference and allow_reuse)
                stage.update(reused=reference is not None, cache_reason=reason)
                if reference is None:
                    reference = compile_reference(sim, client, cfg, OUT)
                tasks = reference['tasks']
                if not tasks:
                    raise RuntimeError('Reference compiler produced no executable tasks')
                reference['scene_signature'] = signature
                save_json(OUT/'alloc_and_reference.json', reference)
                save_json(OUT/'reference_tasks.json', tasks)
                # Selection is derived after fingerprinting the immutable
                # source scene. Restore it on cached and freshly planned runs
                # so the executor and independent audit share the same guard.
                cfg['selected_support'] = deepcopy(reference['allocation']['selected_support'])
                cfg['validation_requirements'] = {
                    **cfg.get('validation_requirements', {}),
                    **reference.get('validation_requirements', {}),
                    'support_part_id': 'TemporarySupport',
                    'gate_blocks_extraction': cfg['selected_support']['candidate'] == 'inner_pocket',
                    'require_actual_applied_load': True,
                    'required_dynamic_connections': ['K1', 'K2', 'Gate', 'Bridge'],
                    'assembly_tool_force_limit_N': cfg.get('runtime', {}).get('assembly_tool_force_limit_N', 40.0),
                    'assembly_tool_torque_limit_Nm': cfg.get('runtime', {}).get('assembly_tool_torque_limit_Nm', 8.0),
                }
                reference_signature = value_hash(tasks)
                manifest.data['reference_signature'] = reference_signature
                for side, q in reference.get('problem', {}).get('initial_joint_configurations', {}).items():
                    cfg['arms'][side]['home_q'] = q
                    for handle, value in zip(cfg['arms'][side]['joints'], q):
                        sim.setJointPosition(handle, value); sim.setJointTargetPosition(handle, value)
                save_json(OUT/'sequential_task_graph.json', {
                    'arms': {arm: [t for t in tasks if t['arm']==arm] for arm in ('A','B')},
                    'allocation': reference['allocation'], 'reference_signature': reference_signature,
                    'scope': 'Sequential feasible reference with explicit task/contact modes; no actual execution claimed'})
                stage['reference_task_count'] = len(tasks)

            with manifest.stage('sampled_conflicts_and_multimodal_TPG') as stage:
                tpg, reason = _cache(OUT/'tpg.json', signature, args.reuse_tpg and allow_reuse,
                                     reference_signature=reference_signature)
                stage.update(reused=tpg is not None, cache_reason=reason)
                if tpg is None:
                    with FastInterarmSampler(sim, client, cfg) as sampler:
                        oracle = SampledConflictOracle(sampler, cfg, OUT, step=args.step)
                        tpg = build_tpg(tasks, oracle, max_joint_span=args.span)
                        tpg['interarm_collision_model'] = sampler.info
                    parameterize(tpg, vmax=args.vmax, amax=args.amax)
                tpg.update(scene_signature=signature, reference_signature=reference_signature)
                runtime_limits = cfg.get('runtime', {})
                requirements = {'support_part_id': 'TemporarySupport', 'goal_poses': _goals(cfg),
                                'support_force_min_N': runtime_limits.get('support_force_min_N', runtime_limits.get('force_min_N', 1.0)),
                                'continuity_force_min_N': runtime_limits.get('continuity_force_min_N', .15),
                                'permanent_force_min_N': runtime_limits.get('permanent_force_min_N', 1.0),
                                'unload_force_max_N': runtime_limits.get('unload_force_max_N', .5),
                                'min_retraction_m': runtime_limits.get('support_extension_m', .008),
                                'lift_tolerance_m': runtime_limits.get('lift_tolerance_m', .0001),
                                'required_validation_load_N': runtime_limits.get('proof_load_N', 8.0),
                                'goal_position_tolerance_m': runtime_limits.get('pose_tolerance_m', .009),
                                'goal_orientation_tolerance_rad': runtime_limits.get('orientation_tolerance_rad', .04),
                                'required_dynamic_objects': [part['id'] for part in cfg['parts']],
                                **cfg.get('validation_requirements', {}),
                                'assembly_tool_force_limit_N': runtime_limits.get('assembly_tool_force_limit_N', 40.0),
                                'assembly_tool_torque_limit_Nm': runtime_limits.get('assembly_tool_torque_limit_Nm', 8.0)}
                tpg['validation_requirements'] = requirements
                save_json(OUT/'tpg.json', tpg)
                stage.update(node_count=len(tpg['nodes']), edge_count=len(tpg['edges']),
                             nominal_makespan_s=tpg['nominal_makespan'])

            with manifest.stage('full_nominal_geometry_and_waiting_pose_validation') as stage:
                with MotionPlanner(sim, client, cfg) as planner:
                    geometry = verify_rollout(tpg, planner, cfg, step_s=args.rollout_step, outdir=OUT)
                save_json(OUT/'nominal_geometry_validation.json', geometry)
                if not geometry.get('valid', geometry.get('ok', False)):
                    raise RuntimeError('Nominal geometry/waiting/world-state validation failed; inspect nominal_geometry_validation.json')
                tpg['certificate']['waiting_endpoints_verified'] = True
                tpg['certificate']['nominal_world_state_verified'] = geometry.get('nominal_world_state_reconstructed', False)
                save_json(OUT/'tpg.json', tpg)
                stage['tested_samples'] = geometry.get('tested_samples')

            with manifest.stage('embed_initial_state_and_dynamic_controller'):
                install_controller(sim, cfg, tpg, scene_name=str(PLANNED.relative_to(ROOT)))
                if not PLANNED.is_file():
                    raise RuntimeError('Runtime did not save the planned initial scene')
                manifest.data['planned_scene_sha256'] = sha256(PLANNED)
                render_figures(tpg, OUT)
            summary.update(planning_success=True, reference_task_count=len(tasks), tpg=tpg['counts'],
                           nominal_makespan_s=tpg['nominal_makespan'])
            if args.plan_only:
                summary.update(success=None, stage='planned_only',
                               explanation='Planning/nominal geometry completed; actual execution was not requested.')
                manifest.data['status'] = 'planned_only'
                save_json(OUT/'summary.json', summary); manifest.write()
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
                return summary

            with manifest.stage('actual_dynamic_execution') as stage:
                manifest.data['execution_performed'] = True; manifest.write()
                summary['execution_performed'] = True
                execution = execute(sim, client, cfg, OUT, video=not args.no_video,
                                    timeout_sim_s=args.timeout_sim, timeout_wall_s=args.timeout_wall)
                execution['run_id'] = manifest.run_id
                execution['scene_signature'] = signature
                execution['reference_signature'] = reference_signature
                save_json(OUT/'execution.json', execution)
                stage.update(controller_success=execution.get('success'), failure=execution.get('failure'),
                             actual_makespan_s=execution.get('makespan', execution.get('time')))

            with manifest.stage('independent_actual_event_load_and_pose_audit') as stage:
                audit = validate_execution(tpg, execution, requirements)
                audit['run_id'] = manifest.run_id
                save_json(OUT/'execution_validation.json', audit)
                execution['independent_validation_passed'] = audit['success']
                execution['success'] = bool(execution.get('success') and audit['success'])
                save_json(OUT/'execution.json', execution)
                plot_execution(tpg, execution, OUT, audit)
                summary.update(success=execution['success'], failure=execution.get('failure'),
                               actual_makespan_s=execution.get('makespan', execution.get('time')),
                               max_goal_position_error_m=audit['max_goal_position_error_m'],
                               max_goal_orientation_error_rad=audit['max_goal_orientation_error_rad'],
                               actual_edges_checked=audit['graph_edges_checked'])
                save_json(OUT/'summary.json', summary)
                stage['violation_count'] = len(audit['violations'])
                if not audit['success'] or not execution['success']:
                    raise RuntimeError('Actual scene3 execution did not pass independent acceptance; inspect execution_validation.json')

            with manifest.stage('promote_only_audited_planned_scene'):
                if source_fingerprint()['sha256'] != frozen_inputs['sha256']:
                    raise RuntimeError('Source scene or code changed during this run; refusing to publish a stale acceptance result')
                if sha256(PLANNED) != manifest.data['planned_scene_sha256']:
                    raise RuntimeError('Planned scene changed after execution began')
                shutil.copy2(PLANNED, MAIN)
                delivery = {'run_id': manifest.run_id, 'main_scene': str(MAIN),
                            'main_scene_sha256': sha256(MAIN), 'source_scene_sha256': sha256(SOURCE),
                            'scene_signature': signature, 'reference_signature': reference_signature,
                            'actual_execution_audited': True, 'initial_state_and_embedded_TPG': True,
                            'execution_validation': str(OUT/'execution_validation.json')}
                save_json(OUT/'delivery.json', delivery)
                summary['main_scene_promoted'] = True
                manifest.data['main_scene_promoted'] = True
            summary['stage'] = 'actual_execution_audited'
            manifest.data['status'] = 'complete'
            save_json(OUT/'summary.json', summary); manifest.write()
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return summary
    except BaseException as error:
        summary.update(success=False, error_type=type(error).__name__, error=str(error))
        manifest.data.update(status='failed', error_type=type(error).__name__, error=str(error),
                             traceback=traceback.format_exc())
        save_json(OUT/'summary.json', summary); manifest.write()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-scene', action='store_true')
    parser.add_argument('--replan', action='store_true')
    parser.add_argument('--reuse-reference', action='store_true')
    parser.add_argument('--reuse-tpg', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--span', type=float, default=.24)
    parser.add_argument('--step', type=float, default=.07)
    parser.add_argument('--rollout-step', type=float, default=.06)
    parser.add_argument('--vmax', type=float, default=.60)
    parser.add_argument('--amax', type=float, default=1.20)
    parser.add_argument('--port', type=int, default=23060)
    parser.add_argument('--timeout-sim', type=float, default=1800.)
    parser.add_argument('--timeout-wall', type=float, default=1200.)
    args = parser.parse_args()
    for key in ('span', 'step', 'rollout_step', 'vmax', 'amax', 'timeout_sim', 'timeout_wall'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error('--'+key.replace('_','-')+' must be positive and finite')
    run(args)


if __name__ == '__main__':
    main()
