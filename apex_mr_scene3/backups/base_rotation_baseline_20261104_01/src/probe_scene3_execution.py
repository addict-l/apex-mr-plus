"""Execute a strict-serial prefix of real Scene3 reference paths for diagnosis.

Examples (from the scene directory)::
    python src/probe_scene3_execution.py --tasks 3
    python src/probe_scene3_execution.py --through-task verify_support
    python src/probe_scene3_execution.py --through-task release_A

No paths are invented or replaced. This intentionally serial diagnostic uses
the real reference's joint samples and the production time parameterization.
Only the in-memory controller's final full-assembly goal guard is replaced by
a prefix-completion marker. All physical, servo, ownership and load guards
remain active. The canonical controller, main scene and acceptance files are
never changed. Diagnostic completion is never reported as full task success.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_connection import Simulator
from apex.scene import resolve
from apex.tpg import build_tpg
from apex.program import parameterize
from apex.runtime import capture


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def choose_task_ids(task_ids, count=None, through=None):
    task_ids = list(task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('Reference contains duplicate task IDs')
    if through is not None:
        if through not in task_ids:
            raise ValueError(f'Prefix endpoint task not found: {through}')
        count = task_ids.index(through) + 1
    if count is None or not 1 <= count <= len(task_ids):
        raise ValueError(f'Prefix task count must be within 1..{len(task_ids)}')
    return task_ids[:count]


def trim_graph(graph, task_ids):
    """Keep an actual reference prefix and impose explicit serial node edges.

    Original joint samples, skill metadata and existing internal edges are
    copied unchanged. Cutting an unsatisfied predecessor raises an error.
    """
    selected = set(task_ids)
    ordered = sorted(graph['nodes'], key=lambda n: n['reference_index'])
    selected_nodes = [node for node in ordered if node['task_id'] in selected]
    if not selected_nodes:
        raise ValueError('Prefix contains no real graph nodes')
    if ordered[:len(selected_nodes)] != selected_nodes:
        raise ValueError('Selected tasks do not form a reference prefix')
    node_ids = {node['id'] for node in selected_nodes}
    for edge in graph['edges']:
        if edge['target'] in node_ids and edge['source'] not in node_ids:
            raise ValueError('Prefix would remove a required predecessor')
    result = deepcopy(graph)
    result['nodes'] = deepcopy(selected_nodes)
    result['edges'] = [deepcopy(edge) for edge in graph['edges']
                       if edge['source'] in node_ids and edge['target'] in node_ids]
    existing = {(edge['source'], edge['target']) for edge in result['edges']}
    for previous, following in zip(selected_nodes, selected_nodes[1:]):
        if (previous['id'], following['id']) not in existing:
            result['edges'].append({'source': previous['id'], 'target': following['id'],
                                    'type': 'diagnostic_serial',
                                    'reason': 'intentional_serial_reference_prefix'})
    result['task_map'] = {key: deepcopy(value) for key, value in graph['task_map'].items() if key in selected}
    result['reference_schedule'] = [deepcopy(row) for row in graph['reference_schedule'] if row['task_id'] in selected]
    result['counts'] = {'nodes': len(selected_nodes), 'edges': len(result['edges']),
                        'tasks': len(selected),
                        'pose_nodes': sum(n['node_type'] == 'pose' for n in selected_nodes),
                        'skill_nodes': sum(n['node_type'] == 'skill' for n in selected_nodes)}
    result['diagnostic'] = {'intentional_prefix': True, 'task_ids': list(task_ids),
                            'serial_reference_execution': True,
                            'full_task_acceptance': False,
                            'motion_source': 'unchanged recorded reference joint paths'}
    result['certificate'] = {'scope': 'strict serial diagnostic prefix only; original path evidence remains in reference file',
                             'asynchronous_execution_certified': False,
                             'full_task_acceptance': False}
    parameterize(result)
    return result


def prefix_controller(source):
    """Alter only completion semantics in a diagnostic copy, fail on drift."""
    original = """  if not flags.support_returned or not flags.gate_closed or not flags.permanent_verified or held.A or held.B then
   fail('TPG finished without lifecycle goal state');return
  end
  done=true;finishTime=t;log('assembly_completed',nil)"""
    replacement = """  done=true;finishTime=t
  log('diagnostic_prefix_completed',nil,{full_task_acceptance=false,intentional_prefix=true})"""
    if source.count(original) != 1:
        raise ValueError('Canonical controller completion block changed; refusing an ambiguous diagnostic patch')
    return source.replace(original, replacement, 1)


def load_graph(reference, count=None, through=None, graph_file=None):
    if graph_file:
        original = json.loads(Path(graph_file).read_text(encoding='utf-8'))
        ordered_ids = [row['task_id'] for row in original['reference_schedule']]
        selected = choose_task_ids(ordered_ids, count, through)
        return trim_graph(original, selected), Path(graph_file), None
    reference = Path(reference)
    content = json.loads(reference.read_text(encoding='utf-8'))
    tasks = content if isinstance(content, list) else content.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Reference must contain real executable tasks')
    tasks = sorted(tasks, key=lambda t: t.get('reference_order', tasks.index(t)))
    selected = choose_task_ids([task['id'] for task in tasks], count, through)
    prefix_tasks = tasks[:len(selected)]
    if any(not task.get('q_path') for task in prefix_tasks):
        raise ValueError('Reference prefix is missing actual joint geometry')
    original = build_tpg(prefix_tasks, max_joint_span=.22)
    return trim_graph(original, selected), reference, content.get('allocation') if isinstance(content, dict) else None


def snapshot(sim, cfg, output, name, time_s):
    capture(sim, cfg['camera'], output / (name + '.png'), f'Scene 3 实际动力学前缀诊断 | t={time_s:.2f}s')
    result = {'time': time_s, 'parts': {}, 'joint_positions': {}, 'dynamic_parts': {}}
    for part, handle in cfg['blocks'].items():
        result['parts'][part] = sim.getObjectPose(handle, sim.handle_world)
        result['dynamic_parts'][part] = sim.isDynamicallyEnabled(handle)
    for side, arm in cfg['arms'].items():
        result['joint_positions'][side] = [sim.getJointPosition(joint) for joint in arm['joints']]
    save(output / (name + '.json'), result)
    return result


def run(args):
    reference = args.reference
    if reference is None and args.graph is None:
        options = [ROOT / 'results/apex/alloc_and_reference.json', ROOT / 'results/apex/reference_tasks.json']
        reference = next((path for path in options if path.is_file()), None)
        if reference is None:
            raise FileNotFoundError('No real reference tasks exist yet. Run the Scene3 compiler first.')
    through = args.through_task or (None if args.tasks else 'verify_support')
    graph, input_path, allocation = load_graph(reference, args.tasks, through, args.graph)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_scene = ROOT / 'scenes/10_Scene3_Dynamic.ttt'
    canonical = (ROOT / 'apex/controller.lua').read_text(encoding='utf-8')
    diagnostic = prefix_controller(canonical)
    manifest = {'stage': 'actual_serial_prefix_diagnostic', 'intentional_prefix': True,
                'full_task_acceptance': False, 'main_scene_promoted': False,
                'reference_file': str(input_path), 'reference_sha256': sha256(input_path),
                'source_scene': str(source_scene), 'source_scene_sha256': sha256(source_scene),
                'canonical_controller_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
                'diagnostic_controller_sha256': hashlib.sha256(diagnostic.encode()).hexdigest(),
                'controller_change': 'Only final full-goal check replaced by diagnostic_prefix_completed event.',
                'selected_task_ids': graph['diagnostic']['task_ids'],
                'selected_nodes': len(graph['nodes']), 'port': args.port,
                'nominal_prefix_duration_s': graph['nominal_makespan']}
    save(output / 'manifest.json', manifest)
    save(output / 'prefix_tpg.json', graph)
    with Simulator(ROOT, port=args.port, timeout=45) as owner:
        client, sim = owner.connect()
        sim.loadScene(str(source_scene))
        cfg = resolve(sim)
        # Never adapt a mismatched source by setting the robot's joints.
        for side, arm in cfg['arms'].items():
            first = next((node for node in graph['nodes'] if node['arm'] == side), None)
            if first:
                actual = [sim.getJointPosition(joint) for joint in arm['joints']]
                errors = [abs(math.atan2(math.sin(a-b), math.cos(a-b))) for a, b in zip(actual, first['q_start'])]
                if max(errors) > .01:
                    raise ValueError(f'{side} reference start differs from source scene by {max(errors):.4f} rad')
        if allocation:
            cfg['selected_support'] = allocation['selected_support']
        cfg['program'] = graph
        sim.writeCustomTableData(cfg['root'], 'Scene3APEX', cfg)
        existing = sim.getObject('/APEX_MR_Scene3/CentralTPGScheduler', {'noError': True})
        if existing != -1:
            sim.removeObjects([existing])
        script = sim.createScript(sim.scripttype_simulation, diagnostic, 0, 'lua')
        sim.setObjectAlias(script, 'CentralTPGScheduler')
        sim.setObjectParent(script, cfg['root'], True)
        saved_scene = ROOT / 'scenes/_Scene3_Prefix_Diagnostic.ttt'
        sim.saveScene(str(saved_scene))
        manifest['diagnostic_scene'] = str(saved_scene)
        save(output / 'manifest.json', manifest)
        snapshot(sim, cfg, output, '00_initial', 0.)
        first_grasp = next((node['id'] for node in graph['nodes']
                            if node.get('metadata', {}).get('action') == 'grasp'), None)
        observed = set()
        wall = time.monotonic()
        last_status = -5.
        timeout_reason = None
        sim.setStepping(True)
        sim.startSimulation()
        try:
            while True:
                sim.step()
                state = sim.callScriptFunction('getState', script)
                t = state['time']
                if t-last_status >= 5 or state.get('failure') or state.get('done'):
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    last_status = t
                # Serial completion counts identify exact actual stage changes.
                completed = int(state.get('completed', 0))
                completed_ids = {node['id'] for node in graph['nodes'][:completed]}
                if first_grasp and first_grasp in completed_ids and 'first_grasp' not in observed:
                    snapshot(sim, cfg, output, '01_first_actual_grasp', t)
                    observed.add('first_grasp')
                if state.get('support_load_verified') and 'fixture_load' not in observed:
                    snapshot(sim, cfg, output, '02_actual_fixture_load', t)
                    observed.add('fixture_load')
                if state.get('done') or state.get('failure'):
                    break
                if t > args.timeout_sim or time.monotonic()-wall > args.timeout_wall:
                    timeout_reason = 'diagnostic simulation/wall time limit exceeded'
                    break
            result = sim.callScriptFunction('getResult', script)
            if timeout_reason:
                result['failure'] = timeout_reason
            prefix_complete = bool(result.get('done') and not result.get('failure') and
                                   result.get('completed') == len(graph['nodes']))
            result.update(stage='actual_serial_prefix_diagnostic', intentional_prefix=True,
                          prefix_completed=prefix_complete, full_task_success=False, success=False,
                          done_meaning='selected reference prefix completed; full lifecycle is intentionally not evaluated',
                          wall_time_s=time.monotonic()-wall, reference_file=str(input_path),
                          main_scene_promoted=False)
            result['final_snapshot'] = snapshot(sim, cfg, output, '03_prefix_final', sim.getSimulationTime())
            save(output / 'execution_partial.json', result)
            summary = {'stage': result['stage'], 'prefix_completed': prefix_complete,
                       'success': False, 'full_task_success': False,
                       'failure': result.get('failure'), 'completed_nodes': result.get('completed'),
                       'expected_prefix_nodes': len(graph['nodes']),
                       'actual_prefix_time_s': result.get('makespan'),
                       'peak_fixture_force_N': result.get('peak_support_normal_N'),
                       'first_grasp_snapshot_available': 'first_grasp' in observed,
                       'output': str(output), 'main_scene_promoted': False}
            save(output / 'summary_partial.json', summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return prefix_complete
        finally:
            sim.stopSimulation()
            sim.setStepping(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tasks', type=int, help='Number of complete reference tasks in the prefix')
    group.add_argument('--through-task', help='Inclusive actual task ID; default verify_support')
    parser.add_argument('--reference', type=Path, help='Recorded tasks list or alloc_and_reference.json')
    parser.add_argument('--graph', type=Path, help='Existing real motion TPG, instead of a task file')
    parser.add_argument('--output', type=Path, default=ROOT/'results/apex/diagnostics/apex_prefix')
    parser.add_argument('--port', type=int, default=23064)
    parser.add_argument('--timeout-sim', type=float, default=300.)
    parser.add_argument('--timeout-wall', type=float, default=600.)
    args = parser.parse_args()
    if args.timeout_sim <= 0 or args.timeout_wall <= 0:
        parser.error('Timeouts must be positive')
    raise SystemExit(0 if run(args) else 1)


if __name__ == '__main__':
    main()
