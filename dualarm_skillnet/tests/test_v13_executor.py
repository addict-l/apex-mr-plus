"""Measured-evidence acceptance and executor plumbing without a live simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apexmr.geometry import verify_rollout
from apexmr.program import parameterize
from apexmr.runtime import execute
from apexmr.tpg import build_tpg
from apexmr.validation import validate_execution
from simbridge.executor import PROFILE, build_requirements, execute_plan
from tests.test_v13_compiler import fixture, sequential


ROOT = Path(__file__).resolve().parents[1]


def synthetic_execution(graph, cfg):
    """Supply explicit fake measurements only for testing the audit boundary."""
    req, events = cfg['validation_requirements'], []
    names = {'grasp': 'grasp', 'release': 'release', 'install': 'part_installed',
             'detach': 'detach', 'support_start': 'support_started', 'support_end': 'support_ended'}
    for node in graph['nodes']:
        times = graph['nominal_timestamps'][node['id']]
        events.extend([{'kind': 'node_started', 'node': node['id'], 'time': times['start']},
                       {'kind': 'node_completed', 'node': node['id'], 'time': times['end']}])
        action = node.get('metadata', {}).get('action')
        semantic = names.get(action)
        if node.get('metadata', {}).get('handover'):
            semantic = {'grasp': 'handover_received', 'release': 'handover_released'}.get(action, semantic)
        if semantic:
            events.append({'kind': semantic, 'node': node['id'], 'time': times['end'],
                           'details': {'part': node.get('part'), 'valid': True, 'force_N': 2.}})
    events.sort(key=lambda row: row['time'])
    poses = deepcopy(cfg['initial'])
    poses.update(deepcopy(req['goal_poses']))
    return {'success': True, 'done': True, 'events': events, 'trace': [],
            'part_poses': poses, 'makespan': graph['nominal_makespan'], 'wall_time_s': .01,
            'final_state': {'held': {'A': None, 'B': None}, 'installed': deepcopy(req['required_installed']), 'supporting': {}},
            'joint_positions': deepcopy(req['home_q']),
            'snap_dynamic': {block: True for block in req['required_installed']},
            'dynamic_enablement': {'A': {'joints': [True]*6}, 'B': {'joints': [True]*6},
                                   'objects': {block: True for block in cfg['initial']}},
            'collision_failures': [], 'tracking_failures': [],
            'collision_monitor': {'enabled': True, 'checks': 10},
            'tracking_monitor': {'enabled': True, 'samples': 10, 'limit_rad': .15, 'peak_error_rad': .001}}


def execution_fixture():
    engine, geometry, compiler = fixture()
    steps = sequential(engine, ['pick(A, red, stock_l1)', 'place(A, red, board_l1)'])
    tasks = compiler.compile(steps)
    graph = parameterize(build_tpg(tasks))
    cfg = compiler.cfg
    req = build_requirements(engine.problem, cfg, tasks)
    cfg.update(profile=PROFILE, validation_requirements=req, program=graph)
    graph['validation_requirements'] = req
    return engine, geometry, compiler, steps, tasks, graph, cfg


class GenericExecutionAuditTests(unittest.TestCase):
    def test_generic_profile_accepts_measured_pick_place_without_scene3_events(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        before = deepcopy(evidence)
        audit = validate_execution(graph, evidence)
        self.assertTrue(audit['success'], audit['violations'])
        self.assertEqual(evidence, before)
        self.assertFalse(any(event['kind'] == 'gate_closed' for event in evidence['events']))

    def test_missing_events_or_broken_causal_order_are_rejected(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        evidence['events'] = [event for event in evidence['events'] if event['kind'] != 'release']
        self.assertIn('missing_or_duplicate_semantic_event',
                      {row['code'] for row in validate_execution(graph, evidence)['violations']})
        evidence = synthetic_execution(graph, cfg)
        first = graph['edges'][0]
        for event in evidence['events']:
            if event.get('node') == first['target'] and event['kind'] == 'node_started':
                event['time'] = 0.
        evidence['events'].sort(key=lambda row: row['time'])
        self.assertIn('predecessor_not_completed',
                      {row['code'] for row in validate_execution(graph, evidence)['violations']})

    def test_pose_empty_hands_home_and_monitor_evidence_are_independent_checks(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        evidence['part_poses']['red'][0] += .006
        evidence['final_state']['held']['A'] = 'red'
        evidence['joint_positions']['B'][0] += .02
        evidence.pop('collision_monitor')
        evidence['tracking_failures'] = [{'error_rad': .4}]
        evidence['tracking_monitor']['peak_error_rad'] = .4
        codes = {row['code'] for row in validate_execution(graph, evidence)['violations']}
        self.assertTrue({'goal_position_error', 'final_hands_not_known_empty', 'final_arm_not_home',
                         'collision_monitoring_not_evidenced', 'tracking_failures', 'tracking_limit_or_peak_invalid'} <= codes)

    def test_temporary_slot_must_be_physically_empty_not_just_uninstalled(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        req = deepcopy(cfg['validation_requirements'])
        req['temporary_slots'] = {'temp': {'pose': list(evidence['part_poses']['red']), 'size': [.12, .14, .076]}}
        audit = validate_execution(graph, evidence, req)
        self.assertIn('temporary_slot_physically_occupied', {row['code'] for row in audit['violations']})

    def test_required_install_needs_slot_flag_and_dynamic_connection(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        req = deepcopy(cfg['validation_requirements'])
        req['required_installed'] = {'red': 'target_slot'}
        audit = validate_execution(graph, evidence, req)
        self.assertTrue({'required_installed_goal_missing', 'required_connection_not_dynamic'}
                        <= {row['code'] for row in audit['violations']})

    def test_acceptance_tolerance_cannot_be_relaxed(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        req = {**cfg['validation_requirements'], 'goal_position_tolerance_m': .1}
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            validate_execution(graph, synthetic_execution(graph, cfg), req)

    def test_goal_requirements_follow_target_slots_and_catalog_layer_heights(self):
        engine, _, compiler = fixture('s2_tower')
        steps = sequential(engine, ['pick(A, b1, stock_l1)', 'insert(A, b1, t1)',
                                    'pick(A, b2, stock_l2)', 'insert(A, b2, t2)',
                                    'pick(B, b3, stock_r1)', 'insert(B, b3, t3)'])
        tasks = compiler.compile(steps)
        req = build_requirements(engine.problem, compiler.cfg, tasks)
        self.assertEqual(req['required_installed'], {'b1': 't1', 'b2': 't2', 'b3': 't3'})
        self.assertAlmostEqual(req['goal_poses']['b3'][2], .772+.076*2+.038)

    def test_handover_nominal_world_waits_for_receiver_then_transfers_once(self):
        engine, geometry, compiler = fixture('s1_reach_handover')
        steps = sequential(engine, ['pick(B, red, stock_r1)', 'handover(B, A, red, hand_mid)',
                                    'place(A, red, board_l1)'])
        graph = parameterize(build_tpg(compiler.compile(steps)))
        compiler.cfg['profile'] = PROFILE
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            audit = verify_rollout(graph, geometry, compiler.cfg, step_s=.5, outdir=outdir)
        self.assertTrue(audit['valid'], audit['world_state_gaps'])
        self.assertTrue(any(sample['payloads'].get('B') for sample in geometry.samples))
        self.assertTrue(any(sample['payloads'].get('A') for sample in geometry.samples))
        self.assertFalse(any(len(sample['payloads']) > 1 for sample in geometry.samples))


class ExecutorPlumbingTests(unittest.TestCase):
    def test_executor_passes_only_remaining_budget_after_geometry_and_install(self):
        engine, geometry, _, steps, tasks, _, cfg = execution_fixture()
        clock = {'now': 100.}
        def make_graph(items, oracle):
            graph = build_tpg(items)
            clock['now'] += 3.
            return graph
        def rollout(*args, **kwargs):
            clock['now'] += 2.
            return {'valid': True}
        def install(sim, current, graph, filename):
            clock['now'] += 1.
            current['program'] = graph
        def measured(sim, client, current, directory, **kwargs):
            self.assertAlmostEqual(kwargs['timeout_wall_s'], 4.)
            return synthetic_execution(current['program'], current)
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with (patch('simbridge.executor.time.monotonic', side_effect=lambda: clock['now']),
                  patch('simbridge.executor.build_tpg', side_effect=make_graph),
                  patch('simbridge.executor.verify_rollout', side_effect=rollout),
                  patch('simbridge.executor.install_controller', side_effect=install),
                  patch('simbridge.executor.execute', side_effect=measured)):
                result = execute_plan(None, None, cfg, tasks, engine.problem, outdir,
                                      planner=geometry, plan_steps=steps, timeout_wall_s=10.)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['executor_wall_seconds'], 6.)

    def test_expired_oracle_stops_before_install_and_actual_execution(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()
        clock = {'now': 100.}
        class ExpiringOracle:
            def __init__(self, *args):
                pass
            def __call__(self, *args):
                clock['now'] += 2.
                return False
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with (patch('simbridge.executor.time.monotonic', side_effect=lambda: clock['now']),
                  patch('simbridge.executor.SampledConflictOracle', ExpiringOracle),
                  patch('simbridge.executor.install_controller') as install,
                  patch('simbridge.executor.execute') as actual):
                result = execute_plan(None, None, cfg, tasks, engine.problem, outdir,
                                      planner=geometry, timeout_wall_s=1.)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['failure_phase'], 'tpg_geometry')
            install.assert_not_called()
            actual.assert_not_called()
            self.assertEqual(json.loads((Path(outdir)/'summary.json').read_text(encoding='utf-8'))['status'], 'timeout')

    def test_rollout_remote_query_checks_same_deadline_after_return(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()
        clock = {'now': 100.}
        def slow_query(samples):
            clock['now'] += 2.
            return [{'ok': True} for sample in samples]
        geometry.sample_state_pairs = slow_query
        def rollout(graph, planner, current, **kwargs):
            planner.sample_state_pairs([{}])
            return {'valid': True}
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with (patch('simbridge.executor.time.monotonic', side_effect=lambda: clock['now']),
                  patch('simbridge.executor.build_tpg', side_effect=lambda items, oracle: build_tpg(items)),
                  patch('simbridge.executor.verify_rollout', side_effect=rollout),
                  patch('simbridge.executor.install_controller') as install,
                  patch('simbridge.executor.execute') as actual):
                result = execute_plan(None, None, cfg, tasks, engine.problem, outdir,
                                      planner=geometry, timeout_wall_s=1.)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['failure_phase'], 'rollout')
            install.assert_not_called()
            actual.assert_not_called()

    def test_tpg_geometry_execution_and_measured_cost_artifacts(self):
        engine, geometry, compiler, steps, tasks, _, cfg = execution_fixture()
        def install(sim, current_cfg, graph, filename):
            current_cfg['program'] = graph
            Path(filename).write_bytes(b'MOCK_SCENE')
        def measured(sim, client, current_cfg, outdir, **kwargs):
            return synthetic_execution(current_cfg['program'], current_cfg)
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with (patch('simbridge.executor.install_controller', side_effect=install),
                  patch('simbridge.executor.execute', side_effect=measured)):
                result = execute_plan(None, None, cfg, tasks, engine.problem, outdir,
                                      planner=geometry, plan_steps=steps)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['phase'], 'complete')
            self.assertTrue(all(row['estimated_actual_ratio'] > 0 for row in result['step_durations']))
            for name in ['tasks.json', 'tpg.json', 'execution.json', 'validation.json', 'summary.json', 'planned.ttt']:
                self.assertTrue((Path(outdir)/name).is_file(), name)
            saved = json.loads((Path(outdir)/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(saved['passed'])

    def test_failed_rollout_stops_before_install_or_actual_execution(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with (patch('simbridge.executor.verify_rollout', return_value={'valid': False, 'first_violation': {'pair': ['A', 'B']}}),
                  patch('simbridge.executor.execute') as actual, patch('simbridge.executor.install_controller') as install):
                result = execute_plan(None, None, cfg, tasks, engine.problem, outdir, planner=geometry)
            self.assertFalse(result['passed'])
            self.assertEqual(result['failure_phase'], 'rollout')
            actual.assert_not_called()
            install.assert_not_called()

    def test_runtime_records_timeout_and_always_stops_its_simulation(self):
        _, _, _, _, _, graph, cfg = execution_fixture()
        evidence = synthetic_execution(graph, cfg)
        class FakeSim:
            simulation_stopped = 0
            def __init__(self):
                self.started = self.stopped = False
            def getObject(self, path):
                return 100
            def setStepping(self, value):
                pass
            def startSimulation(self):
                self.started = True
            def stopSimulation(self):
                self.stopped = True
            def getSimulationState(self):
                return 0
            def callScriptFunction(self, function, script):
                return {'time': 0., 'done': False} if function == 'getState' else deepcopy(evidence)
        sim = FakeSim()
        (ROOT/'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as outdir:
            with patch('apexmr.runtime.time.monotonic', side_effect=[0., 2., 3., 4.]):
                result = execute(sim, None, cfg, outdir, timeout_wall_s=1., video=False)
            self.assertTrue(result['timeout'])
            self.assertFalse(result['success'])
            self.assertTrue(sim.started and sim.stopped)
            self.assertTrue((Path(outdir)/'execution.json').is_file())


if __name__ == '__main__':
    unittest.main()
