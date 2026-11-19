"""Complete reference, final-home and reuse contracts; no simulator process."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.motion.base import MotionResult, NullBackend
from skillnet.pipeline import run
from skillnet.planning.taskgraph import build_task_graph
from skillnet.report import plan_to_dict
from tests.test_coppelia_backend import backend_fixture
from tests.test_motion_v135_feedback import NoProposal, schedule
from tests.test_v13_compiler import sequential


class PipelineCompleteReferenceTests(unittest.TestCase):
    def test_final_home_failure_prevents_solved_and_retains_terminal_evidence(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action('pick(A, red, stock_l1)')
        class FinalHome(NullBackend):
            validation_context = {}
            def finalize_reference(self, schedule, graph):
                self.validation_context = {'phase': 'final_reference', 'failed_segment': 'home_B',
                    'classification': 'home_goal_collision', 'failure_scope': 'terminal',
                    'retryable': False, 'pair': ['B_Finger_pos', 'red']}
                return MotionResult(Status.UNKNOWN, 'fixed home_B collides with final red')
        with patch('skillnet.pipeline.search', return_value=schedule('solved', action, engine.initial)):
            result = run(engine.problem, backend=FinalHome(), decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.graph)
        self.assertEqual(plan_to_dict(result)['search']['status'], 'unknown')
        self.assertTrue(any('home_B' in line for line in result.feedback))

    def test_final_compile_action_failure_can_cut_its_source_and_replan(self):
        engine, _, _ = backend_fixture()
        first = engine.parse_action('pick(A, red, stock_l1)')
        other = engine.parse_action('pick(B, blue, stock_r1)')
        class FinalCompile(NullBackend):
            def __init__(self):
                self.calls = 0
                self.validation_context = {}
            def finalize_reference(self, schedule, graph):
                self.calls += 1
                self.validation_context = {'phase': 'final_reference', 'source_step': 1,
                    'classification': 'rrt_exhausted', 'failure_scope': 'candidate',
                    'temporary_cut': True, 'retryable': False, 'context_id': 'final1'}
                return MotionResult(Status.UNKNOWN if self.calls == 1 else Status.VALID, 'final reference')
        backend = FinalCompile()
        def find(engine, **kwargs):
            return schedule('solved', other if kwargs['cuts'] else first, engine.initial)
        with patch('skillnet.pipeline.search', side_effect=find) as search:
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(search.call_count, 2)
        self.assertEqual(backend.calls, 2)
        self.assertTrue(any('完整参考' in line for line in result.feedback))


class CompleteReferenceReceiptTests(unittest.TestCase):
    def prepared(self):
        engine, geometry, backend = backend_fixture()
        steps = sequential(engine, ['pick(A, red, stock_l1)', 'place(A, red, board_l1)',
                                    'pick(B, blue, stock_r1)', 'place(B, blue, board_r1)'])
        backend.begin_reference(1)
        for step in steps:
            self.assertEqual(backend.validate(step.action, step.state).status, Status.VALID)
        return engine, geometry, backend, steps

    def test_complete_reference_contains_real_homes_and_can_be_exported_without_solving_again(self):
        engine, geometry, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        calls = len(geometry.calls)
        receipt = backend.export_reference(steps, graph)
        self.assertEqual(len(geometry.calls), calls)
        tasks = receipt['tasks']
        self.assertEqual({task['arm'] for task in tasks if task['metadata'].get('final_home')}, {'A', 'B'})
        for arm in ('A', 'B'):
            self.assertEqual(backend.compiler.q[arm], backend.cfg['arms'][arm]['home_q'])
        self.assertTrue(receipt['context_sha256'])
        self.assertTrue(receipt['schedule_sha256'])
        self.assertTrue(receipt['tasks_sha256'])
        receipt['tasks'][0]['q_path'][0][0] += 100.
        self.assertNotEqual(receipt['tasks'], backend.export_reference(steps, graph)['tasks'])

    def test_nominal_evaluation_reuses_exact_complete_reference(self):
        engine, geometry, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        before = deepcopy(backend.compiler._snapshot())
        calls = len(geometry.calls)
        self.assertGreater(backend.evaluate(steps, graph), 0.)
        self.assertEqual(len(geometry.calls), calls)
        self.assertEqual(backend.compiler._snapshot(), before)
        self.assertTrue(backend.last_evaluation['reference_reused'])

    def test_export_rejects_changed_geometry_or_schedule_parameters(self):
        engine, _, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        backend.finalize_reference(steps, graph)
        backend.cfg['geometry_version'] = 'changed-test-geometry'
        with self.assertRaisesRegex(ValueError, '参考|context|上下文'):
            backend.export_reference(steps, graph)

    def test_final_home_goal_collision_is_explicit_and_does_not_discard_action_prefix(self):
        engine, geometry, backend, steps = self.prepared()
        before = deepcopy(backend.compiler._snapshot())
        validate = geometry.validate_joint
        home = backend.cfg['arms']['B']['home_q']
        def final_brick_collision(arm, q, **kwargs):
            if arm == 'B' and q == home:
                return {'ok': False, 'problem': {'reason': 'block_robot_collision:red:B',
                        'pair': ['B_Finger_pos', 'red']}}
            return validate(arm, q, **kwargs)
        geometry.validate_joint = final_brick_collision
        result = backend.finalize_reference(steps, build_task_graph(engine, steps))
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(backend.validation_context['failed_segment'], 'home_B')
        self.assertEqual(backend.validation_context['failure_scope'], 'terminal')
        self.assertIn('red', str(backend.validation_context))
        self.assertEqual(backend.compiler._snapshot(), before)
        self.assertFalse(any(task['metadata'].get('final_home') for task in backend.compiler.tasks))

    def test_full_reference_keeps_cross_arm_causal_dependencies(self):
        _, _, backend, steps = self.prepared()
        graph = {'edges': [{'source': 2, 'target': 3}]}
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        tasks = backend.export_reference(steps, graph)['tasks']
        last_a = [task for task in tasks if task['metadata'].get('source_step') == 2][-1]
        first_b = next(task for task in tasks if task['metadata'].get('source_step') == 3)
        self.assertIn(last_a['id'], first_b['dependencies'])
        by_id = {task['id']: task for task in tasks}
        self.assertTrue(all(by_id[dep]['reference_order'] < task['reference_order']
                            for task in tasks for dep in task['dependencies']))

    def test_reverse_home_order_is_a_distinct_bounded_context_and_is_kept_in_tasks(self):
        engine, geometry, backend, steps = self.prepared()
        validate = geometry.validate_joint
        homes = {arm: backend.cfg['arms'][arm]['home_q'] for arm in ('A', 'B')}
        def first_order_blocked(arm, q, **kwargs):
            if (arm == 'B' and q == homes['B']
                    and kwargs.get('options', {}).get('other_q') == homes['A']):
                return {'ok': False, 'problem': {'reason': 'interarm_collision',
                        'pair': ['A_Finger_pos', 'B_Finger_pos']}}
            return validate(arm, q, **kwargs)
        geometry.validate_joint = first_order_blocked
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        tasks = backend.export_reference(steps, graph)['tasks']
        returns = [task for task in tasks if task['metadata'].get('final_home')]
        self.assertEqual([task['arm'] for task in returns], ['B', 'A'])
        self.assertIn(returns[0]['id'], returns[1]['dependencies'])
        prefix = [task for task in tasks if not task['metadata'].get('final_home')]
        referenced = {dep for task in prefix for dep in task['dependencies']}
        leaves = {task['id'] for task in prefix if task['id'] not in referenced}
        self.assertTrue(leaves.issubset(returns[0]['dependencies']))

    def test_changed_live_post_state_rejects_export(self):
        engine, _, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        backend.compiler.q['A'][0] += .01
        with self.assertRaisesRegex(ValueError, '参考|context|上下文'):
            backend.export_reference(steps, graph)

    def test_changed_graph_rejects_export(self):
        engine, _, backend, steps = self.prepared()
        self.assertEqual(backend.finalize_reference(steps, build_task_graph(engine, steps)).status, Status.VALID)
        with self.assertRaisesRegex(ValueError, '参考|schedule|时间表'):
            backend.export_reference(steps, {'edges': []})

    def test_dependency_cycle_restores_the_complete_prefix_without_querying_homes(self):
        _, geometry, backend, steps = self.prepared()
        before = deepcopy(backend.compiler._snapshot())
        calls = len(geometry.calls)
        result = backend.finalize_reference(steps, {'edges': [{'source': 2, 'target': 1}]})
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(backend.compiler._snapshot(), before)
        self.assertEqual(len(geometry.calls), calls)
        self.assertNotIn('source_step', backend.validation_context)

    def test_finalization_is_idempotent_without_revalidating_or_duplicating_homes(self):
        engine, geometry, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        calls = len(geometry.calls)
        before = deepcopy(backend.compiler._snapshot())
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        self.assertEqual(len(geometry.calls), calls)
        self.assertEqual(backend.compiler._snapshot(), before)

    def test_finalized_evaluation_rejects_changed_hints_instead_of_recompiling(self):
        engine, geometry, backend, steps = self.prepared()
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        calls = len(geometry.calls)
        backend.compiler.qualification_hints = [{'new_policy': True}]
        with self.assertRaisesRegex(ValueError, '参考|context|上下文'):
            backend.evaluate(steps, graph)
        self.assertEqual(len(geometry.calls), calls)


if __name__ == '__main__':
    unittest.main()
