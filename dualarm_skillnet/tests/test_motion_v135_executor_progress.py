"""Stage measurements survive geometry timeouts, without a live simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apexmr.tpg import build_tpg as structural_graph
from simbridge import executor
from tests.test_v13_executor import execution_fixture, synthetic_execution

ROOT = Path(__file__).resolve().parents[1]


class ExecutorProgressTests(unittest.TestCase):
    def test_statistics_rpc_is_not_added_after_a_sample_exhausts_deadline(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()
        clock, statistics_calls = {'now': 100.}, []
        def statistics():
            statistics_calls.append(clock['now'])
            return {'batches': 0, 'geometry_calls': 0}
        def slow_query(samples):
            clock['now'] += 2.
            return [{'ok': True} for _ in samples]
        geometry.sample_batch_statistics = statistics
        geometry.sample_state_pairs = slow_query
        def rollout(graph, planner, current, **kwargs):
            planner.sample_state_pairs([{}])
            return {'valid': True}
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor.time, 'monotonic', side_effect=lambda: clock['now']),
                  patch.object(executor, 'build_tpg', side_effect=lambda items, oracle: structural_graph(items)),
                  patch.object(executor, 'verify_rollout', side_effect=rollout),
                  patch.object(executor, 'execute') as execute):
                result = executor.execute_plan(None, None, cfg, tasks, engine.problem,
                    folder, planner=geometry, timeout_wall_s=1.)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['failure_phase'], 'rollout')
        self.assertTrue(statistics_calls)
        self.assertTrue(all(stamp == 100. for stamp in statistics_calls))
        self.assertEqual(result['server_geometry_statistics']['batches'], 0)
        execute.assert_not_called()

    def test_geometry_timeout_keeps_stage_and_partial_query_statistics(self):
        engine, geometry, _, steps, tasks, _, cfg = execution_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            output = Path(folder)
            def build(items, oracle=None):
                if oracle is None:
                    return structural_graph(items)
                progress = json.loads((output/'executor_progress.json').read_text(encoding='utf-8'))
                self.assertEqual(progress['phase'], 'tpg_geometry')
                self.assertGreater(progress['tpg_nodes'], 0)
                self.assertGreater(progress['candidate_conflict_pairs'], 0)
                raise TimeoutError('retained slow geometry')
            with patch.object(executor, 'build_tpg', side_effect=build):
                result = executor.execute_plan(object(), object(), cfg, tasks, engine.problem,
                    output, planner=geometry, plan_steps=steps)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['failure_phase'], 'tpg_geometry')
            self.assertIn('tpg_geometry', result['stage_seconds'])
            self.assertIn('geometry_statistics', result)
            self.assertEqual(json.loads((output/'summary.json').read_text(encoding='utf-8'))['status'], 'timeout')

    def test_success_has_separate_structure_geometry_parameterize_rollout_and_execution_times(self):
        engine, geometry, _, steps, tasks, graph, cfg = execution_fixture()
        measured = synthetic_execution(graph, cfg)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor, 'install_controller'),
                  patch.object(executor, 'execute', return_value=measured),
                  patch.object(executor, 'validate_execution', return_value={'success': True, 'violations': []})):
                result = executor.execute_plan(object(), object(), cfg, tasks, engine.problem,
                    folder, planner=geometry, plan_steps=steps)
            self.assertTrue(result['passed'], result.get('reason'))
            for stage in ('tpg_structure', 'tpg_geometry', 'parameterize', 'rollout', 'execution'):
                self.assertGreaterEqual(result['stage_seconds'][stage], 0.)
            self.assertGreater(result['geometry_statistics']['oracle_calls'], 0)
            self.assertIn('sampled', result['geometry_scope'])


if __name__ == '__main__':
    unittest.main()
