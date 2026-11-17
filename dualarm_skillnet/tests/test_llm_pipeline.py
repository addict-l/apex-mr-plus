"""Model errors must preserve the task and fall back to complete planning."""
import unittest
from unittest.mock import patch

from skillnet.core.problem import Proposal
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.planning.completion import CompletionReport


class BrokenDecomposer:
    def __init__(self):
        self.calls = []

    def decompose(self, problem, feedback=None):
        self.calls.append({'model': '模拟模型', 'json_valid': False, 'elapsed': 0.01,
                           'raw_output': '', 'prompt': '模拟请求'})
        raise ValueError('未检测到本机 Ollama 服务，请先启动 Ollama')


class ModelPipelineTests(unittest.TestCase):
    def test_connection_error_records_and_discards_file_proposal(self):
        problem = load_problem('tasks/s1_reach_handover.toml')
        result = run(problem, decomposer=BrokenDecomposer())
        self.assertEqual(result.status, 'solved')
        self.assertIsNone(result.proposal)
        self.assertIsNone(result.completion)
        self.assertIn('未检测到本机 Ollama 服务，请先启动 Ollama', result.model_error)
        self.assertTrue(any('不使用模型' in entry for entry in result.feedback))
        self.assertEqual(len(result.model_calls), 1)
        self.assertFalse(result.model_calls[0]['json_valid'])

    def test_failure_during_model_feedback_also_resets_to_no_model(self):
        class FeedbackFailure(BrokenDecomposer):
            def decompose(self, problem, feedback=None):
                if feedback:
                    return super().decompose(problem, feedback)
                return Proposal('', ['at(red, board_l1)'], ['pick(B, red, stock_r1)'], '模拟模型')

        unresolved = CompletionReport(unresolved=['at(red, board_l1)'],
                                      feedback_for_model=['请修正步骤'])
        with patch('skillnet.pipeline.complete', return_value=unresolved):
            result = run(load_problem('tasks/s1_reach_handover.toml'), decomposer=FeedbackFailure())
        self.assertEqual(result.status, 'solved')
        self.assertIsNone(result.proposal)
        self.assertIsNone(result.completion)
        self.assertTrue(result.model_error)
        self.assertEqual(len(result.model_calls), 1)

    def test_wrong_model_goal_is_compared_and_never_replaces_task_goal(self):
        class WrongGoal:
            calls = [{'model': '模拟模型', 'json_valid': True, 'elapsed': 0.01}]

            def decompose(self, problem, feedback=None):
                return Proposal('', ['at(red, stock_r1)'],
                    ['pick(B, red, stock_r1)', 'place(B, red, board_l1)'], '模拟模型')

        problem = load_problem('tasks/s1_reach_handover.toml')
        original_goal = problem.goal
        result = run(problem, decomposer=WrongGoal())
        self.assertEqual(problem.goal, original_goal)
        self.assertEqual(result.status, 'solved')
        # Independent calibrated segments: pick 21.594345679 + handover
        # 19.288926545 + place 16.841305809 seconds.
        self.assertAlmostEqual(result.schedule.makespan, 57.724578033, places=6)
        comparison = result.model_goal_comparison
        self.assertFalse(comparison['covers_explicit'])
        self.assertIn('at(red, board_l1)', comparison['missing'])
        self.assertIn('at(red, stock_r1)', comparison['extra'])
        self.assertEqual(len(result.model_calls), 1)
