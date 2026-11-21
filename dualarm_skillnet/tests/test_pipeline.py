"""End-to-end offline feedback loops and terminal statuses."""

import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.decompose.base import FixedDecomposer
from skillnet.domains.blocks import load_problem
from skillnet.motion.base import MotionResult, NullBackend, ScriptedBackend
from skillnet.pipeline import run
from skillnet.planning.search import SearchResult


class RecordingDecomposer(FixedDecomposer):
    def __init__(self):
        self.calls = []

    def decompose(self, problem, feedback=None):
        self.calls.append(feedback)
        return super().decompose(problem, feedback)


def handover_problem():
    return load_problem("tasks/s1_reach_handover.toml")


class PipelineTests(unittest.TestCase):
    def assert_buffer_route(self, result):
        self.assertEqual(result.status, "solved")
        actions = [step.action for step in result.schedule.steps]
        self.assertFalse(any(action.name == "handover" for action in actions))
        self.assertTrue(any(action.name == "place" and action.args[-1] == "buffer_mid"
                            for action in actions))
        self.assertTrue(any(action.name == "pick" and action.args[-1] == "buffer_mid"
                            for action in actions))

    def test_motion_failure_cuts_failed_state_and_uses_buffer(self):
        backend = ScriptedBackend(fail_once=["handover(B, A, red, hand_mid)"])
        result = run(handover_problem(), backend=backend)
        self.assert_buffer_route(result)
        failures = [line for line in result.feedback if "运动失败" in line]
        self.assertEqual(len(failures), 1)
        self.assertIn("handover(B, A, red, hand_mid)", failures[0])
        self.assertIsNotNone(result.graph)
        self.assertEqual(len(result.motion), len(result.schedule.steps))
        self.assertTrue(all(item.status == Status.VALID for item in result.motion))

    def test_unknown_qualification_replans_to_handover(self):
        problem = handover_problem()
        fact = parse_fact("reach(A, hand_mid)")
        problem.oracle.update(fact, Status.UNKNOWN, "测试")
        result = run(problem, backend=ScriptedBackend(qualify={str(fact): "VALID"}))
        self.assertEqual(result.status, "solved")
        self.assertTrue(any(step.action.name == "handover" for step in result.schedule.steps))
        messages = [line for line in result.feedback if "按需验证" in line]
        self.assertEqual(len(messages), 1)
        self.assertIn(str(fact), messages[0])
        self.assertEqual(problem.oracle.check(fact).status, Status.VALID)
        self.assertEqual(problem.oracle.check(fact).source, "运动后端按需验证")

    def test_unqualified_unknown_still_allows_buffer_and_remains_reported(self):
        problem = handover_problem()
        fact = parse_fact("reach(A, hand_mid)")
        problem.oracle.update(fact, Status.UNKNOWN, "测试")
        result = run(problem, backend=NullBackend())
        self.assert_buffer_route(result)
        self.assertIn(fact, result.schedule.unknown)

    def test_invalid_qualification_is_retained_and_uses_buffer(self):
        problem = handover_problem()
        fact = parse_fact("reach(A, hand_mid)")
        problem.oracle.update(fact, Status.UNKNOWN, "测试")
        result = run(problem, backend=ScriptedBackend(qualify={str(fact): "INVALID"}))
        self.assert_buffer_route(result)
        self.assertNotIn(fact, result.schedule.unknown)
        self.assertEqual(problem.oracle.check(fact).status, Status.INVALID)
        self.assertEqual(sum("按需验证" in line for line in result.feedback), 1)

    def test_qualification_on_final_round_requires_another_search(self):
        problem = handover_problem()
        fact = parse_fact("reach(A, hand_mid)")
        problem.oracle.update(fact, Status.UNKNOWN, "测试")
        result = run(problem, backend=ScriptedBackend(qualify={str(fact): "VALID"}), max_rounds=1)
        self.assertEqual(result.status, "budget")
        self.assertIsNone(result.graph)
        self.assertEqual(result.motion, [])

    def test_relevance_failure_falls_back_to_full_search(self):
        with patch("skillnet.pipeline.relevant_actions", return_value=(frozenset(), {})):
            result = run(handover_problem())
        self.assertEqual(result.status, "solved")
        self.assertTrue(any("完整搜索" in line for line in result.feedback))
        self.assertTrue(any(step.action.name == "handover" for step in result.schedule.steps))

    def test_model_retries_only_when_completion_is_unresolved(self):
        decomposer = RecordingDecomposer()
        repaired = run(handover_problem(), decomposer=decomposer)
        self.assertEqual(repaired.status, "solved")
        self.assertTrue(repaired.completion.goal_reached)
        self.assertTrue(any(entry.kind == "删除" for entry in repaired.completion.entries))
        self.assertEqual(decomposer.calls, [None])

        impossible = handover_problem()
        impossible.oracle.update(parse_fact("reach(B, stock_r1)"), Status.INVALID, "测试")
        decomposer = RecordingDecomposer()
        result = run(impossible, decomposer=decomposer)
        self.assertEqual(result.status, "no_plan")
        self.assertTrue(result.completion.unresolved)
        self.assertEqual(len(decomposer.calls), 3)
        self.assertIsNone(decomposer.calls[0])
        self.assertTrue(all(isinstance(feedback, list) and feedback
                            for feedback in decomposer.calls[1:]))
        self.assertTrue(all(isinstance(line, str) for feedback in decomposer.calls[1:]
                            for line in feedback))

    def test_model_retry_limit_can_disable_retry(self):
        problem = handover_problem()
        problem.oracle.update(parse_fact("reach(B, stock_r1)"), Status.INVALID, "测试")
        decomposer = RecordingDecomposer()
        result = run(problem, decomposer=decomposer, max_model_rounds=0)
        self.assertEqual(result.status, "no_plan")
        self.assertEqual(decomposer.calls, [None])

    def test_absent_proposal_skips_completion(self):
        problem = handover_problem()
        problem.proposal = None
        with patch("skillnet.pipeline.complete", side_effect=AssertionError("不应调用补全")):
            result = run(problem)
        self.assertEqual(result.status, "solved")
        self.assertIsNone(result.proposal)
        self.assertIsNone(result.completion)

    def test_round_exhaustion_does_not_return_unvalidated_solution(self):
        backend = ScriptedBackend(fail_once=["handover(B, A, red, hand_mid)"])
        result = run(handover_problem(), backend=backend, max_rounds=1)
        self.assertEqual(result.status, "budget")
        self.assertIsNone(result.graph)

    def test_unknown_motion_validation_is_not_a_solution(self):
        class UnknownMotion(NullBackend):
            def validate(self, action, state):
                return MotionResult(Status.UNKNOWN, "几何检查尚无结论", [])

        result = run(handover_problem(), backend=UnknownMotion())
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.graph)

    def test_motion_backend_receives_each_actual_start_state(self):
        class RecordingBackend(NullBackend):
            def __init__(self):
                self.calls = []

            def validate(self, action, state):
                self.calls.append((action.key, state))
                return super().validate(action, state)

        backend = RecordingBackend()
        result = run(handover_problem(), backend=backend)
        expected = [(step.action.key, step.state) for step in result.schedule.steps]
        self.assertEqual(backend.calls, expected)

    def test_unsolved_search_statuses_are_preserved_and_skip_motion(self):
        for status in ("budget", "no_plan", "unknown"):
            with self.subTest(status=status):
                problem = handover_problem()
                problem.proposal = None
                schedule = SearchResult(status, [], 0.0, 1, 1, {}, frozenset(), 0.0)
                with patch("skillnet.pipeline.search", return_value=schedule), \
                        patch.object(NullBackend, "validate", side_effect=AssertionError("不应验证无解时间表")):
                    result = run(problem)
                self.assertEqual(result.status, status)
                self.assertEqual(result.schedule.status, status)
                self.assertIsNone(result.graph)


if __name__ == "__main__":
    unittest.main()
