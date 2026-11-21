"""Motion-scope contracts from the new optimization authorization."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from time import monotonic

from skillnet.capability.oracle import Status
from simbridge.compiler import CompileFailure
from tests.test_coppelia_backend import backend_fixture
from skillnet.core.logic import Fact, parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.motion.base import MotionResult, NullBackend
from skillnet.pipeline import run
from skillnet.planning.search import SearchResult, SearchStep
from skillnet.report import plan_to_dict


class TerminalFailureScopeTests(unittest.TestCase):
    def test_current_branch_failure_does_not_inherit_an_earlier_seed_collision(self):
        # Captured pattern: v134 fixed20/07 validations[1], r2. The final
        # candidate violates the 0.25-rad branch bound; an earlier trial hits
        # the table. Neither candidate proves global reach infeasibility.
        failures = [
            {"ok": False, "problem": {"reason": "table_collision:B",
                "pair": ["B_link", "Workbench"]}},
            {"ok": False, "problem": {"reason": "IK_branch_discontinuity",
                "max_joint_delta_rad": 2.0919355885467636, "limit_rad": .25}},
        ]
        engine, _, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        with patch.object(backend.compiler, "compile_action", side_effect=CompileFailure(
                Status.UNKNOWN, "IK_branch_discontinuity", "r2", deepcopy(failures))):
            result = backend.validate(action, engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        record = backend.validation_records[-1]
        self.assertEqual(record["classification"]["kind"], "ik_branch_discontinuity")
        self.assertEqual(record["failure_evidence"], failures)


class NoProposal:
    def decompose(self, problem, messages=None):
        return None


def schedule(status, action=None, state=None):
    return SearchResult(status, [] if action is None else [SearchStep(action, 0., 1., state)],
                        0. if action is None else 1., 1, 1, {}, frozenset(), .001)


class ScopedPipelineFeedbackTests(unittest.TestCase):
    def test_candidate_collision_exhaustion_remains_conditional_unknown(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action('pick(A, red, stock_l1)')
        class Collision(NullBackend):
            validation_context = {'classification': 'ik_collision', 'temporary_cut': True,
                                  'failure_scope': 'candidate', 'context_id': 'scene1'}
            def validate(self, action, state):
                return MotionResult(Status.INVALID, 'this candidate hits the workbench')
        with patch('skillnet.pipeline.search', side_effect=[
                schedule('solved', action, engine.initial), schedule('no_plan')]):
            result = run(engine.problem, backend=Collision(), decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.schedule.status, 'unknown')
        data = plan_to_dict(result)
        self.assertEqual(data['status'], data['search']['status'])
        self.assertTrue(any('临时' in line for line in result.feedback))

    def test_repeated_local_cuts_never_downgrade_global_reach(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action('pick(A, red, stock_l1)')
        class Collision(NullBackend):
            validation_context = {'classification': 'ik_collision', 'temporary_cut': True,
                                  'failure_scope': 'candidate', 'context_id': 'scene1'}
            def validate(self, action, state):
                return MotionResult(Status.INVALID, 'one finite candidate collides')
        states = [engine.initial | {Fact('audit_marker', (str(i),))} for i in range(3)]
        with patch('skillnet.pipeline.search', side_effect=[
                *(schedule('solved', action, state) for state in states), schedule('no_plan')]):
            result = run(engine.problem, backend=Collision(), decomposer=NoProposal(), prune=False)
        fact = parse_fact('reach(A, stock_l1)')
        self.assertEqual(result.problem.oracle.check(fact).status, Status.VALID)
        self.assertEqual(engine.problem.oracle.check(fact).status, Status.VALID)
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(any('三次规避' in line for line in result.feedback))

    def test_unchanged_failed_candidate_stops_without_repeating_rounds(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action('pick(A, red, stock_l1)')
        class Exhausted(NullBackend):
            validation_context = {'classification': 'candidates_exhausted', 'temporary_cut': True,
                'failure_scope': 'candidate', 'context_id': 'scene1', 'candidate_id': 'bank24',
                'retryable': False, 'deduplicated': True}
            def validate(self, action, state):
                return MotionResult(Status.UNKNOWN, 'same candidate already failed')
        with patch('skillnet.pipeline.search', return_value=schedule('solved', action, engine.initial)) as search_mock:
            result = run(engine.problem, backend=Exhausted(), decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'unknown')
        self.assertLessEqual(search_mock.call_count, 2)
        self.assertTrue(any('进展' in line for line in result.feedback))

    def test_search_receives_remaining_task_deadline(self):
        problem = load_problem('tasks/s1_parallel_place.toml')
        received = []
        def bounded_search(engine, **kwargs):
            received.append(kwargs.get('time_limit'))
            return schedule('budget')
        with patch('skillnet.pipeline.search', side_effect=bounded_search):
            result = run(problem, decomposer=NoProposal(), deadline=monotonic()+.5)
        self.assertEqual(result.status, 'budget')
        self.assertTrue(received)
        self.assertTrue(all(value is not None and 0 < value <= .5 for value in received))

    def test_new_relevant_candidate_can_reopen_its_temporary_cut(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action('pick(A, red, stock_l1)')
        class ChangingCandidate(NullBackend):
            def __init__(self):
                self.calls = 0
                self.reopened = False
                self.validation_context = {}
            def validate(self, action, state):
                self.calls += 1
                self.validation_context = {'classification': 'finite_ik', 'temporary_cut': True,
                    'context_id': 'scene1', 'candidate_version': 'v1', 'retryable': False}
                return MotionResult(Status.UNKNOWN if self.calls == 1 else Status.VALID, 'candidate')
            def reconsider_cuts(self, records):
                if self.reopened:
                    return []
                self.reopened = True
                return [{'cut_id': records[0]['cut_id'], 'new_candidate_version': 'v2',
                         'reason': 'new endpoint candidate became available'}]
        backend = ChangingCandidate()
        with patch('skillnet.pipeline.search', side_effect=[
                schedule('solved', action, engine.initial), schedule('no_plan'),
                schedule('solved', action, engine.initial)]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(backend.calls, 2)
        self.assertTrue(backend.reopened)


if __name__ == "__main__":
    unittest.main()
