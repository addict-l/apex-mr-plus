"""Motion avoidance cannot certify optimality in the unrestricted graph."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.motion.base import MotionResult, NullBackend
from skillnet.pipeline import run
from skillnet.report import plan_to_dict
from tests.test_coppelia_backend import backend_fixture
from tests.test_motion_v135_feedback import NoProposal, schedule


class ScopedCertificateTests(unittest.TestCase):
    def fixture(self, *, original_lower=10.):
        engine, _, _ = backend_fixture()
        first = engine.parse_action('pick(A, red, stock_l1)')
        other = engine.parse_action('pick(B, blue, stock_r1)')
        initial = schedule('solved', first, engine.initial)
        initial.makespan = 10.
        initial.lower_bound = original_lower
        initial.suboptimality_bound = None if original_lower is None else 1.
        final = schedule('solved', other, engine.initial)
        final.makespan = 15.
        final.lower_bound = 15.
        final.suboptimality_bound = 1.
        class Backend(NullBackend):
            validation_context = {'failure_scope': 'candidate', 'classification': 'finite_ik'}
            def __init__(self):
                self.records = []
            def validate(self, action, state):
                return MotionResult(Status.UNKNOWN if action.key == first.key else Status.VALID, 'finite bank')
            def record_feedback_round(self, row):
                self.records.append(deepcopy(row))
        return engine, initial, final, Backend()

    def test_motion_cuts_retain_only_the_unrestricted_lower_bound(self):
        engine, first, last, backend = self.fixture()
        with patch('skillnet.pipeline.search', side_effect=[first, last]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(result.schedule.lower_bound, 10.)
        self.assertEqual(result.schedule.suboptimality_bound, 1.5)
        data = plan_to_dict(result)
        self.assertEqual(data['suboptimality_bound'], 1.5)
        self.assertEqual(data['search']['lower_bound'], 10.)
        proof = backend.records[-1]['certificate_scope']
        self.assertEqual(proof['restricted_search']['lower_bound'], 15.)
        self.assertEqual(proof['restricted_search']['suboptimality_bound'], 1.)
        self.assertTrue(proof['temporary_cuts'])
        self.assertIn('symbolic', proof['scope'])

    def test_no_unrestricted_lower_bound_does_not_inherit_a_restricted_proof(self):
        engine, first, last, backend = self.fixture(original_lower=None)
        with patch('skillnet.pipeline.search', side_effect=[first, last]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertIsNone(result.schedule.lower_bound)
        self.assertIsNone(result.schedule.suboptimality_bound)

    def test_no_motion_cut_preserves_the_existing_certificate_exactly(self):
        engine, first, _, backend = self.fixture()
        with patch('skillnet.pipeline.search', return_value=first):
            result = run(engine.problem, backend=NullBackend(), decomposer=NoProposal(), prune=False)
        self.assertEqual((result.schedule.lower_bound, result.schedule.suboptimality_bound), (10., 1.))

    def test_qualification_change_invalidates_an_older_unrestricted_proof(self):
        engine, first, last, backend = self.fixture()
        changed = schedule('unknown')
        fact = Fact('reach', ('A', 'new_candidate'))
        engine.problem.oracle.update(fact, Status.UNKNOWN, 'new candidate not qualified')
        changed.unknown = frozenset({fact})
        backend.qualify = lambda fact: Status.VALID
        with patch('skillnet.pipeline.search', side_effect=[first, changed, last]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertEqual(result.status, 'solved')
        self.assertIsNone(result.schedule.lower_bound)
        self.assertIsNone(result.schedule.suboptimality_bound)

    def test_zero_original_lower_bound_never_divides_by_zero(self):
        engine, first, last, backend = self.fixture(original_lower=0.)
        with patch('skillnet.pipeline.search', side_effect=[first, last]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        self.assertEqual(result.schedule.lower_bound, 0.)
        self.assertIsNone(result.schedule.suboptimality_bound)

    def test_search_boundary_events_survive_a_later_motion_timeout(self):
        engine, first, last, backend = self.fixture()
        events = []
        backend.progress_callback = lambda row: events.append(deepcopy(row))
        with patch('skillnet.pipeline.search', side_effect=[first, last]):
            run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False)
        boundaries = [row for row in events if row['query'] == 'symbolic_search']
        self.assertEqual([row['phase'] for row in boundaries], ['before', 'after', 'before', 'after'])
        self.assertEqual(boundaries[2]['cuts'], 1)
        self.assertEqual(boundaries[-1]['expanded'], last.expanded)
        self.assertEqual(boundaries[-1]['status'], 'solved')

    def test_allowed_subgraph_lower_bound_is_not_an_original_full_graph_proof(self):
        engine, first, last, backend = self.fixture()
        with patch('skillnet.pipeline.search', side_effect=[first, last]):
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=True)
        self.assertIsNone(result.schedule.lower_bound)
        self.assertIsNone(result.schedule.suboptimality_bound)

    def test_expired_deadline_still_commits_search_after_budget_event(self):
        from time import monotonic
        engine, _, _, backend = self.fixture()
        events = []
        backend.progress_callback = events.append
        with patch('skillnet.pipeline.search') as solve:
            result = run(engine.problem, backend=backend, decomposer=NoProposal(), prune=False,
                         deadline=monotonic()-1.)
        solve.assert_not_called()
        self.assertEqual(result.status, 'budget')
        self.assertEqual([row['phase'] for row in events], ['before', 'after'])
        self.assertEqual(events[-1]['status'], 'budget')


if __name__ == '__main__':
    unittest.main()
