"""First-solution guidance must not replace admissible proofs or exact A*."""
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.bounds import HMaxBound

S = importlib.import_module('skillnet.planning.search')
ROOT = Path(__file__).resolve().parents[1]


def fixture():
    problem = load_problem(ROOT/'tasks/s1_nearest_arm.toml')
    problem.proposal = None
    return Engine(problem)


class IncumbentStageTests(unittest.TestCase):
    def test_domain_budget_exhaustion_leaves_an_all_action_attempt(self):
        clock = [0.]
        def observe(state, running, elapsed):
            clock[0] += 1.
        with patch.object(S, 'monotonic', side_effect=lambda:clock[0]):
            result = S.weighted_anytime(fixture(), bound=lambda s,r=():0.,
                time_limit=16., observer=observe)
        self.assertEqual([r['phase'] for r in result.stages[:2]], ['incumbent','incumbent'])
        self.assertEqual([r['action_scope'] for r in result.stages[:2]], ['goal_domain','all_actions'])
        self.assertLessEqual(result.stages[0]['time_budget'], 2.)

    def test_reference_construction_time_cannot_start_work_after_global_deadline(self):
        clock, late = [0.], []
        def build(*args, **kwargs):
            clock[0]=16.
            return lambda state,running=():0.
        def observe(state, running, elapsed):
            if clock[0]>=16.: late.append(True)
        with patch.object(S, 'monotonic', side_effect=lambda:clock[0]), \
             patch('skillnet.planning.heuristic.build_heuristic', side_effect=build):
            result = S.weighted_anytime(fixture(), time_limit=16., observer=observe)
        self.assertEqual(result.status, 'solved')
        self.assertFalse(late)

    def test_blocks_first_stage_then_exact_weight_one_and_shared_work(self):
        engine = fixture()
        zero = lambda state, running=(): 0.
        exact = S._search(engine, bound=zero)
        result = S.weighted_anytime(engine, bound=zero, time_limit=5.)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(result.stages[0]['phase'], 'incumbent')
        self.assertEqual(result.stages[0]['action_scope'], 'goal_domain')
        self.assertIsNone(result.stages[0]['lower_bound'])
        self.assertIsNone(result.stages[0]['suboptimality_bound'])
        self.assertEqual(result.weight, 1.)
        self.assertEqual(result.steps, exact.steps)
        self.assertEqual(result.makespan, exact.makespan)
        self.assertEqual(result.expanded, sum(stage['expanded'] for stage in result.stages))
        self.assertGreater(result.first_solution_expanded, 0)
        self.assertLessEqual(result.first_solution_expanded, result.expanded)
        self.assertGreaterEqual(result.first_solution_seconds, 0.)

    def test_guidance_million_is_never_a_certificate_when_only_first_stage_finishes(self):
        engine, clock = fixture(), [0.]
        def observe(state, running, elapsed):
            if not running and engine.satisfied(engine.goal, state):
                clock[0] = 4.
        with patch.object(HMaxBound, 'hadd', return_value=1_000_000.), \
             patch.object(S, 'monotonic', side_effect=lambda: clock[0]):
            result = S.weighted_anytime(engine, bound=lambda state, running=():0.,
                time_limit=4., observer=observe)
        self.assertEqual(result.status, 'solved')
        self.assertEqual([row['phase'] for row in result.stages], ['incumbent'])
        self.assertEqual(result.lower_bound, 0.)
        self.assertIsNone(result.suboptimality_bound)

    def test_allocation_pruning_is_confined_to_first_solution_stage(self):
        with patch('skillnet.planning.allocation.Allocation.deficiency', return_value=1):
            result = S.weighted_anytime(fixture(), bound=lambda state, running=():0., time_limit=5.)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(result.weight, 1.)
        probes = [row for row in result.stages if row['phase']=='incumbent']
        self.assertTrue(probes)
        self.assertTrue(all(row['status']=='no_plan' for row in probes))
        self.assertTrue(any(row['phase']=='weighted' and row['status']=='solved' for row in result.stages))

    def test_required_unknown_still_returns_before_guidance(self):
        paths = list((ROOT/'results/sim_v1_3/v133/qualification/F3/005').glob('*/*.qualified.toml'))
        result = S.search(Engine(load_problem(paths[0])), time_limit=5.)
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(result.stages)
        self.assertLess(result.elapsed, 5.)


if __name__=='__main__': unittest.main()
