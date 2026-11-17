"""Public relaxed-estimate contracts for FIX v1.3.4 B."""
import unittest
from math import inf

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.domains.blocks import BlocksHooks, load_problem
from skillnet.planning.bounds import HMaxBound
from skillnet.planning.heuristic import build_heuristic
from tests.test_bounds import deletion_engine
from tests.test_search import TestHooks, work_engine


class RelaxedEstimateTests(unittest.TestCase):
    def test_independent_goals_share_relaxation_but_expose_max_and_sum(self):
        engine = work_engine()
        bound = HMaxBound(engine)
        maximum, additive = bound.evaluate(engine.initial)
        # Two independent two-second goals: parallel max=2, ordering sum=4.
        self.assertAlmostEqual(maximum, 2.)
        self.assertAlmostEqual(additive, 4.)
        self.assertEqual(bound(engine.initial), maximum)
        self.assertEqual(bound.hadd(engine.initial), additive)

    def test_allowed_graph_excludes_producers_but_keeps_running_effects(self):
        engine = work_engine()
        first = engine.parse_action("work(A)")
        second = engine.parse_action("work(B)")
        bound = HMaxBound(engine, allowed={first.key})
        self.assertEqual(bound.evaluate(engine.initial), (inf, inf))
        state = engine.start(second, engine.initial)
        maximum, additive = bound.evaluate(state, ((second.key, 1.25),))
        self.assertAlmostEqual(maximum, 2.)
        self.assertAlmostEqual(additive, 3.25)
        self.assertAlmostEqual(build_heuristic(engine, goal=[parse_condition("done(A)")],
            allowed={first.key}, time_limit=0.)(engine.initial), 2.)

    def test_negative_cleanup_and_coarse_running_cache_keep_max_semantics(self):
        engine = deletion_engine()
        action = engine.parse_action("populate(B)")
        state = engine.initial - {parse_fact("filled(x)")}
        bound = HMaxBound(engine, coarse=True)
        for remaining in (1., 4., 10.):
            maximum, additive = bound.evaluate(state, ((action.key, remaining),))
            self.assertAlmostEqual(maximum, remaining + 5.)
            self.assertGreaterEqual(additive, maximum)
        self.assertEqual(len(bound.cache), 1)

    def test_extra_invariant_dependency_rechecks_fired_rule(self):
        class Hooks(TestHooks):
            def invariant_needs(self, action):
                return [parse_fact("load()")] if action.name == "finish" else []
            def optimistic_effect_time(self, fact, earliest, times, action):
                if action.name == "finish":
                    return max(earliest, times.get(parse_fact("load()"), inf))
                return earliest
        engine = simple_engine([
            {"name": "finish", "pre": ["ready()", "ready()"], "add": ["done()"], "op_time": 2.},
            {"name": "prepare", "pre": [], "add": ["load()"], "op_time": 7.},
            {"name": "seed", "pre": [], "add": ["ready()"], "op_time": 1.},
        ], Hooks(), ["ready()"])
        maximum, additive = HMaxBound(engine).evaluate(engine.initial)
        self.assertAlmostEqual(maximum, 7.)
        self.assertAlmostEqual(additive, 7.)

    def test_custom_derived_time_improvement_reopens_enabled_producer(self):
        class EarlierDerived(TestHooks):
            derived_predicates = frozenset({"ready"})

            def evaluate(self, predicate, args, state):
                return False

            def expand(self, fact):
                return [parse_fact("p()"), parse_fact("late()")]

            def optimistic_time(self, fact, times):
                return min(times.get(parse_fact("p()"), inf),
                           1. if parse_fact("late()") in times else inf)

        engine = simple_engine([
            {"name": "seedp", "pre": [], "add": ["p()"], "op_time": 3.},
            {"name": "seedq", "pre": [], "add": ["q()"], "op_time": 4.},
            {"name": "later", "pre": ["q()"], "add": ["late()"], "op_time": 1.},
            {"name": "finish", "pre": ["ready()"], "add": ["done()"], "op_time": 2.},
        ], EarlierDerived())
        # ready first appears at 3, then a declared extra dependency reduces it
        # to 1. The legacy fixed point therefore reaches done at 1+2=3.
        maximum, additive = HMaxBound(engine).evaluate(engine.initial)
        self.assertAlmostEqual(maximum, 3.)
        self.assertAlmostEqual(additive, 3.)

    def test_custom_block_effect_callback_is_never_bypassed(self):
        for subclass in (False, True):
            with self.subTest(subclass=subclass):
                engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))

                def shifted(effect, earliest, times, action=None):
                    return max(42., earliest) if effect.predicate == "arm_at" else earliest

                if subclass:
                    class ShiftedBlocks(BlocksHooks):
                        def optimistic_effect_time(self, effect, earliest, times, action=None):
                            return shifted(effect, earliest, times, action)
                    engine.hooks.__class__ = ShiftedBlocks
                else:
                    engine.hooks.optimistic_effect_time = shifted
                bound = HMaxBound(engine, goal=[parse_condition("arm_at(A, board_c)")])
                self.assertAlmostEqual(bound(engine.initial), 42.)

    def test_replaced_block_class_callback_is_never_bypassed(self):
        original = BlocksHooks.optimistic_effect_time

        def shifted(hooks, effect, earliest, times, action=None):
            return max(42., earliest) if effect.predicate == "arm_at" else earliest

        try:
            BlocksHooks.optimistic_effect_time = shifted
            engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
            bound = HMaxBound(engine, goal=[parse_condition("arm_at(A, board_c)")])
            self.assertAlmostEqual(bound(engine.initial), 42.)
        finally:
            BlocksHooks.optimistic_effect_time = original


def simple_engine(skills, hooks=None, initial=(), goal=("done()",)):
    rows = [{"params": {"arm": "arm"}, "delete": [], "occupies": ["?arm"], **row} for row in skills]
    problem = Problem("relaxation", "松弛传播工作例", {"arm": ("A",), "block": (), "loc": (), "slot": ()},
        frozenset(), frozenset(map(parse_fact, initial)), tuple(map(parse_condition, goal)),
        library_from_data({"skill": rows}), hooks or TestHooks(), CapabilityOracle({}))
    return Engine(problem)


if __name__ == "__main__":
    unittest.main()
