import unittest
from skillnet.core.logic import Fact, Condition, parse_fact
from skillnet.core.library import load_library
from skillnet.core.problem import Problem
from skillnet.core.engine import Engine
from skillnet.capability.oracle import CapabilityOracle, Status


class EmptyHooks:
    derived_predicates = frozenset()
    resource_predicates = frozenset({"free", "arm_at"})
    derived_basis = {}

    def invariant(self, state):
        return None


class EngineTests(unittest.TestCase):
    def make_engine(self):
        static = frozenset(map(parse_fact, ["placeable(src)", "placeable(dst)",
            "arm_pair(A, B)", "two_arm_part(big)"]))
        initial = frozenset(map(parse_fact, ["free(A)", "free(B)", "at(red, src)",
            "at(big, dst)", "arm_at(A, home_A)", "arm_at(B, home_B)"]))
        return Engine(Problem("test", "", {"arm": ("A", "B"), "block": ("red", "big"),
            "loc": ("src", "dst"), "slot": ()}, static, initial, (), load_library(),
            EmptyHooks(), CapabilityOracle({"reach": {"default": "VALID"},
                                           "liftable": {"default": "VALID"}})))

    def test_start_end_and_persistent_hold(self):
        engine = self.make_engine()
        action = engine.parse_action("pick(A, red, src)")
        started = engine.start(action, engine.initial)
        self.assertNotIn(parse_fact("free(A)"), started)
        self.assertNotIn(parse_fact("held(A, red)"), started)
        ended = engine.finish(action, started)
        self.assertIn(parse_fact("held(A, red)"), ended)
        self.assertNotIn(parse_fact("free(A)"), ended)
        self.assertIn(parse_fact("arm_at(A, src)"), ended)
        self.assertNotIn(parse_fact("arm_at(A, home_A)"), ended)

    def test_grounding_and_copick(self):
        engine = self.make_engine()
        with self.assertRaises(ValueError):
            engine.parse_action("pick(A, big, dst)")
        with self.assertRaises(ValueError):
            engine.parse_action("co_pick(A, A, big, dst)")
        action = engine.parse_action("co_pick(A, B, big, dst)")
        self.assertTrue(engine.satisfied(action.pre, engine.initial))
        self.assertEqual(engine.oracle.evaluate(action.capability)[0], Status.VALID)
        engine.oracle.update(parse_fact("reach(B, dst)"), Status.UNKNOWN, "测试")
        self.assertEqual(engine.oracle.evaluate(action.capability)[0], Status.UNKNOWN)
        self.assertEqual(set(action.occupies), {"A", "B"})

    def test_candidate_index_and_predicate_conflict(self):
        engine = self.make_engine()
        expected = {a.key for a in engine.actions.values() if engine.satisfied(a.pre, engine.initial)}
        actual = {a.key for a in engine.candidates(engine.initial) if engine.satisfied(a.pre, engine.initial)}
        self.assertEqual(expected, actual)
        engine.hooks.derived_predicates = frozenset({"free"})
        with self.assertRaisesRegex(ValueError, "动态和推导"):
            Engine(engine.problem)
