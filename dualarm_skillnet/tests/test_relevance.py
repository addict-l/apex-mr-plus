"""Backward reasoning must include prerequisite and invariant producers."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.relevance import relevant_actions
from skillnet.planning.search import search


@dataclass
class ExampleAction:
    key: tuple
    pre: tuple
    add: tuple
    delete: tuple


def action(name, *, pre=(), add=(), delete=()):
    return ExampleAction((name, ()), tuple(map(parse_condition, pre)),
                         tuple(map(parse_fact, add)), tuple(map(parse_fact, delete)))


class ExpansionHooks:
    derived_predicates = frozenset({"ready", "supported", "cycle_a", "cycle_b"})

    def expand(self, fact):
        definitions = {
            "ready": ["supported(beam)"],
            "supported": ["supporting(A, beam)", "filled(platform)"],
            "cycle_a": ["cycle_b()", "seed()"],
            "cycle_b": ["cycle_a()"],
        }
        return list(map(parse_fact, definitions[fact.predicate]))

    def invariant_needs(self, step):
        if step.key[0] == "insert":
            return list(map(parse_fact, ["supported(beam)", "filled(anchor)"]))
        return []


def example_engine(actions, goals):
    return SimpleNamespace(actions={item.key: item for item in actions},
                           goal=tuple(map(parse_condition, goals)), hooks=ExpansionHooks())


class RelevanceTests(unittest.TestCase):
    def test_negative_goal_selects_deleter_and_its_positive_prerequisites(self):
        cleanup = action("cleanup", pre=["held(A, tool)"], delete=["filled(temporary)"])
        acquire = action("acquire", add=["held(A, tool)"])
        irrelevant = action("decorative", add=["painted(block)"])
        engine = example_engine([cleanup, acquire, irrelevant], ["not filled(temporary)"])
        keys, reasons = relevant_actions(engine)
        self.assertEqual(keys, {cleanup.key, acquire.key})
        self.assertEqual(reasons[cleanup.key], [parse_fact("filled(temporary)")])
        self.assertEqual(reasons[acquire.key], [parse_fact("held(A, tool)")])

    def test_recursive_derived_expansion_keeps_all_alternatives(self):
        manual = action("manual", add=["supporting(A, beam)"])
        platform = action("platform", add=["filled(platform)"])
        engine = example_engine([manual, platform], ["ready(beam)"])
        keys, reasons = relevant_actions(engine)
        self.assertEqual(keys, {manual.key, platform.key})
        self.assertEqual(reasons[manual.key], [parse_fact("supporting(A, beam)")])
        self.assertEqual(reasons[platform.key], [parse_fact("filled(platform)")])

    def test_invariant_needs_include_support_and_anchors(self):
        steps = [action("pick", add=["held(A, beam)"]),
                 action("manual", add=["supporting(B, beam)", "supporting(A, beam)"]),
                 action("platform", add=["filled(platform)"]),
                 action("anchor", add=["filled(anchor)"]),
                 action("insert", pre=["held(A, beam)"], add=["filled(beam)"])]
        keys, reasons = relevant_actions(example_engine(steps, ["filled(beam)"]))
        self.assertEqual(keys, {step.key for step in steps})
        self.assertIn(parse_fact("filled(anchor)"), reasons[("anchor", ())])
        self.assertIn(parse_fact("supporting(A, beam)"), reasons[("manual", ())])

    def test_derived_and_action_cycles_terminate(self):
        seed = action("seed", pre=["other()"], add=["seed()"])
        other = action("other", pre=["seed()"], add=["other()"])
        engine = example_engine([seed, other], ["cycle_a()"])
        keys, reasons = relevant_actions(engine)
        self.assertEqual(keys, {seed.key, other.key})
        self.assertEqual(reasons[other.key], [parse_fact("other()")])

    def test_empty_goal_and_negative_preconditions(self):
        goal_step = action("finish", pre=["not blocked()"], add=["done()"])
        unblock = action("unblock", delete=["blocked()"])
        engine = example_engine([goal_step, unblock], ["done()"])
        self.assertEqual(relevant_actions(engine, goal=[]), (frozenset(), {}))
        self.assertEqual(relevant_actions(engine)[0], {goal_step.key})

    def test_reasons_include_later_discovered_targets(self):
        multi = action("multi", add=["first()", "second()"])
        final = action("final", pre=["second()"], add=["last()"])
        engine = example_engine([multi, final], ["first()", "last()"])
        _, reasons = relevant_actions(engine)
        self.assertEqual(reasons[multi.key], list(map(parse_fact, ["first()", "second()"])))

    def test_tower_relevance_preserves_a_solution(self):
        engine = Engine(load_problem("tasks/s2_tower.toml"))
        keys, reasons = relevant_actions(engine)
        result = search(engine, allowed=keys)
        self.assertEqual(result.status, "solved")
        self.assertTrue(all(step.action.key in reasons for step in result.steps))
        inserts = [step.action.args[-1] for step in result.steps if step.action.name == "insert"]
        self.assertEqual(inserts, ["t1", "t2", "t3"])


if __name__ == "__main__":
    unittest.main()
