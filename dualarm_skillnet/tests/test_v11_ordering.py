"""A* must use admissible estimates even without a reference upper bound."""

import unittest

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.planning.search import _search
from tests.test_search import TestHooks


def graph_engine(edges):
    skills = []
    destinations = {}
    for name, source, target, duration in edges:
        skills.append({
            "name": name, "params": {"arm": "arm"},
            "pre": ["free(?arm)", f"node({source})"],
            "delete": ["free(?arm)", f"node({source})"],
            "add": ["free(?arm)", f"node({target})"],
            "occupies": ["?arm"], "op_time": duration,
        })
        destinations[(name, ("A",))] = target
    problem = Problem("heuristic_order", "启发排序验收",
                      {"arm": ("A",), "loc": (), "slot": (), "block": ()},
                      frozenset(), frozenset(map(parse_fact, ["free(A)", "arm_at(A, home_A)", "node(s)"])),
                      (parse_condition("node(g)"),), library_from_data({"skill": skills}),
                      TestHooks(), CapabilityOracle())
    return Engine(problem), destinations


def graph_bound(destinations, estimates):
    def bound(state, running):
        if running:
            return max(remaining + estimates[destinations[key]] for key, remaining in running)
        node = next(fact.args[0] for fact in state if fact.predicate == "node")
        return estimates[node]
    return bound


class AStarOrderingTests(unittest.TestCase):
    def test_estimate_guides_search_without_an_upper_bound(self):
        edges = [(f"decoy_{i}", "s", f"d{i}", 1.0) for i in range(8)]
        edges.extend([("progress", "s", "m", 2.0), ("finish", "m", "g", 2.0)])
        engine, destinations = graph_engine(edges)
        estimates = {"s": 4.0, "m": 2.0, "g": 0.0, **{f"d{i}": 100.0 for i in range(8)}}
        # Dead ends have infinite true remaining cost, so 100 remains admissible.
        guided = _search(engine, bound=graph_bound(destinations, estimates), max_expansions=5)
        self.assertEqual(guided.status, "solved")
        self.assertEqual(guided.makespan, 4.0)
        self.assertEqual([step.action.name for step in guided.steps], ["progress", "finish"])
        self.assertLessEqual(guided.expanded, 5)

    def test_hint_cannot_override_a_better_f_score(self):
        edges = [(f"decoy_{i}", "s", f"d{i}", 1.0) for i in range(8)]
        edges.extend([("progress", "s", "m", 2.0), ("finish", "m", "g", 2.0)])
        engine, destinations = graph_engine(edges)
        estimates = {"s": 4.0, "m": 2.0, "g": 0.0, **{f"d{i}": 100.0 for i in range(8)}}
        result = _search(engine, bound=graph_bound(destinations, estimates),
                         hint={("decoy_0", ("A",))}, max_expansions=5)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 4.0)

    def test_inconsistent_admissible_heuristic_reopens_a_cheaper_state(self):
        edges = [("direct_a", "s", "a", 2.0), ("via_b", "s", "b", 1.0),
                 ("shortcut", "b", "a", 0.5), ("finish", "a", "g", 2.0)]
        engine, destinations = graph_engine(edges)
        # h(b)=2.5 > cost(b,a)+h(a)=0.5, yet each h is admissible.
        bound = graph_bound(destinations, {"s": 0.0, "a": 0.0, "b": 2.5, "g": 0.0})
        result = _search(engine, bound=bound)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 3.5)
        self.assertEqual([step.action.name for step in result.steps], ["via_b", "shortcut", "finish"])

    def test_custom_cost_is_preserved_under_zero_heuristic(self):
        edges = [("expensive", "s", "g", 1.0),
                 ("cheap_first", "s", "m", 9.0), ("cheap_last", "m", "g", 9.0)]
        engine, _ = graph_engine(edges)

        class ActualCost:
            def estimate(self, action, state):
                return 5.0 if action.name == "expensive" else 0.25

        engine.cost = ActualCost()
        result = _search(engine, bound=lambda state, running: 0.0,
                         hint={("expensive", ("A",))})
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 0.5)
        self.assertEqual([step.action.name for step in result.steps], ["cheap_first", "cheap_last"])
