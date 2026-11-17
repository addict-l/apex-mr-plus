"""Task graph provenance and serialization acceptance tests."""

import json
import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import parse_condition
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import SearchStep, search
from skillnet.planning.taskgraph import build_task_graph


def sequential_steps(engine, texts):
    state = engine.initial
    steps = []
    for index, text in enumerate(texts):
        action = engine.parse_action(text)
        steps.append(SearchStep(action, float(index), float(index + 1), state))
        state = engine.execute(action, state)
    return steps


class TaskGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tower_engine = Engine(load_problem("tasks/s2_tower.toml"))
        cls.tower_result = search(cls.tower_engine)
        cls.tower_graph = build_task_graph(cls.tower_engine, cls.tower_result.steps)

    def test_tower_has_derived_filled_dependencies(self):
        self.assertEqual(self.tower_result.status, "solved")
        inserts = {step.action.args[-1]: index for index, step in enumerate(self.tower_result.steps, 1)
                   if step.action.name == "insert"}
        for lower, upper in (("t1", "t2"), ("t2", "t3")):
            self.assertTrue(any(edge.source == inserts[lower] and edge.target == inserts[upper]
                                and edge.kind == "causal" and edge.label == f"filled({lower})"
                                for edge in self.tower_graph.edges))

    def test_resource_facts_do_not_create_causal_edges(self):
        for edge in self.tower_graph.edges:
            if edge.kind == "causal":
                self.assertFalse(edge.label.startswith("free("))
                self.assertFalse(edge.label.startswith("arm_at("))
                self.assertFalse(edge.label.startswith("fits("))

    def test_same_arm_adjacency_and_causal_deduplication(self):
        engine = self.tower_engine
        steps = sequential_steps(engine, [
            "pick(A, b1, stock_l1)", "insert(A, b1, t1)",
            "pick(A, b2, stock_l2)", "insert(A, b2, t2)",
        ])
        graph = build_task_graph(engine, steps)
        self.assertTrue(any((edge.source, edge.target, edge.kind) == (2, 3, "arm") for edge in graph.edges))
        self.assertFalse(any((edge.source, edge.target, edge.kind) == (1, 2, "arm") for edge in graph.edges))
        self.assertTrue(any((edge.source, edge.target, edge.kind) == (1, 2, "causal") for edge in graph.edges))

    def test_latest_readdition_provides_fact_after_deletion(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        steps = sequential_steps(engine, [
            "pick(A, red, stock_c)", "place(A, red, board_c)",
            "pick(A, red, board_c)", "place(A, red, stock_c)",
        ])
        graph = build_task_graph(engine, steps)
        held_edges = [edge for edge in graph.edges if edge.label == "held(A, red)"]
        self.assertEqual([(edge.source, edge.target) for edge in held_edges], [(1, 2), (3, 4)])
        self.assertNotIn((1, 4), [(edge.source, edge.target) for edge in held_edges])

    def test_goals_are_credited_only_to_final_provider(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        steps = sequential_steps(engine, [
            "pick(A, red, stock_c)", "place(A, red, board_c)",
            "pick(A, red, board_c)", "place(A, red, stock_c)",
            "pick(A, red, stock_c)", "place(A, red, board_c)",
        ])
        graph = build_task_graph(engine, steps)
        self.assertNotIn("达成目标 at(red, board_c)", graph.nodes[1].purposes)
        self.assertIn("达成目标 at(red, board_c)", graph.nodes[5].purposes)
        engine.problem.goal = (parse_condition("not at(red, stock_c)"),)
        engine.goal = engine.problem.goal
        graph = build_task_graph(engine, steps)
        self.assertIn("达成目标 not at(red, stock_c)", graph.nodes[4].purposes)

    def test_graph_serialization_and_node_labels(self):
        graph = self.tower_graph
        encoded = json.dumps(graph.to_json(), ensure_ascii=False)
        self.assertEqual(len(json.loads(encoded)["nodes"]), len(self.tower_result.steps))
        mermaid = graph.to_mermaid()
        self.assertTrue(mermaid.startswith("flowchart TD"))
        self.assertIn("插装", mermaid)
        self.assertIn("臂 A", mermaid)
        self.assertIn("filled(t1)", mermaid)
        self.assertTrue(any(node.purposes for node in graph.nodes))

    def test_graph_sorts_steps_and_accepts_empty_schedule(self):
        graph = build_task_graph(self.tower_engine, list(reversed(self.tower_result.steps)))
        self.assertEqual([node.start for node in graph.nodes], sorted(node.start for node in graph.nodes))
        empty = build_task_graph(self.tower_engine, [])
        self.assertEqual(empty.to_json(), {"nodes": [], "edges": []})
        self.assertEqual(empty.to_mermaid(), "flowchart TD\n")


if __name__ == "__main__":
    unittest.main()
