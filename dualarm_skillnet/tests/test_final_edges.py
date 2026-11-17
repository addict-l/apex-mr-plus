"""Independent checks of support transitions and their causal evidence."""

import unittest

from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.domains.blocks import problem_from_data
from skillnet.planning.search import SearchStep, search
from skillnet.planning.taskgraph import build_task_graph


def support_scene():
    return {
        "name": "support_edges", "arms": ["A", "B"], "fill_all_slots": False,
        "arm": [{"id": "A", "home": [-0.3, 0.3, 0.2], "speed": 0.25},
                {"id": "B", "home": [0.3, 0.3, 0.2], "speed": 0.25}],
        "location": [{"id": name, "kind": "stock", "pos": [0, 0, 0]}
                     for name in ("stock_beam", "stock_cw", "stock_cap")],
        "block": [{"id": "pillar", "type": "pillar", "installed_in": "p1"},
                  {"id": "b_beam", "type": "beam", "at": "stock_beam"},
                  {"id": "b_cw", "type": "cw", "at": "stock_cw"},
                  {"id": "b_cap", "type": "cap", "at": "stock_cap"},
                  {"id": "tmp1", "type": "temporary", "installed_in": "tp1"},
                  {"id": "tmp2", "type": "temporary", "installed_in": "tp2"}],
        "slot": [{"id": "p1", "type": "pillar", "pos": [0, 0.1, 0.03]},
                 {"id": "beam", "type": "beam", "pos": [0, 0.1, 0.06],
                  "below": ["p1"], "overhang": True, "anchors": ["cw"], "support_point": True},
                 {"id": "cw", "type": "cw", "pos": [0, 0.1, 0.09], "below": ["beam"]},
                 {"id": "cap", "type": "cap", "pos": [0, 0.1, 0.12], "below": ["beam"],
                  "needs_support_during": ["beam"]},
                 {"id": "tp1", "type": "temporary", "pos": [0, 0, 0.03],
                  "temporary": True, "platform_for": "beam"},
                 {"id": "tp2", "type": "temporary", "pos": [0, 0, 0.06],
                  "temporary": True, "platform_for": "beam", "below": ["tp1"]}],
        "capabilities": {"reach": {"default": "VALID"}, "liftable": {"default": "VALID"}},
    }


def state(*facts):
    return frozenset(map(parse_fact, ["arm_at(A, home_A)", "arm_at(B, home_B)",
                                     "filled(p1)", "installed(pillar, p1)", *facts]))


class SupportTransitionTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(problem_from_data(support_scene()))

    def search_one(self, text, initial, goal):
        action = self.engine.parse_action(text)
        return search(self.engine, start=initial, goal=[parse_condition(goal)], allowed={action.key})

    def test_unsupported_beam_is_rejected_at_finish(self):
        initial = state("free(A)", "held(B, b_beam)")
        action = self.engine.parse_action("insert(B, b_beam, beam)")
        self.assertTrue(self.engine.satisfied(action.pre, initial))
        self.assertIsNone(self.engine.hooks.invariant(self.engine.start(action, initial)))
        ended = self.engine.finish(action, self.engine.start(action, initial))
        self.assertIn("beam", self.engine.hooks.invariant(ended))
        result = self.search_one(str(action), initial, "filled(beam)")
        self.assertEqual(result.status, "no_plan")
        self.assertGreater(result.rejected["invariant"], 0)
        supported = state("supporting(A, beam)", "held(B, b_beam)")
        self.assertEqual(self.search_one(str(action), supported, "filled(beam)").status, "solved")

    def test_support_release_requires_anchors_or_full_platform(self):
        initial = state("supporting(A, beam)", "free(B)", "filled(beam)", "installed(b_beam, beam)")
        text = "support_off(A, beam)"
        self.assertEqual(self.search_one(text, initial, "free(A)").status, "no_plan")
        anchored = initial | {parse_fact("filled(cw)"), parse_fact("installed(b_cw, cw)")}
        self.assertEqual(self.search_one(text, anchored, "free(A)").status, "solved")
        partial = initial | {parse_fact("filled(tp1)")}
        self.assertEqual(self.search_one(text, partial, "free(A)").status, "no_plan")
        full = partial | {parse_fact("filled(tp2)")}
        self.assertEqual(self.search_one(text, full, "free(A)").status, "solved")

    def test_detaching_platform_part_and_anchor_checks_start_stability(self):
        platform = state("free(A)", "free(B)", "filled(beam)", "installed(b_beam, beam)",
                         "filled(tp1)", "installed(tmp1, tp1)", "filled(tp2)", "installed(tmp2, tp2)")
        result = self.search_one("detach(B, tmp2, tp2)", platform, "not filled(tp2)")
        self.assertEqual(result.status, "no_plan")
        self.assertGreater(result.rejected["invariant"], 0)
        supported = (platform - {parse_fact("free(A)")}) | {parse_fact("supporting(A, beam)")}
        self.assertEqual(self.search_one("detach(B, tmp2, tp2)", supported, "not filled(tp2)").status, "solved")
        anchored = state("free(A)", "free(B)", "filled(beam)", "installed(b_beam, beam)",
                         "filled(cw)", "installed(b_cw, cw)")
        result = self.search_one("detach(B, b_cw, cw)", anchored, "not filled(cw)")
        self.assertEqual(result.status, "no_plan")
        self.assertGreater(result.rejected["invariant"], 0)

    def test_anchor_stability_does_not_replace_explicit_support_during_insertion(self):
        initial = state("free(A)", "held(B, b_cap)", "filled(beam)", "installed(b_beam, beam)",
                        "filled(cw)", "installed(b_cw, cw)")
        self.assertIsNone(self.engine.hooks.invariant(initial))
        action = self.engine.parse_action("insert(B, b_cap, cap)")
        self.assertIn(parse_condition("insert_supported(cap)"), self.engine.failed(action, initial))
        self.assertEqual(self.search_one(str(action), initial, "filled(cap)").status, "no_plan")
        supported = (initial - {parse_fact("free(A)")}) | {parse_fact("supporting(A, beam)")}
        self.assertEqual(self.search_one(str(action), supported, "filled(cap)").status, "solved")

    def test_support_during_has_causal_evidence_in_graph(self):
        initial = state("free(A)", "held(B, b_cap)", "filled(beam)", "installed(b_beam, beam)",
                        "filled(cw)", "installed(b_cw, cw)")
        self.engine.initial = initial
        self.engine.goal = (parse_condition("filled(cap)"),)
        support = self.engine.parse_action("support_on(A, beam)")
        after_support = self.engine.execute(support, initial)
        insert = self.engine.parse_action("insert(B, b_cap, cap)")
        steps = [SearchStep(support, 0.0, 2.0, initial), SearchStep(insert, 2.0, 5.0, after_support)]
        graph = build_task_graph(self.engine, steps)
        self.assertTrue(any(edge.source == 1 and edge.target == 2 and edge.kind == "causal"
                            and edge.label == "supporting(A, beam)" for edge in graph.edges))
        self.assertIn("达成目标 filled(cap)", graph.nodes[1].purposes)


class FinalLoadingTests(unittest.TestCase):
    def test_home_id_collision_and_all_slot_reference_fields(self):
        data = support_scene()
        data["location"][0]["id"] = "home_A"
        with self.assertRaisesRegex(ValueError, "home_A.*冲突"):
            problem_from_data(data)
        for index, field, value in ((1, "anchors", ["missing"]),
                                    (3, "needs_support_during", ["missing"]),
                                    (4, "platform_for", "missing")):
            with self.subTest(field=field):
                data = support_scene()
                data["slot"][index][field] = value
                with self.assertRaisesRegex(ValueError, f"{field}.*不存在的槽位 missing"):
                    problem_from_data(data)

    def test_duplicate_installed_slot_and_initial_instability(self):
        data = support_scene()
        data["block"].append({"id": "extra", "type": "pillar", "installed_in": "p1"})
        with self.assertRaisesRegex(ValueError, "p1.*重复安装"):
            problem_from_data(data)
        data = support_scene()
        data["block"][1] = {"id": "b_beam", "type": "beam", "installed_in": "beam"}
        data["block"] = data["block"][:4]
        with self.assertRaisesRegex(ValueError, "初始状态.*beam.*不稳定"):
            problem_from_data(data)


class FinalSearchConclusionTests(unittest.TestCase):
    def test_two_positive_subnanosecond_actions_keep_their_total_time(self):
        problem = problem_from_data(support_scene())
        problem.library = library_from_data({"skill": [
            {"name": "first", "params": {"arm": "arm"},
             "pre": ["free(?arm)", "not ready(?arm)"], "delete": ["free(?arm)"],
             "add": ["ready(?arm)", "free(?arm)"], "occupies": ["?arm"], "op_time": 1e-12},
            {"name": "second", "params": {"arm": "arm"},
             "pre": ["free(?arm)", "ready(?arm)", "not done(?arm)"],
             "delete": ["free(?arm)", "ready(?arm)"],
             "add": ["done(?arm)", "free(?arm)"], "occupies": ["?arm"], "op_time": 1e-12},
        ]})
        problem.goal = (parse_condition("done(A)"),)
        engine = Engine(problem)
        result = search(engine, allowed={("first", ("A",)), ("second", ("A",))})
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2e-12)
        self.assertEqual([(step.start, step.end) for step in result.steps],
                         [(0.0, 1e-12), (1e-12, 2e-12)])

    def test_invalid_capability_dominates_unknown_for_no_plan_conclusion(self):
        data = support_scene()
        data["capabilities"] = {"reach": {"default": "INVALID"},
                                "liftable": {"default": "UNKNOWN"}}
        engine = Engine(problem_from_data(data))
        result = search(engine)
        self.assertGreater(result.rejected["capability_invalid"], 0)
        self.assertEqual(result.rejected["capability_unknown"], 0)
        self.assertEqual(result.status, "no_plan")


if __name__ == "__main__":
    unittest.main()
