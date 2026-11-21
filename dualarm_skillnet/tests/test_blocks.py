import copy
import unittest

from skillnet.core.logic import Fact, parse_condition, parse_fact
from skillnet.domains.blocks import problem_from_data


def scene():
    return {
        "name": "fixture", "arms": ["A", "B"],
        "arm": [{"id": "A", "home": [-0.3, 0.3, 0.2], "speed": 0.25},
                {"id": "B", "home": [0.3, 0.3, 0.2], "speed": 0.25}],
        "location": [{"id": "stock", "kind": "stock", "pos": [0, 0, 0]},
                     {"id": "buffer", "kind": "buffer", "pos": [0, -0.3, 0]},
                     {"id": "hand", "kind": "handover", "pos": [0, 0.1, 0.2]}],
        "block": [{"id": "red", "type": "cube", "at": "stock"}],
        "slot": [{"id": "p1", "type": "cube", "pos": [0, 0.1, 0.03]},
                 {"id": "beam", "type": "beam", "pos": [0, 0.1, 0.06], "below": ["p1"],
                  "overhang": True, "anchors": ["cw"], "support_point": True},
                 {"id": "cw", "type": "cube", "pos": [0, 0.1, 0.09], "below": ["beam"]},
                 {"id": "tp1", "type": "cube", "pos": [0, 0, 0.03],
                  "temporary": True, "platform_for": "beam"},
                 {"id": "tp2", "type": "cube", "pos": [0, 0, 0.06],
                  "temporary": True, "platform_for": "beam", "below": ["tp1"]}],
    }


class BlockLoadingTests(unittest.TestCase):
    def test_generated_facts_and_goals(self):
        problem = problem_from_data(scene())
        self.assertIn(parse_fact("arm_pair(A, B)"), problem.static)
        self.assertIn(parse_fact("below(p1, beam)"), problem.static)
        self.assertIn(parse_fact("fits(red, cw)"), problem.static)
        self.assertIn(parse_fact("platform_part(tp1, beam)"), problem.static)
        self.assertIn(parse_fact("empty(buffer)"), problem.initial)
        self.assertNotIn(parse_fact("empty(hand)"), problem.initial)
        self.assertIn(parse_condition("not filled(tp1)"), problem.goal)
        self.assertIn(parse_condition("filled(beam)"), problem.goal)
        self.assertEqual(problem.hooks.position("home_A"), (-0.3, 0.3, 0.2))

    def test_installed_and_fill_all_false(self):
        data = scene()
        data["block"][0] = {"id": "red", "type": "cube", "installed_in": "p1"}
        data["fill_all_slots"] = False
        problem = problem_from_data(data)
        self.assertIn(parse_fact("installed(red, p1)"), problem.initial)
        self.assertIn(parse_fact("filled(p1)"), problem.initial)
        self.assertNotIn(parse_condition("filled(beam)"), problem.goal)
        self.assertIn(parse_condition("not filled(tp2)"), problem.goal)

    def test_invalid_scenes(self):
        cases = []
        data = scene()
        data["block"].append({"id": "blue", "type": "cube", "at": "stock"})
        cases.append((data, "stock.*重复"))
        data = scene()
        data["slot"][1]["below"] = ["missing"]
        cases.append((data, "不存在的槽位 missing"))
        data = scene()
        data["slot"][1]["anchors"] = []
        cases.append((data, "beam.*anchors"))
        data = scene()
        data["slot"][0]["below"] = ["cw"]
        cases.append((data, "below.*成环"))
        data = scene()
        data["location"][0]["id"] = "A"
        cases.append((data, "id 全局重复.*A"))
        data = scene()
        data["block"][0] = {"id": "red", "type": "cube", "installed_in": "beam"}
        cases.append((data, "类型不匹配"))
        data = scene()
        data["slot"][0]["platform_for"] = "beam"
        cases.append((data, "platform_for.*临时"))
        data = scene()
        data["arm"][0]["speed"] = 0
        cases.append((data, "speed.*大于 0"))
        data = scene()
        data["location"][0]["pos"] = [0, float("inf"), 0]
        cases.append((data, "有限数值"))
        for data, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                problem_from_data(data)


class BlockHooksTests(unittest.TestCase):
    def setUp(self):
        self.problem = problem_from_data(scene())
        self.hooks = self.problem.hooks

    def test_ready_and_clear_use_direct_connections(self):
        state = frozenset({parse_fact("filled(p1)"), parse_fact("filled(beam)")})
        self.assertTrue(self.hooks.evaluate("slot_ready", ("beam",), state))
        self.assertFalse(self.hooks.evaluate("slot_clear", ("p1",), state))
        self.assertTrue(self.hooks.evaluate("slot_clear", ("beam",), state))
        self.assertEqual(self.hooks.justify(parse_fact("slot_ready(beam)"), state), [parse_fact("filled(p1)")])

    def test_support_needs_all_platform_parts_or_one_arm(self):
        self.assertFalse(self.hooks.evaluate("has_support", ("p1",), frozenset()))
        partial = frozenset({parse_fact("filled(tp1)")})
        self.assertFalse(self.hooks.evaluate("has_support", ("beam",), partial))
        complete = partial | {parse_fact("filled(tp2)")}
        self.assertTrue(self.hooks.evaluate("has_support", ("beam",), complete))
        arm = frozenset({parse_fact("supporting(A, beam)")})
        self.assertTrue(self.hooks.evaluate("has_support", ("beam",), arm))
        self.assertEqual(self.hooks.justify(parse_fact("has_support(beam)"), arm), [parse_fact("supporting(A, beam)")])

    def test_invariant_and_support_during(self):
        state = frozenset({parse_fact("filled(beam)")})
        self.assertIn("beam", self.hooks.invariant(state))
        self.assertIsNone(self.hooks.invariant(state | {parse_fact("filled(cw)")}))
        self.assertIsNone(self.hooks.invariant(state | {parse_fact("supporting(B, beam)")}))
        self.hooks.slots["cw"]["needs_support_during"] = ["beam"]
        self.assertFalse(self.hooks.evaluate("insert_supported", ("cw",), state))
        self.assertTrue(self.hooks.evaluate("insert_supported", ("cw",), state | {parse_fact("supporting(A, beam)")}))
        expanded = self.hooks.expand(parse_fact("insert_supported(cw)"))
        self.assertIn(parse_fact("filled(tp2)"), expanded)
        self.assertIn(parse_fact("supporting(B, beam)"), expanded)

    def test_invariant_needs_includes_both_alternatives(self):
        action = self.problem.library.by_name["insert"].ground(("A", "red", "beam"))
        self.assertEqual(self.hooks.invariant_needs(action),
                         [parse_fact("has_support(beam)"), parse_fact("filled(cw)")])


if __name__ == "__main__":
    unittest.main()
