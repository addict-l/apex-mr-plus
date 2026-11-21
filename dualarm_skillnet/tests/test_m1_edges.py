"""Specification boundary checks for the generic core."""

from copy import deepcopy
import unittest

from skillnet.capability.oracle import CapabilityOracle, Status
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data, load_library
from skillnet.core.logic import Fact, parse_fact
from skillnet.core.problem import Problem


SIMPLE_SKILL = {
    "name": "transport",
    "params": {"arm": "arm", "block": "block", "source": "loc", "target": "loc"},
    "pre": ["free(?arm)", "at(?block, ?source)", "placeable(?target)",
            "not forbidden(?target)"],
    "capability": ["reach(?arm, ?target)"],
    "add": ["at(?block, ?target)", "free(?arm)"],
    "delete": ["at(?block, ?source)", "free(?arm)"],
    "occupies": ["?arm"],
    "move_to": {"?arm": "?target"},
    "op_time": 1.0,
}


class CoreHooks:
    derived_predicates = frozenset({"slot_ready", "insert_supported"})
    derived_basis = {"slot_ready": frozenset({"filled"}),
                     "insert_supported": frozenset({"filled", "supporting"})}
    resource_predicates = frozenset({"free", "arm_at"})

    def evaluate(self, pred, args, state):
        return True

    def invariant(self, state):
        return None


def make_engine(library, static, initial, objects):
    return Engine(Problem("core_edges", "核心边界测试", objects,
                          frozenset(map(parse_fact, static)),
                          frozenset(map(parse_fact, initial)), (), library, CoreHooks(),
                          CapabilityOracle({"reach": {"default": "VALID"}})))


class DefinitionBoundaryTests(unittest.TestCase):
    def test_duplicate_names_and_invalid_identifier(self):
        with self.assertRaisesRegex(ValueError, "transport.*name.*重复"):
            library_from_data({"skill": [deepcopy(SIMPLE_SKILL), deepcopy(SIMPLE_SKILL)]})
        bad = deepcopy(SIMPLE_SKILL)
        bad["name"] = "bad-name"
        with self.assertRaisesRegex(ValueError, "bad-name.*name"):
            library_from_data({"skill": [bad]})

    def test_parameter_type_and_move_mapping(self):
        mutations = [
            ("params", {"arm": "robot"}, "params"),
            ("move_to", {"?block": "?target"}, "move_to"),
            ("move_to", {"?arm": "?block"}, "move_to"),
            ("move_to", {"?arm": "?missing"}, "move_to"),
            ("occupies", [], "occupies"),
        ]
        for field, value, error_field in mutations:
            with self.subTest(field=field, value=value):
                bad = deepcopy(SIMPLE_SKILL)
                bad[field] = value
                with self.assertRaisesRegex(ValueError, f"transport.*{error_field}"):
                    library_from_data({"skill": [bad]})

    def test_operation_time_must_be_positive_finite_number(self):
        for value in (0, -1, float("nan"), float("inf"), True, "2"):
            with self.subTest(value=value):
                bad = deepcopy(SIMPLE_SKILL)
                bad["op_time"] = value
                with self.assertRaisesRegex(ValueError, "transport.*op_time"):
                    library_from_data({"skill": [bad]})

    def test_negative_effects_and_undeclared_primitive_variables(self):
        for field, expression in (("delete", "not free(?arm)"),
                                  ("capability", "not reach(?arm, ?target)"),
                                  ("primitives", "approach(?missing)")):
            with self.subTest(field=field):
                bad = deepcopy(SIMPLE_SKILL)
                bad[field] = [expression]
                with self.assertRaisesRegex(ValueError, f"transport.*{field}"):
                    library_from_data({"skill": [bad]})

    def test_network_without_derived_basis(self):
        library = library_from_data({"skill": [deepcopy(SIMPLE_SKILL)]})
        edges = library.network()
        self.assertTrue(any(edge.source == "transport" and edge.target == "transport"
                            and edge.label == "at" for edge in edges))
        self.assertTrue(library.to_mermaid().startswith("flowchart TD"))

    def test_dynamic_derived_conflict_at_load(self):
        with self.assertRaisesRegex(ValueError, "动态.*推导.*at"):
            library_from_data({"skill": [deepcopy(SIMPLE_SKILL)]},
                              derived_basis={"at": frozenset({"filled"})})


class GroundingBoundaryTests(unittest.TestCase):
    def test_all_same_type_parameters_are_distinct_and_static_filters_apply(self):
        engine = make_engine(
            library_from_data({"skill": [deepcopy(SIMPLE_SKILL)]}),
            ["placeable(src)", "placeable(dst)", "placeable(blocked)", "forbidden(blocked)"],
            ["free(A)", "at(red, src)", "arm_at(A, home_A)"],
            {"arm": ("A",), "block": ("red",),
             "loc": ("src", "dst", "blocked", "unplaceable"), "slot": ()},
        )
        self.assertIn(("transport", ("A", "red", "src", "dst")), engine.actions)
        for action in engine.actions.values():
            self.assertNotEqual(action.args[2], action.args[3])
            self.assertNotIn(action.args[3], {"blocked", "unplaceable"})
        for text in ("missing(A)", "transport(A, red)",
                     "transport(red, A, src, dst)", "transport(A, red, src, absent)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                engine.parse_action(text)

    def test_coinsert_grounding_capabilities_and_two_arm_end_effects(self):
        engine = make_engine(
            load_library(), ["fits(beam, socket)", "arm_pair(A, B)", "two_arm_part(beam)"],
            ["coheld(A, B, beam)", "arm_at(A, home_A)", "arm_at(B, home_B)"],
            {"arm": ("A", "B"), "block": ("beam",), "loc": (), "slot": ("socket",)},
        )
        action = engine.parse_action("co_insert(A, B, beam, socket)")
        self.assertTrue(engine.satisfied(action.pre, engine.initial))
        self.assertEqual(action.occupies, ("A", "B"))
        self.assertEqual(action.move_to, {"A": "socket", "B": "socket"})
        self.assertEqual(engine.oracle.evaluate(action.capability)[0], Status.VALID)
        engine.oracle.update(Fact("reach", ("B", "socket")), Status.INVALID, "测试")
        self.assertEqual(engine.oracle.evaluate(action.capability)[0], Status.INVALID)
        ended = engine.execute(action, engine.initial)
        for text in ("installed(beam, socket)", "filled(socket)", "free(A)", "free(B)",
                     "arm_at(A, socket)", "arm_at(B, socket)"):
            self.assertIn(parse_fact(text), ended)
        self.assertNotIn(parse_fact("coheld(A, B, beam)"), ended)
        self.assertFalse(any("?" in str(item) for item in action.primitives))


if __name__ == "__main__":
    unittest.main()
