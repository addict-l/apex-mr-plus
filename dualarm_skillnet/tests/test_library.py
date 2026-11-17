import copy
import tomllib
import unittest

from skillnet.core.library import library_from_data, load_library
from skillnet.core.logic import Fact, parse_condition, parse_fact


class LogicTests(unittest.TestCase):
    def test_parse_and_substitute(self):
        fact = parse_fact(" held( ?arm, red ) ")
        self.assertEqual(fact.substitute({"?arm": "A"}), Fact("held", ("A", "red")))
        condition = parse_condition("not held(?arm, red)")
        self.assertTrue(condition.negated)
        self.assertEqual(str(condition.substitute({"?arm": "B"})), "not held(B, red)")

    def test_invalid_syntax(self):
        for text in ("not free(A)", "held(A,)", "held(A", "held((A))", "1bad(A)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_fact(text)


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("skills/blocks_v1.toml", "rb") as source:
            cls.data = tomllib.load(source)

    def test_nine_skills_and_grounding(self):
        library = load_library()
        self.assertEqual(len(library.skills), 9)
        action = library.by_name["handover"].ground(("B", "A", "red", "hand_mid"))
        self.assertEqual(action.key, ("handover", ("B", "A", "red", "hand_mid")))
        self.assertEqual(action.occupies, ("B", "A"))
        self.assertEqual(action.move_to, {"A": "hand_mid", "B": "hand_mid"})
        self.assertEqual(str(action.primitives[2]), "grasp(A, red)")
        self.assertEqual(hash(action), hash(action.key))

    def test_network_is_derived_from_predicates(self):
        basis = {"slot_ready": frozenset({"filled"}),
                 "insert_supported": frozenset({"supporting", "filled"})}
        library = load_library(derived_basis=basis)
        edges = library.network()
        for pair in (("pick", "place"), ("pick", "insert"), ("insert", "insert"), ("support_on", "insert")):
            self.assertTrue(any((edge.source, edge.target) == pair for edge in edges), pair)
        self.assertTrue(any(edge.source == edge.target == "insert" and "slot_ready" in edge.label for edge in edges))
        self.assertTrue(library.to_mermaid().startswith("flowchart TD\n"))

    def test_reject_undeclared_variable(self):
        data = copy.deepcopy(self.data)
        data["skill"][0]["pre"].append("free(?ghost)")
        with self.assertRaisesRegex(ValueError, r"pick.*pre.*ghost"):
            library_from_data(data)

    def test_not_is_only_allowed_in_pre(self):
        data = copy.deepcopy(self.data)
        data["skill"][0]["add"].append("not free(?arm)")
        with self.assertRaisesRegex(ValueError, r"pick.*add.*not"):
            library_from_data(data)

    def test_occupies_requires_arm(self):
        data = copy.deepcopy(self.data)
        data["skill"][0]["occupies"] = ["?block"]
        with self.assertRaisesRegex(ValueError, r"pick.*occupies.*block"):
            library_from_data(data)

    def test_dynamic_and_derived_cannot_overlap(self):
        with self.assertRaisesRegex(ValueError, "动态条件与推导条件.*held"):
            load_library(derived_basis={"held": frozenset()})


if __name__ == "__main__":
    unittest.main()
