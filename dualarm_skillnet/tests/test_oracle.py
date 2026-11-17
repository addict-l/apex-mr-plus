import unittest

from skillnet.capability.oracle import CapabilityOracle, Status
from skillnet.core.logic import Fact


class CapabilityTests(unittest.TestCase):
    def test_default_and_both_explicit_formats(self):
        oracle = CapabilityOracle({
            "reach": {
                "default": "INVALID",
                "valid": {"A": ["left", "middle"]},
                "unknown": [["B", "middle"]],
            },
            "liftable": {"default": "VALID"},
        })
        self.assertEqual(oracle.check(Fact("reach", ("A", "left"))).status, Status.VALID)
        self.assertEqual(oracle.check(Fact("reach", ("A", "right"))).status, Status.INVALID)
        self.assertEqual(oracle.check(Fact("reach", ("B", "middle"))).status, Status.UNKNOWN)
        self.assertEqual(oracle.check(Fact("liftable", ("B", "red"))).status, Status.VALID)
        self.assertIn("reach.valid", oracle.check(Fact("reach", ("A", "left"))).source)

    def test_missing_predicate_and_default_are_unknown(self):
        oracle = CapabilityOracle({"reach": {}})
        self.assertEqual(oracle.check(Fact("reach", ("A", "left"))).status, Status.UNKNOWN)
        self.assertEqual(oracle.check(Fact("undeclared", ("A",))).status, Status.UNKNOWN)

    def test_invalid_dominates_unknown_and_all_evidence_is_returned(self):
        oracle = CapabilityOracle({"reach": {"default": "INVALID"}})
        unknown = Fact("liftable", ("A", "red"))
        invalid = Fact("reach", ("A", "left"))
        status, evidence = oracle.evaluate([unknown, invalid])
        self.assertEqual(status, Status.INVALID)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].status, Status.UNKNOWN)
        self.assertEqual(oracle.evaluate([unknown])[0], Status.UNKNOWN)
        self.assertEqual(oracle.evaluate([]), (Status.VALID, ()))

    def test_feedback_update_overrides_defaults_and_retains_source(self):
        oracle = CapabilityOracle({"reach": {"default": "INVALID"}})
        fact = Fact("reach", ("A", "left"))
        oracle.update(fact, "VALID", "运动后端按需验证")
        self.assertEqual(oracle.check(fact).status, Status.VALID)
        self.assertEqual(oracle.check(fact).source, "运动后端按需验证")
        oracle.update(fact, Status.UNKNOWN, "重新验证")
        self.assertEqual(oracle.check(fact).status, Status.UNKNOWN)

    def test_conflicting_explicit_statuses_fail(self):
        with self.assertRaisesRegex(ValueError, r"reach.*reach\(A, left\).*VALID.*UNKNOWN"):
            CapabilityOracle({"reach": {"valid": {"A": ["left"]}, "unknown": [["A", "left"]]}})

    def test_invalid_table_format_and_status_fail(self):
        for table in ({"default": "YES"}, {"valid": "A"}, {"valid": [["A", 1]]}, {"valid": {"A": "left"}}):
            with self.subTest(table=table), self.assertRaisesRegex(ValueError, "reach"):
                CapabilityOracle({"reach": table})


if __name__ == "__main__":
    unittest.main()
