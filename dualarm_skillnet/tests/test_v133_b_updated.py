"""Updated B policy: no implicit reach claims and public qualification APIs."""
from copy import deepcopy
import unittest

from gen.generator import generate, task_to_toml
from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.domains.blocks import problem_from_data


class UpdatedCapabilityPolicyTests(unittest.TestCase):
    def test_default_generator_has_no_reach_or_source_claim_and_loads_as_unknown(self):
        result = generate("F2", 20261115, 0)
        data = result["task"]
        self.assertNotIn("reach", data["capabilities"])
        self.assertNotIn("capability_source", data)
        self.assertIn("liftable", data["capabilities"])
        self.assertNotIn("capabilities.reach", task_to_toml(data))
        problem = problem_from_data(data)
        for arm in data["arms"]:
            for place in (*problem.hooks.locations, *problem.hooks.slots):
                self.assertEqual(problem.oracle.check(Fact("reach", (arm, place))).status, Status.UNKNOWN)

    def test_explicit_distance_prior_is_pure_preserves_layout_and_marks_source(self):
        from gen.geometry import distance_prior_copy
        data = generate("F1", 11)["task"]
        data["metadata"] = {"preserve": ["额外字段", 17]}
        original = deepcopy(data)
        prior = distance_prior_copy(data)
        self.assertEqual(data, original)
        self.assertEqual(prior["capability_source"], "distance_prior")
        self.assertIn("reach", prior["capabilities"])
        actual = deepcopy(prior)
        actual.pop("capability_source")
        actual["capabilities"].pop("reach")
        expected = deepcopy(original)
        expected["capabilities"].pop("reach", None)
        self.assertEqual(actual, expected)

    def test_generator_prior_requires_explicit_switch(self):
        data = generate("F1", 11, distance_prior=True)["task"]
        self.assertEqual(data["capability_source"], "distance_prior")
        self.assertIn("reach", data["capabilities"])
        self.assertIn('capability_source = "distance_prior"', task_to_toml(data))

    def test_missing_entire_capabilities_is_supported_without_changing_loader(self):
        data = generate("F1", 11)["task"]
        data.pop("capabilities")
        problem = problem_from_data(data)
        self.assertEqual(problem.oracle.check(Fact("reach", ("A", "board_1"))).status, Status.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
