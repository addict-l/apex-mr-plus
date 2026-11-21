"""Behavioral tests for the production opt-in TPG collision oracle."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest

from apexmr.geometry import BroadphaseConflictOracle, SampledConflictOracle


def q(first_joint):
    return [first_joint, 0., 0., 0., 0., 0.]


def node(arm, suffix, values, *, metadata=None, payload=None, kind="transit"):
    return {
        "id": f"{arm}-{suffix}", "arm": arm, "name": suffix, "kind": kind,
        "node_type": "pose" if kind == "transit" else "skill",
        "q_samples": [q(value) for value in values], "metadata": metadata or {},
        "payload": payload,
    }


def binding(name, x=0.):
    return {"name": name, "offset": [x, 0., 0.], "quaternion": [0., 0., 0., 1.]}


class SyntheticFullScene:
    """Offers the production pair-sampling boundary with a deterministic mesh predicate."""

    def __init__(self, collision=None):
        self.batches = []
        self.collision = collision or (lambda sample: None)

    @property
    def samples(self):
        return [sample for batch in self.batches for sample in batch]

    def sample_state_pairs(self, samples):
        self.batches.append(deepcopy(samples))
        results = []
        for sample in samples:
            problem = self.collision(sample)
            results.append({"ok": problem is None, "problem": problem})
        return results

    def validate_joint(self, *args, **kwargs):
        raise AssertionError("The broadphase oracle must use full-scene joint pair sampling")


class SyntheticBounds:
    def __init__(self, right_min=2., reject=False):
        self.right_min = right_min
        self.reject = reject
        self.nodes = []

    def bounds_for_node(self, item):
        self.nodes.append(item["id"])
        if self.reject and item["arm"] == "B":
            raise RuntimeError("Observed corner exceeds assumed revolute-joint point radius")
        if item["arm"] == "A":
            return {"ok": True, "min": [-1., -1., -1.], "max": [1., 1., 1.]}
        return {"ok": True, "min": [self.right_min, -1., -1.], "max": [3., 1., 1.]}


class BroadphaseOracleTests(unittest.TestCase):
    LEFT = (0., .04, .08)
    RIGHT = (.01, .05)
    PLACEMENT = [1., 2., 3., 0., 0., 0., 1.]
    COLLISION = {"reason": "block_robot_collision:beam:A", "pair": ["/A/link", "/beam"]}

    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.left = node("A", "one", self.LEFT)
        self.right = node("B", "one", self.RIGHT)

    def oracle(self, planner, bounds, name="broadphase"):
        return BroadphaseConflictOracle(planner, {}, self.directory / name, bounds_sampler=bounds)

    @staticmethod
    def coordinate_pairs(planner):
        return [(sample["A"][0], sample["B"][0]) for sample in planner.samples]

    def test_disjoint_cold_cross_then_exact_positive_reuse(self):
        planner, bounds = SyntheticFullScene(), SyntheticBounds()
        oracle = self.oracle(planner, bounds)
        self.assertFalse(oracle(self.left, self.right)["conflict"])
        self.assertEqual(len(planner.batches), 1)
        self.assertCountEqual(self.coordinate_pairs(planner),
                              [(0., .01), (.04, .01), (.08, .01), (0., .05)])

        self.assertFalse(oracle(deepcopy(self.left), deepcopy(self.right))["conflict"])
        self.assertCountEqual(bounds.nodes, [self.left["id"], self.right["id"]])
        revised = node("B", "same-states", self.RIGHT)
        self.assertFalse(oracle(self.left, revised)["conflict"])
        self.assertEqual(len(planner.batches), 1)

    def test_concurrent_placement_is_checked_even_when_arms_are_disjoint(self):
        def collision(sample):
            if sample["placed"].get("beam") == self.PLACEMENT and sample["A"] == q(.04):
                return self.COLLISION
            return None

        planner = SyntheticFullScene(collision)
        oracle = self.oracle(planner, SyntheticBounds())
        self.assertFalse(oracle(self.left, self.right)["conflict"])
        placed_right = node("B", "after-place", self.RIGHT,
                            metadata={"placed": {"beam": self.PLACEMENT}})
        planner.batches.clear()
        result = oracle(self.left, placed_right)
        baseline = SampledConflictOracle(SyntheticFullScene(collision), {}, self.directory / "baseline")
        self.assertTrue(baseline(self.left, placed_right)["conflict"])
        self.assertTrue(result["conflict"])
        self.assertEqual(result["evidence"]["problem"], self.COLLISION)
        self.assertEqual(result["evidence"]["qA"], q(.04))
        self.assertIn(result["evidence"]["qB"], [q(value) for value in self.RIGHT])
        self.assertTrue(all(sample["placed"]["beam"] == self.PLACEMENT for sample in planner.samples))
        self.assertLessEqual(len(planner.samples), len(self.LEFT) + len(self.RIGHT) - 1)

    def test_a_free_part_removed_by_other_payload_name_keeps_separate_positive_entries(self):
        # The ambient beam is hazardous only when B is not holding it.
        def collision(sample):
            carrying = sample["payloads"]["B"]["name"]
            return self.COLLISION if carrying != "beam" and sample["A"] == q(.04) else None

        planner = SyntheticFullScene(collision)
        oracle = self.oracle(planner, SyntheticBounds())
        carrying_beam = node("B", "holding-beam", self.RIGHT, payload=binding("beam"))
        carrying_other = node("B", "holding-other", self.RIGHT, payload=binding("other"))
        self.assertFalse(oracle(self.left, carrying_beam)["conflict"])
        planner.batches.clear()
        result = oracle(self.left, carrying_other)
        self.assertTrue(result["conflict"])
        self.assertEqual(result["evidence"]["problem"], self.COLLISION)
        self.assertEqual(result["evidence"]["qA"], q(.04))
        self.assertTrue(all(sample["placed"] == {} for sample in planner.samples))

    def test_component_cache_can_use_one_anchor_when_full_context_is_new(self):
        planner = SyntheticFullScene()
        oracle = self.oracle(planner, SyntheticBounds())
        a1 = node("A", "offset-1", self.LEFT, payload=binding("left", -.1))
        a2 = node("A", "offset-2", self.LEFT, payload=binding("left", -.2))
        b1 = node("B", "offset-1", self.RIGHT, payload=binding("right", .1))
        b2 = node("B", "offset-2", self.RIGHT, payload=binding("right", .2))
        self.assertFalse(oracle(a1, b1)["conflict"])
        self.assertFalse(oracle(a2, b2)["conflict"])
        planner.batches.clear()
        self.assertFalse(oracle(a1, b2)["conflict"])
        self.assertEqual(self.coordinate_pairs(planner), [(self.LEFT[0], self.RIGHT[0])])

    def test_touching_overlapping_and_refused_bounds_use_complete_grid(self):
        for case, bounds in (("touch", SyntheticBounds(right_min=1.)),
                             ("overlap", SyntheticBounds(right_min=.5)),
                             ("radius-refused", SyntheticBounds(reject=True))):
            with self.subTest(case=case):
                planner = SyntheticFullScene()
                oracle = self.oracle(planner, bounds, case)
                self.assertFalse(oracle(self.left, self.right)["conflict"])
                self.assertEqual(self.coordinate_pairs(planner), list(product(self.LEFT, self.RIGHT)))
                if case == "radius-refused":
                    planner.batches.clear()
                    self.assertFalse(oracle(self.left, self.right)["conflict"])
                    self.assertEqual(bounds.nodes.count(self.right["id"]), 1)

    def test_closed_chain_and_nonordinary_node_cannot_use_cross_reduction(self):
        paired = node("B", "closed-chain", self.RIGHT,
                      payload=binding("beam"),
                      metadata={"options": {"shared_payload": "beam"}})
        skill = node("B", "gripper", self.RIGHT, kind="grip")
        for special in (paired, skill):
            with self.subTest(node=special["id"]):
                planner = SyntheticFullScene()
                oracle = self.oracle(planner, SyntheticBounds(), special["id"])
                self.assertFalse(oracle(self.left, special)["conflict"])
                self.assertEqual(self.coordinate_pairs(planner), list(product(self.LEFT, self.RIGHT)))

    def test_incompatible_modes_keep_original_conflict_without_mesh_queries(self):
        pairs = [(node("A", "one", self.LEFT, payload=binding("beam")),
                  node("B", "two", self.RIGHT, payload=binding("beam"))),
                 (node("A", "one", self.LEFT, metadata={"placed": {"part": [1., 1., 1.]}}),
                  node("B", "two", self.RIGHT, metadata={"placed": {"part": [2., 2., 2.]}}))]
        for index, (left, right) in enumerate(pairs):
            with self.subTest(index=index):
                planner, bounds = SyntheticFullScene(), SyntheticBounds()
                oracle = self.oracle(planner, bounds, str(index))
                result = oracle(left, right)
                self.assertTrue(result["conflict"])
                self.assertEqual(result["evidence"]["problem"]["reason"], "inconsistent_mode")
                self.assertEqual(planner.batches, [])
                self.assertEqual(bounds.nodes, [])


    def test_full_grid_fallback_never_consults_the_planners_bounds(self):
        class LegacyBoundsPlanner(SyntheticFullScene):
            def bounds_for_node(self, item):
                raise AssertionError("Only the explicit sampler can supply an AABB")

        def collision(sample):
            return self.COLLISION if sample["A"] == q(.04) and sample["B"] == q(.05) else None

        planner = LegacyBoundsPlanner(collision)
        overlapping = SyntheticBounds(right_min=.5)
        result = self.oracle(planner, overlapping)(self.left, self.right)
        self.assertTrue(result["conflict"])
        self.assertEqual(result["evidence"]["problem"], self.COLLISION)
        self.assertEqual(self.coordinate_pairs(planner), list(product(self.LEFT, self.RIGHT)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
