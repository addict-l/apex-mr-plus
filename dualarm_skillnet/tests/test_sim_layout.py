"""Offline layout and constructive-rule checks; no search or simulator."""
import hashlib
from pathlib import Path
import tomllib
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.cost.model import DistanceCost
from skillnet.domains.blocks import load_problem, problem_from_data
from simbridge.frames import BOARD_TOP, Frames
from simbridge.layout import check_layout


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("s2_tower_sim", "s3a_cantilever_sim", "s3a_platform_sim")
COLUMN = ["pick(B, pillar_base, stock_r0)", "insert(B, pillar_base, p0)",
          "pick(B, pillar, stock_r1)", "insert(B, pillar, p1)"]
BEAM_AND_WEIGHT = ["pick(B, beam_block, stock_r2)", "insert(B, beam_block, beam)",
                   "pick(B, counterweight, stock_r3)", "insert(B, counterweight, cw)"]
WITNESSES = {
    "s2_tower_sim": ["pick(A, b1, stock_l1)", "insert(A, b1, t1)",
                     "pick(A, b2, stock_l2)", "insert(A, b2, t2)",
                     "pick(B, b3, stock_r1)", "insert(B, b3, t3)"],
    "s3a_cantilever_sim": COLUMN+["support_on(A, beam)"]+BEAM_AND_WEIGHT+["support_off(A, beam)"],
    "s3a_platform_sim": COLUMN+[
        "pick(A, temp_block1, stock_l1)", "insert(A, temp_block1, tp1)",
        "pick(A, temp_block2, stock_l2)", "insert(A, temp_block2, tp2)"]+BEAM_AND_WEIGHT+[
        "detach(A, temp_block2, tp2)", "place(A, temp_block2, stock_l2)",
        "detach(A, temp_block1, tp1)", "place(A, temp_block1, stock_l1)"]}


def fixture(name="s3a_platform_sim"):
    return tomllib.loads((ROOT/"tasks/sim"/(name+".toml")).read_text(encoding="utf-8"))


def replay_fixture(name, *, problem=None):
    """Conditional rule witness; UNKNOWN assumptions never become IK proof."""
    problem = problem if problem is not None else load_problem(ROOT/"tasks/sim"/(name+".toml"))
    engine, state = Engine(problem), problem.initial
    if engine.hooks.invariant(state):
        raise AssertionError("Initial invariant fails")
    cost, duration = DistanceCost(problem.hooks), 0.
    unknown, trace = set(), []
    for text in WITNESSES[name]:
        action = engine.parse_action(text)
        if not engine.satisfied(action.pre, state):
            raise AssertionError(f"Preconditions fail: {text}")
        status, evidence = engine.oracle.evaluate(action.capability)
        if status == Status.INVALID:
            raise AssertionError(f"Capabilities fail: {text}: {evidence}")
        unknown.update(str(item.fact) for item in evidence if item.status == Status.UNKNOWN)
        trace.append({"action": text, "capability_status": status.value,
                      "required_unknown_facts": [str(item.fact) for item in evidence if item.status == Status.UNKNOWN]})
        duration += cost.estimate(action, state)
        state = engine.execute(action, state)
        if engine.hooks.invariant(state):
            raise AssertionError(f"Invariant fails: {text}")
    if not engine.satisfied(engine.goal, state):
        raise AssertionError("Constructive witness does not reach task goals")
    return {"task": name, "rules_replayed": True, "steps": WITNESSES[name],
            "serial_witness_estimate_seconds": duration,
            "search_performed": False, "optimal_makespan": None,
            "goal_satisfied": True, "initial_invariant_valid": True,
            "conditional": bool(unknown), "qualification_needed": bool(unknown),
            "required_unknown_facts": sorted(unknown), "all_capabilities_valid": not unknown,
            "all_witness_capabilities_noninvalid": True, "trace": trace,
            "motion_verified": False, "ik_performed": False, "dynamics_performed": False}


class SimLayoutTests(unittest.TestCase):
    def test_loaded_problem_retains_explicit_goal_geometry(self):
        data = fixture()
        data["fill_all_slots"] = False
        data["location"].append({"id": "outboard", "kind": "board", "pos": [.8, .12, 0.]})
        data["goal"] = ["at(pillar_base, outboard)"]
        problem = problem_from_data(data)
        report = check_layout(problem)
        self.assertIn("goal_outside_board", [row["code"] for row in report["violations"]])

    def assert_code(self, data, code):
        report = check_layout(data)
        self.assertFalse(report["valid"])
        self.assertIn(code, [item["code"] for item in report["violations"]])

    def test_three_fixtures_load_have_stable_initial_state_and_pass_geometry(self):
        for name in NAMES:
            with self.subTest(name=name):
                path = ROOT/"tasks/sim"/(name+".toml")
                problem = load_problem(path)
                self.assertIsNone(problem.hooks.invariant(problem.initial))
                report = check_layout(path)
                self.assertTrue(report["valid"], report["violations"])
                self.assertGreaterEqual(report["metrics"]["minimum_reference_spacing_m"], .20-1e-9)

    def test_cantilever_and_platform_have_tall_columns_and_matching_support_top(self):
        for name in NAMES[1:]:
            frames = Frames(ROOT/"tasks/sim"/(name+".toml"))
            beam_bottom = frames.slot_pose("beam")[2]-.04/2
            self.assertAlmostEqual(beam_bottom, BOARD_TOP+.152)
            self.assertAlmostEqual(frames.slot_pose("tp2")[2]+.076/2, beam_bottom)
            report = check_layout(frames.task)
            self.assertEqual(report["metrics"]["overhangs"]["beam"]["free_direction"], -1)
            self.assertAlmostEqual(report["metrics"]["overhangs"]["beam"]["length_m"], .32)

    def test_constructive_symbolic_witnesses_reach_all_goals_without_search(self):
        for name in NAMES:
            with self.subTest(name=name):
                report = replay_fixture(name)
                self.assertFalse(report["search_performed"])
                self.assertGreater(report["serial_witness_estimate_seconds"], 0)
                self.assertTrue(report["goal_satisfied"])
                self.assertTrue(report["initial_invariant_valid"])
                self.assertTrue(report["conditional"])
                self.assertTrue(report["qualification_needed"])
                self.assertTrue(report["required_unknown_facts"])
                self.assertFalse(report["all_capabilities_valid"])
                self.assertTrue(report["all_witness_capabilities_noninvalid"])
                self.assertFalse(report["motion_verified"])
                self.assertFalse(report["ik_performed"])
                self.assertFalse(report["dynamics_performed"])

    def test_conditional_witness_never_updates_unknown_oracle_evidence(self):
        problem = load_problem(ROOT/"tasks/sim/s2_tower_sim.toml")
        before = dict(problem.oracle.entries)
        required = Fact("reach", ("A", "stock_l1"))
        self.assertEqual(problem.oracle.check(required).status, Status.UNKNOWN)
        report = replay_fixture("s2_tower_sim", problem=problem)
        self.assertIn(str(required), report["required_unknown_facts"])
        self.assertEqual(problem.oracle.entries, before)
        self.assertEqual(problem.oracle.check(required).status, Status.UNKNOWN)

    def test_conditional_witness_still_rejects_explicit_invalid(self):
        problem = load_problem(ROOT/"tasks/sim/s2_tower_sim.toml")
        problem.oracle.update(Fact("reach", ("A", "stock_l1")), Status.INVALID, "synthetic invalid proof")
        with self.assertRaisesRegex(AssertionError, "Capabilities fail"):
            replay_fixture("s2_tower_sim", problem=problem)

    def test_reference_spacing_collision_is_rejected(self):
        data = fixture()
        data["location"][1]["pos"] = [-.50, -.21, 0.]
        self.assert_code(data, "reference_spacing")

    def test_stock_owner_range_rejects_suggested_point_just_outside_radius(self):
        data = fixture()
        data["location"][1]["pos"] = [-.72, -.52, 0.]
        self.assert_code(data, "stock_base_range")

    def test_loadbearing_pad_must_not_overlap_board_even_when_body_only_touches(self):
        data = fixture()
        data["location"][0]["pos"] = [-.4, -.2, 0.]
        self.assert_code(data, "pad_board_footprint")

    def test_pad_outside_table_is_rejected(self):
        data = fixture()
        data["location"][0]["pos"] = [-.90, -.20, 0.]
        self.assert_code(data, "pad_outside_table")

    def test_real_mount_overlap_is_rejected(self):
        data = fixture()
        data["location"][0]["pos"] = [-.70, .13, 0.]
        report = check_layout(data)
        self.assertTrue(any(item["code"] == "initial_overlap" and "A_Mount" in item["objects"]
                            for item in report["violations"]))

    def test_long_beam_can_collide_despite_reference_spacing(self):
        data = fixture()
        rows = {row["id"]: row for row in data["location"]}
        rows["stock_r2"]["pos"] = [.50, -.20, 0.]
        rows["stock_r0"]["pos"] = [.72, -.30, 0.]
        report = check_layout(data)
        self.assertTrue(any(item["code"] == "initial_overlap" and
                            set(item["objects"]) == {"Block_beam_block", "Block_pillar_base"}
                            for item in report["violations"]))

    def test_slot_body_must_fit_board(self):
        data = fixture("s2_tower_sim")
        for row in data["slot"]:
            row["pos"][0] = .33
        self.assert_code(data, "slot_outside_board")

    def test_too_short_platform_is_rejected(self):
        data = fixture()
        next(row for row in data["slot"] if row["id"] == "tp2")["type"] = "temporary_type2"
        self.assert_code(data, "platform_height")

    def test_platform_under_column_instead_of_overhang_is_rejected(self):
        data = fixture()
        for row in data["slot"]:
            if row["id"].startswith("tp"):
                row["pos"][0] = .12
        self.assert_code(data, "platform_under_wrong_end")

    def test_counterweight_on_free_end_is_rejected(self):
        data = fixture()
        next(row for row in data["slot"] if row["id"] == "cw")["pos"][0] = -.18
        self.assert_code(data, "counterweight_geometry")

    def test_low_column_is_rejected(self):
        data = fixture()
        next(row for row in data["slot"] if row["id"] == "p1")["below"] = []
        self.assert_code(data, "low_beam")

    def test_original_planning_fixtures_remain_byte_identical(self):
        expected = {"s2_tower": "2b9bef99b441d7ff1afe4195c9e370e928d89e28311bb2284919cc9c6704f47b",
                    "s3a_cantilever": "70a4b42f9fa9fc6214cd8e48311a40f0a419873928bd2a106ed6891d30707af5",
                    "s3a_platform": "743373b671126ad7213b489e3348357e0fe61506780e71fd30a697f6c41f7179"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/"tasks"/(name+".toml")).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
