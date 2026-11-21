"""D qualification public contracts; geometry is a surrogate, never native."""
from copy import deepcopy
import math
import json
from pathlib import Path
import tomllib
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact, parse_condition
from simbridge.backend import CoppeliaBackend
from simbridge import qualifier
from skillnet.capability.oracle import CapabilityOracle
from tests.test_v13_compiler import fixture
from tests import test_v133_qualifier as old_qualifier_tests


class QualificationCandidateTests(unittest.TestCase):
    def test_thirty_two_distinct_seeds_and_raw_answers_without_main_mutation(self):
        _, geometry, compiler = fixture()
        backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames)
        before = deepcopy((backend.cfg, backend.compiler.q, backend.compiler.held))
        geometry.ik_failure = {"ok": False, "stats": {"candidates": 0}, "problem": {"reason": "no IK solution"}}
        answer = backend.qualify_candidate(parse_fact("reach(A, board_l1)"), seed_limit=32)
        self.assertEqual(answer, Status.UNKNOWN)
        record = backend.qualification_records[-1]
        stage = record["target_contexts"][0]["stages"][0]
        self.assertEqual(stage["ik_attempts"], 32)
        self.assertEqual(len({tuple(row["seed"]) for row in stage["raw_results"]}), 32)
        self.assertEqual(len(stage["seed_bank_sha256"]), 64)
        self.assertTrue(record["candidate_exhausted"])
        self.assertEqual((backend.cfg, backend.compiler.q, backend.compiler.held), before)

    def test_side_pose_uses_physical_slot_height_and_same_grasp_approach_candidate(self):
        from scipy.spatial.transform import Rotation
        _, geometry, compiler = fixture("sim/s2_tower_sim")
        backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames)
        original = geometry.solve_ik
        side = backend.frames.tcp_pose(backend.frames.block_pose("b3", "t3"),
                                      backend.frames.side_payload("b3", "A", math.radians(45)))
        observed = []
        def only_side(arm, position, quaternion, **kwargs):
            observed.append(list(position)+list(quaternion))
            if (Rotation.from_quat(quaternion).inv()*Rotation.from_quat(side[3:])).magnitude() < 1e-9:
                return original(arm, position, quaternion, **kwargs)
            return {"ok": False, "stats": {"candidates": 0}}
        geometry.solve_ik = only_side
        answer = backend.qualify_candidate(parse_fact("reach(A, t3)"), pitch_degrees=45)
        self.assertEqual(answer, Status.VALID)
        record = backend.qualification_records[-1]
        self.assertEqual(record["pose_mode"], "side_45")
        stages = record["target_contexts"][0]["stages"]
        self.assertEqual([stage["label"] for stage in stages], ["grasp", "approach"])
        self.assertAlmostEqual(stages[0]["pose"][2], side[2])
        self.assertAlmostEqual(stages[1]["pose"][2]-side[2], .1)
        self.assertEqual(stages[0]["pose"][3:], stages[1]["pose"][3:])
        self.assertIn("不等于", record["qualification_scope"])

    def test_deadline_after_last_seed_and_error_answers_are_not_complete_exhaustion(self):
        for late in (False, True):
            _, geometry, compiler = fixture()
            backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames)
            backend.compiler.deadline = 200.
            clock, calls = [100.], [0]
            def fail(*args, **kwargs):
                calls[0] += 1
                if late and calls[0] == 32:
                    clock[0] = 201.
                return {"ok": False, "stats": {"candidates": 0}, **({} if late else {"error": "temporary solver error"})}
            geometry.solve_ik = fail
            with patch("simbridge.compiler.time.monotonic", side_effect=lambda: clock[0]):
                self.assertEqual(backend.qualify_candidate(parse_fact("reach(A, board_l1)")), Status.UNKNOWN)
            self.assertFalse(backend.qualification_records[-1]["candidate_exhausted"])


class UpgradeBackend:
    def __init__(self, accepted=None, incomplete=False):
        self.compiler = SimpleNamespace(deadline=None)
        self.fatal_error = None
        self.qualification_records = []
        self.calls = []
        self.facts = []
        self.accepted, self.incomplete = accepted, incomplete

    def qualify(self, fact):
        self.facts.append(str(fact))
        self.calls.append((8, None))
        self.qualification_records.append({"status": "UNKNOWN", "seed_limit": 8})
        return Status.UNKNOWN

    def qualify_candidate(self, fact, *, seed_limit=32, pitch_degrees=None):
        self.calls.append((seed_limit, pitch_degrees))
        status = Status.VALID if pitch_degrees == self.accepted else Status.UNKNOWN
        self.qualification_records.append({"status": status.value, "pose_mode": "catalog" if pitch_degrees is None else f"side_{pitch_degrees}",
            "candidate_exhausted": status == Status.UNKNOWN and not self.incomplete,
            "seed_limit": seed_limit, "raw": "synthetic complete mock evidence"})
        return status


def one_fact_problem(depth=1):
    slots = {f"layer{i}": {"below": [f"layer{i-1}"] if i > 1 else []} for i in range(1, depth+1)}
    return SimpleNamespace(hooks=SimpleNamespace(arms=["A"], locations=[], slots=[f"layer{depth}"]),
                           oracle=CapabilityOracle(), data={"slot": [{"id": name, **row} for name, row in slots.items()]})


class QualificationUpgradeTests(unittest.TestCase):
    def check(self, backend, depth=1):
        return qualifier.qualify_workspace(one_fact_problem(depth), backend, deadline=time.monotonic()+5)

    def test_unknown_catalog_upgrades_to_thirty_two_and_succeeds(self):
        backend = UpgradeBackend()
        result = self.check(backend)
        self.assertEqual(backend.calls, [(8, None), (32, None)])
        self.assertEqual(result["updates"][0]["status"], "VALID")
        self.assertEqual(result["escalation"][0]["accepted_pose_mode"], "catalog")

    def test_fourth_layer_tries_side_modes_in_order_and_stops_on_matching_success(self):
        backend = UpgradeBackend(accepted=45)
        result = self.check(backend, 4)
        self.assertEqual(backend.calls, [(8, None), (32, None), (32, 30), (32, 45)])
        self.assertEqual(result["updates"][0]["status"], "VALID")
        self.assertEqual(result["escalation"][0]["accepted_pose_mode"], "side_45")

    def test_complete_high_layer_exhaustion_is_bounded_invalid_but_partial_unknown(self):
        for incomplete, expected in ((False, "INVALID"), (True, "UNKNOWN")):
            with self.subTest(incomplete=incomplete):
                backend = UpgradeBackend(accepted=99, incomplete=incomplete)
                result = self.check(backend, 4)
                self.assertEqual(backend.calls, [(8, None), (32, None), (32, 30), (32, 45), (32, 60)])
                self.assertEqual(result["updates"][0]["status"], expected)
                if not incomplete:
                    self.assertIn("顶抓 32 种子 + 侧抓 3 姿态 × 32 种子无解", result["escalation"][0]["detail"])

    def test_third_layer_has_only_catalog_and_interrupted_upgrade_stays_unknown(self):
        backend = UpgradeBackend(accepted=99)
        result = self.check(backend, 3)
        self.assertEqual(backend.calls, [(8, None), (32, None)])
        self.assertEqual(result["updates"][0]["status"], "INVALID")
        backend = UpgradeBackend(accepted=99)
        original = backend.qualify_candidate
        def disconnected(*args, **kwargs):
            answer = original(*args, **kwargs)
            backend.fatal_error = "disconnected"
            return answer
        backend.qualify_candidate = disconnected
        result = self.check(backend, 4)
        self.assertEqual(result["updates"][0]["status"], "UNKNOWN")
        self.assertEqual(len(result["escalation"][0]["attempts"]), 1)

    def test_real_f3_high_goal_rejected_without_enumeration_within_five_seconds(self):
        from skillnet.domains.blocks import load_problem
        from skillnet.core.engine import Engine
        from skillnet.planning.search import search
        path = qualifier.ROOT/"results/sim_v1_3/v133/qualification/F3/005/gen_f3_s20261115_005/gen_f3_s20261115_005.qualified.toml"
        before = path.read_bytes()
        problem = deepcopy(load_problem(path))
        high = [slot for slot in problem.hooks.slots if qualifier._slot_depth(problem, slot) >= 4]
        self.assertTrue(high)
        for slot in high:
            for arm in problem.hooks.arms:
                problem.oracle.update(parse_fact(f"reach({arm}, {slot})"), Status.INVALID, "mock完整D限定候选无解")
        began = time.monotonic()
        result = search(Engine(problem), time_limit=5.)
        self.assertEqual(result.status, "no_plan")
        self.assertLessEqual(time.monotonic()-began, 5.)
        self.assertEqual(path.read_bytes(), before)

    def test_actual_backend_records_all_four_high_pose_banks_before_invalid(self):
        from simbridge.frames import Frames
        _, geometry, compiler = fixture("sim/s2_tower_sim")
        raw = deepcopy(compiler.frames.task)
        raw["slot"] = [{"id": f"layer{i}", "type": "tower_top", "pos": [0, .12, 99+i],
                        "below": [f"layer{i-1}"] if i > 1 else []} for i in range(1, 5)]
        backend = CoppeliaBackend(geometry, compiler.cfg, Frames(raw))
        geometry.ik_failure = {"ok": False, "stats": {"candidates": 0}}
        result = self.check(backend, 4)
        self.assertEqual(result["updates"][0]["status"], "INVALID")
        row = result["escalation"][0]
        self.assertEqual([attempt["pose_mode"] for attempt in row["attempts"]], ["catalog", "side_30", "side_45", "side_60"])
        self.assertEqual(len([call for call in geometry.calls if call[0] == "ik"]), 8+4*32)
        for attempt in row["attempts"]:
            context = attempt["backend_record"]["target_contexts"][0]
            self.assertEqual(context["type"], "tower_top")
            self.assertEqual(context["stages"][0]["ik_attempts"], 32)
            self.assertEqual(len(context["stages"][0]["raw_results"]), 32)
            self.assertLess(context["target_pose"][2], 2.)  # symbolic z=103 is not physical stacking height
            self.assertEqual(len(attempt["backend_record"]["seed_bank_sha256"]), 64)

    def test_explicit_prior_prioritizes_high_positive_goals_without_reordering_updates(self):
        slots = {f"layer{i}": {"below": [f"layer{i-1}"] if i > 1 else []} for i in range(1, 6)}
        problem = SimpleNamespace(hooks=SimpleNamespace(arms=["A", "B"], locations=["stock"], slots=slots),
            oracle=CapabilityOracle(), goal=(parse_condition("filled(layer4)"), parse_condition("filled(layer5)")))
        known = {parse_fact("reach(B, layer5)"): Status.INVALID}
        backend = UpgradeBackend()
        result = qualifier.qualify_workspace(problem, backend, deadline=time.monotonic()+5, inherited=known)
        self.assertEqual(backend.facts[:3], ["reach(A, layer5)", "reach(A, layer4)", "reach(B, layer4)"])
        self.assertNotIn("reach(B, layer5)", backend.facts)
        self.assertEqual(result["updates"][0]["place"], "stock")
        self.assertEqual(result["updates"][-1]["status"], "INVALID")
        self.assertIn("正目标高层UNKNOWN", result["query_priority"])
        self.assertEqual(result["query_order"][:3], backend.facts[:3])


class PriorQualificationTests(unittest.TestCase):
    def setUp(self):
        self.env = old_qualifier_tests.QualifierTests()
        self.env.setUp()
        self.addCleanup(self.env.doCleanups)

    def make_prior(self):
        with self.env.mocks():
            return qualifier.qualify_task(self.env.task, out=self.env.output)

    def test_explicit_prior_inherits_only_known_cells_and_preserves_old_source(self):
        first = self.make_prior()
        prior = Path(first["qualified_task"])
        before = prior.read_bytes()
        old_log = Path(first["log_file"]).read_bytes()
        self.env.backend = old_qualifier_tests.FakeBackend()
        with self.env.mocks():
            result = qualifier.qualify_task(prior, prior_qualification=prior, out=self.env.directory/"upgraded")
        self.assertEqual([str(fact) for fact in self.env.backend.calls], ["reach(B, board_l1)"])
        self.assertEqual(result["inherited_count"], result["total"]-1)
        self.assertEqual(result["queried_count"], 1)
        self.assertEqual(prior.read_bytes(), before)
        self.assertEqual(Path(first["log_file"]).read_bytes(), old_log)
        self.assertTrue(result["prior_qualification"]["sha256"])

    def test_prior_source_hash_or_layout_mismatch_rejected_before_native_start(self):
        first = self.make_prior()
        prior = Path(first["qualified_task"])
        self.env.task.write_text(self.env.task.read_text(encoding="utf-8")+'\nchanged = "layout"\n', encoding="utf-8")
        with patch.object(qualifier, "_simulator") as start:
            with self.assertRaises(ValueError):
                qualifier.qualify_task(prior, prior_qualification=prior, out=self.env.directory/"bad")
        start.assert_not_called()

    def test_prior_layout_and_log_projection_tampering_are_rejected(self):
        first = self.make_prior()
        prior = Path(first["qualified_task"])
        raw = prior.read_text(encoding="utf-8")
        prior.write_text(raw.replace("qualifier_fixture", "other_layout", 1), encoding="utf-8")
        with patch.object(qualifier, "_simulator") as start:
            with self.assertRaises(ValueError):
                qualifier.qualify_task(self.env.task, prior_qualification=prior, out=self.env.directory/"layout_bad")
        start.assert_not_called()
        prior.write_text(raw, encoding="utf-8")
        log_path = Path(first["log_file"])
        log = json.loads(log_path.read_text(encoding="utf-8"))
        log["updates"][0]["status"] = "UNKNOWN"
        log_path.write_text(json.dumps(log), encoding="utf-8")
        with self.assertRaises(ValueError):
            qualifier.qualify_task(prior, prior_qualification=prior, out=self.env.directory/"projection_bad")

    def test_real_frozen_prior_validates_without_writing_or_querying(self):
        prior = qualifier.ROOT/"results/sim_v1_3/v133/qualification/F3/005/gen_f3_s20261115_005/gen_f3_s20261115_005.qualified.toml"
        before = prior.read_bytes()
        with patch.object(qualifier, "_simulator") as start:
            result = qualifier.qualify_task(prior, prior_qualification=prior, out=self.env.directory/"frozen", deadline=0.)
        start.assert_not_called()
        self.assertEqual(result["queried_count"], 0)
        self.assertGreater(result["inherited_count"], 0)
        self.assertEqual(result["checked"], result["inherited_count"])
        self.assertEqual(prior.read_bytes(), before)

    def test_default_version_output_is_v134_and_old_root_rejected(self):
        self.assertEqual(qualifier.VERSION_ROOT.name, "v134")
        with self.assertRaises(ValueError):
            qualifier.qualify_task(self.env.task, out=qualifier.ROOT/"results/sim_v1_3/v133/no_new_D")

    def test_cli_explicit_prior_routes_to_public_supervisor(self):
        from simbridge.__main__ import main
        with patch.object(qualifier, "qualify_tasks", return_value={"status": "completed", "results": []}) as run:
            self.assertEqual(main(["qualify", "--tasks", str(self.env.task), "--out", str(self.env.output),
                                   "--prior-qualification", str(self.env.task)]), 0)
        run.assert_called_once_with([str(self.env.task)], out=str(self.env.output),
                                    prior_qualification=[str(self.env.task)])


if __name__ == "__main__":
    unittest.main()
