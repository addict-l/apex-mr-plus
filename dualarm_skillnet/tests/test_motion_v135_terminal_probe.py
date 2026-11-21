"""Nominal final-home necessary conditions, never dynamics success claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skillnet.core.logic import Condition, Fact
from skillnet.domains.blocks import load_problem
from simbridge.frames import Frames
from simbridge.compiler import CompileFailure
from tests.test_v13_compiler import fixture
from tools import motion_v135_terminal_probe as probe


ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = ROOT / "results/sim_v1_3/v133/qualification"


class NominalTerminalTests(unittest.TestCase):
    def problem(self, family, number):
        name = f"gen_{family.lower()}_s20261115_{number}"
        task = QUALIFIED/family/number/name/(name+".qualified.toml")
        return task, load_problem(task, ROOT/"skills/blocks_v1.toml")

    def test_same_type_filled_goals_have_one_exact_union_and_no_native_success_claim(self):
        task, problem = self.problem("F3", "021")
        before = hashlib.sha256(task.read_bytes()).hexdigest()
        frames = Frames(problem)
        terminal = probe.build_nominal_terminal(problem, frames)
        self.assertEqual(terminal["status"], "constructed")
        self.assertTrue(terminal["exact_nominal_goals"])
        self.assertEqual(len(terminal["required_goal_parts"]), 5)
        self.assertEqual(terminal["held"], {})
        self.assertEqual(len(terminal["placed"]), 5)
        self.assertEqual({tuple(pose) for pose in terminal["placed"].values()},
                         {tuple(frames.slot_pose(slot)) for slot in frames.slots})
        self.assertEqual(terminal["identity_equivalence"][0]["identity_permutations"], 120)
        self.assertFalse(terminal["robust_to_goal_tolerances"])
        self.assertFalse(terminal["native_execution_passed"])
        self.assertEqual(hashlib.sha256(task.read_bytes()).hexdigest(), before)

    def test_f5_temporary_parts_at_original_stocks_are_only_one_unconstrained_candidate(self):
        _, problem = self.problem("F5", "028")
        frames = Frames(problem)
        terminal = probe.build_nominal_terminal(problem, frames)
        self.assertTrue(terminal["exact_nominal_goals"])
        self.assertEqual(set(terminal["unconstrained_parts"]), {"temp_block1", "temp_block2"})
        for part in terminal["unconstrained_parts"]:
            self.assertEqual(terminal["placed"][part], frames.block_pose(part))
        self.assertFalse(any(text in terminal["facts"] for text in ("filled(tp1)", "filled(tp2)")))
        self.assertTrue(terminal["unconstrained_placement_is_one_candidate"])

    def test_explicit_at_goals_pin_part_identity_and_unsupported_goals_remain_unknown(self):
        engine, _, compiler = fixture()
        terminal = probe.build_nominal_terminal(engine.problem, compiler.frames)
        self.assertEqual(terminal["placed"]["red"], compiler.frames.block_pose("red", "board_l1"))
        self.assertEqual(terminal["placed"]["blue"], compiler.frames.block_pose("blue", "board_r1"))
        problem = deepcopy(engine.problem)
        problem.goal += (Condition(Fact("held", ("A", "red"))),)
        unknown = probe.build_nominal_terminal(problem, compiler.frames)
        self.assertEqual(unknown["status"], "unknown")
        self.assertFalse(unknown["exact_nominal_goals"])

    def test_duplicate_initial_slot_occupant_cannot_be_hidden_by_filled_goal(self):
        _, problem = self.problem("F3", "021")
        problem.initial = (problem.initial-{Fact("at", ("b1_1", "stock_b1"))}
                           |{Fact("installed", ("b1_1", "s1_1")), Fact("filled", ("s1_1",))})
        problem.goal = (Condition(Fact("free", ("A",))), Condition(Fact("free", ("B",))),
                        Condition(Fact("filled", ("s1_1",))),
                        Condition(Fact("installed", ("b2_1", "s1_1"))))
        terminal = probe.build_nominal_terminal(problem, Frames(problem))
        self.assertEqual(terminal["status"], "unknown")
        self.assertTrue(any("multiple" in reason for reason in terminal["unresolved"]))


class HomeTargetScopeTests(unittest.TestCase):
    def checked(self, collision_part):
        engine, geometry, compiler = fixture("s3a_cantilever")
        terminal = probe.build_nominal_terminal(engine.problem, compiler.frames)
        validate = geometry.validate_joint
        def goal_collision(arm, q, **kwargs):
            answer = validate(arm, q, **kwargs)
            if arm == "B":
                return {"ok": False, "problem": {"reason": "block_robot_collision",
                    "pair": ["/B_UR5/B_GripperPalm", collision_part]}}
            return answer
        geometry.validate_joint = goal_collision
        result = probe.check_home_targets(geometry, compiler.cfg, compiler.frames, terminal)
        return result, geometry, compiler, terminal

    def test_required_part_collision_has_exact_nominal_scope_and_checks_both_homes(self):
        result, geometry, compiler, terminal = self.checked("beam_block")
        self.assertEqual(len(result["home_checks"]), 2)
        bad = result["home_checks"][-1]
        self.assertEqual(bad["arm"], "B")
        self.assertEqual(bad["status"], "target_configuration_collision")
        self.assertTrue(bad["collision_involves_required_goal_part"])
        self.assertFalse(bad["conditional_on_unconstrained_placement"])
        self.assertEqual(bad["q"], {arm: compiler.cfg["arms"][arm]["home_q"] for arm in ("A", "B")})
        self.assertEqual(bad["placed"], terminal["placed"])
        self.assertEqual(bad["held"], {})
        self.assertFalse(result["native_execution_passed"])
        self.assertFalse(result["global_infeasibility_proven"])
        self.assertFalse(any(call[0] in {"ik", "plan"} for call in geometry.calls))
        self.assertTrue(all(call[2]["options"]["other_q"] == compiler.cfg["arms"]["B" if call[1] == "A" else "A"]["home_q"]
                            for call in geometry.calls if call[0] == "validate"))
        json.dumps(result, allow_nan=False)

    def test_collision_with_unconstrained_temp_does_not_prove_required_goal_conflict(self):
        result, _, _, _ = self.checked("temp_block1")
        bad = result["home_checks"][-1]
        self.assertFalse(bad["collision_involves_required_goal_part"])
        self.assertTrue(bad["conditional_on_unconstrained_placement"])
        self.assertFalse(result["global_infeasibility_proven"])

    def test_deadline_and_transport_loss_do_not_issue_further_geometry_calls(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        terminal = probe.build_nominal_terminal(engine.problem, compiler.frames)
        with self.assertRaises(TimeoutError):
            probe.check_home_targets(geometry, compiler.cfg, compiler.frames, terminal,
                                     deadline=time.monotonic()-1.)
        self.assertFalse(geometry.calls)
        attempts = []
        def channel_lost(*args, **kwargs):
            attempts.append(args)
            raise OSError("connection lost")
        geometry.validate_joint = channel_lost
        with self.assertRaises(CompileFailure):
            probe.check_home_targets(geometry, compiler.cfg, compiler.frames, terminal)
        self.assertEqual(len(attempts), 1)


class OwnedProbeBoundaryTests(unittest.TestCase):
    def test_task_identity_is_original_manifest_sha_and_original_home_evidence(self):
        case = NominalTerminalTests()
        task, _ = case.problem("F3", "021")
        identity = probe.task_identity(task)
        self.assertEqual(identity["baseline_job_id"], "v134-fixed20-08")
        self.assertEqual(identity["source_job_id"], "sim20-f3-0")
        self.assertEqual(identity["task_sha256"], "941de06c1a30217afda2bf2375017a6d516e6ad6a3b7dc54f2640232371fad83")
        self.assertEqual(set(identity["home_q"]), {"A", "B"})
        self.assertEqual(identity["original_task_budget_seconds"], 1200.)
        self.assertTrue(Path(identity["baseline_workcell"]).is_file())
        with self.assertRaisesRegex(ValueError, "正式 manifest"):
            probe.task_identity(ROOT/"tasks/s1_parallel_place.toml")

    def test_cli_uses_owned_hard_budget_and_keeps_worker_explicit(self):
        case = NominalTerminalTests()
        task, _ = case.problem("F3", "021")
        with tempfile.TemporaryDirectory(dir=probe.BASE) as folder:
            output = Path(folder)/"new_attempt"
            with patch.object(probe.control, "run_job", return_value={"status": "passed"}) as run:
                self.assertEqual(probe.main(["--task", str(task), "--out", str(output), "--seconds", "90"]), 0)
            args, kwargs = run.call_args
            self.assertEqual(args[2], 90.)
            self.assertTrue(kwargs["live"])
            self.assertIn("--worker", args[1])
            self.assertIn(str(output), args[1])
        with self.assertRaisesRegex(ValueError, "motion_v135"):
            probe.confined_output(ROOT/"out/invalid_terminal_probe")

    def test_worker_lifecycle_uses_new_instance_but_original_fixed_home_targets(self):
        case = NominalTerminalTests()
        task, _ = case.problem("F3", "021")
        _, geometry, compiler = fixture()
        owner = SimpleNamespace(process=SimpleNamespace(pid=4242, poll=lambda: 0), _instance="owned-test",
            log_path="local-test.log", install=Path("unused"),
            start=lambda **kwargs: None, connect=lambda: (object(), object()), close=lambda: None)
        geometry.close = lambda: None
        with tempfile.TemporaryDirectory(dir=probe.BASE) as folder:
            output = Path(folder)/"worker"
            with patch("apexmr.sim_connection.Simulator", return_value=owner), \
                 patch("apexmr.motion.MotionPlanner", return_value=geometry), \
                 patch("simbridge.runner.check_environment"), \
                 patch("simbridge.workcell.build", return_value=deepcopy(compiler.cfg)), \
                 patch("tools.overnight_batch.process_birth", return_value="test-birth"), \
                 patch("tools.overnight_regression.health", return_value={"passed": True}):
                result = probe.run(task, output, 90.)
            self.assertEqual(result["status"], "nominal_target_checks_passed")
            self.assertEqual(result["simulator_nonce"], "owned-test")
            self.assertTrue(result["simulator_closed"])
            self.assertTrue(result["cleanup"]["passed"])
            self.assertEqual(result["home_checks"][0]["q"], probe.task_identity(task)["home_q"])
            self.assertFalse(result["native_execution_passed"])
            self.assertTrue((output/"probe.json").is_file())
            self.assertFalse(any(call[0] in {"ik", "plan"} for call in geometry.calls))


if __name__ == "__main__":
    unittest.main()
