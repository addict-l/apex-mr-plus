"""Mock-only preflight safety and evidence checks; no native/default run."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools import motion_v135_preflight as preflight


VERSION = {"core": {"simbridge/compiler.py": "new"}, "tools": {"supervisor": "new"}}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class SerialPreflightTests(unittest.TestCase):
    def execute(self, folder, *, failures=None, selected=None, version=None, stop=None, calls=None):
        calls = [] if calls is None else calls

        def perform(step, output, attempt, snapshot, tpg_profiles):
            calls.append(step)
            return {"passed": step not in (failures or {}), "scope": "mock_only",
                    "startup_failure": (failures or {}).get(step) == "startup",
                    "connected": (failures or {}).get(step) != "startup", "evidence_files": {}}

        result = preflight.run("mock", base=Path(folder), steps=selected,
            tpg_profiles=[], snapshotter=lambda: deepcopy(version or VERSION), performer=perform,
            stop_writer=(stop if stop is not None else lambda detail: None))
        return result, calls

    def test_all_current_version_proofs_follow_fixed_serial_order(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            result, calls = self.execute(folder)
            proof = preflight.old.read_json(Path(folder)/"m6/preflight_mock/state/proofs.json")
        self.assertEqual(calls, list(preflight.STEPS))
        self.assertTrue(result["formal_ready"])
        self.assertEqual(len(proof["steps"]), len(preflight.STEPS))

    def test_default_failure_stops_originals_and_all_live_dependencies(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            result, calls = self.execute(folder, failures={"default": "failure"})
        self.assertEqual(calls, ["tpg_profiles", "default"])
        self.assertFalse(result["formal_ready"])
        self.assertEqual(result["stop_reason"], "default_failed")

    def test_normal_gate_failure_requires_rollback_and_stops_other_live_steps(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            result, calls = self.execute(folder, failures={"gate_parallel": "failure"})
        self.assertEqual(calls[-1], "gate_parallel")
        self.assertTrue(result["requires_rollback"])
        self.assertFalse(result["formal_ready"])

    def test_other_gate_failure_preserves_denominator_and_runs_independent_remaining_gates(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            result, calls = self.execute(folder, failures={"gate_handover": "failure"})
        self.assertEqual(calls, list(preflight.STEPS))
        self.assertFalse(result["formal_ready"])
        self.assertFalse(result["steps"]["gate_handover"]["passed"])

    def test_resume_keeps_completed_proofs_and_does_not_redo_default_or_originals(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            calls = []
            first, _ = self.execute(folder, calls=calls, selected=["tpg_profiles", "default", "original11"])
            self.assertFalse(first["formal_ready"])
            result, _ = self.execute(folder, calls=calls)
        self.assertTrue(result["formal_ready"])
        self.assertEqual(calls, list(preflight.STEPS))

    def test_changed_version_is_rejected_without_reset_or_new_jobs(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            self.execute(folder, selected=["default"])
            changed = deepcopy(VERSION)
            changed["core"]["simbridge/compiler.py"] = "other"
            calls = []
            with self.assertRaisesRegex(ValueError, "版本|version"):
                self.execute(folder, calls=calls, version=changed)
            self.assertEqual(calls, [])

    def test_selected_native_step_cannot_skip_missing_current_prerequisites(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            result, calls = self.execute(folder, selected=["gate_tower"])
        self.assertEqual(calls, [])
        self.assertFalse(result["formal_ready"])
        self.assertEqual(result["steps"]["gate_tower"]["status"], "blocked")

    def test_three_consecutive_connect_failures_set_global_stop(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            stops = []
            result, calls = self.execute(folder, failures={name: "startup" for name in
                ("gate_handover", "gate_tower", "gate_cantilever")}, stop=stops.append)
        self.assertEqual(len(stops), 1)
        self.assertEqual(calls[-1], "gate_cantilever")
        self.assertTrue(result["global_stop"])
        self.assertFalse(result["formal_ready"])


class NativeEvidenceTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        output = root/"native"
        evidence = output/"simulation/s1_parallel_place"
        task = preflight.ROOT/"tasks/s1_parallel_place.toml"
        version = {"core": {"simbridge/compiler.py": "new"}, "tools": {}}
        code = put(root/"code.json", version["core"])
        job = {"status": "passed", "code_unchanged_during_job": True,
               "cleanup": {"passed": True}, "code_version": str(code)}
        put(output/"worker_result.json", {"passed": True, "status": "passed", "tests_run": 1,
            "test_skipped": 0, "test_failures": 0, "test_errors": 0,
            "test_name": preflight.GATES["gate_parallel"][0], "evidence_directory": str(evidence),
            "task_sha256": preflight.old.sha256(task)})
        put(evidence/"summary.json", {"passed": True, "status": "passed", "success": True,
            "task_sha256": preflight.old.sha256(task), "simulator_closed": True,
            "real_planning_status": "solved", "fatal_environment": False})
        put(evidence/"validation.json", {"success": True, "violations": [],
            "expected_nodes": 1, "actual_node_intervals": 1})
        put(evidence/"execution.json", {"success": True, "done": True,
            "independent_validation_passed": True, "completed": 1, "total": 1,
            "timeout": False, "remote_interface_failure": False,
            "tracking_failures": [], "collision_failures": [], "task_schedule": [{"id": "one"}]})
        put(evidence/"geometry/rollout_verification.json", {"valid": True, "coverage_complete": True,
            "tested_samples": 2, "planned_samples": 2, "no_sampled_collision": True,
            "stationary_waiting_poses_included": True})
        for name in ("tasks.json", "tpg.json", "reference_receipt.json", "execution_validation.json"):
            put(evidence/name, {"mock_only": True})
        return output, evidence, job, version

    def test_complete_native_evidence_is_required_and_read(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            output, _, job, version = self.fixture(folder)
            proof = preflight.verify_native("gate_parallel", output, job, version)
        self.assertTrue(proof["passed"])
        self.assertGreaterEqual(len(proof["evidence_files"]), 8)
        self.assertEqual(proof["scope"], "native_dynamics_and_independent_validation")

    def test_returncode_zero_without_execution_artifact_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            output, evidence, job, version = self.fixture(folder)
            (evidence/"execution.json").unlink()
            proof = preflight.verify_native("gate_parallel", output, job, version)
        self.assertFalse(proof["passed"])
        self.assertIn("execution", proof["reason"])

    def test_independent_validation_failure_is_preserved_despite_passed_summary(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            output, evidence, job, version = self.fixture(folder)
            put(evidence/"validation.json", {"success": False, "violations": ["collision"]})
            proof = preflight.verify_native("gate_parallel", output, job, version)
        self.assertFalse(proof["passed"])

    def test_native_proof_from_old_core_version_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT/"out") as folder:
            output, _, job, version = self.fixture(folder)
            version["core"]["simbridge/compiler.py"] = "newer"
            proof = preflight.verify_native("gate_parallel", output, job, version)
        self.assertFalse(proof["passed"])


if __name__ == "__main__":
    unittest.main()
