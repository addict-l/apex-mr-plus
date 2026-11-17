"""Explicit registration-attempt changes against disposable mock workspaces."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_v134_tools as fixtures
from tools import v134_batch as batch


class RegistrationAttemptTests(unittest.TestCase):
    setUp = fixtures.V134ToolsTests.setUp
    save = fixtures.V134ToolsTests.save
    register = fixtures.V134ToolsTests.register

    def previous(self):
        self.register()
        path = batch.prepare_gates("motion_gate", base=self.base, project=self.project)
        manifest = batch.old.load_manifest(path)
        self.save(Path(manifest["state_dir"]) / "state.json", {"status": "phase_complete", "active_job": None})
        output = Path(manifest["jobs"][0]["output"])
        self.save(output / "raw.json", {"original_native": True})
        self.save(self.base / "experiments" / batch.NAMESPACE / "workflow_planning.json", {"status": "interrupted", "stages": {}})
        self.save(self.base / "last_motion_gate.json", {"passed": True, "motion_code_sha256": batch.motion_fingerprint(project=self.project)})
        return path, manifest

    def fresh_proofs(self, attempt="runtime_fix1"):
        full = json.loads(self.full.read_text(encoding="utf-8"))
        original = json.loads(self.original.read_text(encoding="utf-8"))
        original["label"] = "four_fixes_"+attempt
        return (self.save(self.base / "regressions" / ("full_"+attempt+".json"), full),
                self.save(self.base / ("original11_four_fixes_"+attempt+".json"), original))

    def test_explicit_attempt_archives_only_active_controls_and_uses_fresh_paths(self):
        path, manifest = self.previous()
        old_selection = (self.base / "experiment_selection.json").read_bytes()
        old_manifest = path.read_bytes()
        previous_sha = batch.sha256(self.base / "experiment_selection.json")
        protected = {p: p.read_bytes() for directory in [self.source, self.base / "states", self.base / "experiments"]
                     for p in directory.rglob("*") if p.is_file()}
        proofs = self.fresh_proofs()
        selected = batch.register(*proofs, base=self.base, project=self.project,
            attempt="runtime_fix1", replace_previous_sha256=previous_sha)
        data = json.loads(selected.read_text(encoding="utf-8"))
        self.assertEqual(data["selection_id"], "c_wall_v134")
        self.assertEqual(data["registration_attempt"], "runtime_fix1")
        history = self.base / "registration_history" / previous_sha
        self.assertEqual((history / "experiment_selection.json").read_bytes(), old_selection)
        self.assertEqual((history / "manifests" / "motion_gate.json").read_bytes(), old_manifest)
        self.assertFalse(path.exists())
        self.assertFalse((self.base / "last_motion_gate.json").exists())
        self.assertEqual(protected, {p: p.read_bytes() for p in protected})
        fresh = batch.old.load_manifest(batch.prepare_gates("motion_gate", base=self.base, project=self.project))
        self.assertEqual(Path(fresh["state_dir"]), self.base / "states/runtime_fix1/motion_gate")
        self.assertTrue(Path(fresh["jobs"][0]["output"]).is_relative_to(self.base / "experiments/c_wall_v134/runtime_fix1"))

    def test_wrong_sha_and_plain_register_never_archive_or_replace(self):
        self.previous(); proofs = self.fresh_proofs()
        path = self.base / "experiment_selection.json"; before = path.read_bytes()
        for kwargs in ({}, {"attempt": "runtime_fix1", "replace_previous_sha256": "0"*64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                batch.register(*proofs, base=self.base, project=self.project, **kwargs)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((self.base / "registration_history").exists())

    def test_running_cursor_stage_and_live_process_each_reject_before_archive(self):
        _, manifest = self.previous(); proofs = self.fresh_proofs()
        selected = self.base / "experiment_selection.json"; expected = batch.sha256(selected)
        cursor = self.base / "experiments/c_wall_v134/workflow_planning.json"
        active = Path(manifest["state_dir"]) / "state.json"
        process = self.base / "experiments/c_wall_v134/process_planning.json"
        for mode in ("cursor", "stage", "active_job", "process", "process_no_birth"):
            self.save(cursor, {"status": "running" if mode == "cursor" else "interrupted"})
            self.save(active, {"status": "running" if mode == "stage" else "phase_complete",
                "active_job": {"pid": 99} if mode == "active_job" else None})
            if mode.startswith("process"):
                self.save(process, {"pid": 77, **({"birth": "same"} if mode == "process" else {})})
            with self.subTest(mode=mode), patch.object(batch.old, "process_birth", return_value="same"), self.assertRaises(ValueError):
                batch.register(*proofs, base=self.base, project=self.project,
                    attempt="runtime_fix1", replace_previous_sha256=expected)
            self.assertEqual(batch.sha256(selected), expected)
            self.assertFalse((self.base / "registration_history").exists())

    def test_failed_new_proof_and_old_proof_paths_cannot_replace(self):
        self.previous(); proofs = self.fresh_proofs()
        selected = self.base / "experiment_selection.json"; expected = batch.sha256(selected)
        full = json.loads(proofs[0].read_text(encoding="utf-8")); full["status"] = "failed"
        self.save(proofs[0], full)
        with self.assertRaisesRegex(ValueError, "未通过"):
            batch.register(*proofs, base=self.base, project=self.project,
                attempt="runtime_fix1", replace_previous_sha256=expected)
        with self.assertRaisesRegex(ValueError, "新回归证明路径"):
            batch.register(self.full, self.original, base=self.base, project=self.project,
                attempt="runtime_fix1", replace_previous_sha256=expected)
        self.assertEqual(batch.sha256(selected), expected)
        self.assertFalse((self.base / "registration_history").exists())

    def test_interrupted_publish_resumes_without_touching_old_outputs_or_states(self):
        path, manifest = self.previous(); proofs = self.fresh_proofs()
        selected = self.base / "experiment_selection.json"; expected = batch.sha256(selected)
        output = Path(manifest["jobs"][0]["output"]) / "native.json"
        self.save(output, {"status": "old_failure"})
        old_bytes = output.read_bytes()
        atomic = batch.old.atomic_json

        def fail_publish(target, value):
            if Path(target) == selected:
                raise OSError("mock atomic publish interruption")
            return atomic(target, value)

        with patch.object(batch.old, "atomic_json", side_effect=fail_publish), self.assertRaises(OSError):
            batch.register(*proofs, base=self.base, project=self.project,
                attempt="runtime_fix1", replace_previous_sha256=expected)
        self.assertFalse(path.exists())
        self.assertEqual(batch.sha256(selected), expected)
        batch.register(*proofs, base=self.base, project=self.project,
            attempt="runtime_fix1", replace_previous_sha256=expected)
        before = selected.read_bytes()
        batch.register(*proofs, base=self.base, project=self.project,
            attempt="runtime_fix1", replace_previous_sha256=expected)
        self.assertEqual(selected.read_bytes(), before)
        self.assertEqual(output.read_bytes(), old_bytes)

    def test_workflow_cursor_and_progress_journal_use_attempt_directory(self):
        from tools import v134_workflow as workflow
        self.previous(); proofs = self.fresh_proofs()
        selected = self.base / "experiment_selection.json"; expected = batch.sha256(selected)
        old_cursor = self.base / "experiments/c_wall_v134/workflow_planning.json"
        cursor_before = old_cursor.read_bytes()
        batch.register(*proofs, base=self.base, project=self.project,
            attempt="runtime_fix1", replace_previous_sha256=expected)
        result = workflow.run(base=self.base, project=self.project, phase="qualification",
            ready=lambda: None, preparer=lambda stage: None,
            runner=lambda stage: {"status": "phase_complete"}, checker=lambda: {"passed": True}, reporter=lambda: None)
        self.assertEqual(result["status"], "finished")
        self.assertTrue((self.base / "experiments/c_wall_v134/runtime_fix1/workflow_qualification.json").is_file())
        self.assertEqual(old_cursor.read_bytes(), cursor_before)
        progress = json.loads((self.base / "progress_state.json").read_text(encoding="utf-8"))
        events = [x for x in progress["steps"] if x.get("journal")]
        self.assertTrue(events)
        self.assertTrue(all(Path(x["journal"]).is_relative_to(self.base / "states/runtime_fix1") for x in events))

    def test_nested_startup_failures_still_stop_and_old_gate_is_not_reused(self):
        for i in range(3):
            path = self.base / "states/runtime_fix1/qualify5/completed.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": str(i), "phase": "sim", "status": "failed", "executed": True,
                    "startup_failure_phase": "simulator_connect", "completed_at": str(i)})+"\n")
        self.assertIn("三次", batch.stop_reason(self.base))
        self.save(self.base / "experiment_selection.json", {"registration_attempt": "runtime_fix1"})
        self.save(self.base / "last_motion_gate.json", {"passed": True, "motion_code_sha256": {}})
        with patch.object(batch, "motion_fingerprint", return_value={}), self.assertRaisesRegex(ValueError, "新attempt"):
            batch.require_motion_gate({"kind": "sim", "motion_code_sha256": {}}, base=self.base)


if __name__ == "__main__":
    unittest.main()
