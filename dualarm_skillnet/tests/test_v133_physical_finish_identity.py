"""Immutable completion identities cannot be replaced by old same-ID rows."""
from pathlib import Path
import tempfile
import unittest

from tools import overnight_batch as old
from tools import v133_batch as batch
from tools import v133_finish_physical as finish


class PhysicalFinishIdentityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=old.ROOT/"out")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.task = self.base/"task.toml"
        self.task.write_text('name="mock"\n', encoding="utf-8")
        old.atomic_json(self.base/"physical_stage_outcome.json", {
            "finished": False, "deadline_epoch": 10000., "fixtures": {"kept": "proof"}})
        path = batch.prepare(finish.DEFAULT_REQUIRED_STAGE, [{"id": "same-id", "phase": "plan", "kind": "plan",
            "task": str(self.task), "output": str(self.base/"proof"), "timeout_seconds": 60.}], base=self.base)
        self.manifest = old.load_manifest(path)

    def invoke(self):
        self.calls = []
        return finish.run(base=self.base,
            runner=lambda stage: self.calls.append(stage) or {"status": "phase_complete"},
            preparer=lambda stage: self.calls.append("prepare:"+stage), checker=lambda: {"passed": True},
            reporter=lambda: None, recorder=lambda text: None, now=lambda: 100.,
            late=lambda: self.calls.append("late") or {"status": "finished"})

    def test_same_id_completion_with_old_definition_is_a_global_stop(self):
        old.Journal(self.manifest["state_dir"]).append({
            "id": "same-id", "definition_sha256": "old-definition", "status": "failed", "passed": False})
        result = self.invoke()
        self.assertEqual(result["status"], "global_stop")
        self.assertIn("定义", result["reason"])
        self.assertEqual(self.calls, [])
        self.assertFalse(old.read_json(self.base/"physical_stage_outcome.json")["finished"])

    def test_changed_input_cannot_reuse_previously_committed_definition(self):
        old.Journal(self.manifest["state_dir"]).append({
            "id": "same-id", "definition_sha256": self.manifest["jobs"][0]["definition_sha256"],
            "status": "failed", "passed": False})
        self.task.write_text('name="changed"\n', encoding="utf-8")
        result = self.invoke()
        self.assertEqual(result["status"], "global_stop")
        self.assertIn("SHA", result["reason"])
        self.assertEqual(self.calls, [])
        self.assertEqual(old.read_json(self.base/"physical_stage_outcome.json")["fixtures"], {"kept": "proof"})


if __name__ == "__main__":
    unittest.main()
