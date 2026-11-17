"""Stop conditions and failure evidence for project-owned simulator runs."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from simbridge.compiler import CompileFailure
from skillnet.capability.oracle import Status
from simbridge import runner
from tests.test_v13_compiler import fixture, sequential


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/s1_parallel_place.toml"


class RunnerTests(unittest.TestCase):
    def test_connect_failure_survives_close_timeout_with_original_stage(self):
        for raises in (False, True):
            with self.subTest(close_raises=raises), tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
                owner = MagicMock()
                owner.process.pid = 12345
                owner.process.poll.return_value = None
                owner.log_path = Path(folder) / "native.log"
                owner.connect.side_effect = TimeoutError("mock initial connect timeout")
                if raises:
                    owner.close.side_effect = TimeoutError("mock close timeout")
                plan = MagicMock(status="solved", feedback=[])
                with (patch.object(runner, "check_environment", return_value={}),
                      patch.object(runner, "run", return_value=plan),
                      patch.object(runner, "plan_to_dict", return_value={}),
                      patch.object(runner, "Simulator", return_value=owner)):
                    result = runner.run_task(TASK, out=folder)
                self.assertFalse(result["passed"])
                self.assertTrue(result["fatal_environment"])
                self.assertEqual(result["failure_phase"], "simulator_close")
                self.assertEqual(result["original_failure_phase"], "simulator_connect")
                self.assertEqual(result["startup_failure_phase"], "simulator_connect")
                self.assertIn("initial connect timeout", result["original_failure_reason"])
                saved = json.loads((Path(folder) / TASK.stem / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["startup_failure_phase"], "simulator_connect")

    def test_connect_checkpoint_is_durable_before_the_first_connect_call(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            owner, observed = MagicMock(), []
            owner.process.pid = 12345
            owner.process.poll.return_value = 0
            owner.log_path = Path(folder) / "native.log"
            def connect():
                observed.append(json.loads((Path(folder) / TASK.stem / "run_progress.json").read_text(encoding="utf-8")))
                raise TimeoutError("mock stop after reading connect checkpoint")
            owner.connect.side_effect = connect
            with (patch.object(runner, "check_environment", return_value={}),
                  patch.object(runner, "run", return_value=MagicMock(status="solved", feedback=[])),
                  patch.object(runner, "plan_to_dict", return_value={}),
                  patch.object(runner, "Simulator", return_value=owner)):
                runner.run_task(TASK, out=folder)
            self.assertEqual(len(observed), 1)
            progress = observed[0]
            self.assertEqual(progress["phase"], "simulator_connect")
            self.assertEqual(progress["pid"], 12345)
            self.assertEqual(progress["task"], TASK.stem)
            self.assertEqual(progress["task_file"], str(TASK.resolve()))
            self.assertRegex(progress["task_sha256"], "^[a-f0-9]{64}$")
            self.assertGreaterEqual(progress["elapsed_seconds"], 0.)

    def test_workcell_checkpoint_marks_a_completed_connection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            owner, observed = MagicMock(), []
            owner.process.pid = 12345
            owner.process.poll.return_value = 0
            owner.log_path = Path(folder) / "native.log"
            owner.connect.return_value = (object(), object())
            def build(*args):
                observed.append(json.loads((Path(folder) / TASK.stem / "run_progress.json").read_text(encoding="utf-8")))
                raise RuntimeError("mock stop after reading workcell checkpoint")
            with (patch.object(runner, "check_environment", return_value={}),
                  patch.object(runner, "run", return_value=MagicMock(status="solved", feedback=[])),
                  patch.object(runner, "plan_to_dict", return_value={}),
                  patch.object(runner, "Simulator", return_value=owner),
                  patch.object(runner, "build", side_effect=build)):
                result = runner.run_task(TASK, out=folder)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["phase"], "workcell")
            self.assertEqual(result["original_failure_phase"], "workcell")
            self.assertNotIn("startup_failure_phase", result)

    def test_output_outside_project_rejected_before_loading_or_launch(self):
        with patch.object(runner, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "项目根目录"):
                runner.run_task(TASK, out=ROOT.parent / "forbidden")
            simulator.assert_not_called()

    def test_low_c_space_is_fatal_without_starting_simulator(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            with (patch.object(runner.shutil, "disk_usage", return_value=MagicMock(free=4*1024**3)),
                  patch.object(runner, "Simulator") as simulator):
                result = runner.run_task(TASK, out=folder)
            self.assertFalse(result["passed"])
            self.assertTrue(result["fatal_environment"])
            self.assertEqual(result["failure_phase"], "environment")
            simulator.assert_not_called()
            saved = json.loads((Path(folder) / TASK.stem / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("不足 5 GB", saved["reason"])

    def test_launch_failure_is_fatal_and_preserves_reason(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            with (patch.object(runner, "check_environment", return_value={}),
                  patch.object(runner, "Simulator") as factory):
                owner = factory.return_value
                owner.start.side_effect = RuntimeError("模拟器启动失败")
                owner.process, owner.log_path = None, None
                result = runner.run_task(TASK, out=folder)
            self.assertTrue(result["fatal_environment"])
            self.assertEqual(result["failure_phase"], "simulator_start")
            owner.close.assert_called_once()

    def test_expired_compiler_deadline_stops_before_remote_call(self):
        engine, geometry, compiler = fixture()
        compiler.deadline = time.monotonic()-1
        steps = sequential(engine, ["pick(A, red, stock_l1)"])
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile(steps)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertTrue(failed.exception.evidence["timeout"])
        self.assertFalse(geometry.calls)
        self.assertFalse(compiler.tasks)

    def test_batch_continues_after_timeout_and_stops_on_environment_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            manifest = Path(folder) / "manifest.toml"
            manifest.write_text('[[run]]\ntask="tasks/s1_parallel_place.toml"\n'
                                '[[run]]\ntask="tasks/s1_reach_handover.toml"\n'
                                '[[run]]\ntask="tasks/s2_tower.toml"\n', encoding="utf-8")
            with (patch.object(runner, "DEFAULT_OUT", Path(folder)),
                  patch.object(runner, "run_task", side_effect=[
                      {"passed": False, "status": "timeout", "failure_phase": "execute", "reason": "超时"},
                      {"passed": False, "fatal_environment": True, "reason": "无法连接"}]) as execute):
                result = runner.batch(manifest)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(result["completed"], 2)
            self.assertFalse(result["passed"])

    def test_video_limit_disables_recording_at_two_gib(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as folder:
            path = Path(folder) / "record.mp4"
            path.write_bytes(b"video")
            self.assertEqual(runner.VIDEO_LIMIT, 2*1024**3)
            with patch.object(runner, "VIDEO_LIMIT", 5):
                self.assertFalse(runner.video_allowed(True, folder))


if __name__ == "__main__":
    unittest.main()
