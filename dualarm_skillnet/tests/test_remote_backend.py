"""Transport-loss mocks never contact or launch a simulator."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from simbridge import runner
from simbridge.backend import CoppeliaBackend
from simbridge.errors import is_transport_failure
from tests.test_v13_compiler import fixture


ROOT = Path(__file__).resolve().parents[1]


class RemoteBackendTests(unittest.TestCase):
    def make_backend(self):
        engine, geometry, compiler = fixture()
        geometry.solve_ik = MagicMock(side_effect=ConnectionResetError("mock remote closed"))
        return engine, CoppeliaBackend(geometry, compiler.cfg, compiler.frames)

    def test_validation_keeps_unknown_and_marks_transport_loss_fatal(self):
        engine, backend = self.make_backend()
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertIn("mock remote closed", backend.fatal_error)
        self.assertEqual(backend.audit()["fatal_error"], backend.fatal_error)

    def test_qualification_marks_transport_loss_fatal(self):
        _, backend = self.make_backend()
        self.assertEqual(backend.qualify(parse_fact("reach(A, stock_l1)")), Status.UNKNOWN)
        self.assertIn("mock remote closed", backend.fatal_error)

    def test_transport_loss_prevents_further_remote_queries(self):
        engine, backend = self.make_backend()
        backend.qualify(parse_fact("reach(A, stock_l1)"))
        calls = backend.planner.solve_ik.call_count
        self.assertEqual(backend.qualify(parse_fact("reach(B, stock_r1)")), Status.UNKNOWN)
        self.assertEqual(backend.validate(engine.parse_action("pick(A, red, stock_l1)"),
                                          engine.initial).status, Status.UNKNOWN)
        self.assertEqual(backend.planner.solve_ik.call_count, calls)

    def test_search_deadline_is_not_a_transport_failure(self):
        self.assertFalse(is_transport_failure(TimeoutError("task deadline")))
        self.assertFalse(is_transport_failure(RuntimeError("IK seeds exhausted")))
        try:
            try:
                raise ConnectionResetError("socket closed")
            except ConnectionResetError as exc:
                raise RuntimeError("query failed") from exc
        except RuntimeError as exc:
            self.assertTrue(is_transport_failure(exc))

    def test_runner_stops_after_mid_validation_remote_loss(self):
        self.check_runner_loss("real_planning")

    def test_runner_propagates_executor_transport_stop_flag(self):
        self.check_runner_loss("execution")

    def check_runner_loss(self, phase):
        engine, geometry, compiler = fixture()
        problem = engine.problem
        if phase == "real_planning":
            geometry.solve_ik = MagicMock(side_effect=ConnectionResetError("mock remote closed"))
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            owner = MagicMock()
            owner.connect.return_value = (object(), object())
            owner.process.pid = 12345
            owner.process.poll.return_value = 0
            owner.log_path = Path(folder)/"fake.log"
            with (patch.object(runner, "load_problem", return_value=problem),
                  patch.object(runner, "check_environment", return_value={"port": 23210}),
                  patch.object(runner, "Simulator", return_value=owner),
                  patch.object(runner, "build", return_value=deepcopy(compiler.cfg)),
                  patch.object(runner, "MotionPlanner") as planner,
                  patch("simbridge.executor.execute_plan", return_value={
                      "passed": False, "remote_interface_failure": True,
                      "failure_phase": "execution", "reason": "mock remote closed"}) as execute):
                planner.return_value.__enter__.return_value = geometry
                result = runner.run_task(ROOT/"tasks"/(problem.name+".toml"), out=folder)
            self.assertFalse(result["passed"])
            self.assertTrue(result["fatal_environment"], result)
            self.assertEqual(result["failure_phase"], phase)
            if phase == "real_planning":
                execute.assert_not_called()
            else:
                execute.assert_called_once()
            owner.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
