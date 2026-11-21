"""Transport failures stay visible across generic runtime and executor mocks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apexmr.runtime import execute
from simbridge.executor import execute_plan
from tests.test_v13_executor import execution_fixture, synthetic_execution


ROOT = Path(__file__).resolve().parents[1]


class RemoteSimFake:
    simulation_stopped = 0

    def __init__(self, evidence, *, polling_error=None, result_error=None, stop_error=None):
        self.evidence = evidence
        self.polling_error, self.result_error, self.stop_error = polling_error, result_error, stop_error
        self.stopped = False

    def getObject(self, path):
        return 100

    def setStepping(self, enabled):
        pass

    def startSimulation(self):
        pass

    def callScriptFunction(self, function, script):
        if function == "getState":
            if self.polling_error:
                raise self.polling_error
            return {"done": True, "time": .02}
        if self.result_error:
            raise self.result_error
        return deepcopy(self.evidence)

    def stopSimulation(self):
        self.stopped = True
        if self.stop_error:
            raise self.stop_error

    def getSimulationState(self):
        return self.simulation_stopped


class RuntimeTransportTests(unittest.TestCase):
    def check_runtime(self, **errors):
        _, _, _, _, _, graph, cfg = execution_fixture()
        sim = RemoteSimFake(synthetic_execution(graph, cfg), **errors)
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as directory:
            result = execute(sim, None, cfg, directory, video=False)
            saved = json.loads((Path(directory)/"execution.json").read_text(encoding="utf-8"))
        self.assertTrue(sim.stopped)
        return result, saved

    def test_poll_connection_loss_is_returned_and_persisted_as_transport_failure(self):
        result, saved = self.check_runtime(polling_error=ConnectionError("remote socket closed"))
        self.assertTrue(result["remote_interface_failure"])
        self.assertFalse(result["success"])
        self.assertIn("remote socket closed", result["failure"])
        self.assertTrue(saved["remote_interface_failure"])

    def test_result_connection_loss_is_returned_and_persisted(self):
        result, saved = self.check_runtime(result_error=OSError("RPC disconnected"))
        self.assertTrue(result["remote_interface_failure"])
        self.assertFalse(result["success"])
        self.assertTrue(saved["remote_interface_failure"])

    def test_cleanup_connection_loss_cannot_turn_into_a_successful_run(self):
        result, saved = self.check_runtime(stop_error=ConnectionError("stop RPC disconnected"))
        self.assertTrue(result["remote_interface_failure"])
        self.assertFalse(result["success"])
        self.assertFalse(saved["success"])
        self.assertIn("stop RPC disconnected", result["failure"])

    def test_controller_program_error_is_failed_without_transport_outage_label(self):
        result, _ = self.check_runtime(polling_error=RuntimeError("controller rejected action"))
        self.assertFalse(result.get("remote_interface_failure", False))
        self.assertFalse(result["success"])


class ExecutorTransportTests(unittest.TestCase):
    def test_runtime_transport_flag_survives_even_if_fake_measurements_claim_success(self):
        engine, geometry, _, steps, tasks, _, cfg = execution_fixture()

        def install(sim, current, graph, filename):
            current["program"] = graph

        def actual(sim, client, current, output, **kwargs):
            evidence = synthetic_execution(current["program"], current)
            evidence["remote_interface_failure"] = True
            evidence["remote_interface_errors"] = [{"stage": "poll", "message": "closed"}]
            return evidence

        with tempfile.TemporaryDirectory(dir=ROOT/"out") as directory:
            with (patch("simbridge.executor.install_controller", side_effect=install),
                  patch("simbridge.executor.execute", side_effect=actual)):
                result = execute_plan(None, None, cfg, tasks, engine.problem, directory,
                                      planner=geometry, plan_steps=steps)
            saved = json.loads((Path(directory)/"summary.json").read_text(encoding="utf-8"))
        self.assertTrue(result["remote_interface_failure"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_phase"], "execution")
        self.assertTrue(saved["remote_interface_failure"])

    def test_geometry_query_transport_cause_is_propagated_before_execution(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()

        def disconnected(*args, **kwargs):
            try:
                raise ConnectionError("geometry socket closed")
            except ConnectionError as cause:
                raise RuntimeError("remote query wrapper") from cause

        with tempfile.TemporaryDirectory(dir=ROOT/"out") as directory:
            with (patch("simbridge.executor.build_tpg", side_effect=disconnected),
                  patch("simbridge.executor.execute") as actual):
                result = execute_plan(None, None, cfg, tasks, engine.problem, directory, planner=geometry)
        self.assertTrue(result["remote_interface_failure"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_phase"], "tpg")
        actual.assert_not_called()

    def test_budget_timeout_does_not_claim_transport_outage(self):
        engine, geometry, _, _, tasks, _, cfg = execution_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as directory:
            with patch("simbridge.executor.build_tpg", side_effect=TimeoutError("budget exhausted")):
                result = execute_plan(None, None, cfg, tasks, engine.problem, directory, planner=geometry)
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result.get("remote_interface_failure", False))


if __name__ == "__main__":
    unittest.main()
