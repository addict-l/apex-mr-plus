"""The stopped-scene bounds sampler must be removed before any dynamics."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simbridge import executor
from tests.test_v13_executor import execution_fixture, synthetic_execution


ROOT = Path(__file__).resolve().parents[1]
ALIAS = "Scene3FastInterarmSampler"


class FakeScene:
    simulation_stopped = 0
    handle_scene = -1
    object_script_type = 16

    def __init__(self, stopped=True):
        self.state = 0 if stopped else 16
        self.scripts = {}

    def getSimulationState(self):
        return self.state

    def createScript(self, *_args):
        raise AssertionError("The fixture supplies the native sampler factory")

    def getObjectsInTree(self, root, kind, _options):
        assert root == self.handle_scene and kind == self.object_script_type
        return list(self.scripts)

    def getObjectAlias(self, handle):
        return self.scripts[handle]


class BoundsSamplerLifecycleTests(unittest.TestCase):
    def case(self, *, stopped=True, error=None, invalid_rollout=False, leak=False):
        engine, geometry, _, steps, tasks, _, cfg = execution_fixture()
        sim = FakeScene(stopped)
        client = SimpleNamespace(socket=SimpleNamespace(getsockopt=lambda *_: 0, setsockopt=lambda *_: None))
        trace = []
        full_planners = []
        context = {"restored": False}
        originals = {"oracle": executor.SampledConflictOracle, "rollout": executor.verify_rollout}

        class FakeSampler:
            def __init__(self, received_sim, received_client, received_cfg):
                assert (received_sim, received_client, received_cfg) == (sim, client, cfg)
                assert trace == ["idle_entered"]
                self.info = {"name": "native_bounds_test_adapter"}
                self.script = 101
                sim.scripts[self.script] = ALIAS
                trace.append("sampler_created")

            def close(self):
                trace.append("sampler_closed")
                if not leak:
                    sim.scripts.pop(self.script)
                self.script = None

        def opt_in(planner, current, directory, *, bounds_sampler):
            self.assertEqual(bounds_sampler.info, {"name": "native_bounds_test_adapter"})
            self.assertEqual(trace[-1], "sampler_created")
            full_planners.append(planner)
            if error is not None:
                raise error
            return originals["oracle"](planner, current, directory)

        @contextmanager
        def idle():
            trace.append("idle_entered")
            try:
                yield context
            finally:
                trace.append("idle_exited")
                context["restored"] = True

        def rollout(graph, planner, current, **kwargs):
            if stopped:
                self.assertEqual(len(full_planners), 1)
                self.assertIs(planner, full_planners[0])
            if invalid_rollout:
                return {"valid": False, "first_violation": {"reason": "full-scene collision"}}
            return originals["rollout"](graph, planner, current, **kwargs)

        def install(_sim, current, graph, _filename):
            self.assertFalse(sim.scripts)
            self.assertEqual(trace[-1], "idle_exited")
            trace.append("installed")
            current["program"] = graph

        def measured(_sim, _client, current, _output, **_kwargs):
            self.assertFalse(sim.scripts)
            trace.append("executed")
            return synthetic_execution(current["program"], current)

        (ROOT/"out").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            with (patch.object(executor, "FastInterarmSampler", FakeSampler, create=True),
                  patch.object(executor, "BroadphaseConflictOracle", side_effect=opt_in, create=True),
                  patch.object(executor, "verify_rollout", side_effect=rollout),
                  patch.object(executor, "install_controller", side_effect=install),
                  patch.object(executor, "execute", side_effect=measured)):
                result = executor.execute_plan(sim, client, cfg, tasks, engine.problem, folder,
                    planner=geometry, plan_steps=steps, stopped_geometry_context=idle)
            saved=json.loads((Path(folder)/"summary.json").read_text(encoding="utf-8"))
            progress=json.loads((Path(folder)/"executor_progress.json").read_text(encoding="utf-8"))
        return trace, result, saved, progress

    def test_stopped_sampler_is_created_and_removed_before_install_and_execution(self):
        trace, result, saved, progress = self.case()
        self.assertTrue(result["passed"], result.get("reason"))
        self.assertEqual(trace, ["idle_entered", "sampler_created", "sampler_closed",
                                 "idle_exited", "installed", "executed"])
        receipt = {"created": True, "closed": True, "script_absent": True,
                   "script_handle": 101, "script_alias": ALIAS,
                   "lua_sha256": sha256((ROOT/"apexmr/fast_interarm.lua").read_bytes()).hexdigest()}
        for data in (result, saved, progress):
            self.assertEqual({key: data["tpg_bounds_sampler"][key] for key in receipt}, receipt)

    def test_exceptions_and_invalid_rollout_close_the_sampler_without_installing(self):
        for options, phase in (({"error": RuntimeError("oracle construction interrupted")}, "tpg_geometry"),
                               ({"invalid_rollout": True}, "rollout")):
            with self.subTest(phase=phase):
                trace, result, _, _ = self.case(**options)
                self.assertEqual(trace, ["idle_entered", "sampler_created", "sampler_closed", "idle_exited"])
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure_phase"], phase)
                self.assertTrue(result["tpg_bounds_sampler"]["script_absent"])

    def test_a_script_still_present_after_close_blocks_dynamics(self):
        trace, result, _, _ = self.case(leak=True)
        self.assertEqual(trace, ["idle_entered", "sampler_created", "sampler_closed", "idle_exited"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_phase"], "geometry_context_restore")
        self.assertFalse(result["tpg_bounds_sampler"]["script_absent"])

    def test_non_stopped_connection_uses_the_original_geometry_route(self):
        trace, result, _, _ = self.case(stopped=False)
        self.assertEqual(trace, ["idle_entered", "idle_exited", "installed", "executed"])
        self.assertTrue(result["passed"], result.get("reason"))
        self.assertFalse(result["tpg_bounds_sampler"]["created"])


if __name__ == "__main__":
    unittest.main()

