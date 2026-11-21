"""Runner orchestration tests with a fake owner, scene, geometry and executor."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.pipeline import run as pipeline_run
from simbridge import runner
from simbridge.backend import CoppeliaBackend
from tests.test_v13_compiler import fixture


ROOT = Path(__file__).resolve().parents[1]


class RunnerBackendTests(unittest.TestCase):
    def run_mock(self, *, feedback=False, initial_unknown=False, missing_reach=False, distance_prior=False):
        engine, geometry, compiler = fixture("s1_reach_handover" if feedback else "s1_parallel_place")
        problem = engine.problem
        problem.proposal = None
        if missing_reach:
            problem.oracle.defaults.pop("reach", None)
            problem.oracle.entries = {fact: evidence for fact, evidence in problem.oracle.entries.items()
                                      if fact.predicate != "reach"}
        if distance_prior:
            problem.oracle.defaults["reach"] = Status.INVALID
            problem.oracle.entries = {fact: evidence for fact, evidence in problem.oracle.entries.items()
                                      if fact.predicate != "reach"}
        if initial_unknown:
            problem.oracle.update(parse_fact("reach(A, stock_l1)"), Status.UNKNOWN,
                                  "mock capability awaiting geometry qualification")
        if feedback:
            for action in engine.actions.values():
                for fact in action.capability:
                    problem.oracle.update(fact, Status.VALID, "mock capability table")
            solve = geometry.solve_ik

            def workspace(arm, position, quaternion, **kwargs):
                if (arm == "A" and position[0] > .1) or (arm == "B" and position[0] < -.1):
                    return {"ok": False, "infeasible": True,
                            "problem": {"reason": "mock exclusive workspace"}}
                return solve(arm, position, quaternion, **kwargs)

            geometry.solve_ik = workspace
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            owner = MagicMock()
            owner.connect.return_value = (object(), object())
            owner.process.pid = 12345
            owner.process.poll.return_value = 0
            owner.log_path = Path(folder)/"fake.log"
            scene = deepcopy(compiler.cfg)
            with (patch.object(runner, "load_problem", return_value=problem),
                  patch.object(runner, "check_environment", return_value={"port": 23210}),
                  patch.object(runner, "Simulator", return_value=owner),
                  patch.object(runner, "build", return_value=scene),
                  patch.object(runner, "MotionPlanner") as planner,
                  patch.object(runner, "read_task", wraps=runner.read_task) as read_task,
                  patch.object(runner, "run", wraps=pipeline_run) as planning,
                  patch("simbridge.executor.execute_plan", return_value={"passed": True}) as execute):
                planner.return_value.__enter__.return_value = geometry
                if distance_prior:
                    task_data = read_task(ROOT/"tasks"/(problem.name+".toml"))
                    task_data["capability_source"] = "distance_prior"
                    read_task.return_value = task_data
                result = runner.run_task(ROOT/"tasks"/(problem.name+".toml"), out=folder)
            self.assertTrue(result["passed"], result.get("reason"))
            self.assertEqual(planning.call_count, 2)
            self.assertNotIn("backend", planning.call_args_list[0].kwargs)
            self.assertIsInstance(planning.call_args_list[1].kwargs["backend"], CoppeliaBackend)
            self.assertEqual(result["real_planning_status"], "solved")
            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs["planning_seconds"],
                             result["initial_planning_seconds"]+result["real_planning_seconds"])
            expected = len(problem.hooks.arms)*(len(problem.hooks.locations)+len(problem.hooks.slots)) if missing_reach or distance_prior else 1 if initial_unknown else 0
            self.assertEqual(result["qualification_calls"], expected)
            owner.close.assert_called_once()
            output = Path(folder)/problem.name
            self.assertTrue((output/"initial_plan.json").is_file())
            self.assertTrue((output/"motion_backend.json").is_file())
            saved = json.loads((output/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["feedback"], result["feedback"])
            self.assertGreater(saved["motion_backend_cache"]["entries"], 0)
            return result

    def test_runner_replans_with_backend_before_final_compile(self):
        self.run_mock()

    def test_runner_retains_real_backend_cut_and_replanning_feedback(self):
        result = self.run_mock(feedback=True)
        self.assertTrue(any("运动失败" in message and "重新搜索" in message
                            for message in result["feedback"]))
        self.assertGreater(result["motion_feedback_count"], 0)

    def test_initial_unknown_is_qualified_after_scene_creation_and_then_executed(self):
        result = self.run_mock(initial_unknown=True)
        self.assertEqual(result["planning_status"], "unknown")
        self.assertEqual(result["real_planning_status"], "solved")
        self.assertTrue(any("按需验证" in message for message in result["feedback"]))

    def test_missing_reach_prequalifies_every_combination_in_one_scene_before_planning(self):
        result = self.run_mock(missing_reach=True)
        self.assertEqual(result["prequalification"]["status"], "completed")
        self.assertEqual(result["prequalification"]["checked"], result["qualification_calls"])

    def test_distance_prior_is_replaced_by_real_prequalification_before_planning(self):
        result = self.run_mock(distance_prior=True)
        self.assertEqual(result["prequalification"]["status"], "completed")
        self.assertEqual(result["real_planning_status"], "solved")


if __name__ == "__main__":
    unittest.main()
