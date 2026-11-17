"""Bounded qualification persistence and CLI contracts with no simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from simbridge import qualifier
from simbridge.taskio import task_data_to_toml, write_task_data


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self):
        self.compiler = SimpleNamespace(deadline=None)
        self.fatal_error = None
        self.qualification_records = []
        self.calls = []

    def qualify(self, fact):
        self.calls.append(fact)
        status = (Status.INVALID if fact.args == ("A", "stock_l1") else
                  Status.UNKNOWN if fact.args == ("B", "board_l1") else Status.VALID)
        self.qualification_records.append({"fact": str(fact), "status": status.value,
            "evidence": [{"query": "IK", "answer": {"ok": status == Status.VALID}}],
            "failure_evidence": {"proven_infeasible": True} if status == Status.INVALID else None})
        return status


class QualifierTests(unittest.TestCase):
    def setUp(self):
        qualifier.VERSION_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="mock_qualification_", dir=qualifier.VERSION_ROOT)
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.data = tomllib.loads((ROOT / "tasks/s1_parallel_place.toml").read_text(encoding="utf-8-sig"))
        self.data["name"] = "qualifier_fixture"
        self.data["metadata"] = {"nested": {"text": "保留原字段\n逗号,引号\"", "numbers": [1, 2.5]}}
        self.task = self.directory / "input.toml"
        write_task_data(self.task, self.data)
        self.output = self.directory / "qualified"
        self.backend = FakeBackend()
        self.owner = MagicMock()
        self.owner.process.pid = 12345
        self.owner.process.poll.return_value = 0
        self.owner.install = ROOT
        self.owner.connect.return_value = (object(), object())

    def mocks(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(qualifier, "_environment", return_value={"port": 23210}))
        stack.enter_context(patch.object(qualifier, "_simulator", return_value=self.owner))
        self.build = stack.enter_context(patch.object(qualifier, "_build", return_value={}))
        planner = stack.enter_context(patch.object(qualifier, "_planner"))
        planner.return_value.__enter__.return_value = object()
        stack.enter_context(patch.object(qualifier, "_backend", return_value=self.backend))
        return stack

    def test_all_combinations_checked_even_when_input_explicitly_valid_or_invalid(self):
        source = self.task.read_bytes()
        with self.mocks():
            result = qualifier.qualify_task(self.task, out=self.output)
        expected = {Fact("reach", (arm, row["id"])) for arm in self.data["arms"]
                    for row in (*self.data["location"], *self.data.get("slot", []))}
        self.assertEqual(set(self.backend.calls), expected)
        self.assertEqual(result["total"], len(expected))
        self.assertEqual(result["checked"], len(expected))
        self.assertEqual(result["status"], "completed")
        self.build.assert_called_once()
        self.owner.close.assert_called_once()
        self.assertEqual(self.task.read_bytes(), source)
        qualified = tomllib.loads(Path(result["qualified_task"]).read_text(encoding="utf-8"))
        self.assertEqual(qualified["capability_source"], "ik_qualified")
        original, copy = deepcopy(self.data), deepcopy(qualified)
        original["capabilities"].pop("reach")
        copy["capabilities"].pop("reach")
        copy.pop("capability_source")
        self.assertEqual(copy, original)
        problem = load_problem(result["qualified_task"])
        self.assertEqual(problem.oracle.check(Fact("reach", ("A", "stock_l1"))).status, Status.INVALID)
        self.assertEqual(problem.oracle.check(Fact("reach", ("B", "board_l1"))).status, Status.UNKNOWN)
        self.assertIsNone(self.backend.compiler.deadline)

    def test_all_unknown_snapshot_and_log_exist_before_simulator_start(self):
        def check_snapshot(**kwargs):
            root = self.output / self.data["name"]
            problem = load_problem(root / (self.data["name"]+".qualified.toml"))
            self.assertTrue(all(problem.oracle.check(Fact("reach", (arm, row["id"]))).status == Status.UNKNOWN
                                for arm in self.data["arms"] for row in self.data["location"]))
            raw = tomllib.loads((root / (self.data["name"]+".qualified.toml")).read_text(encoding="utf-8"))
            self.assertNotIn("capability_source", raw)
            self.assertEqual(json.loads((root / "qualify_log.json").read_text(encoding="utf-8"))["checked"], 0)
        self.owner.start.side_effect = check_snapshot
        with self.mocks():
            qualifier.qualify_task(self.task, out=self.output)

    def test_expired_deadline_preserves_every_unchecked_fact_unknown_without_starting(self):
        with self.mocks(), patch.object(qualifier, "_simulator") as simulator:
            result = qualifier.qualify_task(self.task, out=self.output, deadline=0.)
        simulator.assert_not_called()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["checked"], 0)
        self.assertEqual(result["counts"]["UNKNOWN"], result["total"])
        self.assertTrue(Path(result["qualified_task"]).exists())

    def test_partial_task_output_is_adopted_without_another_experimental_budget(self):
        with self.mocks():
            first = qualifier.qualify_task(self.task, out=self.output, deadline=0.)
        with patch.object(qualifier, "_simulator") as simulator:
            second = qualifier.qualify_task(self.task, out=self.output)
        simulator.assert_not_called()
        self.assertEqual(first, second)

    def test_running_log_recovery_rebuilds_lagging_task_projection_without_queries(self):
        with self.mocks():
            result = qualifier.qualify_task(self.task, out=self.output, deadline=0.)
        log = json.loads(Path(result["log_file"]).read_text(encoding="utf-8"))
        log.update(status="running", checked=1)
        log["updates"][0]["status"] = "VALID"
        Path(result["log_file"]).write_text(json.dumps(log), encoding="utf-8")
        with patch.object(qualifier, "_simulator") as simulator:
            recovered = qualifier.qualify_task(self.task, out=self.output)
        simulator.assert_not_called()
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["status"], "partial")
        problem = load_problem(recovered["qualified_task"])
        self.assertEqual(problem.oracle.check(Fact("reach", ("A", "stock_l1"))).status, Status.VALID)

    def test_different_input_and_existing_output_are_never_overwritten(self):
        with self.mocks():
            result = qualifier.qualify_task(self.task, out=self.output, deadline=0.)
        before = Path(result["qualified_task"]).read_bytes()
        self.task.write_text(self.task.read_text(encoding="utf-8")+'extra = "changed"\n', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            qualifier.qualify_task(self.task, out=self.output)
        self.assertEqual(before, Path(result["qualified_task"]).read_bytes())

    def test_timeout_after_a_query_does_not_promote_its_late_answer(self):
        problem = load_problem(self.task)
        clock = [100.]
        original = self.backend.qualify
        def late(fact):
            answer = original(fact)
            clock[0] = 161.
            return answer
        self.backend.qualify = late
        snapshots = []
        with patch.object(qualifier.time, "monotonic", side_effect=lambda: clock[0]):
            result = qualifier.qualify_workspace(problem, self.backend, deadline=160., checkpoint=snapshots.append)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["counts"]["UNKNOWN"], result["total"])
        self.assertTrue(result["records"][0]["backend_record"]["deadline_exhausted"])
        self.assertEqual(snapshots[0]["checked"], 0)

    def test_sixty_second_limit_and_project_version_outputs_are_enforced(self):
        with self.assertRaises(ValueError):
            qualifier.qualify_task(self.task, out=self.output, timeout_wall_s=60.1)
        with self.assertRaises(ValueError):
            qualifier.qualify_task(self.task, out=ROOT / "out")
        with self.assertRaises(ValueError):
            qualifier.qualify_task(self.task, out=self.output, port=23010)

    def test_cli_dispatch_uses_public_qualification_entry_without_real_work(self):
        from simbridge.__main__ import main
        with patch.object(qualifier, "qualify_tasks", return_value={"status": "completed", "results": []}) as run:
            self.assertEqual(main(["qualify", "--tasks", str(self.task), "--out", str(self.output)]), 0)
        run.assert_called_once_with([str(self.task)], out=str(self.output))


if __name__ == "__main__":
    unittest.main()
