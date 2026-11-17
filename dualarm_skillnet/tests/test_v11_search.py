"""v1.1 acceptance at the planner boundary, with an independent UCS oracle."""

import json
from pathlib import Path
from time import monotonic
import tomllib
import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.planning.completion import complete
from skillnet.planning.search import _search, search
from tests.test_replay import replay


BASIC_TASKS = (
    "s1_parallel_place", "s1_nearest_arm", "s1_reach_handover",
    "s1_far_handover", "s2_tower", "s2_tower_same_type",
    "s3a_platform_demo_needs_support",
)


def load_without_proposal(name):
    path = Path("tasks") / f"{name}.toml"
    problem = load_problem(path)
    problem.proposal = None
    return problem


def statistics(result, elapsed):
    return {"status": result.status, "makespan": result.makespan,
            "expanded": result.expanded, "generated": result.generated,
            "search_elapsed": result.elapsed, "wall_elapsed": elapsed}


def record(name, value):
    directory = Path("out/v1_1_acceptance")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ProposalIndependentSupportTests(unittest.TestCase):
    def test_cantilever_infers_support_without_proposal(self):
        problem = load_without_proposal("s3a_cantilever")
        began = monotonic()
        result = run(problem)
        elapsed = monotonic() - began
        record("s3a_cantilever", statistics(result.schedule, elapsed))
        self.assertIsNone(result.proposal)
        self.assertEqual(result.status, "solved", result.feedback)
        self.assertLess(elapsed, 60.0)
        names = [step.action.name for step in result.schedule.steps]
        self.assertIn("support_on", names)
        self.assertIn("support_off", names)
        counterweight = next(step for step in result.schedule.steps
                             if step.action.name == "insert" and step.action.args[-1] == "cw")
        for step in result.schedule.steps:
            if step.action.name == "support_off":
                self.assertGreaterEqual(step.start, counterweight.end)
            if step.action.name in {"insert", "detach"}:
                self.assertNotIn(step.action.args[-1], {"tp1", "tp2"})
        replay(self, Engine(problem), result.schedule)

    def test_platform_infers_build_and_cleanup_without_proposal(self):
        problem = load_without_proposal("s3a_platform")
        began = monotonic()
        result = run(problem)
        elapsed = monotonic() - began
        record("s3a_platform", statistics(result.schedule, elapsed))
        self.assertIsNone(result.proposal)
        self.assertEqual(result.status, "solved", result.feedback)
        self.assertLess(elapsed, 60.0)
        self.assertNotIn("support_on", [step.action.name for step in result.schedule.steps])
        beam = next(step for step in result.schedule.steps
                    if step.action.name == "insert" and step.action.args[-1] == "beam")
        for slot in ("tp1", "tp2"):
            inserts = [step for step in result.schedule.steps
                       if step.action.name == "insert" and step.action.args[-1] == slot]
            detaches = [step for step in result.schedule.steps
                        if step.action.name == "detach" and step.action.args[-1] == slot]
            self.assertTrue(inserts)
            self.assertTrue(detaches)
            self.assertLessEqual(inserts[0].end, beam.start)
            self.assertGreaterEqual(detaches[-1].start, beam.end)
        final = replay(self, Engine(problem), result.schedule)
        self.assertNotIn(Fact("filled", ("tp1",)), final)
        self.assertNotIn(Fact("filled", ("tp2",)), final)
        for arm in problem.objects["arm"]:
            self.assertIn(Fact("free", (arm,)), final)


class OptimalityAgainstBasicUCSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for name in BASIC_TASKS:
            engine = Engine(load_without_proposal(name))
            began = monotonic()
            basic = _search(engine)
            baseline_wall = monotonic() - began
            began = monotonic()
            revised = search(engine)
            revised_wall = monotonic() - began
            cls.results[name] = (engine, basic, revised)
            record(name, {"basic_ucs": statistics(basic, baseline_wall),
                          "search": statistics(revised, revised_wall)})

    def test_all_seven_tasks_match_unpruned_basic_ucs(self):
        for name, (engine, basic, revised) in self.results.items():
            with self.subTest(task=name):
                self.assertEqual(basic.status, "solved")
                self.assertEqual(revised.status, "solved")
                self.assertLessEqual(abs(revised.makespan - basic.makespan), 1e-6)
                replay(self, engine, revised)

    def test_same_type_tower_selects_blocks_bottom_to_top(self):
        engine, _, result = self.results["s2_tower_same_type"]
        self.assertEqual(result.status, "solved")
        slots = [step.action.args[-1] for step in result.steps if step.action.name == "insert"]
        self.assertEqual(slots, ["t1", "t2", "t3"])
        self.assertLessEqual(result.makespan, self.results["s2_tower"][2].makespan + 1e-6)
        self.assertEqual(len({row["type"] for row in engine.hooks.slots.values()}), 1)
        self.assertEqual(len({row["type"] for row in engine.hooks.blocks.values()}), 1)


class RevisedTaskInputTests(unittest.TestCase):
    def test_support_tasks_keep_original_proposals_only_in_separate_files(self):
        for name, count in (("s3a_cantilever", 8), ("s3a_platform", 14)):
            with self.subTest(task=name):
                with open(f"tasks/{name}.toml", "rb") as source:
                    base = tomllib.load(source)
                with open(f"tasks/{name}_with_proposal.toml", "rb") as source:
                    proposed = tomllib.load(source)
                self.assertNotIn("proposal", base)
                self.assertIn("proposal", proposed)
                original_proposal = proposed.pop("proposal")
                proposed["name"] = base["name"]
                self.assertEqual(proposed, base)
                self.assertEqual(len(original_proposal["steps"]), count)

    def test_both_preserved_proposals_keep_v1_completion_results(self):
        for name, count in (("s3a_cantilever", 8), ("s3a_platform", 14)):
            with self.subTest(task=name):
                problem = load_problem(f"tasks/{name}_with_proposal.toml")
                report = complete(Engine(problem), problem.proposal.steps)
                self.assertTrue(report.goal_reached)
                self.assertFalse(report.unresolved)
                self.assertEqual([entry.kind for entry in report.entries], ["保留"] * count)
                self.assertEqual([str(action) for action in report.plan], problem.proposal.steps)


if __name__ == "__main__":
    unittest.main()
