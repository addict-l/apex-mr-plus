"""Real backend protocol exercised with geometry doubles, never a simulator."""
from copy import deepcopy
from pathlib import Path
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact, parse_fact
from skillnet.pipeline import run
from skillnet.planning.taskgraph import build_task_graph
from simbridge.backend import CoppeliaBackend
from simbridge.compiler import MotionCompiler
from tests.test_v13_compiler import fixture, sequential


def backend_fixture(name="s1_parallel_place", **options):
    engine, geometry, compiler = fixture(name)
    backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames, **options)
    return engine, geometry, backend


class BackendStatusTests(unittest.TestCase):
    def test_proven_ik_infeasibility_is_invalid_with_original_evidence(self):
        engine, geometry, backend = backend_fixture()
        geometry.ik_failure = {"ok": False, "infeasible": True,
                               "problem": {"reason": "outside certified workspace", "arm": "A"}}
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.INVALID)
        self.assertIn("outside certified workspace", result.detail)
        self.assertEqual(backend.validation_records[-1]["failure_evidence"], geometry.ik_failure)

    def test_finite_ik_seed_exhaustion_is_unknown(self):
        engine, geometry, backend = backend_fixture(ik_seeds=3)
        geometry.ik_failure = {"ok": False, "solutions": []}
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(sum(row[0] == "ik" for row in geometry.calls), 3)

    def test_two_stage_rrt_timeout_is_unknown(self):
        engine, geometry, backend = backend_fixture()
        geometry.plan_failure = {"ok": False, "error": "No exact collision-free solution within budget"}
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(sum(row[0] == "plan" for row in geometry.calls), 2)
        attempts = backend.validation_records[-1]["failure_evidence"]["attempts"]
        self.assertEqual([row["stage"] for row in attempts], ["local_box", "global"])

    def test_collision_is_invalid_with_pair_and_rollback(self):
        engine, geometry, backend = backend_fixture()
        geometry.collision_failure = {"ok": False, "problem": {
            "reason": "block_robot_collision", "pair": ["red", "A_link"]}}
        before = deepcopy(backend.compiler.q)
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.INVALID)
        self.assertIn("A_link", result.detail)
        self.assertEqual(backend.compiler.q, before)
        self.assertFalse(backend.compiler.tasks)

    def test_failure_starts_next_validation_at_home_without_losing_valid_cache(self):
        engine, geometry, backend = backend_fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        self.assertEqual(backend.validate(steps[0].action, steps[0].state).status, Status.VALID)
        geometry.plan_failure = {"ok": False, "error": "Invalid goal", "problem": {"reason": "collision"}}
        self.assertEqual(backend.validate(steps[1].action, steps[1].state).status, Status.INVALID)
        geometry.plan_failure = None
        calls = len(geometry.calls)
        self.assertEqual(backend.validate(steps[0].action, steps[0].state).status, Status.VALID)
        self.assertEqual(len(geometry.calls), calls)
        self.assertEqual(len(backend.compiler.tasks), 4)
        self.assertGreater(backend.compiler.cache_hits, 0)

    def test_parallel_start_home_facts_do_not_reset_previous_arm_geometry(self):
        engine, _, backend = backend_fixture()
        first = engine.parse_action("pick(A, red, stock_l1)")
        second = engine.parse_action("pick(B, blue, stock_r1)")
        self.assertEqual(backend.validate(first, engine.initial).status, Status.VALID)
        q_after_first = list(backend.compiler.q["A"])
        parallel_state = engine.start(first, engine.initial)
        self.assertEqual(backend.validate(second, parallel_state).status, Status.VALID)
        self.assertEqual(backend.compiler.q["A"], q_after_first)
        self.assertEqual(set(backend.compiler.held), {"A", "B"})


class QualificationAndEvaluationTests(unittest.TestCase):
    def test_reach_checks_grasp_and_approach_without_advancing_world(self):
        _, geometry, backend = backend_fixture()
        snapshot = deepcopy((backend.compiler.q, backend.compiler.placed, backend.compiler.held))
        self.assertEqual(backend.qualify(parse_fact("reach(A, stock_l1)")), Status.VALID)
        self.assertEqual(sum(row[0] == "ik" for row in geometry.calls), 2)
        self.assertEqual((backend.compiler.q, backend.compiler.placed, backend.compiler.held), snapshot)

    def test_liftable_uses_catalog_mass_and_configured_arm_limit(self):
        engine, _, compiler = fixture()
        compiler.cfg["payload_limits_kg"] = {"A": .1, "B": 1.}
        backend = CoppeliaBackend(compiler.planner, compiler.cfg, compiler.frames)
        self.assertEqual(backend.qualify(parse_fact("liftable(A, red)")), Status.INVALID)
        self.assertEqual(backend.qualify(parse_fact("liftable(B, red)")), Status.VALID)

    def test_evaluate_records_nominal_duration_without_changing_reference_or_search_cost(self):
        engine, _, backend = backend_fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        for step in steps:
            self.assertEqual(backend.validate(step.action, step.state).status, Status.VALID)
        before = deepcopy((backend.compiler.q, backend.compiler.held, backend.compiler.placed))
        graph = build_task_graph(engine, steps)
        duration = backend.evaluate(steps, graph)
        self.assertGreater(duration, 0.)
        self.assertEqual((backend.compiler.q, backend.compiler.held, backend.compiler.placed), before)
        self.assertIsNone(engine.cost)


class PathCacheTests(unittest.TestCase):
    def test_final_compile_reuses_validated_paths_and_certificates(self):
        engine, geometry, backend = backend_fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        for step in steps:
            backend.validate(step.action, step.state)
        calls = len(geometry.calls)
        tasks = backend.compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len(geometry.calls), calls)
        self.assertTrue(any(row["query"] == "path_cache" for row in backend.compiler.evidence))
        self.assertTrue(backend.compiler.path_cache)

    def test_other_arm_pose_change_invalidates_cached_action(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, engine.initial)
        backend.reset()
        backend.compiler.q["B"][0] += .1
        calls = len(geometry.calls)
        backend.validate(action, engine.initial)
        self.assertGreater(len(geometry.calls), calls)

    def test_placed_object_change_invalidates_cached_action(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, engine.initial)
        backend.reset()
        state = set(engine.initial)
        state.remove(Fact("at", ("blue", "stock_r1")))
        state.add(Fact("at", ("blue", "board_r1")))
        calls = len(geometry.calls)
        backend.validate(action, frozenset(state))
        self.assertGreater(len(geometry.calls), calls)
        self.assertEqual(backend.compiler.placed["blue"], backend.frames.block_pose("blue", "board_r1"))

    def test_payload_binding_change_invalidates_cache_even_with_same_symbolic_state(self):
        engine, geometry, original = fixture()
        original.cfg["initial_payloads"] = {"B": original.frames.payload("blue", "B")}
        backend = CoppeliaBackend(geometry, original.cfg, original.frames)
        state = set(engine.initial)
        state.remove(Fact("at", ("blue", "stock_r1")))
        state.discard(Fact("free", ("B",)))
        state.add(Fact("held", ("B", "blue")))
        state = frozenset(state)
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, state)
        backend.reset()
        backend.compiler.held["B"]["offset"][0] += .02
        calls = len(geometry.calls)
        backend.validate(action, state)
        self.assertGreater(len(geometry.calls), calls)

    def test_contact_pair_change_invalidates_cache(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, engine.initial)
        backend.reset()
        backend.cfg["pads"]["stock_l1"] = "A_different_bearing_pad"
        calls = len(geometry.calls)
        backend.validate(action, engine.initial)
        self.assertGreater(len(geometry.calls), calls)

    def test_symbolic_mode_change_invalidates_cache_at_same_geometry(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, engine.initial)
        backend.reset()
        state = engine.initial - {Fact("free", ("B",))}
        calls = len(geometry.calls)
        backend.validate(action, state)
        self.assertGreater(len(geometry.calls), calls)

    def test_cached_paths_are_not_reused_by_a_different_planner_scene(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        backend.validate(action, engine.initial)
        _, other_geometry, _ = fixture()
        compiler = MotionCompiler(other_geometry, backend.cfg, backend.frames,
                                  path_cache=backend.compiler.path_cache)
        compiler.compile_action(action, engine.initial)
        self.assertTrue(other_geometry.calls)
        self.assertEqual(compiler.cache_hits, 0)

    def test_certificate_evidence_is_kept_with_each_cache_entry(self):
        engine, _, backend = backend_fixture()
        backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        for entry in backend.compiler.path_cache.values():
            self.assertTrue(entry["certificate"])
            self.assertTrue(entry["path"])
            self.assertEqual(set(entry["context"]["start_q"]), {"A", "B"})
        self.assertTrue(any(record["query"] == "RRTConnect"
                            for entry in backend.compiler.path_cache.values()
                            for record in entry["certificate"]))


class PipelineFeedbackTests(unittest.TestCase):
    def test_all_valid_capability_table_is_cut_and_replanned_using_geometry(self):
        engine, geometry, compiler = fixture("s1_reach_handover")
        problem = engine.problem
        problem.proposal = None
        for action in engine.actions.values():
            for fact in action.capability:
                problem.oracle.update(fact, Status.VALID, "mock misleading capability table")
        solve = geometry.solve_ik

        def workspace(arm, position, quaternion, **kwargs):
            if (arm == "B" and position[0] < -.1) or (arm == "A" and position[0] > .1):
                return {"ok": False, "infeasible": True,
                        "problem": {"reason": "mock certified exclusive workspace", "arm": arm}}
            return solve(arm, position, quaternion, **kwargs)

        geometry.solve_ik = workspace
        backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames)
        result = run(problem, backend=backend, max_rounds=6)
        self.assertEqual(result.status, "solved", result.feedback)
        self.assertTrue(any("运动失败" in line and "重新搜索" in line for line in result.feedback))
        names = [step.action.name for step in result.schedule.steps]
        self.assertTrue("handover" in names or len(names) >= 4)
        self.assertTrue(any(record["status"] == "INVALID" for record in backend.validation_records))
        self.assertTrue(all(motion.status == Status.VALID for motion in result.motion))


if __name__ == "__main__":
    unittest.main()
