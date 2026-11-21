"""Complete-action candidate contracts with the existing Cartesian fake.

The fake has two distinct joint branches for one pose. These tests establish
control flow, rollback and evidence scope; they make no robot-geometry claim.
Qualification hints use the first-version, catalog-only adapter schema below.
"""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from scipy.spatial.transform import Rotation

from skillnet.capability.oracle import Status
from simbridge.compiler import CompileFailure, MotionCompiler
from simbridge.primitives import expand
from tests.test_idle_parking import blocked_until_idle_home, tower_prefix
from tests.test_v13_compiler import GeometryFake, fixture


class FakeClock:
    def __init__(self):
        self.now = 100.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CandidatePlanner(GeometryFake):
    """A's second branch stores x+1 in q1; FK decodes the same TCP pose.

    Normal task x values are negative for A. The marker stays inside joint
    bounds, and both decoded branches satisfy the existing dense line check.
    """
    def __init__(self, approach, *, clock=None, fail_first_rrt=False,
                 fail_all_rrt=False, fail_first_lift=False, fail_all_lifts=False):
        super().__init__()
        self.approach = list(approach)
        self.clock = clock
        self.fail_first_rrt = fail_first_rrt
        self.fail_all_rrt = fail_all_rrt
        self.fail_first_lift = fail_first_lift
        self.fail_all_lifts = fail_all_lifts
        self.compiler = None
        self.transit_seed_calls = 0
        self.plan_receipts = []
        self.lift_rejections = 0
        self.invalid_hint_q = None

    @staticmethod
    def second_branch(q):
        return q[0] > .2

    def solve_ik(self, arm, position, quaternion, **kwargs):
        answer = super().solve_ik(arm, position, quaternion, **kwargs)
        if arm != "A" or not answer.get("ok"):
            return answer
        is_approach = (math.dist(position, self.approach[:3]) < 1e-9
                       and not kwargs.get("payload"))
        if is_approach:
            self.transit_seed_calls += 1
            alternative = self.transit_seed_calls % 2 == 0
        else:
            alternative = self.second_branch(kwargs.get("options", {}).get("seed", [0.]*6))
        if alternative:
            answer["q"][0] += 1.
        return answer

    def fk(self, arm, q):
        self.calls.append(("fk", arm, {"q": list(q)}))
        canonical = list(q)
        if arm == "A" and self.second_branch(q):
            canonical[0] -= 1.
        return super().fk(arm, canonical)

    def plan(self, arm, start, goal, **kwargs):
        answer = super().plan(arm, start, goal, **kwargs)
        receipt = {"arm": arm, "start": list(start), "goal": list(goal),
                   "solve_time": kwargs["solve_time"],
                   "snapshot": deepcopy(self.compiler._snapshot())}
        self.plan_receipts.append(receipt)
        failed = arm == "A" and (self.fail_all_rrt or
                                self.fail_first_rrt and not self.second_branch(goal))
        if self.clock is not None:
            # Failed searches consume their allotted solver slice. Success
            # returns early, as a real bounded solver may also do.
            self.clock.advance(kwargs["solve_time"] if failed else min(.01, kwargs["solve_time"]))
        if failed:
            return {"ok": False, "error": "No exact collision-free solution within budget"}
        return answer

    def validate_joint(self, arm, q, **kwargs):
        answer = super().validate_joint(arm, q, **kwargs)
        hinted = self.invalid_hint_q is not None and math.dist(q, self.invalid_hint_q) < 1e-8
        lift = arm == "A" and (kwargs.get("payload") or {}).get("name") == "red"
        blocked = lift and (self.fail_all_lifts or
                           self.fail_first_lift and not self.second_branch(q))
        if hinted or blocked:
            self.lift_rejections += int(bool(blocked))
            return {"ok": False, "problem": {"reason": "block_robot_collision:red:A",
                    "pair": ["red", "A_link"]}}
        return answer


def candidate_fixture(**planner_options):
    engine, _, compiler = fixture(ik_seeds=2)
    target = compiler.frames.grasp_pose("red", "stock_l1", "A")
    planner = CandidatePlanner(compiler.frames.approach_pose(target), **planner_options)
    compiler.planner = planner
    planner.compiler = compiler
    return engine, planner, compiler


def hint_for(compiler, planner):
    pose = planner.approach
    q = list(pose[:3])+Rotation.from_quat(pose[3:]).as_euler("xyz").tolist()
    q[0] += 1.
    return {"arm": "A", "part": "red", "block_type": "2x4", "place": "stock_l1",
            "pose_mode": "catalog", "grasp_strategy": "catalog_top",
            "pose": list(pose), "q": q}


class CompleteActionCandidatesTests(unittest.TestCase):
    def test_constructor_accepts_finite_limit_and_optional_qualification_hints(self):
        _, _, original = fixture()
        compiler = MotionCompiler(original.planner, original.cfg, original.frames,
                                  action_candidate_limit=4, qualification_hints=[])
        self.assertEqual(compiler.action_candidate_limit, 4)
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MotionCompiler(original.planner, original.cfg, original.frames,
                               action_candidate_limit=limit)

    def test_alternative_valid_transit_endpoint_connects_when_first_rrt_fails(self):
        engine, planner, compiler = candidate_fixture(fail_first_rrt=True)
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        attempts = [row for row in planner.plan_receipts if row["arm"] == "A"]
        self.assertTrue(any(not planner.second_branch(row["goal"]) for row in attempts))
        self.assertTrue(any(planner.second_branch(row["goal"]) for row in attempts))
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        self.assertEqual(len(tasks), 4)
        self.assertEqual(compiler.held["A"]["name"], "red")

    def test_failed_lift_restores_entire_action_checkpoint_before_alternative(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        blue = engine.parse_action("pick(B, blue, stock_r1)")
        compiler.compile_action(blue, engine.initial, step_id="blue_prefix")
        state = engine.execute(blue, engine.initial)
        before = compiler._snapshot()
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), state,
                                        step_id="red_candidate")
        self.assertGreater(planner.lift_rejections, 0, "the rejected branch must reach the held-object lift")
        alternatives = [row for row in planner.plan_receipts
                        if row["arm"] == "A" and planner.second_branch(row["goal"])]
        self.assertTrue(alternatives)
        # q, held, placed, physical tasks, last-arm dependencies and touched
        # objects must all return to the same action start. Symbolic context
        # is deliberately the new action's state rather than the old prefix.
        self.assertEqual(alternatives[0]["snapshot"][:6], before[:6])
        self.assertEqual(alternatives[0]["snapshot"][6], tuple(sorted(map(str, state))))
        self.assertEqual(compiler.tasks[:len(before[3])], before[3])
        self.assertEqual(len(tasks), 4, "failed-candidate grasp/lift tasks must not leak")
        self.assertEqual(len({row["id"] for row in compiler.tasks}), len(compiler.tasks))
        self.assertEqual(compiler.q["B"], before[0]["B"])
        self.assertEqual(set(compiler.held), {"A", "B"})
        self.assertNotIn("red", compiler.placed)

    def test_finite_candidates_exhausted_restore_outer_checkpoint_and_stay_unknown(self):
        engine, planner, compiler = candidate_fixture(fail_all_lifts=True)
        before = compiler._snapshot()
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN,
                         "collisions of finite sampled branches are not global infeasibility")
        self.assertGreaterEqual(planner.lift_rejections, 2)
        self.assertEqual(compiler._snapshot(), before)
        # No identical geometric endpoint should be rerun after the same
        # complete candidate already failed in this unchanged context.
        successful_endpoints = [tuple(row["goal"]) for row in planner.plan_receipts if row["arm"] == "A"]
        self.assertEqual(len(successful_endpoints), len(set(successful_endpoints)))

    def test_endpoint_attempts_share_original_ten_second_rrt_budget(self):
        clock = FakeClock()
        engine, planner, compiler = candidate_fixture(clock=clock, fail_first_rrt=True)
        with patch("simbridge.compiler.time.monotonic", clock.monotonic):
            tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        attempts = [row for row in planner.plan_receipts if row["arm"] == "A"]
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        self.assertLessEqual(sum(row["solve_time"] for row in attempts), 10.+1e-8)
        self.assertLessEqual(clock.now-100., 10.+1e-8)
        self.assertGreater(len({tuple(row["goal"]) for row in attempts}), 1)

    def test_task_deadline_covers_all_candidates_without_reset(self):
        clock = FakeClock()
        engine, planner, compiler = candidate_fixture(clock=clock, fail_all_rrt=True)
        compiler.deadline = 100.6
        before = compiler._snapshot()
        with patch("simbridge.compiler.time.monotonic", clock.monotonic):
            with self.assertRaises(CompileFailure) as failed:
                compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertLessEqual(sum(row["solve_time"] for row in planner.plan_receipts), .6+1e-8)
        self.assertLessEqual(clock.now, compiler.deadline+1e-8)
        self.assertEqual(compiler._snapshot(), before)

    def test_small_endpoint_slices_still_try_local_then_global(self):
        clock = FakeClock()
        engine, planner, compiler = candidate_fixture(clock=clock, fail_all_rrt=True)
        compiler.ik_seeds = 4
        original = planner.solve_ik
        calls = 0

        def four_solutions(arm, position, quaternion, **kwargs):
            nonlocal calls
            answer = original(arm, position, quaternion, **kwargs)
            calls += 1
            answer["q"][0] = position[0]+calls-1
            return answer

        planner.solve_ik = four_solutions
        with patch("simbridge.compiler.time.monotonic", clock.monotonic):
            with self.assertRaises(CompileFailure):
                compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        by_endpoint = {}
        for row in compiler.evidence:
            if row.get("query") == "RRTConnect":
                by_endpoint.setdefault(tuple(row["endpoint_q"]), []).append(row["search_stage"])
        self.assertEqual(len(by_endpoint), 4)
        self.assertTrue(all(stages == ["local_box", "global"] for stages in by_endpoint.values()))
        self.assertLessEqual(sum(row["solve_time"] for row in planner.plan_receipts), 10.+1e-8)


class QualificationHintCandidatesTests(unittest.TestCase):
    def test_grasp_stage_hint_seeds_a_coherent_approach_without_using_grasp_as_endpoint(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        hint = hint_for(compiler, planner)
        hint["stage"] = "grasp"
        hint["pose"] = compiler.frames.grasp_pose("red", "stock_l1", "A")
        hint["q"][2] = hint["pose"][2]
        compiler.qualification_hints = [hint]
        original = planner.solve_ik

        def seed_sensitive(arm, position, quaternion, **kwargs):
            answer = original(arm, position, quaternion, **kwargs)
            if (arm == "A" and math.dist(position, planner.approach[:3]) < 1e-9
                    and not kwargs.get("payload")):
                seed = kwargs["options"]["seed"]
                coherent = math.dist(seed, hint["q"]) < 1e-8
                answer["q"][0] = position[0]+int(coherent)
            return answer

        planner.solve_ik = seed_sensitive
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        self.assertNotEqual(tasks[0]["q_path"][-1], hint["q"])
        self.assertAlmostEqual(planner.fk("A", tasks[0]["q_path"][-1])["pose"][2], planner.approach[2])
        transit_seeds = [row["answer"]["seed"] for row in compiler.evidence
                         if row.get("query") == "IK_seed" and row["answer"].get("request_kind") == "transit"]
        self.assertEqual(len(transit_seeds), compiler.ik_seeds)
        self.assertEqual(transit_seeds[0], hint["q"])

    def test_matching_catalog_hint_is_revalidated_before_current_endpoint_ranking(self):
        engine, planner, compiler = candidate_fixture()
        blue = engine.parse_action("pick(B, blue, stock_r1)")
        compiler.compile_action(blue, engine.initial)
        state = engine.execute(blue, engine.initial)
        hint = hint_for(compiler, planner)
        compiler.qualification_hints = [deepcopy(hint)]
        other_q = list(compiler.q["B"])
        first = len(planner.calls)
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), state)
        calls = planner.calls[first:]
        checked = [(i, row[2]) for i, row in enumerate(calls)
                   if row[:2] == ("validate", "A") and math.dist(row[2]["q"], hint["q"]) < 1e-8]
        self.assertTrue(checked, "a qualification solution must be rechecked, never trusted as a current certificate")
        index, collision = checked[0]
        first_plan = next(i for i, row in enumerate(calls) if row[:2] == ("plan", "A"))
        self.assertLess(index, first_plan)
        self.assertEqual(collision["options"]["other_q"], other_q)
        self.assertEqual(collision["options"]["other_payload"]["name"], "blue")
        self.assertNotIn("blue", collision["placed"])
        self.assertTrue(any(row[:2] == ("fk", "A") and math.dist(row[2]["q"], hint["q"]) < 1e-8
                            for row in calls[:first_plan]))
        self.assertLess(math.dist(tasks[0]["q_path"][-1], tasks[0]["q_path"][0]),
                        math.dist(hint["q"], tasks[0]["q_path"][0]),
                        "point qualification does not outrank closer current-context IK")

    def test_now_colliding_hint_is_discarded_without_rejecting_other_candidates(self):
        engine, planner, compiler = candidate_fixture()
        hint = hint_for(compiler, planner)
        compiler.qualification_hints = [hint]
        planner.invalid_hint_q = hint["q"]
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertTrue(any(row[:2] == ("validate", "A") and math.dist(row[2]["q"], hint["q"]) < 1e-8
                            for row in planner.calls))
        self.assertNotEqual(tasks[0]["q_path"][-1], hint["q"])
        self.assertEqual(compiler.held["A"]["name"], "red")

    def test_mismatched_strategy_type_or_pose_hint_does_not_change_catalog_action(self):
        for field, value in (("pose_mode", "side_45"), ("grasp_strategy", "side_opposed"),
                             ("block_type", "1x2"), ("arm", "B"), ("part", "blue"),
                             ("place", "board_l1"),
                             ("pose", [0., 0., 2., 0., 0., 0., 1.])):
            with self.subTest(field=field):
                engine, planner, compiler = candidate_fixture()
                hint = hint_for(compiler, planner)
                hint[field] = value
                compiler.qualification_hints = [hint]
                tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
                self.assertFalse(planner.second_branch(tasks[0]["q_path"][-1]))
                self.assertTrue(all(row["metadata"]["grasp_strategy"] == "catalog_top" for row in tasks))

    def test_missing_qualification_hints_keeps_normal_catalog_pick_compatible(self):
        engine, planner, compiler = candidate_fixture()
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(compiler.held["A"]["name"], "red")
        self.assertTrue(all(row["metadata"]["grasp_strategy"] == "catalog_top" for row in tasks))
        self.assertFalse(planner.second_branch(tasks[0]["q_path"][-1]))

    def test_winning_complete_action_policy_reuses_only_its_validated_endpoints(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        action = engine.parse_action("pick(A, red, stock_l1)")
        first_tasks = compiler.compile_action(action, engine.initial)
        self.assertTrue(planner.second_branch(first_tasks[0]["q_path"][-1]))
        compiler.reset()
        before = len(planner.calls)
        second_tasks = compiler.compile_action(action, engine.initial)
        self.assertEqual(second_tasks, first_tasks)
        self.assertEqual(len(planner.calls), before,
                         "the winning policy must not retry the failed branch during later compilation")


class IdleParkingReturnTests(unittest.TestCase):
    def test_rescued_action_returns_idle_arm_to_exact_symbolic_reference_with_dependencies(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        initial_q = deepcopy(compiler.q)
        blocked_until_idle_home(geometry, compiler)
        first = len(geometry.calls)
        tasks = compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        returns = [row for row in tasks if row["metadata"].get("idle_arm_return")]
        self.assertEqual(len(returns), 1)
        returning = returns[0]
        self.assertEqual(returning["arm"], "A")
        self.assertTrue(returning["metadata"].get("restores_symbolic_reference"))
        self.assertEqual(returning["q_path"][0], compiler.cfg["arms"]["A"]["home_q"])
        self.assertEqual(returning["q_path"][-1], initial_q["A"])
        self.assertEqual(compiler.q["A"], initial_q["A"])
        self.assertGreater(returning["duration"], 0.)
        rescued = [row for row in tasks if row["arm"] == "B"]
        parents = {dependency for row in rescued for dependency in row["dependencies"]}
        leaves = {row["id"] for row in rescued if row["id"] not in parents}
        parked = next(row for row in tasks if row["metadata"].get("idle_arm_parking"))
        self.assertTrue(leaves | {parked["id"]} <= set(returning["dependencies"]))
        goal_calls = [row[2] for row in geometry.calls[first:]
                      if row[:2] == ("plan", "A") and row[2]["goal"] == initial_q["A"]]
        self.assertEqual(len(goal_calls), 1)
        query = goal_calls[0]
        self.assertEqual(query["placed"]["b3"], compiler.frames.block_pose("b3", "t3"))
        self.assertFalse(query["options"]["other_payload"])
        self.assertEqual(query["other_q"], compiler.q["B"])
        self.assertTrue(any(row[:2] == ("validate", "A") and row[2]["q"] == initial_q["A"]
                            and row[2]["placed"].get("b3") == compiler.frames.block_pose("b3", "t3")
                            for row in geometry.calls[first:]))

    def test_return_goal_collision_is_unknown_and_rolls_back_the_whole_rescued_action(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        before = compiler._snapshot()
        blocked_until_idle_home(geometry, compiler)
        original = geometry.validate_joint

        def collision_after_install(arm, q, **kwargs):
            if (arm == "A" and math.dist(q, before[0]["A"]) < 1e-8
                    and kwargs.get("placed", {}).get("b3") == compiler.frames.block_pose("b3", "t3")):
                geometry.calls.append(("validate", arm, {"q": list(q), **deepcopy(kwargs)}))
                return {"ok": False, "problem": {"reason": "block_robot_collision:b3:A",
                        "pair": ["A_link", "b3"]}}
            return original(arm, q, **kwargs)

        geometry.validate_joint = collision_after_install
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertEqual(compiler._snapshot(), before)
        self.assertEqual(failed.exception.evidence["recovery_phase"], "return")
        self.assertIn("b3", str(failed.exception.evidence))

    def test_support_lifecycle_uses_metadata_events_when_task_kind_is_transit(self):
        engine, _, compiler, state, _ = tower_prefix()
        action = engine.parse_action("insert(B, b3, t3)")
        requests = expand(action, state, compiler.frames, compiler.cfg,
                          placed=compiler.placed, payloads=compiler.held)
        compiler.tasks.append({"id": "metadata_support", "arm": "A", "kind": "transit",
                               "metadata": {"action": "support_start"}})
        self.assertIsNone(compiler._idle_other(action, state, requests))
        compiler.tasks.append({"id": "metadata_release", "arm": "A", "kind": "transit",
                               "metadata": {"action": "support_end"}})
        self.assertEqual(compiler._idle_other(action, state, requests), "A")


if __name__ == "__main__":
    unittest.main()
