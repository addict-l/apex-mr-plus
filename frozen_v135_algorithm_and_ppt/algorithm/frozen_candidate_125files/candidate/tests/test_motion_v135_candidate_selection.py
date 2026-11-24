"""Complete-action selection contracts; deterministic geometry, no live RPC."""
from copy import deepcopy
import math
import unittest

from scipy.spatial.transform import Rotation
from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.planning.search import SearchStep
from simbridge.compiler import CompileFailure

from tests.test_motion_v135_action_candidates import candidate_fixture, hint_for
from tests.test_v13_compiler import GeometryFake, fixture


class MotionCandidateSelectionTests(unittest.TestCase):
    def test_nearer_fresh_endpoint_precedes_farther_qualification_hint(self):
        engine, planner, compiler = candidate_fixture()
        hint = hint_for(compiler, planner)
        compiler.qualification_hints = [hint]
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        endpoint = tasks[0]["q_path"][-1]
        self.assertFalse(planner.second_branch(endpoint),
                         "hint provenance must not outrank a closer same-family solution")
        self.assertLess(math.dist(endpoint, tasks[0]["q_path"][0]),
                        math.dist(hint["q"], tasks[0]["q_path"][0]))
        # Qualification remains a point observation, checked in the current
        # scene even though another endpoint wins the complete action.
        self.assertTrue(any(row.get("query") == "qualification_hint"
                            and row["answer"].get("ok") for row in compiler.evidence))

    def test_near_duplicates_do_not_hide_a_distinct_complete_action_branch(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        compiler.ik_seeds = 6
        solve = planner.solve_ik
        calls = 0

        def repeated_branch(arm, position, quaternion, **kwargs):
            nonlocal calls
            if arm == "A" and math.dist(position, planner.approach[:3]) < 1e-9 and not kwargs.get("payload"):
                calls += 1
                answer = GeometryFake.solve_ik(planner, arm, position, quaternion, **kwargs)
                answer["q"][0] += (calls-1)*1e-6 if calls < 5 else 1.
                return answer
            return solve(arm, position, quaternion, **kwargs)

        planner.solve_ik = repeated_branch
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        attempted = [row["goal"] for row in planner.plan_receipts if row["arm"] == "A"]
        self.assertEqual(len(attempted), 2,
                         "numerical duplicates must share one finite candidate slot")
        self.assertEqual(len(tasks), 4, "failed grasp/lift prefix must be rolled back")
        clusters = [row for row in compiler.evidence if row.get("query") == "IK_candidate_clusters"]
        self.assertTrue(clusters)
        self.assertGreaterEqual(clusters[0]["answer"]["suppressed_count"], 3)

    def test_same_type_representative_grasp_is_only_a_fresh_ik_seed(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        hint = hint_for(compiler, planner)
        hint.update(part="blue", stage="grasp",
                    pose=compiler.frames.grasp_pose("blue", "stock_l1", "A"))
        hint["q"][2] = hint["pose"][2]
        compiler.qualification_hints = [hint]
        original = planner.solve_ik

        def only_representative_connects(arm, position, quaternion, **kwargs):
            if arm == "A" and math.dist(position, planner.approach[:3]) < 1e-9 and not kwargs.get("payload"):
                answer = GeometryFake.solve_ik(planner, arm, position, quaternion, **kwargs)
                if math.dist(kwargs["options"]["seed"], hint["q"]) < 1e-8:
                    answer["q"][0] += 1.
                return answer
            return original(arm, position, quaternion, **kwargs)

        planner.solve_ik = only_representative_connects
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        self.assertNotEqual(tasks[0]["q_path"][-1], hint["q"], "grasp q cannot become the approach endpoint")
        self.assertEqual(compiler.held["A"]["name"], "red")
        seeds = [row["answer"] for row in compiler.evidence if row.get("query") == "IK_seed"]
        representative = next(row for row in seeds if row.get("source_part") == "blue")
        self.assertEqual(representative["requested_part"], "red")
        self.assertEqual(len(seeds), compiler.ik_seeds)
        self.assertFalse(any(row.get("query") == "qualification_hint" for row in compiler.evidence),
                         "cross-part observations must not enter the direct endpoint pool")

    def test_independent_hint_remains_a_bounded_fallback_after_closer_failures(self):
        engine, planner, compiler = candidate_fixture(fail_first_lift=True)
        compiler.ik_seeds = 8
        compiler.qualification_hints = [hint_for(compiler, planner)]
        solve = planner.solve_ik
        count = 0

        def many_near_branches(arm, position, quaternion, **kwargs):
            nonlocal count
            if arm == "A" and math.dist(position, planner.approach[:3]) < 1e-9 and not kwargs.get("payload"):
                count += 1
                answer = GeometryFake.solve_ik(planner, arm, position, quaternion, **kwargs)
                answer["q"][0] += count*.02
                return answer
            return solve(arm, position, quaternion, **kwargs)

        planner.solve_ik = many_near_branches
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertTrue(planner.second_branch(tasks[0]["q_path"][-1]))
        receipts = [row for row in compiler.evidence if row.get("query") == "IK_selection"]
        self.assertTrue(receipts[0]["answer"]["candidate_details"][3]["retained_hint_fallback"])
        self.assertLessEqual(len([r for r in planner.plan_receipts if r["arm"] == "A"]), 4)

    def test_cross_part_seed_requires_current_pose_type_and_strategy(self):
        for field, value in (("pose", [0., 0., 5., 0., 0., 0., 1.]),
                             ("block_type", "1x2"), ("arm", "B"),
                             ("place", "board_l1"), ("pose_mode", "side_45"),
                             ("grasp_strategy", "side_opposed"), ("stage", "approach")):
            with self.subTest(field=field):
                engine, planner, compiler = candidate_fixture()
                hint = hint_for(compiler, planner)
                hint.update(part="blue", stage="grasp",
                            pose=compiler.frames.grasp_pose("blue", "stock_l1", "A"))
                hint[field] = value
                compiler.qualification_hints = [hint]
                compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
                self.assertFalse(any(row.get("query") == "IK_seed"
                                     and row["answer"].get("source_part") == "blue"
                                     for row in compiler.evidence))

    def test_wrist_family_precedes_a_shorter_opposite_wrist_jump(self):
        engine, planner, compiler = candidate_fixture()
        compiler.q["A"][4] = .1
        count = 0
        def two_wrists(arm, position, quaternion, **kwargs):
            nonlocal count
            answer = GeometryFake.solve_ik(planner, arm, position, quaternion, **kwargs)
            if arm == "A" and math.dist(position, planner.approach[:3]) < 1e-9 and not kwargs.get("payload"):
                count += 1
                answer["q"][4] = -.1 if count == 1 else math.pi/2
            else:
                answer["q"][4] = kwargs["options"]["seed"][4]
            return answer
        def decoded_fk(arm, q):
            decoded = list(q)
            decoded[4] = 0.
            return GeometryFake.fk(planner, arm, decoded)
        planner.solve_ik, planner.fk = two_wrists, decoded_fk
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertAlmostEqual(tasks[0]["q_path"][-1][4], math.pi/2)
        selected = next(row["answer"] for row in compiler.evidence if row.get("query") == "IK_selection")
        self.assertGreater(selected["candidate_details"][0]["joint_distance"],
                           selected["candidate_details"][1]["joint_distance"])
        self.assertEqual(selected["candidate_details"][0]["families"]["wrist"], 1)

    def test_representative_seed_is_rejected_for_a_changed_held_binding(self):
        engine, planner, compiler = candidate_fixture()
        pick = engine.parse_action("pick(A, red, stock_l1)")
        state = engine.execute(pick, engine.initial)
        compiler.held["A"] = compiler.frames.payload("red", "A")
        compiler.held["A"]["offset"][0] += .03
        hint = hint_for(compiler, planner)
        hint.update(part="blue", stage="grasp", place="board_l1",
                    pose=compiler.frames.grasp_pose("blue", "board_l1", "A"))
        compiler.qualification_hints = [hint]
        compiler.compile_action(engine.parse_action("place(A, red, board_l1)"), state)
        self.assertFalse(any(row.get("query") == "IK_seed"
                             and row["answer"].get("source_part") == "blue"
                             for row in compiler.evidence))


class SupportWithdrawalCandidatesTests(unittest.TestCase):
    def test_colliding_optional_lower_retries_release_and_horizontal_retreat_transactionally(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        target = compiler.frames.support_pose("beam", "A", arm_config=compiler.cfg["arms"])
        compiler.q["A"] = target[:3]+Rotation.from_quat(target[3:]).as_euler("xyz").tolist()
        state = frozenset([*engine.initial, Fact("supporting", ("A", "beam"))])
        validate = geometry.validate_joint
        rejected = []

        def lower_hits_table(arm, q, **kwargs):
            if arm == "A" and q[2] < target[2]-1e-6:
                rejected.append(list(q))
                return {"ok": False, "problem": {"reason": "table_collision:A",
                                                "pair": ["A_SupportShoe", "Workbench"]}}
            return validate(arm, q, **kwargs)

        geometry.validate_joint = lower_hits_table
        tasks = compiler.compile_action(engine.parse_action("support_off(A, beam)"), state)
        self.assertTrue(rejected, "the unchanged default lower route must be tried first")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(sum(row["metadata"].get("action") == "support_end" for row in tasks), 1)
        self.assertIn(tasks[0]["id"], tasks[1]["dependencies"])
        self.assertAlmostEqual(tasks[1]["q_path"][-1][2], target[2])
        self.assertTrue(all(row["metadata"].get("support_exit_strategy") == "horizontal_only-v1"
                            for row in tasks))
        # A later compilation may reuse the certified alternative; it must
        # not silently reintroduce the failed lower primitive.
        winning = [entry for entry in compiler.path_cache.values()
                   if entry.get("complete_action_policies")]
        self.assertTrue(any(policy.get("support_exit_strategy") == "horizontal_only-v1"
                            for entry in winning for policy in entry["complete_action_policies"].values()))
        # Matching context replays the selected strategy, without restoring
        # the now-rejected extra lower or changing the release event.
        compiler.q["A"] = target[:3]+Rotation.from_quat(target[3:]).as_euler("xyz").tolist()
        compiler.tasks, compiler._last_arm = [], {}
        before = len(rejected)
        again = compiler.compile_action(engine.parse_action("support_off(A, beam)"), state)
        self.assertEqual(len(again), 2)
        self.assertEqual(len(rejected), before)

    def test_successful_default_exit_keeps_the_original_extra_lower(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        target = compiler.frames.support_pose("beam", "A", arm_config=compiler.cfg["arms"])
        compiler.q["A"] = target[:3]+Rotation.from_quat(target[3:]).as_euler("xyz").tolist()
        state = frozenset([*engine.initial, Fact("supporting", ("A", "beam"))])
        tasks = compiler.compile_action(engine.parse_action("support_off(A, beam)"), state)
        self.assertEqual(len(tasks), 3)
        self.assertAlmostEqual(tasks[-1]["q_path"][-1][2], target[2]-.1)
        self.assertFalse(any(row["metadata"].get("support_exit_strategy") for row in tasks))

    def test_failed_horizontal_retreat_does_not_skip_collision_or_leak_release(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        target = compiler.frames.support_pose("beam", "A", arm_config=compiler.cfg["arms"])
        compiler.q["A"] = target[:3]+Rotation.from_quat(target[3:]).as_euler("xyz").tolist()
        state = frozenset([*engine.initial, Fact("supporting", ("A", "beam"))])
        before = compiler._snapshot()
        original = geometry.validate_joint

        def bad_retreat(arm, q, **kwargs):
            if arm == "A" and math.dist(q[:3], target[:3]) > 1e-5:
                return {"ok": False, "problem": {"reason": "table_collision:A",
                                                "pair": ["A_SupportShoe", "Workbench"]}}
            return original(arm, q, **kwargs)

        geometry.validate_joint = bad_retreat
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("support_off(A, beam)"), state)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(compiler._snapshot(), before)
        trials = caught.exception.evidence["action_candidate_attempts"]
        self.assertEqual(len(trials), 1, "r2 failure cannot trigger the r3-only alternate")

    def test_alternate_support_exit_does_not_bypass_final_home_goal_validation(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        support = engine.parse_action("support_on(A, beam)")
        release = engine.parse_action("support_off(A, beam)")
        state = frozenset([*engine.initial, Fact("supporting", ("A", "beam"))])
        target = compiler.frames.support_pose("beam", "A", arm_config=compiler.cfg["arms"])
        original = geometry.validate_joint
        lower_failed = False

        def collision(arm, q, **kwargs):
            nonlocal lower_failed
            if arm == "A" and len(compiler.tasks) >= 4 and q[2] < target[2]-1e-6:
                lower_failed = True
                return {"ok": False, "problem": {"reason": "table_collision:A", "pair": ["A", "Table"]}}
            return original(arm, q, **kwargs)

        geometry.validate_joint = collision
        plan = geometry.plan
        home_checked = []
        def blocked_home(arm, start, goal, **kwargs):
            answer = plan(arm, start, goal, **kwargs)
            if arm == "B" and goal == compiler.cfg["arms"]["B"]["home_q"]:
                home_checked.append(list(goal))
                return {"ok": False, "error": "Invalid goal", "problem": {
                    "reason": "block_robot_collision:beam_block:B", "pair": ["B", "beam_block"]}}
            return answer
        geometry.plan = blocked_home
        steps = [SearchStep(support, 0., 2., engine.initial), SearchStep(release, 2., 4., state)]
        with self.assertRaises(CompileFailure):
            compiler.compile(steps)
        self.assertTrue(lower_failed)
        self.assertTrue(home_checked)
        self.assertFalse(compiler.tasks)


if __name__ == "__main__":
    unittest.main()
