"""Compiler integration contracts using real grounded skills and fake geometry."""
from copy import deepcopy
from pathlib import Path
import unittest
import tempfile

from scipy.spatial.transform import Rotation

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import SearchStep
from skillnet.planning.taskgraph import build_task_graph
from simbridge.compiler import CompileFailure, MotionCompiler
from simbridge.frames import Frames


ROOT = Path(__file__).resolve().parents[1]


class GeometryFake:
    """A deterministic Cartesian surrogate; it proves call/context flow only."""

    def __init__(self):
        self.calls = []
        self.ik_failure = self.plan_failure = self.collision_failure = None

    def solve_ik(self, arm, position, quaternion, **kwargs):
        self.calls.append(("ik", arm, deepcopy(kwargs)))
        if self.ik_failure:
            return deepcopy(self.ik_failure)
        import math
        import warnings
        # Three independent cyclic Euler coordinates match the compiler's
        # per-joint winding semantics. Rotvec components are not independent
        # cyclic angles, so unwrapping one used to corrupt this fake's FK.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            angles = Rotation.from_quat(quaternion).as_euler("xyz").tolist()
        alternatives = [angles, [angles[0]+math.pi, math.pi-angles[1], angles[2]+math.pi]]
        seed = kwargs.get("options", {}).get("seed", [0.]*6)[3:]
        period = 2*math.pi
        candidates = [[value+period*round((current-value)/period)
                       for value, current in zip(row, seed)] for row in alternatives]
        nearest = min(candidates, key=lambda row: sum((value-current)**2
                                                      for value, current in zip(row, seed)))
        return {"ok": True, "q": list(position)+nearest}

    def plan(self, arm, start, goal, **kwargs):
        self.calls.append(("plan", arm, {"start": list(start), "goal": list(goal), **deepcopy(kwargs)}))
        return (deepcopy(self.plan_failure) if self.plan_failure else
                {"ok": True, "path": [list(start), list(goal)]})

    def fk(self, arm, q):
        return {"ok": True, "pose": list(q[:3])+Rotation.from_euler("xyz", q[3:]).as_quat().tolist()}

    def validate_joint(self, arm, q, **kwargs):
        self.calls.append(("validate", arm, {"q": list(q), **deepcopy(kwargs)}))
        return deepcopy(self.collision_failure) if self.collision_failure else {"ok": True}

    def sample_state_pairs(self, samples):
        self.samples = getattr(self, "samples", [])+deepcopy(samples)
        return [{"ok": True} for sample in samples]


def fixture(name="s1_parallel_place", **options):
    problem = load_problem(ROOT/"tasks"/(name+".toml"), ROOT/"skills/blocks_v1.toml")
    engine, frames, geometry = Engine(problem), Frames(problem), GeometryFake()
    cfg = {"arms": {arm: {"home_q": frames.home_pose(arm)[:3]
                  +Rotation.from_quat(frames.home_pose(arm)[3:]).as_euler("xyz").tolist(),
                  "base_pose": [-.70 if arm == "A" else .70, .13, .783, 0, 0, 0, 1]}
                    for arm in ("A", "B")},
           "initial": {block: frames.block_pose(block) for block in frames.blocks},
           "parts": [{"id": block, "source": frames.block_pose(block)} for block in frames.blocks],
           "baseplate": "Baseplate", "pads": {name: "Pad_"+name for name in frames.locations}}
    return engine, geometry, MotionCompiler(geometry, cfg, frames, **options)


def sequential(engine, texts):
    state, steps = engine.initial, []
    for index, text in enumerate(texts):
        action = engine.parse_action(text)
        steps.append(SearchStep(action, index*5., (index+1)*5., state))
        state = engine.execute(action, state)
    return steps


class MotionCompilerTests(unittest.TestCase):
    def test_transit_can_select_later_connected_elbow_family_at_same_pose(self):
        engine, geometry, compiler = fixture(ik_seeds=2)
        original_ik, original_plan, original_fk = geometry.solve_ik, geometry.plan, geometry.fk
        candidates = 0

        def two_elbows(arm, position, quaternion, **kwargs):
            nonlocal candidates
            answer = original_ik(arm, position, quaternion, **kwargs)
            candidates += 1
            # Both elbow signs map to the same Cartesian pose in this external
            # geometry surrogate. The first candidate is valid but disconnected.
            if candidates == 1:
                answer["q"][2] = -answer["q"][2]
            return answer

        def elbow_fk(arm, q):
            canonical = list(q)
            canonical[2] = abs(canonical[2])
            return original_fk(arm, canonical)

        def connected_plan(arm, start, goal, **kwargs):
            if goal[2] < 0:
                return {"ok": False, "error": "No exact collision-free solution within budget"}
            return original_plan(arm, start, goal, **kwargs)

        geometry.solve_ik, geometry.fk, geometry.plan = two_elbows, elbow_fk, connected_plan
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(len(tasks), 4)
        self.assertGreater(tasks[0]["q_path"][-1][2], 0)
        self.assertEqual(compiler.held["A"]["name"], "red")

    def test_real_grounded_pick_place_uses_previous_q_and_finishes_at_home(self):
        engine, geometry, compiler = fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps))
        by_id = {task["id"]: task for task in tasks}
        place = next(task for task in tasks if task["metadata"]["source_step"] == 2)
        prior = [task for task in tasks if task["metadata"]["source_step"] == 1][-1]
        self.assertEqual(place["q_path"][0], prior["q_path"][-1])
        self.assertIn(prior["id"], place["dependencies"])
        self.assertEqual({task["arm"] for task in tasks if task["metadata"].get("final_home")}, {"A", "B"})
        self.assertFalse(compiler.held)
        self.assertEqual(compiler.placed["red"], compiler.frames.block_pose("red", "board_l1"))
        for task in tasks:
            for predecessor in task["dependencies"]:
                self.assertLess(by_id[predecessor]["reference_order"], task["reference_order"])
        for arm in ("A", "B"):
            self.assertEqual(compiler.q[arm], compiler.cfg["arms"][arm]["home_q"])
        # Stage 1 (local joint box) succeeds in the fake, so each free motion
        # is a single bounded call inside the 10 s budget.
        plans = [row for row in geometry.calls if row[0] == "plan"]
        self.assertTrue(all(row[2]["solve_time"] == 4. for row in plans))
        self.assertTrue(all(row[2]["options"]["joint_bound_margin"] == .6 for row in plans))
        self.assertTrue(all(row[2]["options"]["direct_first"] is False
                            for row in geometry.calls if row[0] == "plan"))

    def test_line_interpolation_checks_each_emitted_joint_state(self):
        engine, geometry, compiler = fixture(linear_step=.02)
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        checked = [row[2]["q"] for row in geometry.calls if row[0] == "validate"]
        lines = [task for task in tasks if task["metadata"]["request_kind"] == "linear"]
        self.assertEqual(len(lines), 2)
        self.assertGreater(len(lines[0]["q_path"]), 3)
        for task in lines:
            for q in task["q_path"]:
                self.assertIn(q, checked)
            for q0, q1 in zip(task["q_path"], task["q_path"][1:]):
                self.assertLessEqual(max(abs(a-b) for a,b in zip(q0,q1)), .03500001)

    def test_parallel_starts_keep_independent_dependencies_and_other_payload(self):
        engine, geometry, compiler = fixture()
        pick_a = engine.parse_action("pick(A, red, stock_l1)")
        pick_b = engine.parse_action("pick(B, blue, stock_r1)")
        steps = [SearchStep(pick_a, 0., 4., engine.initial),
                 SearchStep(pick_b, 0., 4., engine.start(pick_a, engine.initial))]
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        first_b = next(task for task in tasks if task["arm"] == "B")
        self.assertFalse(first_b["dependencies"])
        b_plan = next(row[2] for row in geometry.calls if row[:2] == ("plan", "B"))
        a_last = [task for task in tasks if task["arm"] == "A"][-1]
        self.assertEqual(b_plan["other_q"], a_last["q_path"][-1])
        self.assertEqual(b_plan["options"]["other_payload"]["name"], "red")
        self.assertNotIn("red", b_plan["placed"])
        self.assertEqual(set(compiler.held), {"A", "B"})

    def test_stale_parallel_place_snapshot_does_not_reacquire_released_object(self):
        engine, _, compiler = fixture()
        picks = [engine.parse_action("pick(A, red, stock_l1)"), engine.parse_action("pick(B, blue, stock_r1)")]
        state = engine.execute(picks[1], engine.execute(picks[0], engine.initial))
        place_a = engine.parse_action("place(A, red, board_l1)")
        place_b = engine.parse_action("place(B, blue, board_r1)")
        steps = [SearchStep(picks[0], 0., 4., engine.initial),
                 SearchStep(picks[1], 0., 4., engine.start(picks[0], engine.initial)),
                 SearchStep(place_a, 4., 8., state), SearchStep(place_b, 4., 8., state)]
        tasks = compiler.compile(steps, return_home=False)
        b_place = next(task for task in tasks if task["metadata"]["source_step"] == 4)
        self.assertEqual(b_place["metadata"]["placed"]["red"], compiler.frames.block_pose("red", "board_l1"))
        self.assertFalse(compiler.held)

    def test_handover_cross_arm_grasp_release_and_retreat_dependencies(self):
        engine, _, compiler = fixture("s1_reach_handover")
        steps = sequential(engine, ["pick(B, red, stock_r1)", "handover(B, A, red, hand_mid)",
                                    "place(A, red, board_l1)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        hand = [task for task in tasks if task["metadata"]["source_step"] == 2]
        take = next(task for task in hand if task["metadata"]["action"] == "grasp")
        give = next(task for task in hand if task["metadata"]["action"] == "release")
        self.assertIn(take["id"], give["dependencies"])
        self.assertEqual(take["arm"], "A")
        self.assertEqual(give["arm"], "B")
        retreats = [task for task in hand if task["metadata"]["request_kind"] == "linear"]
        self.assertEqual({task["arm"] for task in retreats}, {"A", "B"})
        self.assertTrue(all(give["id"] in task["dependencies"] for task in retreats))
        self.assertFalse(compiler.held)

    def test_ik_budget_is_unknown_and_exactly_bounded(self):
        engine, geometry, compiler = fixture(ik_seeds=3)
        geometry.ik_failure = {"ok": False, "problem": {"reason": "no candidate"}}
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertEqual(sum(row[0] == "ik" for row in geometry.calls), 3)
        self.assertFalse(compiler.tasks)
        self.assertFalse(compiler.held)

    def test_proven_ik_failure_is_invalid(self):
        engine, geometry, compiler = fixture()
        geometry.ik_failure = {"ok": False, "infeasible": True, "problem": {"reason": "outside workspace"}}
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.INVALID)
        self.assertEqual(len(geometry.calls), 1)

    def test_rrt_budget_unknown_but_invalid_start_collision_invalid(self):
        for answer, status in [({"ok": False, "error": "No exact collision-free solution within budget"}, Status.UNKNOWN),
                               ({"ok": False, "error": "Invalid start", "problem": {"pair": ["red", "table"]}}, Status.INVALID)]:
            with self.subTest(status=status):
                engine, geometry, compiler = fixture()
                geometry.plan_failure = answer
                with self.assertRaises(CompileFailure) as failed:
                    compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
                self.assertEqual(failed.exception.status, status)
                self.assertFalse(compiler.tasks)

    def test_collision_rolls_back_action_and_reports_pair(self):
        engine, geometry, compiler = fixture()
        geometry.collision_failure = {"ok": False, "problem": {"reason": "block_robot_collision", "pair": ["red", "A_link"]}}
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.INVALID)
        self.assertIn("A_link", failed.exception.detail)
        self.assertFalse(compiler.tasks)
        self.assertEqual(compiler.q["A"], compiler.cfg["arms"]["A"]["home_q"])

    def test_complete_compile_failure_does_not_leave_successful_prefix(self):
        engine, geometry, compiler = fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        original = geometry.plan
        motions = []
        def fail_second(arm, start, goal, **kwargs):
            # The second distinct free motion fails in both search stages.
            key = (tuple(start), tuple(goal))
            if key not in motions:
                motions.append(key)
            if motions.index(key) == 1:
                return {"ok": False, "error": "budget"}
            return original(arm, start, goal, **kwargs)
        geometry.plan = fail_second
        with self.assertRaises(CompileFailure):
            compiler.compile(steps, return_home=False)
        self.assertFalse(compiler.tasks)
        self.assertTrue(compiler.evidence)

    def test_coheld_skills_are_explicitly_rejected_before_geometry(self):
        engine, geometry, compiler = fixture()
        for name, args in [("co_pick", ("A", "B", "red", "stock_l1")),
                           ("co_insert", ("A", "B", "red", "unused_slot"))]:
            with self.subTest(skill=name):
                action = engine.library.by_name[name].ground(args)
                with self.assertRaisesRegex(CompileFailure, "共持尚未接入运动后端") as failed:
                    compiler.compile_action(action, engine.initial)
                self.assertEqual(failed.exception.status, Status.INVALID)
        self.assertFalse(geometry.calls)

    def test_generated_effects_reconstruct_ordinary_world_for_tpg_rollout(self):
        from apexmr.geometry import verify_rollout
        from apexmr.program import parameterize
        from apexmr.tpg import build_tpg

        engine, geometry, compiler = fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        tasks = compiler.compile(steps)
        tpg = parameterize(build_tpg(tasks))
        (ROOT/"out").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as outdir:
            report = verify_rollout(tpg, geometry, compiler.cfg, step_s=.5, outdir=outdir)
        self.assertTrue(report["nominal_world_state_reconstructed"], report["world_state_gaps"])
        self.assertTrue(report["valid"])
        self.assertTrue(any(row["payloads"].get("A", {}).get("name") == "red" for row in geometry.samples))
        self.assertFalse(geometry.samples[-1]["payloads"])
        self.assertEqual(geometry.samples[-1]["placed"]["red"], compiler.frames.block_pose("red", "board_l1"))

    def test_final_home_expires_gripper_exclusions(self):
        engine, _, compiler = fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "place(A, red, board_l1)"])
        tasks = compiler.compile(steps)
        home = [task for task in tasks if task["metadata"].get("final_home")]
        self.assertEqual(len(home), 2)
        for task in home:
            pairs = task["metadata"]["contact_exclusions"]
            self.assertIn(["red", "Baseplate"], pairs)
            self.assertFalse(any("Finger" in str(item) or "Palm" in str(item) for pair in pairs for item in pair))

    def test_linear_ik_branch_jump_is_unknown_with_finite_retries(self):
        engine, geometry, compiler = fixture(ik_seeds=3)
        original = geometry.solve_ik
        def jump_branch(arm, position, quaternion, **kwargs):
            answer = original(arm, position, quaternion, **kwargs)
            if len([call for call in geometry.calls if call[0] == "ik"]) > 1:
                answer["q"][5] += 3.
            return answer
        geometry.solve_ik = jump_branch
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertIn("IK_branch_discontinuity", failed.exception.detail)
        # Three transit endpoint candidates, then exactly three bounded retries
        # of the discontinuous first Cartesian line sample.
        self.assertEqual(sum(call[0] == "ik" for call in geometry.calls), 6)
        self.assertFalse(compiler.tasks)
        self.assertEqual(compiler.q["A"], compiler.cfg["arms"]["A"]["home_q"])

    def test_equivalent_ur5_winding_is_unwrapped_before_continuity_check(self):
        import math
        from simbridge.primitives import MotionRequest
        _, geometry, compiler = fixture()
        geometry.solve_ik = lambda *args, **kwargs: {
            "ok": True, "q": [0., 0., 0., 0., 0., 2*math.pi-.1]}
        request = MotionRequest("cyclic", "A", "linear")
        q = compiler._ik(request, [0, 0, 0, 0, 0, 0, 1], [0.]*6,
                         max_joint_delta=.25)
        self.assertAlmostEqual(q[5], -.1)
        self.assertEqual(len(compiler.evidence), 1)

    def test_collision_free_joint_interpolation_must_stay_on_cartesian_line(self):
        engine, geometry, compiler = fixture(joint_step=.003)
        solve, fk = geometry.solve_ik, geometry.fk
        solved_q = set()
        def remember_ik(*args, **kwargs):
            answer = solve(*args, **kwargs)
            solved_q.add(tuple(answer["q"]))
            return answer
        def curved_fk(arm, q):
            answer = fk(arm, q)
            # Endpoints satisfy IK exactly, but collision-free interpolated
            # joint states make a lateral excursion beyond the 2 mm bound.
            if not any(max(abs(a-b) for a,b in zip(q, goal)) < 1e-10 for goal in solved_q):
                answer["pose"][0] += .004
            return answer
        geometry.solve_ik, geometry.fk = remember_ik, curved_fk
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertIn("Cartesian_line_deviation", failed.exception.detail)
        self.assertFalse(compiler.tasks)
        self.assertFalse(compiler.held)


if __name__ == "__main__":
    unittest.main()


class TwoStageTransitSearchTests(unittest.TestCase):
    def test_local_box_failure_falls_back_to_global_search_within_budget(self):
        engine, geometry, compiler = fixture()
        original_plan = geometry.plan

        calls = []

        def boxed_then_global(arm, start, goal, **kwargs):
            calls.append((kwargs["solve_time"], deepcopy(kwargs["options"])))
            if kwargs.get("options", {}).get("joint_bound_margin"):
                return {"ok": False, "error": "No exact collision-free solution within budget"}
            return original_plan(arm, start, goal, **kwargs)

        geometry.plan = boxed_then_global
        tasks = compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], 4.)
        self.assertEqual(calls[0][1]["joint_bound_margin"], .6)
        self.assertNotIn("joint_bound_margin", calls[1][1])
        # The global stage receives whatever wall time the box stage left of
        # the single 10 s budget (the fake answers instantly).
        self.assertGreaterEqual(calls[1][0], 6.-1e-9)
        self.assertLessEqual(calls[1][0], 10.+1e-9)
        stages = [row["search_stage"] for row in compiler.evidence if row["query"] == "RRTConnect"]
        self.assertEqual(stages, ["local_box", "global"])
        self.assertEqual(tasks[0]["kind"], "transit")
        stages = [row["search_stage"] for row in compiler.evidence if row["query"] == "RRTConnect"]
        self.assertEqual(stages, ["local_box", "global"])
        self.assertEqual(tasks[0]["kind"], "transit")

    def test_both_stages_failing_is_unknown_with_both_attempts_recorded(self):
        engine, geometry, compiler = fixture()
        geometry.plan_failure = {"ok": False, "error": "No exact collision-free solution within budget"}
        with self.assertRaises(CompileFailure) as failed:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        attempts = failed.exception.evidence["attempts"]
        self.assertEqual([a["stage"] for a in attempts], ["local_box", "global"])
        self.assertIn("local_box、global", str(failed.exception))
        self.assertEqual(len([row for row in geometry.calls if row[0] == "plan"]), 2)

    def test_invalid_box_failure_does_not_retry_globally(self):
        engine, geometry, compiler = fixture()
        geometry.plan_failure = {"ok": False, "error": "invalid goal state",
                                 "problem": {"reason": "table_collision:A"}}
        with self.assertRaises(CompileFailure):
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(len([row for row in geometry.calls if row[0] == "plan"]), 1)

    def test_local_box_can_be_disabled(self):
        engine, geometry, compiler = fixture(local_box_margin=None)
        compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        plans = [row for row in geometry.calls if row[0] == "plan"]
        self.assertTrue(all(row[2]["solve_time"] == 10. for row in plans))
        self.assertTrue(all("joint_bound_margin" not in row[2]["options"] for row in plans))

