"""One bounded, geometry-certified idle-arm recovery; no simulator process."""
from copy import deepcopy
import math
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.planning.taskgraph import build_task_graph
from simbridge.compiler import CompileFailure
from tests.test_v13_compiler import fixture, sequential


INTERARM = {"ok": False, "solutions": {}, "problem": {
    "reason": "interarm_collision", "pair": ["/A_UR5/A_joint_4/link", "/B_UR5/B_GripperPalm"]}}


def tower_prefix():
    engine, geometry, compiler = fixture("sim/s2_tower_sim")
    texts = ["pick(A, b1, stock_l1)", "insert(A, b1, t1)",
             "pick(A, b2, stock_l2)", "insert(A, b2, t2)", "pick(B, b3, stock_r1)"]
    state = engine.initial
    for number, text in enumerate(texts, 1):
        action = engine.parse_action(text)
        compiler.compile_action(action, state, step_id=number)
        state = engine.execute(action, state)
    return engine, geometry, compiler, state, texts


def blocked_until_idle_home(geometry, compiler, *, retry_failure=False):
    original = geometry.solve_ik
    failed_queries = []
    def solve(arm, position, quaternion, **kwargs):
        if arm == "B" and (retry_failure or math.dist(kwargs["options"]["other_q"], compiler.cfg["arms"]["A"]["home_q"]) > 1e-6):
            geometry.calls.append(("ik", arm, deepcopy(kwargs)))
            failed_queries.append(deepcopy(kwargs))
            return deepcopy(INTERARM)
        return original(arm, position, quaternion, **kwargs)
    geometry.solve_ik = solve
    return failed_queries


def payload_blocked_until_idle_home(geometry, compiler, *, retry_failure=False, malformed=None):
    """Three empty seeds plus five payload/idle-arm rejections, as recorded."""
    original, failed_queries = geometry.solve_ik, []
    empty = {"ok": False, "solutions": {}, "stats": {"candidates": 0, "rejections": {}}}
    problem = {"reason": "block_robot_collision:b3:A", "pair": ["/A_UR5/A_Finger_neg", "/b3"]}
    blocked = {"ok": False, "solutions": {}, "problem": deepcopy(problem),
               "stats": {"candidates": 1, "rejections": [deepcopy(problem)]}}
    if malformed is not None:
        blocked = malformed(blocked)
    def solve(arm, position, quaternion, **kwargs):
        if arm == "B" and (retry_failure or math.dist(kwargs["options"]["other_q"], compiler.cfg["arms"]["A"]["home_q"]) > 1e-6):
            geometry.calls.append(("ik", arm, deepcopy(kwargs)))
            answer = deepcopy(empty if len(failed_queries) % 8 < 3 else blocked)
            failed_queries.append(answer)
            return answer
        return original(arm, position, quaternion, **kwargs)
    geometry.solve_ik = solve
    return failed_queries


class IdleParkingTests(unittest.TestCase):
    def test_current_held_payload_collision_with_idle_arm_parks_once_after_eight_seeds(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        prior = dict(compiler._last_arm)
        failures = payload_blocked_until_idle_home(geometry, compiler)
        tasks = compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        parked = [task for task in tasks if task["metadata"].get("idle_arm_parking")]
        self.assertEqual(len(failures), 8)
        self.assertEqual(len(parked), 1)
        self.assertEqual(parked[0]["q_path"][-1], compiler.cfg["arms"]["A"]["home_q"])
        self.assertTrue(set(prior.values()) <= set(parked[0]["dependencies"]))
        self.assertIn(parked[0]["id"], next(t for t in tasks if t["arm"] == "B")["dependencies"])
        self.assertTrue(any(e.get("query") == "idle_arm_parking" for e in compiler.evidence))

    def test_payload_parking_replays_in_final_compile_cache_without_remote_queries(self):
        engine, geometry, compiler, state, texts = tower_prefix()
        payload_blocked_until_idle_home(geometry, compiler)
        compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        calls = len(geometry.calls)
        steps = sequential(engine, texts + ["insert(B, b3, t3)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        self.assertEqual(len(geometry.calls), calls)
        self.assertEqual(sum(bool(t["metadata"].get("idle_arm_parking")) for t in tasks), 1)

    def test_payload_retry_failure_remains_unknown_and_rolls_back_with_one_parking(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        before, first = compiler._snapshot(), len(geometry.calls)
        failures = payload_blocked_until_idle_home(geometry, compiler, retry_failure=True)
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(caught.exception.evidence["recovery_phase"], "retry")
        self.assertEqual(compiler._snapshot(), before)
        self.assertEqual(len(failures), 16)
        self.assertEqual(sum(c[:2] == ("plan", "A") for c in geometry.calls[first:]), 1)

    def test_payload_parking_never_moves_other_held_coheld_or_supporting_arm(self):
        for mode in ("held", "coheld", "supporting", "compiled_support"):
            with self.subTest(mode=mode):
                engine, geometry, compiler, state, _ = tower_prefix()
                state = set(state)
                if mode == "held":
                    compiler.held["A"] = {"name": "b1", "offset": [0, 0, .1], "quaternion": [0, 0, 0, 1]}
                elif mode == "coheld":
                    state.add(Fact("coheld", ("A", "B", "b3")))
                elif mode == "supporting":
                    state.add(Fact("supporting", ("A", "t2")))
                else:
                    compiler.tasks.append({"id": "active_support", "arm": "A", "kind": "support_start"})
                payload_blocked_until_idle_home(geometry, compiler)
                first = len(geometry.calls)
                with self.assertRaises(CompileFailure) as caught:
                    compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), frozenset(state), step_id=6)
                self.assertEqual(caught.exception.status, Status.UNKNOWN)
                self.assertFalse(any(c[:2] == ("plan", "A") for c in geometry.calls[first:]))

    def test_payload_recovery_does_not_expand_a_smaller_seed_budget(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        compiler.ik_seeds = 4
        failures = payload_blocked_until_idle_home(geometry, compiler)
        first = len(geometry.calls)
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(len(failures), 4)
        self.assertFalse(any(c[:2] == ("plan", "A") for c in geometry.calls[first:]))

    def test_payload_recovery_requires_matching_held_body_arm_paths_and_unmixed_evidence(self):
        def modify(answer, mode):
            problem = answer["problem"]
            if mode == "other_body":
                problem.update(reason="block_robot_collision:b1:A", pair=["/A_UR5/A_Finger_neg", "/b1"])
            elif mode == "own_arm":
                problem.update(reason="block_robot_collision:b3:B", pair=["/B_UR5/B_Finger_neg", "/b3"])
            elif mode == "mismatched_pair":
                problem["pair"] = ["/B_UR5/B_Finger_neg", "/b3"]
            elif mode == "unstructured":
                answer.pop("problem")
                answer["stats"] = {"candidates": 1}
                return answer
            answer["stats"]["rejections"] = [deepcopy(problem)]
            if mode == "mixed_reason":
                answer["stats"]["rejections"].append({"reason": "payload_table_collision:b3", "pair": ["/b3", "/Table"]})
            return answer
        for mode in ("other_body", "own_arm", "mismatched_pair", "mixed_reason", "unstructured"):
            with self.subTest(mode=mode):
                engine, geometry, compiler, state, _ = tower_prefix()
                payload_blocked_until_idle_home(geometry, compiler, malformed=lambda a: modify(a, mode))
                first = len(geometry.calls)
                with self.assertRaises(CompileFailure) as caught:
                    compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
                self.assertEqual(caught.exception.status, Status.UNKNOWN)
                self.assertFalse(any(c[:2] == ("plan", "A") for c in geometry.calls[first:]))

    def test_real_tower_mixture_of_four_empty_candidates_and_four_interarm_rejections_recovers(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        original = geometry.solve_ik
        empty = {"ok": False, "solutions": {}, "stats": {"candidates": 0, "rejections": {}}}
        blocked = deepcopy(INTERARM)
        blocked["stats"] = {"candidates": 1, "rejections": [deepcopy(blocked["problem"])]}
        answers = [empty, empty, empty, blocked, blocked, blocked, blocked, empty]
        calls = []
        def solve(arm, position, quaternion, **kwargs):
            if arm == "B" and math.dist(kwargs["options"]["other_q"], compiler.cfg["arms"]["A"]["home_q"]) > 1e-6:
                geometry.calls.append(("ik", arm, deepcopy(kwargs)))
                answer = deepcopy(answers[len(calls)])
                calls.append(answer)
                return answer
            return original(arm, position, quaternion, **kwargs)
        geometry.solve_ik = solve
        tasks = compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(len(calls), 8)
        self.assertEqual(sum(answer.get("problem", {}).get("reason") == "interarm_collision" for answer in calls), 4)
        self.assertEqual(sum(bool(task["metadata"].get("idle_arm_parking")) for task in tasks), 1)

    def test_public_compile_parks_free_other_arm_then_retries_with_cross_arm_dependencies(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        prior = dict(compiler._last_arm)
        failed = blocked_until_idle_home(geometry, compiler)
        tasks = compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        parking = [task for task in tasks if task["metadata"].get("idle_arm_parking")]
        self.assertEqual(len(parking), 1)
        parked = parking[0]
        self.assertEqual(parked["arm"], "A")
        self.assertEqual(parked["q_path"][-1], compiler.cfg["arms"]["A"]["home_q"])
        self.assertTrue(set(prior.values()) <= set(parked["dependencies"]))
        moving = next(task for task in tasks if task["arm"] == "B")
        self.assertIn(parked["id"], moving["dependencies"])
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len(failed), compiler.ik_seeds)
        parking_query = next(call[2] for call in geometry.calls if call[0] == "plan" and call[1] == "A"
                             and call[2]["goal"] == compiler.cfg["arms"]["A"]["home_q"])
        self.assertEqual(parking_query["options"]["other_payload"]["name"], "b3")

    def test_final_compile_replays_successful_parking_from_cache_without_geometry_calls(self):
        engine, geometry, compiler, state, texts = tower_prefix()
        blocked_until_idle_home(geometry, compiler)
        compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        calls = len(geometry.calls)
        steps = sequential(engine, texts+["insert(B, b3, t3)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        self.assertEqual(len(geometry.calls), calls)
        self.assertEqual(len([task for task in tasks if task["metadata"].get("idle_arm_parking")]), 1)
        self.assertEqual(len(tasks), 25)

    def test_other_held_or_supporting_arm_is_not_parked(self):
        for mode in ("held", "supporting", "compiled_support"):
            with self.subTest(mode=mode):
                engine, geometry, compiler, state, _ = tower_prefix()
                modified = set(state)
                if mode == "held":
                    compiler.held["A"] = {"name": "b1", "offset": [0, 0, .1], "quaternion": [0, 0, 0, 1]}
                elif mode == "supporting":
                    modified.discard(Fact("free", ("A",)))
                    modified.add(Fact("supporting", ("A", "t2")))
                else:
                    compiler.tasks.append({"id": "active_support", "arm": "A", "kind": "support_start"})
                blocked_until_idle_home(geometry, compiler)
                before, first = deepcopy(compiler.q), len(geometry.calls)
                with self.assertRaises(CompileFailure) as caught:
                    compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), frozenset(modified), step_id=6)
                self.assertEqual(caught.exception.status, Status.UNKNOWN)
                self.assertEqual(compiler.q, before)
                self.assertFalse(any(call[:2] == ("plan", "A") for call in geometry.calls[first:]))

    def test_failed_retry_rolls_back_whole_action_and_never_parks_twice(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        before = compiler._snapshot()
        first = len(geometry.calls)
        failures = blocked_until_idle_home(geometry, compiler, retry_failure=True)
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(compiler._snapshot(), before)
        self.assertEqual(sum(call[:2] == ("plan", "A") for call in geometry.calls[first:]), 1)
        self.assertEqual(len(failures), 2*compiler.ik_seeds)

    def test_parking_route_invalid_does_not_turn_original_unknown_into_invalid(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        before = compiler._snapshot()
        blocked_until_idle_home(geometry, compiler)
        original_plan = geometry.plan
        def invalid_parking(arm, start, goal, **kwargs):
            if arm == "A":
                geometry.calls.append(("plan", arm, deepcopy(kwargs)))
                return {"ok": False, "error": "invalid goal state", "problem": {
                    "reason": "interarm_collision", "pair": ["/A_UR5/link", "/B_UR5/B_GripperPalm"]}}
            return original_plan(arm, start, goal, **kwargs)
        geometry.plan = invalid_parking
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(caught.exception.evidence["recovery_status"], "INVALID")
        self.assertEqual(caught.exception.evidence["recovery_phase"], "parking")
        self.assertEqual(compiler._snapshot(), before)

    def test_other_arm_already_home_is_not_moved_again(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        compiler.q["A"] = list(compiler.cfg["arms"]["A"]["home_q"])
        geometry.ik_failure = deepcopy(INTERARM)
        first = len(geometry.calls)
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertFalse(any(call[:2] == ("plan", "A") for call in geometry.calls[first:]))

    def test_no_structured_interarm_evidence_does_not_start_recovery(self):
        engine, geometry, compiler, state, _ = tower_prefix()
        first = len(geometry.calls)
        geometry.ik_failure = {"ok": False, "error": "no candidate found"}
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("insert(B, b3, t3)"), state, step_id=6)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertFalse(any(call[:2] == ("plan", "A") for call in geometry.calls[first:]))

    def test_normal_parallel_fixture_keeps_original_eighteen_tasks(self):
        engine, geometry, compiler = fixture()
        steps = sequential(engine, ["pick(A, red, stock_l1)", "pick(B, blue, stock_r1)",
                                    "place(A, red, board_l1)", "place(B, blue, board_r1)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps))
        self.assertEqual(len(tasks), 18)
        self.assertFalse(any(task["metadata"].get("idle_arm_parking") for task in tasks))


if __name__ == "__main__":
    unittest.main()
