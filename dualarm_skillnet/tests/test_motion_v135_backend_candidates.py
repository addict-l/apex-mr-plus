"""Context-scoped, finite motion candidates through the real backend boundary.

GeometryFake is an external Cartesian surrogate: these tests prove feedback,
transaction and evidence behavior, not physical UR5 reachability. The public
begin_reference boundary explicitly starts a sequential geometric reference;
it must retain candidate history and exact-context path certificates.
"""
from copy import deepcopy
import math
import time
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.planning.search import SearchStep
from tests.test_coppelia_backend import backend_fixture


FINITE_IK_FAILURE = {"ok": False, "solutions": []}


class ContextScopedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.geometry, self.backend = backend_fixture()
        self.action = self.engine.parse_action("pick(A, red, stock_l1)")
        self.state = self.engine.initial

    def begin(self, number, *, deadline=None):
        self.backend.begin_reference(number, deadline=deadline)

    def assert_conditional_failure(self):
        context = self.backend.validation_context
        self.assertRegex(context["context_id"], r"^[0-9a-f]{64}$")
        self.assertTrue(context["candidate_id"])
        self.assertTrue(context["candidate_version"])
        self.assertEqual(context["failure_scope"], "candidate")
        self.assertTrue(context["temporary_cut"])
        return deepcopy(context)

    def fail_first_reference(self):
        self.geometry.ik_failure = deepcopy(FINITE_IK_FAILURE)
        self.begin(1)
        result = self.backend.validate(self.action, self.state)
        self.assertEqual(result.status, Status.UNKNOWN)
        return self.assert_conditional_failure()

    def test_begin_reference_resets_world_but_preserves_exact_path_certificates(self):
        self.begin(1)
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.VALID)
        self.assertEqual(self.backend.compiler.held["A"]["name"], "red")
        self.assertTrue(self.backend.compiler.tasks)
        calls = len(self.geometry.calls)

        self.begin(2)
        self.assertFalse(self.backend.compiler.held)
        self.assertFalse(self.backend.compiler.tasks)
        for arm in ("A", "B"):
            self.assertEqual(self.backend.compiler.q[arm], self.backend.cfg["arms"][arm]["home_q"])
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.VALID)
        self.assertEqual(len(self.geometry.calls), calls)

    def test_later_reference_cannot_extend_an_expired_task_deadline(self):
        self.begin(1, deadline=time.monotonic()-1.)
        self.begin(2, deadline=time.monotonic()+60.)
        result = self.backend.validate(self.action, self.state)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(self.backend.validation_context["classification"], "deadline")
        self.assertFalse(self.geometry.calls)
        self.assertFalse(self.backend.validation_context["retryable"])

    def test_repeated_failed_candidate_is_reused_across_references_without_rpc(self):
        first = self.fail_first_reference()
        calls = len(self.geometry.calls)
        self.assertGreater(calls, 0)
        self.begin(2)
        result = self.backend.validate(self.action, self.state)
        self.assertEqual(result.status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertEqual(second["context_id"], first["context_id"])
        self.assertEqual(second["candidate_id"], first["candidate_id"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(len(self.geometry.calls), calls)

    def test_same_symbolic_state_with_changed_other_arm_pose_gets_a_new_attempt(self):
        first = self.fail_first_reference()
        calls = len(self.geometry.calls)
        self.begin(2)
        self.backend.compiler.q["B"][0] += .1
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["context_id"], first["context_id"])
        self.assertFalse(second["deduplicated"])
        self.assertGreater(len(self.geometry.calls), calls)

    def test_same_symbolic_held_fact_with_changed_binding_gets_a_new_attempt(self):
        self.backend.cfg["initial_payloads"] = {"B": self.backend.frames.payload("blue", "B")}
        self.state = (self.state - {Fact("at", ("blue", "stock_r1")), Fact("free", ("B",))}
                      | {Fact("held", ("B", "blue"))})
        first = self.fail_first_reference()
        calls = len(self.geometry.calls)
        self.begin(2)
        self.backend.compiler.held["B"]["offset"][0] += .02
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["context_id"], first["context_id"])
        self.assertGreater(len(self.geometry.calls), calls)

    def test_changed_contact_rule_gets_a_new_attempt(self):
        first = self.fail_first_reference()
        calls = len(self.geometry.calls)
        self.begin(2)
        self.backend.cfg["pads"]["stock_l1"] = "DifferentBearingPad"
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["context_id"], first["context_id"])
        self.assertGreater(len(self.geometry.calls), calls)

    def test_changed_target_pose_gets_a_new_attempt(self):
        first = self.fail_first_reference()
        calls = len(self.geometry.calls)
        self.begin(2)
        # Modify only this in-memory scene double, never a task or result file.
        self.backend.frames.references["stock_l1"][0] += .02
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["context_id"], first["context_id"])
        self.assertGreater(len(self.geometry.calls), calls)

    def test_equivalent_scene_inputs_have_reproducible_context_ids(self):
        first = self.fail_first_reference()
        other_engine, other_geometry, other_backend = backend_fixture()
        other_geometry.ik_failure = deepcopy(FINITE_IK_FAILURE)
        other_backend.begin_reference(1)
        other_backend.validate(other_engine.parse_action("pick(A, red, stock_l1)"), other_engine.initial)
        self.assertEqual(other_backend.validation_context["context_id"], first["context_id"])

    def test_failed_action_retains_the_entire_successful_prefix(self):
        self.begin(1)
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.VALID)
        other = self.engine.parse_action("pick(B, blue, stock_r1)")
        parallel_state = self.engine.start(self.action, self.state)
        compiler = self.backend.compiler
        before = deepcopy((compiler.q, compiler.held, compiler.placed, compiler.tasks))
        self.geometry.ik_failure = deepcopy(FINITE_IK_FAILURE)
        result = self.backend.validate(other, parallel_state)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual((compiler.q, compiler.held, compiler.placed, compiler.tasks), before)
        self.assertEqual(compiler.held["A"]["name"], "red")
        self.assert_conditional_failure()

    def test_a_new_seed_strategy_can_succeed_from_the_same_action_start(self):
        # The first eight trials are current-q plus seven configured seeds.
        # The eighth configured seed is available only in the expanded bank.
        # This deterministic external solver accepts that extra seed at the
        # approach pose; its geometry does not change between the attempts.
        self.backend.cfg["ik_seeds"] = [[.11*index, 0., 0., 0., 0., 0.]
                                        for index in range(1, 9)]
        target = self.backend.frames.approach_pose(
            self.backend.frames.grasp_pose("red", "stock_l1", "A"))[:3]
        solve_ik = self.geometry.solve_ik

        def seeded_geometry(arm, position, quaternion, **kwargs):
            seed = kwargs.get("options", {}).get("seed", [])
            at_approach = all(abs(a-b) <= 1e-10 for a, b in zip(position, target))
            extra_seed = bool(seed) and math.isclose(seed[0], .88, abs_tol=1e-10)
            if at_approach and not extra_seed:
                self.geometry.calls.append(("ik", arm, deepcopy(kwargs)))
                return deepcopy(FINITE_IK_FAILURE)
            return solve_ik(arm, position, quaternion, **kwargs)

        self.geometry.solve_ik = seeded_geometry
        self.begin(1)
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        first = self.assert_conditional_failure()
        before_q = deepcopy(self.backend.compiler.q)
        self.assertEqual(sum(row[0] == "ik" for row in self.geometry.calls), 8)
        self.assertEqual(self.backend.validate_escalated(self.action, self.state).status, Status.VALID)
        second = self.backend.validation_context
        self.assertEqual(second["context_id"], first["context_id"])
        self.assertNotEqual(second["candidate_id"], first["candidate_id"])
        self.assertEqual(second["ik_seed_limit"], 24)
        self.assertEqual(self.backend.compiler.tasks[0]["q_path"][0], before_q["A"])
        self.assertEqual(self.backend.compiler.held["A"]["name"], "red")
        # Final compilation must select the accepted context-specific policy,
        # and commit its complete world through exact path certificates.
        calls = len(self.geometry.calls)
        tasks = self.backend.compiler.compile(
            [SearchStep(self.action, 0., 1., self.state)], return_home=False)
        self.assertEqual(len(self.geometry.calls), calls)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(self.backend.compiler.held["A"]["name"], "red")

    def test_two_finite_strategies_exhaust_as_unknown_without_a_third_rpc_attempt(self):
        first = self.fail_first_reference()
        self.assertEqual(self.backend.validate_escalated(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["candidate_id"], first["candidate_id"])
        self.assertFalse(second["retryable"])
        calls = len(self.geometry.calls)
        self.assertLessEqual(sum(row[0] == "ik" for row in self.geometry.calls), 32)
        self.assertEqual(self.backend.validate_escalated(self.action, self.state).status, Status.UNKNOWN)
        self.assertEqual(len(self.geometry.calls), calls)
        self.assertEqual(self.backend.validation_context["candidate_id"], second["candidate_id"])
        self.assertFalse(self.backend.validation_context["retryable"])

    def test_escalation_of_one_context_does_not_change_another_context_baseline(self):
        first = self.fail_first_reference()
        self.assertEqual(self.backend.validate_escalated(self.action, self.state).status, Status.UNKNOWN)
        self.begin(2)
        self.backend.compiler.q["B"][0] += .1
        before = len(self.geometry.calls)
        self.assertEqual(self.backend.validate(self.action, self.state).status, Status.UNKNOWN)
        second = self.assert_conditional_failure()
        self.assertNotEqual(second["context_id"], first["context_id"])
        self.assertEqual(second["ik_seed_limit"], 8)
        self.assertLessEqual(len(self.geometry.calls)-before, 8)


class CandidateFailureClassificationTests(unittest.TestCase):
    def test_explicit_infeasibility_retains_invalid_without_sampling_escalation(self):
        engine, geometry, backend = backend_fixture()
        geometry.ik_failure = {"ok": False, "proven_infeasible": True,
            "problem": {"reason": "outside certified workspace"}}
        backend.begin_reference(1)
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.INVALID)
        self.assertEqual(backend.validation_context["classification"], "proven_infeasible")
        self.assertFalse(backend.validation_context["retryable"])
        self.assertEqual(sum(row[0] == "ik" for row in geometry.calls), 1)

    def test_finite_collision_candidates_remain_unknown_with_candidate_scope(self):
        engine, geometry, backend = backend_fixture()
        geometry.ik_failure = {"ok": False, "problem": {
            "reason": "table_collision:A", "pair": ["A_joint_2/link", "Workbench"]}}
        backend.begin_reference(1)
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(backend.validation_context["classification"], "ik_collision")
        self.assertEqual(backend.validation_context["failure_scope"], "candidate")
        self.assertTrue(backend.validation_context["temporary_cut"])
        record = backend.audit()["validations"][-1]
        raw = record["failure_evidence"]
        while isinstance(raw, dict) and "terminal_failure" in raw:
            raw = raw["terminal_failure"]["evidence"]
        self.assertEqual(raw[-1]["problem"]["pair"],
                         ["A_joint_2/link", "Workbench"])

    def test_rrt_exhaustion_is_not_reclassified_by_its_last_sample_collision(self):
        engine, geometry, backend = backend_fixture()
        geometry.plan_failure = {"ok": False,
            "error": "No exact collision-free solution within budget", "problem": {
                "reason": "table_collision:A", "pair": ["A_joint_3/link", "Workbench"]}}
        backend.begin_reference(1)
        result = backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(backend.validation_context["classification"], "rrt_exhausted")
        self.assertEqual(backend.validation_context["failure_scope"], "candidate")
        self.assertTrue(backend.validation_context["temporary_cut"])
        record = backend.audit()["validations"][-1]
        attempts = record["failure_evidence"]["attempts"]
        self.assertEqual([attempt["stage"] for attempt in attempts],
                         ["local_box", "global"]*(len(attempts)//2))
        self.assertGreaterEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()
