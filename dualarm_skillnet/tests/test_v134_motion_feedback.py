"""C feedback contracts: controlled schedules and fake geometry only."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact, parse_fact
from skillnet.motion.base import MotionResult, NullBackend
from skillnet.pipeline import run
from skillnet.planning.search import SearchResult, SearchStep
from simbridge import backend as backend_module
from simbridge.backend import CoppeliaBackend
from simbridge.compiler import CompileFailure
from tests.test_coppelia_backend import backend_fixture


def schedule(action=None, state=frozenset(), status="solved"):
    steps = [] if action is None else [SearchStep(action, 0., 1., state)]
    return SearchResult(status, steps, float(bool(steps)), 1, 1,
                        {"cut": 0}, frozenset(), 0.)


class SequenceBackend(NullBackend):
    def __init__(self, bad=(), upgraded=Status.UNKNOWN):
        self.bad = set(bad)
        self.upgraded = upgraded
        self.calls = []
        self.rounds = []

    def validate(self, action, state):
        self.calls.append((action.key, frozenset(state), 8))
        status = Status.UNKNOWN if action.key in self.bad else Status.VALID
        return MotionResult(status, "mock finite seeds", [])

    def validate_escalated(self, action, state):
        self.calls.append((action.key, frozenset(state), 24))
        return MotionResult(self.upgraded, "mock upgraded finite seeds", [])

    def record_feedback_round(self, record):
        self.rounds.append(deepcopy(record))


class MotionReasonMappingTests(unittest.TestCase):
    def check_failure(self, evidence, *, queries=(), detail="mock failed IK"):
        engine, _, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        def fail(*args, **kwargs):
            backend.compiler.evidence.extend(deepcopy(queries))
            raise CompileFailure(Status.UNKNOWN, detail, "r1", deepcopy(evidence))
        with patch.object(backend.compiler, "compile_action", side_effect=fail):
            result = backend.validate(action, engine.initial)
        return result, backend.validation_records[-1]

    def test_independent_collision_seed_maps_invalid_and_keeps_raw_evidence(self):
        evidence = [{"ok": False, "stats": {"candidates": 0}}] * 7 + [
            {"ok": False, "stats": {"rejections": [{"reason": "finger_contact", "pair": ["A_finger", "beam"]}]}}]
        result, record = self.check_failure(evidence)
        self.assertEqual(result.status, Status.INVALID)
        self.assertEqual(record["failure_evidence"], evidence)
        self.assertEqual(record["original_status"], "UNKNOWN")
        self.assertEqual(record["classification"]["kind"], "ik_collision")
        self.assertTrue(record["classification"]["basis"])

    def test_collision_reason_prefix_does_not_require_pair(self):
        for reason in ("table_collision:A", "interarm_collision"):
            with self.subTest(reason=reason):
                result, _ = self.check_failure([{"ok": False, "problem": {"reason": reason}}])
                self.assertEqual(result.status, Status.INVALID)

    def test_rrt_nested_pair_and_earlier_ik_rejection_stay_unknown(self):
        problem = {"reason": "table_collision:A", "pair": ["A_link", "Workbench"]}
        evidence = {"attempts": [{"stage": "local_box", "problem": problem},
                                  {"stage": "global", "problem": problem}],
                    "last_answer": {"ok": False, "problem": problem}}
        queries = [{"request": "r1", "arm": "A", "query": "IK",
                    "answer": {"ok": False, "problem": problem}},
                   {"request": "r1", "arm": "A", "query": "RRTConnect",
                    "answer": evidence["last_answer"], "search_stage": "global"}]
        result, record = self.check_failure(evidence, queries=queries, detail="RRT-Connect failed")
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(record["classification"]["kind"], "rrt_exhausted")

    def test_branch_wrapper_does_not_inherit_nested_pair_but_other_seed_does(self):
        branch = {"ok": False, "problem": {"reason": "IK_branch_discontinuity", "limit_rad": .25},
                  "answer": {"ok": True, "problem": {"pair": ["stale_a", "stale_b"]}}}
        result, _ = self.check_failure([branch])
        self.assertEqual(result.status, Status.UNKNOWN)
        result, _ = self.check_failure([branch, {"ok": False, "problem": {
            "reason": "obstacle_collision:A", "pair": ["A_link", "Baseplate"]}}])
        self.assertEqual(result.status, Status.INVALID)

    def test_timeout_wrapper_does_not_inherit_last_answer_pair(self):
        result, _ = self.check_failure({"timeout": True, "last_answer": {
            "problem": {"reason": "table_collision:A", "pair": ["A_link", "table"]}}})
        self.assertEqual(result.status, Status.UNKNOWN)


class PipelineMotionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.engine, _, self.real = backend_fixture()
        self.problem = self.engine.problem
        self.problem.proposal = None
        self.first = self.engine.parse_action("pick(A, red, stock_l1)")
        self.other = self.engine.parse_action("pick(B, blue, stock_r1)")

    def execute(self, backend, search_function, **kwargs):
        with patch("skillnet.pipeline.search", side_effect=search_function) as searches, \
                patch("skillnet.pipeline.build_task_graph", return_value=object()):
            result = run(self.problem, backend=backend, prune=False, **kwargs)
        return result, searches

    def test_collision_unknown_is_mapped_invalid_cut_then_alternative_succeeds(self):
        first = True
        def compile_fake(action, state, **kwargs):
            nonlocal first
            if first:
                first = False
                raise CompileFailure(Status.UNKNOWN, "mock collision", "r1", [
                    {"ok": False, "problem": {"reason": "interarm_collision", "pair": ["A_link", "B_link"]}}])
            return []
        def searches(engine, **kwargs):
            if kwargs["cuts"]:
                self.assertIn((self.engine.initial, self.first.key), kwargs["cuts"])
                return schedule(self.other, self.engine.initial)
            return schedule(self.first, self.engine.initial)
        with patch.object(self.real.compiler, "compile_action", side_effect=compile_fake):
            result, searches = self.execute(self.real, searches)
        self.assertEqual(result.status, "solved")
        self.assertEqual(searches.call_count, 2)
        self.assertEqual([r["status"] for r in self.real.validation_records], ["INVALID", "VALID"])

    def test_noncollision_unknown_upgraded_valid_continues_without_cut(self):
        backend = SequenceBackend([self.first.key], upgraded=Status.VALID)
        result, searches = self.execute(backend, lambda *a, **k: schedule(self.first, self.engine.initial))
        self.assertEqual(result.status, "solved")
        self.assertEqual(searches.call_count, 1)
        self.assertEqual([c[2] for c in backend.calls], [8, 24])
        self.assertTrue(all(item.status == Status.VALID for item in result.motion))

    def test_persistent_unknown_is_cut_and_alternative_route_succeeds(self):
        backend = SequenceBackend([self.first.key])
        def searches(engine, **kwargs):
            return schedule(self.other if kwargs["cuts"] else self.first, self.engine.initial)
        result, calls = self.execute(backend, searches)
        self.assertEqual(result.status, "solved")
        self.assertEqual(calls.call_count, 2)
        self.assertTrue(any("已规避" in line and "重新搜索" in line for line in result.feedback))
        self.assertTrue(backend.rounds)

    def test_all_unknown_routes_exhaust_as_unknown_not_global_no_plan(self):
        backend = SequenceBackend([self.first.key, self.other.key])
        def searches(engine, **kwargs):
            for action in (self.first, self.other):
                if (self.engine.initial, action.key) not in kwargs["cuts"]:
                    return schedule(action, self.engine.initial)
            return schedule(status="no_plan")
        result, searches = self.execute(backend, searches)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.schedule.status, "no_plan")
        self.assertEqual(searches.call_count, 3)
        self.assertIsNone(result.graph)

    def test_three_distinct_state_cuts_downgrade_only_this_run_reach(self):
        backend = SequenceBackend([self.first.key])
        fact = parse_fact("reach(A, stock_l1)")
        original = self.problem.oracle.check(fact)
        def searches(engine, **kwargs):
            count = len(kwargs["cuts"])
            if count == 3:
                self.assertEqual(engine.oracle.check(fact).status, Status.INVALID)
                self.assertIn("运动级三次规避", engine.oracle.check(fact).source)
                return schedule(status="no_plan")
            state = self.engine.initial | {Fact("marker", ("abc"[count],))}
            return schedule(self.first, state)
        result, searches = self.execute(backend, searches, max_rounds=4)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(searches.call_count, 4)
        self.assertEqual(self.problem.oracle.check(fact), original)
        self.assertEqual(result.problem.oracle.check(fact).status, Status.INVALID)
        self.assertTrue(any("运动级三次规避" in line for line in result.feedback))

    def test_round_limit_is_budget_and_explicit_limit_is_preserved(self):
        backend = SequenceBackend([self.first.key])
        result, searches = self.execute(backend, lambda *a, **k: schedule(self.first, self.engine.initial), max_rounds=1)
        self.assertEqual(result.status, "budget")
        self.assertEqual(searches.call_count, 1)

    def test_default_round_limit_real_twelve_legacy_six_explicit_wins(self):
        for backend, explicit, expected in [(self.real, None, 12), (SequenceBackend(), None, 6),
                                             (self.real, 2, 2)]:
            with self.subTest(real=isinstance(backend, CoppeliaBackend), explicit=explicit):
                kwargs = {} if explicit is None else {"max_rounds": explicit}
                result, _ = self.execute(backend, lambda *a, **k: schedule(status="budget"), **kwargs)
                self.assertEqual(result.status, "budget")
                rounds = backend.audit()["feedback_rounds"] if isinstance(backend, CoppeliaBackend) else backend.rounds
                self.assertEqual(rounds[-1]["round_limit"], expected)

    def test_support_off_downgrade_maps_to_its_slot_without_liftable(self):
        engine, _, backend = backend_fixture("s3a_cantilever")
        problem = engine.problem
        problem.proposal = None
        action = engine.parse_action("support_off(A, beam)")
        fake = SequenceBackend([action.key])
        def searches(local, **kwargs):
            n = len(kwargs["cuts"])
            if n == 3:
                self.assertEqual(local.oracle.check(parse_fact("reach(A, beam)")).status, Status.INVALID)
                return schedule(status="no_plan")
            return schedule(action, engine.initial | {Fact("marker", ("abc"[n],))})
        with patch("skillnet.pipeline.search", side_effect=searches), \
                patch("skillnet.pipeline.build_task_graph", return_value=object()):
            result = run(problem, backend=fake, prune=False, max_rounds=4)
        self.assertEqual(result.status, "unknown")
        downgraded = [line for line in result.feedback if "运动级三次规避" in line]
        self.assertTrue(any("reach(A, beam)" in line for line in downgraded))
        self.assertFalse(any("liftable" in line for line in downgraded))


class EscalationCertificateTests(unittest.TestCase):
    def test_exhausted_upgrade_runs_exactly_24_distinct_trials_once(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        geometry.ik_failure = {"ok": False, "stats": {"candidates": 0}}
        self.assertEqual(backend.validate(action, engine.initial).status, Status.UNKNOWN)
        self.assertEqual(len(geometry.calls), 8)
        self.assertEqual(backend.validate_escalated(action, engine.initial).status, Status.UNKNOWN)
        self.assertEqual(len(geometry.calls), 32)
        seeds = [tuple(row[2]["options"]["seed"]) for row in geometry.calls[8:]]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(backend.validate_escalated(action, engine.initial).status, Status.UNKNOWN)
        self.assertEqual(len(geometry.calls), 32)
        self.assertEqual(backend.validation_context["escalation_skipped"], "already_at_least_24")

    def test_seed_helper_is_deterministic_unique_and_does_not_mutate_cfg(self):
        cfg = {"ik_seeds": [[0.] * 6, [0.] * 6, [1.] * 6]}
        old = deepcopy(cfg)
        for count in (24, 32):
            rows = backend_module.deterministic_ik_seeds(cfg, count)
            self.assertGreaterEqual(len(rows), count)
            self.assertEqual(len({tuple(row) for row in rows}), len(rows))
            self.assertEqual(rows, backend_module.deterministic_ik_seeds(cfg, count))
            self.assertEqual(rows[:2], [[0.] * 6, [1.] * 6])
        self.assertEqual(cfg, old)

    def test_escalation_preserves_prefix_uses_24_distinct_seeds_and_final_cache(self):
        engine, geometry, backend = backend_fixture()
        compiler = backend.compiler
        first = engine.parse_action("pick(A, red, stock_l1)")
        second = engine.parse_action("pick(B, blue, stock_r1)")
        state = engine.start(first, engine.initial)
        self.assertEqual(backend.validate(first, engine.initial).status, Status.VALID)
        prefix = deepcopy(compiler._snapshot())
        solve = geometry.solve_ik
        seen = []
        def finite_seeds(arm, position, quaternion, **kwargs):
            if arm == "B":
                seen.append((compiler.ik_seeds, tuple(kwargs["options"]["seed"])))
                if compiler.ik_seeds == 8:
                    return {"ok": False, "stats": {"candidates": 0}}
            return solve(arm, position, quaternion, **kwargs)
        geometry.solve_ik = finite_seeds
        self.assertEqual(backend.validate(second, state).status, Status.UNKNOWN)
        self.assertEqual(compiler._snapshot(), prefix)
        self.assertEqual(backend.validate_escalated(second, state).status, Status.VALID)
        self.assertIs(backend.compiler, compiler)
        self.assertEqual(backend.validation_records[-1]["start_q"], prefix[0])
        self.assertEqual(backend.validation_records[-1]["held"], prefix[1])
        self.assertEqual(len({seed for limit, seed in seen if limit == 8}), 8)
        expanded = [seed for limit, seed in seen if limit == 24]
        self.assertGreaterEqual(len(expanded), 24)
        self.assertEqual(len(set(expanded[:24])), 24)
        steps = [SearchStep(first, 0., 1., engine.initial), SearchStep(second, 0., 1., state)]
        calls = len(geometry.calls)
        compiler.compile(steps, return_home=False)
        self.assertEqual(len(geometry.calls), calls)
        scratch = backend._scratch_compiler()
        scratch.compile(steps, return_home=False)
        self.assertEqual(len(geometry.calls), calls)
        expanded_certificates = [entry for entry in compiler.path_cache.values()
                                 if entry["context"]["limits"][0] == 24]
        self.assertTrue(expanded_certificates)
        self.assertTrue(all("seed" in str(entry["context"]["geometry_version"])
                            for entry in expanded_certificates))
        backend.reset()
        backend.compiler.q["A"][0] += .1
        backend.validate(second, state)
        self.assertGreater(len(geometry.calls), calls)

    def test_escalation_rejects_a_different_action_or_state(self):
        engine, geometry, backend = backend_fixture()
        first = engine.parse_action("pick(A, red, stock_l1)")
        second = engine.parse_action("pick(B, blue, stock_r1)")
        geometry.ik_failure = {"ok": False, "stats": {"candidates": 0}}
        backend.validate(first, engine.initial)
        calls = len(geometry.calls)
        with self.assertRaises(ValueError):
            backend.validate_escalated(second, engine.initial)
        self.assertEqual(len(geometry.calls), calls)


if __name__ == "__main__":
    unittest.main()
