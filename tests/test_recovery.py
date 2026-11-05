"""Recovery behavior: physical role transitions, partial progress and safe reuse."""
from copy import deepcopy
import unittest

from recovery.__main__ import run_case
from recovery.model import Action, Problem, apply_event
from recovery.oracles import DeclaredCapabilities, EvidenceCache, PlanarWrenchOracle
from recovery.planner import affected_tasks, plan_recovery
from recovery.scenarios import cases, get_case
from recovery.schedule import action_schedule, build_schedule, validate_schedule


def changed(name):
    before = get_case(name)
    return apply_event(before, before.data["event"])[0]


class RecoveryPlanningTests(unittest.TestCase):
    def test_blocked_predecessor_does_not_block_independent_work_or_become_success(self):
        result = run_case(get_case("independent_work"))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["final_state"]["completed"], ["block3"])
        self.assertEqual(result["unfinished_tasks"], ["block1", "block2"])
        self.assertEqual(result["schedule"][0]["start"], 0.)
        self.assertTrue(result["validation"]["valid"])
        self.assertFalse(result["validation"]["all_goals_completed_and_resources_released"])

    def test_ready_work_overlaps_slow_predecessor_then_dependent_task_waits(self):
        problem = get_case("independent_work")  # Before its event.
        plan = plan_recovery(problem)
        graph = build_schedule(problem, plan)
        rows = {r["target"]: r for r in action_schedule(plan, graph)}
        self.assertEqual(rows["block1"]["start"], 0.)
        self.assertEqual(rows["block3"]["start"], 0.)
        self.assertGreaterEqual(rows["block2"]["start"], rows["block1"]["end"])
        self.assertEqual(plan["predicted_makespan_s"], 7.)

    def test_simple_reassignment_uses_existing_source_and_preserves_done_facts(self):
        result = run_case(get_case("direct_swap"), policy="reassign")
        self.assertTrue(result["solved"])
        self.assertEqual([r["kind"] for r in result["actions"]], ["pick", "install"])
        self.assertTrue(all(r["arm"] == "A" for r in result["actions"]))
        self.assertIn("beam_ready", result["final_state"]["completed"])
        self.assertTrue(result["allocation_hint"]["solver_called"])

    def test_held_object_requires_real_handover_actions_in_the_discrete_model(self):
        fixed_network = run_case(get_case("held_handover"), policy="reassign")
        self.assertEqual(fixed_network["status"], "no_plan")
        result = run_case(get_case("held_handover"))
        self.assertTrue(result["solved"])
        self.assertEqual(result["actions"][0]["kind"], "handover")
        self.assertEqual(set(result["actions"][0]["resources"]), {"A", "B"})
        self.assertEqual(result["actions"][1]["arm"], "A")

    def test_support_transfer_overlaps_and_final_support_outlasts_installation(self):
        result = run_case(get_case("support_swap"))
        self.assertEqual([r["kind"] for r in result["actions"]],
                         ["place", "acquire_support", "release_support", "pick", "install", "release_support"])
        intervals = {r["arm"]: r for r in result["validation"]["support_intervals"]}
        install = next(r for r in result["schedule"] if r["kind"] == "install")
        self.assertLess(intervals["B"]["start"], intervals["A"]["end"])
        self.assertLessEqual(intervals["B"]["start"], install["start"])
        self.assertGreaterEqual(intervals["B"]["end"], install["end"])
        self.assertTrue(result["validation"]["valid"])
        self.assertFalse(result["final_state"]["supports"])
        self.assertEqual(result["predicted_makespan_s"], 7.)

    def test_endpoints_declared_valid_do_not_override_insufficient_support_capacity(self):
        problem = changed("weak_support")
        self.assertEqual(problem.status("B", "support", "right"), "VALID")
        result = plan_recovery(problem)
        self.assertEqual(result["status"], "no_plan")
        self.assertTrue(result["search"]["rejections"]["wrench_invalid"] > 0)
        self.assertEqual(result["final_state"], problem.initial.to_dict())

    def test_search_selects_an_alternative_contact_without_a_scripted_recovery_chain(self):
        result = run_case(get_case("alternative_contact"))
        self.assertTrue(result["solved"])
        selected = next(r for r in result["actions"] if r["kind"] == "acquire_support")
        self.assertEqual(selected["target"], "right_far")

    def test_no_buffer_means_no_fabricated_release_or_teleportation(self):
        problem = changed("no_buffer")
        result = plan_recovery(problem)
        self.assertEqual(result["status"], "no_plan")
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["final_state"], problem.initial.to_dict())

    def test_unknown_and_search_limit_are_not_infeasibility_proofs(self):
        self.assertEqual(plan_recovery(changed("unknown_substitute"))["status"], "unknown")
        result = plan_recovery(changed("support_swap"), max_expansions=1)
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["search"]["budget_exhausted"])
        self.assertEqual(result["actions"], [])

    def test_completed_installation_is_never_repeated_after_an_event(self):
        data = cases()["support_swap"]
        data["snapshot"]["locations"]["cap"] = "installed"
        data["snapshot"]["completed"].append("fit_cap")
        result = run_case(Problem(data))
        self.assertTrue(result["solved"])
        self.assertEqual([a["kind"] for a in result["actions"]], ["release_support"])
        self.assertEqual(result["repair"]["preserved_completed"], ["beam_ready", "fit_cap"])

    def test_tpg_never_claims_abstract_graph_is_motion_validated(self):
        result = run_case(get_case("support_swap"))
        certificate = result["tpg"]["certificate"]
        self.assertFalse(certificate["collision_oracle_supplied"])
        self.assertFalse(certificate["all_nodes_have_geometry"])
        self.assertTrue(certificate["motion_compilation_required"])
        self.assertFalse(result["executable_in_coppeliasim"])


class RecoveryScheduleTests(unittest.TestCase):
    def test_delayed_acquisition_and_installation_do_not_release_support_early(self):
        problem = changed("support_swap")
        plan = plan_recovery(problem)
        graph = build_schedule(problem, plan)
        acquire = next(a for a in plan["actions"] if a["kind"] == "acquire_support")
        install = next(a for a in plan["actions"] if a["kind"] == "install")
        delays = {f"{acquire['id']}:B": 2., f"{install['id']}:A": 9.}
        rows = action_schedule(plan, graph, delays)
        audit = validate_schedule(problem, rows, expected_final=plan["final_state"])
        self.assertTrue(audit["valid"], audit["failures"])
        self.assertEqual(audit["predicted_makespan_s"], 18.)

    def test_handover_reserves_both_arms_until_both_completion_events(self):
        data = cases()["held_handover"]
        data["tasks"].append({"id": "independent", "kind": "work", "original_arm": "B", "duration": 2.,
                              "dependencies": ["beam_ready"], "independent_of_assembly": True})
        data["capabilities"]["B"]["work"]["independent"] = "VALID"
        data["capabilities"]["A"]["work"]["independent"] = "INVALID"
        problem = apply_event(Problem(data), data["event"])[0]
        plan = plan_recovery(problem)
        self.assertTrue(plan["solved"])
        graph = build_schedule(problem, plan)
        handover = next(r for r in plan["actions"] if r["kind"] == "handover")
        rows = action_schedule(plan, graph, {f"{handover['id']}:B": 4.})
        transfer_end = next(r["end"] for r in rows if r["id"] == handover["id"])
        for row in rows:
            if row["kind"] in {"install", "work"}:
                self.assertGreaterEqual(row["start"], transfer_end)
        audit = validate_schedule(problem, rows, expected_final=plan["final_state"])
        self.assertTrue(audit["valid"], audit["failures"])

    def test_independent_audit_rejects_early_support_release(self):
        result = run_case(get_case("support_swap"))
        rows = deepcopy(result["schedule"])
        release = next(r for r in rows if r["kind"] == "release_support" and r["arm"] == "A")
        release.update(start=0., end=.5)
        audit = validate_schedule(changed("support_swap"), rows)
        self.assertFalse(audit["valid"])
        self.assertTrue(any("equilibrium" in f for f in audit["failures"]))

    def test_support_arm_is_occupied_between_establishment_and_release(self):
        result = run_case(get_case("support_swap"))
        data = changed("support_swap").data
        data["tasks"].append({"id": "bad_work", "kind": "work", "original_arm": "B", "duration": .1,
                              "dependencies": [], "independent_of_assembly": True})
        data["capabilities"]["B"]["work"]["bad_work"] = "VALID"
        rows = deepcopy(result["schedule"])
        rows.append({**Action("work", "B", "bad_work", .1).to_dict(), "id": "bad", "start": 4., "end": 4.1})
        audit = validate_schedule(Problem(data), rows)
        self.assertFalse(audit["valid"])
        self.assertTrue(any("invalid precondition" in f for f in audit["failures"]))

    def test_declared_skill_cannot_be_completed_early_without_feedback(self):
        result = run_case(get_case("direct_swap"))
        rows = deepcopy(result["schedule"])
        rows[-1]["end"] -= 1.
        audit = validate_schedule(changed("direct_swap"), rows)
        self.assertFalse(audit["valid"])
        self.assertTrue(any("declared duration" in f for f in audit["failures"]))


class RecoveryEvidenceTests(unittest.TestCase):
    def test_changed_capability_invalidates_only_dependent_evidence(self):
        before = get_case("support_swap")
        cache = EvidenceCache()
        oracle = DeclaredCapabilities(before, cache)
        oracle.check("A", "install", "fit_cap")
        oracle.check("B", "install", "fit_cap")
        PlanarWrenchOracle(before, cache).check(before.initial)
        after, facts = apply_event(before, before.data["event"])
        invalidated = cache.invalidate(facts)
        self.assertEqual(len(invalidated), 1)
        self.assertEqual(invalidated[0]["kind"], "capability")
        self.assertEqual(DeclaredCapabilities(after, cache).check("B", "install", "fit_cap")["status"], "INVALID")
        self.assertEqual(DeclaredCapabilities(after, cache).check("A", "install", "fit_cap")["status"], "VALID")
        PlanarWrenchOracle(after, cache).check(after.initial)
        self.assertEqual(cache.counts["wrench_hits"], 1)
        self.assertEqual(before.status("B", "install", "fit_cap"), "VALID")
        self.assertEqual(after.initial, before.initial)

    def test_physics_change_invalidates_lp_but_not_capability_facts(self):
        before = get_case("support_swap")
        cache = EvidenceCache()
        DeclaredCapabilities(before, cache).check("A", "install", "fit_cap")
        PlanarWrenchOracle(before, cache).check(before.initial)
        event = {"updates": [{"fact": "physics/contacts/left/normal_limit_N", "value": 0.}]}
        after, facts = apply_event(before, event)
        self.assertEqual([e["kind"] for e in cache.invalidate(facts)], ["wrench"])
        self.assertEqual(PlanarWrenchOracle(after, cache).check(after.initial)["status"], "INVALID")
        DeclaredCapabilities(after, cache).check("A", "install", "fit_cap")
        self.assertEqual(cache.counts["capability_hits"], 1)

    def test_warm_and_cold_search_agree_while_warm_search_reuses_lp_witnesses(self):
        warm = run_case(get_case("support_swap"), reuse=True)
        cold = run_case(get_case("support_swap"), reuse=False)
        self.assertEqual(warm["final_state"], cold["final_state"])
        self.assertEqual(warm["predicted_makespan_s"], cold["predicted_makespan_s"])
        self.assertLess(warm["evidence_queries"].get("wrench_misses", 0), cold["evidence_queries"]["wrench_misses"])

    def test_unknown_solver_answer_is_not_cached_as_a_permanent_cut(self):
        cache = EvidenceCache()
        calls = []
        def compute():
            calls.append(1)
            return {"status": "UNKNOWN"}
        for _ in range(2):
            cache.query("test", "x", {"model": "v1"}, compute)
        self.assertEqual(len(calls), 2)
        self.assertFalse(cache.entries)

    def test_task_impact_propagates_to_descendants_and_preserves_completed_history(self):
        before = get_case("independent_work")
        _, facts = apply_event(before, before.data["event"])
        report = affected_tasks(before, facts)
        self.assertEqual(report["affected_tasks"], ["block1", "block2"])
        self.assertEqual(report["other_pending_tasks"], ["block3"])

    def test_horizontal_balance_and_friction_are_checked_in_addition_to_moment(self):
        data = cases()["independent_work"]
        data["physics"] = {"gravity_m_s2": 9.81,
                           "connection": {"force_x_limit_N": 0., "force_z_limit_N": 0., "moment_limit_Nm": 0.},
                           "loads": [{"fx_N": 4., "fz_N": -5., "x_m": 0., "z_m": 0.}],
                           "contacts": {"pad": {"arm": "A", "x_m": 0., "z_m": 0.,
                                                "normal_limit_N": 10., "friction": .5}}}
        data["snapshot"]["supports"] = {"A": "pad"}
        low = Problem(data)
        self.assertEqual(PlanarWrenchOracle(low, EvidenceCache()).check(low.initial)["status"], "INVALID")
        data["physics"]["contacts"]["pad"]["friction"] = 1.
        high = Problem(data)
        answer = PlanarWrenchOracle(high, EvidenceCache()).check(high.initial)
        self.assertEqual(answer["status"], "VALID")
        self.assertLess(answer["balance_residual"], 1e-7)


class RecoveryInputTests(unittest.TestCase):
    def test_reject_running_snapshot_and_holding_support_conflict(self):
        data = cases()["support_swap"]
        data["snapshot"]["running"] = ["unfinished_skill"]
        with self.assertRaisesRegex(ValueError, "quiescent"):
            Problem(data)
        del data["snapshot"]["running"]
        data["snapshot"]["locations"]["cap"] = "held:A"
        with self.assertRaisesRegex(ValueError, "empty arm"):
            Problem(data)

    def test_no_history_edits_or_unknown_fact_paths_in_events(self):
        for fact in ("snapshot/completed", "capabilities/A/install/nonexistent"):
            with self.assertRaises(ValueError):
                apply_event(get_case("direct_swap"), {"updates": [{"fact": fact, "value": "VALID"}]})

    def test_dependency_cycles_and_nonfinite_physics_fail_fast(self):
        data = cases()["independent_work"]
        data["tasks"][0]["dependencies"] = ["block2"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            Problem(data)
        data = cases()["support_swap"]
        data["physics"]["contacts"]["right"]["normal_limit_N"] = float("nan")
        with self.assertRaises(ValueError):
            Problem(data)

    def test_missing_capability_is_unknown_and_scenario_name_is_not_a_path(self):
        problem = get_case("direct_swap")
        self.assertEqual(problem.status("A", "pick", "undeclared"), "UNKNOWN")
        data = cases()["direct_swap"]
        data["name"] = "../../somewhere"
        with self.assertRaises(ValueError):
            Problem(data)


if __name__ == "__main__":
    unittest.main()
