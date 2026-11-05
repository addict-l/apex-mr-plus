"""Constraint tests for the finite design model, not simulator acceptance."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifecycle import (CANDIDATES, InvalidPlan, check_resource_capacity, evaluate,
                       make_actions, nominal_schedule, qualify, search, validate_plan)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.outer, self.inner, self.weak = CANDIDATES

    def test_selected_candidate_is_best_finite_nominal_plan(self):
        result = search()
        selected = result["selected_plan"]
        self.assertEqual(result["counts"], {"evaluated": 12, "feasible": 3, "rejected": 9})
        self.assertEqual(selected["candidate"]["id"], "outer_front")
        self.assertEqual(selected["removal_order"], "after_gate")
        feasible = [d["objective"] for d in result["candidate_decisions"] if d["feasible"]]
        self.assertEqual(selected["nominal_metrics"]["objective"], min(feasible))
        self.assertTrue(selected["validation"]["valid_symbolic_plan"])

    def test_inner_is_trapped_only_when_gate_closes_first(self):
        with self.assertRaisesRegex(InvalidPlan, "support_trapped_by_closed_gate"):
            evaluate(self.inner, "after_gate")
        self.assertTrue(evaluate(self.inner, "before_gate")["validation"]["valid_symbolic_plan"])

    def test_weak_contact_and_insufficient_load_rating_are_rejected(self):
        with self.assertRaisesRegex(InvalidPlan, "insufficient_declared_contact_overlap"):
            qualify(self.weak)
        with self.assertRaisesRegex(InvalidPlan, "insufficient_declared_load_rating"):
            qualify(replace(self.outer, rated_load_N=10.0))

    def test_no_buffer_capacity_means_reverse_supply_deadlocks(self):
        with self.assertRaisesRegex(InvalidPlan, "dependency_cycle_or_buffer_deadlock"):
            evaluate(self.outer, "after_gate", ("K2", "K1"))
        with self.assertRaisesRegex(InvalidPlan, "resource_capacity_exceeded:shared_buffer"):
            check_resource_capacity([
                {"resource": "shared_buffer", "owner": "K1", "start_s": 0, "end_s": 5},
                {"resource": "shared_buffer", "owner": "K2", "start_s": 4, "end_s": 6}])

    def test_release_guard_survives_deleted_dependency(self):
        plan = evaluate(self.outer, "after_gate")
        actions = make_actions(self.outer, "after_gate")
        next(a for a in actions if a["id"] == "release_A")["predecessors"] = []
        plan["actions"] = nominal_schedule(actions)
        with self.assertRaisesRegex(InvalidPlan, "premature_A_release_without_load_handoff"):
            validate_plan(plan)

    def test_unload_requires_permanent_structure_even_without_edge(self):
        plan = evaluate(self.outer, "before_gate")
        actions = make_actions(self.outer, "before_gate")
        next(a for a in actions if a["id"] == "unload_support")["predecessors"] = ["release_A"]
        plan["actions"] = nominal_schedule(actions)
        with self.assertRaisesRegex(InvalidPlan, "premature_support_unload"):
            validate_plan(plan)

    def test_source_access_cannot_be_assigned_to_B(self):
        plan = copy.deepcopy(evaluate(self.outer, "after_gate"))
        next(a for a in plan["actions"] if a["id"] == "fetch_K1")["arm"] = "B"
        with self.assertRaises(InvalidPlan):
            validate_plan(plan)

    def test_missing_return_is_not_a_success(self):
        plan = evaluate(self.outer, "after_gate")
        plan["actions"] = [a for a in plan["actions"] if a["id"] != "return_support"]
        with self.assertRaisesRegex(InvalidPlan, "missing_or_unknown_lifecycle_action"):
            validate_plan(plan)

    def test_resources_include_continuous_initial_hold_and_buffer_storage(self):
        plan = evaluate(self.outer, "after_gate")
        hold = next(r for r in plan["resource_occupancy"] if r["owner"] == "initial_beam_hold")
        release = next(a for a in plan["actions"] if a["id"] == "release_A")
        self.assertEqual((hold["start_s"], hold["end_s"]), (0.0, release["start_s"]))
        buffer = [r for r in plan["resource_occupancy"] if r["resource"] == "shared_buffer"]
        self.assertEqual(len(buffer), 2)
        self.assertLessEqual(buffer[0]["end_s"], buffer[1]["start_s"])
        self.assertEqual(plan["validation"]["final_state"]["support"], "parked")


if __name__ == "__main__":
    unittest.main()
