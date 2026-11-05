"""Tests use the real scipy HiGHS solver, including retry and coupled choices."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.allocation import solve_allocation


def example_problem():
    supports = []
    for candidate, cost, overlap, blocked in [
            ("outer_front", 8., .04, False), ("inner_pocket", 6., .04, True),
            ("weak_narrow", .1, .005, False)]:
        for order in ["before_gate", "after_gate"]:
            supports.append({"id": candidate + ":" + order, "candidate": candidate,
                             "removal_order": order, "placer_arm": "B", "remover_arm": "B",
                             "feasible": True, "cost": cost, "rated_load_N": 40.,
                             "contact_overlap_m": overlap, "blocked_after_gate": blocked})
    tasks, modes, handoffs = [], [], []
    for part in ["K1", "K2"]:
        for kind in ["supply", "install"]:
            task = kind + "_" + part
            tasks.append({"id": task, "kind": kind, "part": part})
            for arm in ["A", "B"]:
                cost = (4. if arm == "A" else .1) if kind == "supply" else (12. if arm == "A" else 2.)
                modes.append({"id": task + ":" + arm, "task": task,
                              "arm": arm, "feasible": True, "cost": cost, "grasp": "top"})
        handoffs.append({"id": "handoff_"+part, "part": part,
                         "supply_task": "supply_"+part, "install_task": "install_"+part,
                         "from_arm": "A", "to_arm": "B", "feasible": True, "cost": 1.})
    tasks.append({"id": "close_gate", "kind": "gate", "part": "Gate"})
    modes += [{"id": "close_gate:"+arm, "task": "close_gate", "arm": arm,
               "feasible": True, "cost": 2. if arm == "B" else 6.} for arm in ["A", "B"]]
    return {"support_modes": supports, "tasks": tasks, "task_modes": modes,
            "handoffs": handoffs, "assembly_order": ["K1", "K2"], "load_balance_weight": .15}


class AllocationTests(unittest.TestCase):
    def test_actual_solver_joint_selection_and_buffer_contract(self):
        result = solve_allocation(example_problem())
        self.assertTrue(result["solver_called"])
        self.assertTrue(result["optimal"])
        self.assertEqual(result["solver_status"], 0)
        self.assertEqual(result["mip_gap"], 0.)
        self.assertEqual(result["selected_support"]["candidate"], "inner_pocket")
        self.assertEqual(result["selected_support"]["removal_order"], "before_gate")
        self.assertEqual(result["task_assignments"]["supply_K1"], "A")
        self.assertEqual(result["task_assignments"]["install_K1"], "B")
        self.assertEqual(len(result["handoffs"]), 2)
        self.assertIn({"from": "pick_K1", "to": "deposit_K2", "reason": "shared_buffer_capacity_one"},
                      result["required_buffer_edges"])
        self.assertAlmostEqual(result["objective"], result["solver_objective"])
        json.dumps(result, allow_nan=False)

    def test_geometry_failure_changes_arm_choice_and_avoids_handoff(self):
        problem = example_problem()
        next(m for m in problem["task_modes"] if m["id"] == "install_K1:B")["feasible"] = False
        result = solve_allocation(problem)
        self.assertTrue(result["solved"])
        self.assertEqual(result["task_assignments"]["install_K1"], "A")
        self.assertEqual([h["part"] for h in result["handoffs"]], ["K2"])

    def test_unqualified_handoff_forces_direct_A_transport(self):
        problem = example_problem()
        problem["handoffs"] = []
        result = solve_allocation(problem)
        self.assertTrue(result["solved"])
        self.assertEqual(result["task_assignments"]["install_K1"], "A")
        self.assertEqual(result["handoffs"], [])

    def test_false_handoff_and_infeasible_direct_mode_makes_problem_infeasible(self):
        problem = example_problem()
        problem["handoffs"][0]["feasible"] = False
        next(m for m in problem["task_modes"] if m["id"] == "install_K1:A")["feasible"] = False
        result = solve_allocation(problem)
        self.assertFalse(result["solved"])
        self.assertEqual(result["status"], "infeasible")

    def test_initial_holder_contact_and_gate_trap_are_hard_cuts(self):
        problem = example_problem()
        problem["support_modes"][0]["placer_arm"] = "A"
        result = solve_allocation(problem)
        reasons = {r["reason"] for r in result["rejected_modes"]}
        self.assertTrue({"initial_holder_cannot_place_support", "insufficient_contact_overlap",
                         "closed_gate_traps_support", "A_exclusive_supply_access"} <= reasons)

    def test_insufficient_load_does_not_pass_due_to_low_cost(self):
        problem = example_problem()
        for mode in problem["support_modes"]:
            mode["rated_load_N"] = 10.
        result = solve_allocation(problem)
        self.assertFalse(result["solved"])
        self.assertEqual(result["solver_status"], 2)

    def test_single_mode_ban_replans_without_banning_candidate_family(self):
        first = solve_allocation(example_problem())
        selected = first["selected_support"]["id"]
        second = solve_allocation(example_problem(), banned_modes=[selected])
        self.assertTrue(second["solved"])
        self.assertEqual(second["selected_support"]["candidate"], "outer_front")
        self.assertNotIn(selected, second["selected_mode_ids"])

    def test_no_good_cut_excludes_exact_joint_solution_only(self):
        problem = example_problem()
        first = solve_allocation(problem)
        second = solve_allocation(problem, banned_combinations=[first["selected_mode_ids"]])
        self.assertTrue(second["solved"])
        self.assertNotEqual(set(first["selected_mode_ids"]), set(second["selected_mode_ids"]))
        self.assertGreaterEqual(second["objective"], first["objective"] - 1e-6)

    def test_support_compatibility_couples_task_and_support_decisions(self):
        problem = example_problem()
        for mode in problem["task_modes"]:
            if mode["id"] == "install_K1:B":
                mode["compatible_support_modes"] = ["outer_front:after_gate"]
        result = solve_allocation(problem)
        self.assertEqual(result["selected_support"]["id"], "outer_front:after_gate")
        self.assertEqual(result["task_assignments"]["install_K1"], "B")

    def test_bad_retry_id_and_nonfinite_cost_are_input_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown mode"):
            solve_allocation(example_problem(), banned_modes=["does_not_exist"])
        problem = example_problem()
        problem["task_modes"][0]["cost"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_allocation(problem)


if __name__ == "__main__":
    unittest.main()
