"""Allocation checks against changes in abilities, support and objective."""
from copy import deepcopy
import unittest

from normal.allocation import solve_allocation, support_requirement


def part(name, *, support=False):
    return {
        "id": name, "mass": .10, "insertion_force": 8.,
        "lever_arm": .12 if support else .01,
        "connection_moment_limit": .30,
        "grasps": [{"id": "top"}, {"id": "side"}],
    }


def support(arm, feasible=True, cost=1.):
    return {"arm": arm, "feasible": feasible, "cost": cost,
            "tcp_position": [.1, .2, .8], "quaternion": [0, 0, 0, 1]}


def candidate(name, arm, *, grasp="top", cost=1., feasible=True, supports=None):
    return {"part": name, "arm": arm, "grasp": grasp, "cost": cost,
            "feasible": feasible, "support_candidates": supports or []}


class AllocationTests(unittest.TestCase):
    def test_lost_capability_reassigns_every_part(self):
        problem = {"arms": ["A", "B"], "parts": [part("p1"), part("p2")],
                   "candidates": [candidate(p, a, feasible=a == "B")
                                  for p in ["p1", "p2"] for a in ["A", "B"]]}
        result = solve_allocation(problem)
        self.assertTrue(result["solved"])
        self.assertEqual(result["status"], "optimal")
        self.assertEqual([choice["arm"] for choice in result["choices"]], ["B", "B"])
        self.assertTrue(result["solver_called"])
        self.assertEqual(result["mip_gap"], 0.)

    def test_required_support_cannot_be_omitted_even_if_cheap(self):
        problem = {"arms": ["A", "B"], "parts": [part("beam", support=True)],
                   "candidates": [candidate("beam", "A", supports=[support("B", cost=3.)]),
                                  candidate("beam", "B", cost=.1, supports=[support("A", False)])]}
        result = solve_allocation(problem)
        self.assertTrue(result["solved"])
        chosen = result["choices"][0]
        self.assertEqual((chosen["arm"], chosen["support_arm"]), ("A", "B"))
        self.assertTrue(chosen["support_required"])
        self.assertAlmostEqual(chosen["mode_cost"], 4.)
        self.assertTrue(any("load-moment" in row["reason"] for row in result["rejected_candidates"]))

    def test_same_arm_support_is_rejected_and_no_fake_solution(self):
        problem = {"arms": ["A", "B"], "parts": [part("beam", support=True)],
                   "candidates": [candidate("beam", "A", supports=[support("A")])]}
        result = solve_allocation(problem)
        self.assertFalse(result["solved"])
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["choices"], [])
        self.assertEqual(result["uncovered_parts"], ["beam"])

    def test_balance_weight_changes_allocation_when_it_pays_off(self):
        problem = {"arms": ["A", "B"], "parts": [part("p1"), part("p2")],
                   "cost_unit": "s", "load_balance_weight": 0.,
                   "candidates": [candidate(p, arm, cost=10. if arm == "A" else 11.)
                                  for p in ["p1", "p2"] for arm in ["A", "B"]]}
        unbalanced = solve_allocation(problem)
        self.assertEqual([c["arm"] for c in unbalanced["choices"]], ["A", "A"])
        problem["load_balance_weight"] = 1.
        balanced = solve_allocation(problem)
        self.assertEqual(set(c["arm"] for c in balanced["choices"]), {"A", "B"})
        self.assertAlmostEqual(balanced["objective"], 32.)
        self.assertEqual(balanced["cost_unit"], "s")
        self.assertFalse(balanced["objective_is_makespan"])

    def test_grasp_is_selected_from_candidate_cost_and_feasibility(self):
        problem = {"arms": ["A", "B"], "parts": [part("p")],
                   "candidates": [candidate("p", "A", grasp="top", cost=.1, feasible=False),
                                  candidate("p", "A", grasp="side", cost=2.),
                                  candidate("p", "B", grasp="top", cost=8.)]}
        chosen = solve_allocation(problem)["choices"][0]
        self.assertEqual((chosen["arm"], chosen["grasp"]), ("A", "side"))
        self.assertEqual(chosen["candidate_index"], 1)

    def test_force_boundary_and_invalid_data(self):
        p = part("p", support=True)
        exact = (p["mass"] * 9.81 + p["insertion_force"]) * p["lever_arm"]
        p["connection_moment_limit"] = exact
        self.assertFalse(support_requirement(p)["support_required"])
        p["connection_moment_limit"] = exact / 2
        self.assertTrue(support_requirement(p)["support_required"])
        bad = {"arms": ["A", "B"], "parts": [p],
               "candidates": [candidate("p", "A", supports=[support("B")])]}
        copy = deepcopy(bad)
        copy["candidates"][0]["cost"] = float("nan")
        with self.assertRaises(ValueError):
            solve_allocation(copy)
        copy = deepcopy(bad)
        copy["candidates"][0]["support_candidates"][0]["quaternion"] = [0, 0, 0, 2]
        with self.assertRaises(ValueError):
            solve_allocation(copy)


if __name__ == "__main__":
    unittest.main()
