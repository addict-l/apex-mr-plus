"""Finite symbolic support-lifecycle search; no robot motion or dynamics.

Run from the scene directory with ``python src/lifecycle.py``. Durations,
support ratings and contact requirements are declared design assumptions.
The event DAG is an input for a future motion-qualified APEX-MR pipeline;
it is NOT a motion TPG, a collision certificate or measured execution.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-9
REQUIRED_CONTACT_OVERLAP_M = 0.010
REQUIRED_REACTION_N = (0.45 * 9.81 * 0.26 + 8.0 * 0.46) / 0.28
REACTION_SAFETY_FACTOR = 1.5
DESIGN_REACTION_N = REQUIRED_REACTION_N * REACTION_SAFETY_FACTOR


class InvalidPlan(ValueError):
    """A symbolic guard, dependency or resource constraint was violated."""


@dataclass(frozen=True)
class Candidate:
    id: str
    rated_load_N: float
    contact_overlap_m: float
    place_duration_s: float
    extract_duration_s: float
    blocked_after_gate: bool = False


CANDIDATES = (
    Candidate("outer_front", 40.0, 0.040, 6.0, 4.0),
    Candidate("inner_pocket", 40.0, 0.040, 4.0, 10.0, True),
    Candidate("weak_narrow", 40.0, 0.005, 3.0, 3.0),
)


def qualify(candidate: Candidate) -> None:
    if candidate.rated_load_N + EPSILON < DESIGN_REACTION_N:
        raise InvalidPlan("insufficient_declared_load_rating")
    if candidate.contact_overlap_m + EPSILON < REQUIRED_CONTACT_OVERLAP_M:
        raise InvalidPlan("insufficient_declared_contact_overlap")


def make_actions(candidate: Candidate, removal_order: str,
                 supply_order: tuple[str, str] = ("K1", "K2")) -> list[dict]:
    """Generate an event DAG including buffer-token and lifecycle edges."""
    if removal_order not in {"before_gate", "after_gate"}:
        raise ValueError("removal_order must be before_gate or after_gate")
    if sorted(supply_order) != ["K1", "K2"]:
        raise ValueError("supply_order must contain K1 and K2 once each")
    actions = []

    def add(name, arm, duration, predecessors=(), part=None):
        actions.append({"id": name, "arm": arm, "nominal_duration_s": duration,
                        "predecessors": list(predecessors), "part": part})

    add("place_support", "B", candidate.place_duration_s)
    add("verify_support_load", "B", 2.0, ["place_support"])
    add("release_A", "A", 1.0, ["verify_support_load"])
    previous = None
    for part in supply_order:
        add(f"fetch_{part}", "A", 4.0,
            ["release_A"] if previous is None else [f"deposit_{previous}"] , part)
        deposit_dependencies = [f"fetch_{part}"]
        if previous is not None:
            # Capacity-one storage: reserve the next deposit only after pickup.
            deposit_dependencies.append(f"pick_{previous}")
        add(f"deposit_{part}", "A", 1.0, deposit_dependencies, part)
        previous = part
    for part in ["K1", "K2"]:
        dependencies = [f"deposit_{part}"]
        if part == "K2":
            dependencies.append("install_K1")
        add(f"pick_{part}", "B", 1.0, dependencies, part)
        add(f"install_{part}", "B", 4.0, [f"pick_{part}"], part)
    add("verify_permanent_structure", "B", 2.0, ["install_K2"])
    permanent = "verify_permanent_structure"
    add("close_gate", "B", 4.0,
        [permanent, "return_support"] if removal_order == "before_gate" else [permanent])
    add("unload_support", "B", 1.0,
        [permanent, "close_gate"] if removal_order == "after_gate" else [permanent])
    add("extract_support", "B", candidate.extract_duration_s, ["unload_support"])
    add("return_support", "B", 2.0, ["extract_support"])
    return actions


def topological_order(actions: list[dict]) -> list[str]:
    indexed = {a["id"]: a for a in actions}
    if len(indexed) != len(actions):
        raise InvalidPlan("duplicate_action_id")
    for action in actions:
        if any(p not in indexed for p in action["predecessors"]):
            raise InvalidPlan("unknown_predecessor")
        if action["nominal_duration_s"] <= 0:
            raise InvalidPlan("nonpositive_duration")
    result = []
    remaining = set(indexed)
    while remaining:
        ready = sorted(k for k in remaining
                       if all(p in result for p in indexed[k]["predecessors"]))
        if not ready:
            raise InvalidPlan("dependency_cycle_or_buffer_deadlock")
        result.extend(ready)
        remaining.difference_update(ready)
    return result


def nominal_schedule(actions: list[dict]) -> list[dict]:
    """Deterministic earliest-ready list scheduling for the two arm resources."""
    topological_order(actions)
    pending = {a["id"]: a for a in actions}
    finish = {}
    arm_free = {"A": 0.0, "B": 0.0}
    schedule = []
    while pending:
        ready = [a for a in pending.values()
                 if all(p in finish for p in a["predecessors"])]
        options = []
        for action in ready:
            earliest = max([arm_free[action["arm"]]] +
                           [finish[p] for p in action["predecessors"]])
            options.append((earliest, action["id"], action))
        start, key, action = min(options, key=lambda item: item[:2])
        end = start + action["nominal_duration_s"]
        schedule.append({**action, "start_s": start, "end_s": end})
        arm_free[action["arm"]] = end
        finish[key] = end
        del pending[key]
    return sorted(schedule, key=lambda a: (a["start_s"], a["id"]))


def resource_occupancy(schedule: list[dict]) -> list[dict]:
    indexed = {a["id"]: a for a in schedule}
    intervals = [{"resource": a["arm"], "owner": a["id"],
                  "start_s": a["start_s"], "end_s": a["end_s"]}
                 for a in schedule]
    release = indexed["release_A"]["start_s"]
    if release > 0:
        intervals.append({"resource": "A", "owner": "initial_beam_hold",
                          "start_s": 0.0, "end_s": release})
    for part in ["K1", "K2"]:
        intervals.append({"resource": "shared_buffer", "owner": part,
                          "start_s": indexed[f"deposit_{part}"]["start_s"],
                          "end_s": indexed[f"pick_{part}"]["end_s"]})
    intervals.append({"resource": "support_fixture", "owner": "temporary_support",
                      "start_s": indexed["place_support"]["start_s"],
                      "end_s": indexed["return_support"]["end_s"]})
    return sorted(intervals, key=lambda r: (r["resource"], r["start_s"]))


def check_resource_capacity(intervals: list[dict]) -> None:
    for resource in {r["resource"] for r in intervals}:
        occupied = sorted((r for r in intervals if r["resource"] == resource),
                          key=lambda r: r["start_s"])
        previous_end = 0.0
        for item in occupied:
            if item["start_s"] < -EPSILON or item["end_s"] < item["start_s"]:
                raise InvalidPlan("invalid_resource_interval")
            if item["start_s"] + EPSILON < previous_end:
                raise InvalidPlan(f"resource_capacity_exceeded:{resource}")
            previous_end = max(previous_end, item["end_s"])


def validate_plan(plan: dict) -> dict:
    """Independently replay symbolic guards and effects, and check resources."""
    candidate = Candidate(**plan["candidate"])
    qualify(candidate)
    actions = plan["actions"]
    topological_order(actions)
    indexed = {a["id"]: a for a in actions}
    expected_ids = {a["id"] for a in make_actions(candidate, "before_gate")}
    if set(indexed) != expected_ids:
        raise InvalidPlan("missing_or_unknown_lifecycle_action")
    for action in actions:
        if abs(action["end_s"] - action["start_s"] - action["nominal_duration_s"]) > EPSILON:
            raise InvalidPlan("duration_mismatch")
        for predecessor in action["predecessors"]:
            if indexed[predecessor]["end_s"] > action["start_s"] + EPSILON:
                raise InvalidPlan("dependency_not_complete")
    # Recompute this evidence rather than trusting a submitted occupancy list.
    occupancy = resource_occupancy(actions)
    check_resource_capacity(occupancy)
    state = {"holding_initial_beam": True, "support": "parked",
             "load_verified": False, "permanent_verified": False,
             "gate_closed": False, "held_A": None, "held_B": None,
             "buffer": None, "installed": set(), "supply": {"K1", "K2"}}

    def require(condition, message):
        if not condition:
            raise InvalidPlan(message)

    def start_guard(action):
        key, arm, part = action["id"], action["arm"], action.get("part")
        expected_arm = "A" if key == "release_A" or key.startswith(("fetch_", "deposit_")) else "B"
        require(arm == expected_arm, "arm_assignment_or_A_exclusive_supply_violation")
        if arm == "A" and key != "release_A":
            require(not state["holding_initial_beam"], "A_still_occupied_supporting")
        if key == "place_support":
            require(state["support"] == "parked" and state["holding_initial_beam"],
                    "support_place_requires_continuous_A_hold")
        elif key == "verify_support_load":
            require(state["support"] == "placed", "support_not_placed")
        elif key == "release_A":
            require(state["load_verified"], "premature_A_release_without_load_handoff")
        elif key.startswith("fetch_"):
            require(part in state["supply"] and state["held_A"] is None,
                    "fetch_requires_available_part_and_empty_A")
        elif key.startswith("deposit_"):
            require(state["held_A"] == part and state["buffer"] is None,
                    "deposit_requires_held_part_and_empty_buffer")
        elif key.startswith("pick_"):
            require(state["buffer"] == part and state["held_B"] is None,
                    "pick_requires_buffer_part_and_empty_B")
        elif key.startswith("install_"):
            require(state["held_B"] == part, "install_requires_held_part")
            require(part != "K2" or "K1" in state["installed"], "K2_requires_K1")
            require(state["load_verified"], "installation_requires_verified_support")
        elif key == "verify_permanent_structure":
            require(state["installed"] == {"K1", "K2"}, "permanent_structure_incomplete")
        elif key == "close_gate":
            require(state["permanent_verified"] and state["held_B"] is None,
                    "gate_requires_permanent_structure_and_empty_B")
        elif key == "unload_support":
            require(state["permanent_verified"], "premature_support_unload")
            require(state["support"] == "placed", "support_not_in_place")
        elif key == "extract_support":
            require(state["support"] == "unloaded", "extract_requires_unloaded_support")
            require(not (candidate.blocked_after_gate and state["gate_closed"]),
                    "support_trapped_by_closed_gate")
        elif key == "return_support":
            require(state["held_B"] == "support", "return_requires_extracted_support")

    def complete(action):
        key, part = action["id"], action.get("part")
        if key == "place_support": state["support"] = "placed"
        elif key == "verify_support_load": state["load_verified"] = True
        elif key == "release_A": state["holding_initial_beam"] = False
        elif key.startswith("fetch_"):
            state["supply"].remove(part); state["held_A"] = part
        elif key.startswith("deposit_"):
            state["held_A"] = None; state["buffer"] = part
        elif key.startswith("pick_"):
            state["buffer"] = None; state["held_B"] = part
        elif key.startswith("install_"):
            state["held_B"] = None; state["installed"].add(part)
        elif key == "verify_permanent_structure": state["permanent_verified"] = True
        elif key == "close_gate": state["gate_closed"] = True
        elif key == "unload_support":
            state["support"] = "unloaded"; state["load_verified"] = False
        elif key == "extract_support":
            state["support"] = "extracted"; state["held_B"] = "support"
        elif key == "return_support":
            state["support"] = "parked"; state["held_B"] = None

    # At a common timestamp, predecessor completions apply before new starts.
    events = [(a["start_s"], 1, a) for a in actions] + [(a["end_s"], 0, a) for a in actions]
    for _, phase, action in sorted(events, key=lambda e: (e[0], e[1], e[2]["id"])):
        (start_guard if phase else complete)(action)
    require(state["support"] == "parked" and state["gate_closed"] and
            state["permanent_verified"] and not state["holding_initial_beam"] and
            state["held_A"] is None and state["held_B"] is None and
            state["buffer"] is None, "goal_not_complete_or_support_not_returned")
    return {"valid_symbolic_plan": True, "checked": ["acyclic_dependencies",
            "predecessor_completion", "single_arm_capacity", "buffer_capacity_one",
            "load_handoff_before_release", "A_exclusive_supply", "permanent_before_unload",
            "gate_extraction_guard", "support_returned"],
            "final_state": {**state, "installed": sorted(state["installed"]),
                            "supply": sorted(state["supply"])}}


def evaluate(candidate: Candidate, removal_order: str,
             supply_order: tuple[str, str] = ("K1", "K2")) -> dict:
    qualify(candidate)
    actions = nominal_schedule(make_actions(candidate, removal_order, supply_order))
    plan = {"candidate": asdict(candidate), "removal_order": removal_order,
            "supply_order": list(supply_order), "actions": actions}
    plan["validation"] = validate_plan(plan)
    plan["edges"] = [{"from": p, "to": a["id"]}
                     for a in actions for p in a["predecessors"]]
    plan["resource_occupancy"] = resource_occupancy(actions)
    makespan = max(a["end_s"] for a in actions)
    hold = next(a["end_s"] for a in actions if a["id"] == "release_A")
    # Fixed weights are illustrative; all time terms are nominal assumptions.
    plan["nominal_metrics"] = {"makespan_s": makespan, "A_support_occupancy_s": hold,
                               "action_count": len(actions),
                               "objective": round(makespan + 0.25 * hold + 0.10 * len(actions), 6)}
    return plan


def search(candidates=CANDIDATES) -> dict:
    feasible, decisions = [], []
    for candidate, order, supply in itertools.product(
            candidates, ["after_gate", "before_gate"], itertools.permutations(["K1", "K2"])):
        decision = {"candidate": candidate.id, "removal_order": order, "supply_order": list(supply)}
        try:
            plan = evaluate(candidate, order, supply)
            feasible.append(plan)
            decision.update(feasible=True, **plan["nominal_metrics"])
        except InvalidPlan as error:
            decision.update(feasible=False, reason=str(error))
        decisions.append(decision)
    if not feasible:
        raise InvalidPlan("no_symbolically_feasible_plan")
    selected = min(feasible, key=lambda p: (p["nominal_metrics"]["objective"],
                                           p["candidate"]["id"], p["removal_order"]))
    return {"schema_version": 1, "stage": "symbolic_design_only",
            "evidence_scope": "Finite action-DAG search and symbolic event replay; not motion TPG or dynamics.",
            "assumptions": {
                "initial_state": "Beam held by A; B empty; support parked; K1/K2 in A-only supply.",
                "assembly": "K1 then K2 establish permanent support; gate is a later non-load-bearing closure.",
                "A_exclusive_supply": "Declared access constraint; not an IK reachability certificate.",
                "buffer_capacity": 1, "duration_source": "Illustrative nominal seconds; not robot measurements.",
                "required_reaction_N": REQUIRED_REACTION_N, "reaction_safety_factor": REACTION_SAFETY_FACTOR,
                "design_reaction_N": DESIGN_REACTION_N,
                "reaction_model": "(.45*9.81*.26 + 8*.46)/.28; vertical moment balance only.",
                "minimum_contact_overlap_m": REQUIRED_CONTACT_OVERLAP_M,
                "candidate_contact_and_load": "Declared design inputs, not measured or geometrically certified.",
                "unload": "Permanent structure verified before 8 mm jack lowering and extraction.",
                "objective": "nominal makespan + .25*A support occupancy + .10*action count",
                "search_scope": "12 candidate/order/supply combinations with deterministic list scheduling; no global TAMP optimality claim.",
                "gate_trap": "Declared inner-pocket extraction guard; no swept-volume or robot collision proof."},
            "candidate_decisions": decisions, "selected_plan": selected,
            "counts": {"evaluated": len(decisions), "feasible": len(feasible),
                       "rejected": len(decisions) - len(feasible)},
            "remaining_validation": ["UR5 IK and collision-free approach/withdrawal paths",
                                     "Support and connector force/moment/stability model",
                                     "Measured load handoff and permanent support acceptance",
                                     "Dynamic execution and repeatability"]}


def main():
    result = search()
    output = ROOT / "results" / "lifecycle_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = result["selected_plan"]
    print(json.dumps({"output": str(output), "stage": result["stage"],
                      "candidate": selected["candidate"]["id"],
                      "removal_order": selected["removal_order"],
                      "nominal_metrics": selected["nominal_metrics"],
                      "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
