"""Reuse normal.tpg for abstract schedules, with an independent state audit.

No q_path is invented. These graphs have geometry_available=False and must
not be installed in normal.controller.lua. A motion compiler/validator is a
separate, still-missing step before simulator or robot execution.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from normal.tpg import build_tpg, simulate_tpg

from .model import Action, Problem, number, transition
from .oracles import EvidenceCache, PlanarWrenchOracle
from .planner import actions, qualifications


def build_schedule(problem: Problem, plan: dict) -> dict:
    tasks, last_assembly, last_arm, barriers, completed_by = [], [], {}, defaultdict(list), {}
    for row in plan["actions"]:
        action = Action.from_dict(row)
        deps = set(last_assembly if action.kind != "work" else [])
        if action.kind in {"install", "work"}:
            for dep in problem.tasks[action.target].get("dependencies", []):
                deps.update(completed_by.get(dep, []))
        for arm in action.actors:
            deps.update(barriers[arm])
        if action.kind == "handover":
            # Both reservation nodes share exactly the same incoming barrier.
            # All later actions on EITHER arm wait for BOTH acknowledgments.
            deps.update(last_arm[arm] for arm in action.actors if arm in last_arm)
        ids = []
        for arm in action.actors:
            tid = f"{row['id']}:{arm}"
            tasks.append({"id": tid, "arm": arm, "kind": f"offline_{action.kind}",
                          "part": action.part, "duration": action.duration,
                          "dependencies": sorted(deps), "reference_order": len(tasks),
                          "metadata": {"recovery_action_id": row["id"], "abstract_only": True,
                                       "motion_verified": False}})
            ids.append(tid)
            last_arm[arm] = tid
        if action.kind != "work":
            last_assembly = ids
        if action.kind == "handover":
            for arm in action.actors:
                barriers[arm] = ids
        if action.kind in {"install", "work"}:
            completed_by[action.target] = ids
    graph = build_tpg(tasks)
    graph.update(executable_in_coppeliasim=False, planning_level="abstract_recovery_schedule")
    graph["certificate"].update(
        geometric_reference_paths_unchanged=False,
        motion_compilation_required=True,
        scope="abstract dependencies and resource reservations only; no geometry or dynamics certificate")
    return graph


def action_schedule(plan: dict, graph: dict, delays=None) -> list[dict]:
    prediction = simulate_tpg(graph, delays)
    grouped = defaultdict(list)
    for task in prediction["task_schedule"]:
        source = graph["task_map"][task["id"]]
        grouped[source["metadata"]["recovery_action_id"]].append(task)
    result = []
    for row in plan["actions"]:
        group = grouped[row["id"]]
        if len(group) != len(row["resources"]) or len({task["arm"] for task in group}) != len(group):
            raise ValueError("missing or duplicate arm reservation")
        starts = [task["start"] for task in group]
        if max(starts)-min(starts) > 1e-8:
            raise ValueError("handover arm reservations must start together")
        result.append({**deepcopy(row), "start": min(starts), "end": max(task["end"] for task in group)})
    return result


def validate_schedule(problem: Problem, rows: list[dict], *, policy="recovery",
                      expected_final: dict | None = None) -> dict:
    """Replay completion events, holding/support occupancy and applied loads.

This audit reconstructs state from the input snapshot; it never trusts the
planner's recorded final state or an LP boolean stored in the output plan.
"""
    failures, witnesses, events = [], [], defaultdict(lambda: {"start": [], "end": []})
    ids = set()
    for row in rows:
        number(row["start"], "schedule start")
        number(row["end"], "schedule end")
        if row["id"] in ids or row["start"] < 0 or row["end"] <= row["start"]:
            raise ValueError("invalid action identity or time interval")
        number(row["duration"], "declared skill duration")
        if row["end"]-row["start"] < row["duration"]-1e-8:
            failures.append(f"{row['id']}: skill completed before its declared duration")
        ids.add(row["id"])
        events[round(row["start"], 9)]["start"].append(row)
        events[round(row["end"], 9)]["end"].append(row)
    state = problem.initial
    running, occupied = {}, {}
    support_open = {arm: {"arm": arm, "contact": cid, "start": 0., "initial": True}
                    for arm, cid in state.supports}
    support_intervals = []
    wrench = PlanarWrenchOracle(problem, EvidenceCache())

    def equilibrium(now, phase):
        assembly = [a for a in running.values() if a.kind != "work"]
        if len(assembly) > 1:
            failures.append(f"{now}: overlapping assembly-changing skills")
        answer = wrench.check(state, assembly[0] if assembly else None)
        if answer["status"] != "VALID":
            failures.append(f"{now}: {phase} equilibrium {answer['status']}")
        witnesses.append({"time": now, "phase": phase, **answer})

    equilibrium(0., "snapshot")
    for now, batch in sorted(events.items()):
        # Skills have positive duration. Simultaneous finishes are applied
        # before starts, which makes predecessor completion the release event.
        for row in batch["end"]:
            action = running.pop(row["id"], None)
            if action is None:
                failures.append(f"{now}: finishing an action that was never started")
                continue
            try:
                state = transition(problem, state, action)
            except (KeyError, ValueError) as exc:
                failures.append(f"{now}: illegal completed effect: {exc}")
                continue
            for arm in action.actors:
                occupied.pop(arm, None)
            if action.kind == "acquire_support":
                support_open[action.arm] = {"arm": action.arm, "contact": action.target,
                                            "start": now, "initial": False}
            elif action.kind == "release_support":
                support_intervals.append({**support_open.pop(action.arm), "end": now})
        equilibrium(now, "after_completion")
        for row in batch["start"]:
            action = Action.from_dict(row)
            if any(arm in occupied for arm in action.actors):
                failures.append(f"{now}: arm occupied by another running skill")
                continue
            if action not in actions(problem, state, policy):
                failures.append(f"{now}: invalid precondition for {row['id']} {action.kind}")
                continue
            if any(problem.status(*spec) != "VALID" for spec in qualifications(action)):
                failures.append(f"{now}: undeclared or invalid capability for {row['id']}")
                continue
            if any(a.target == action.target and a.kind in {"install", "work"}
                   for a in running.values()) and action.kind in {"install", "work"}:
                failures.append(f"{now}: a task cannot be executed twice concurrently")
                continue
            running[row["id"]] = action
            for arm in action.actors:
                occupied[arm] = row["id"]
        equilibrium(now, "during_skills")
    for value in support_open.values():
        support_intervals.append({**value, "end": None})
    if running:
        failures.append("unfinished running actions")
    if expected_final is not None and state.to_dict() != expected_final:
        failures.append("replayed final state differs from the search result")
    return {"valid": not failures, "failures": failures, "final_state": state.to_dict(),
            "all_goals_completed_and_resources_released": problem.finished(state),
            "support_intervals": support_intervals, "wrench_witnesses": witnesses,
            "predicted_makespan_s": max((r["end"] for r in rows), default=0.),
            "level": "abstract event/state/planar-physics audit", "robot_execution_verified": False,
            "ik_verified": False, "path_verified": False, "collision_verified": False}
