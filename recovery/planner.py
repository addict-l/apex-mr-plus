"""Bounded event-state search with persistent support and auxiliary actions.

This is inspired by D-ITAGS/WRAP, not a reproduction of either implementation.
The existing normal MILP ranks nominal role choices. A uniform-cost search
then chooses actual bindings, recovery actions and an abstract time schedule.
Only one assembly-changing skill may run at once; independent work on the
other arm may proceed. This conservative mutex is explicit, not geometric
collision checking. Search-frontier reuse across events is not implemented.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import heapq
import itertools
import time

from normal.allocation import solve_allocation

from .model import ARMS, Action, Problem, State, number, transition
from .oracles import DeclaredCapabilities, EvidenceCache, PlanarWrenchOracle


POLICIES = ("fixed", "reassign", "recovery")


def actions(problem: Problem, state: State, policy: str = "recovery") -> list[Action]:
    """Enumerate syntactically applicable primitives; no hardcoded recovery chain."""
    locations, supports = dict(state.locations), dict(state.supports)
    result = []
    for arm in ARMS:
        held = state.held(arm)
        if arm in supports:
            result.append(Action("release_support", arm, supports[arm], problem.durations["release_support"]))
        elif held:
            if policy == "recovery":
                for slot in problem.buffers:
                    if slot not in locations.values():
                        result.append(Action("place", arm, f"{held}@{slot}", problem.durations["place"],
                                             part=held, destination=slot))
                other = "B" if arm == "A" else "A"
                if state.free(other):
                    result.append(Action("handover", arm, f"{held}@{other}", problem.durations["handover"],
                                         part=held, receiver=other))
        else:
            for pid, location in state.locations:
                if location in problem.slots and (policy == "recovery" or location not in problem.buffers):
                    result.append(Action("pick", arm, f"{pid}@{location}", problem.durations["pick"], part=pid))
            for cid, contact in problem.physics.get("contacts", {}).items():
                if contact["arm"] == arm:
                    result.append(Action("acquire_support", arm, cid, problem.durations["acquire_support"]))
        for tid, task in problem.tasks.items():
            if tid in state.completed or not set(task.get("dependencies", [])) <= state.completed:
                continue
            if policy == "fixed" and task["original_arm"] != arm:
                continue
            if task["kind"] == "install" and held == task["part"] and arm not in supports:
                result.append(Action("install", arm, tid, task["duration"], part=task["part"]))
            elif task["kind"] == "work" and state.free(arm):
                result.append(Action("work", arm, tid, task["duration"]))
    return result


def qualifications(action: Action) -> list[tuple[str, str, str]]:
    if action.kind == "release_support":
        # Withdrawal is an available abstract primitive; stability is checked
        # before/during/after. No sensor feedback is fabricated here.
        return []
    operation = "support" if action.kind == "acquire_support" else action.kind
    result = [(action.arm, operation, action.target)]
    if action.kind == "handover":
        result.append((action.receiver, "handover", f"{action.part}@{action.arm}"))
    return result


def allocation_hint(problem: Problem, capability: DeclaredCapabilities) -> dict:
    """Reuse the original MILP, as a role-ranking hint, never a feasibility gate.

Candidates here are declared discrete capabilities, NOT measured IK endpoints.
Every selected action still goes through the state and LP checks in search.
The original MILP objective is a workload proxy, not the recovery makespan.
"""
    parts, candidates = [], []
    for tid, task in problem.tasks.items():
        if task["kind"] != "install" or tid in problem.initial.completed:
            continue
        pid = task["part"]
        parts.append({"id": pid, "mass": problem.parts[pid]["mass"],
                      "insertion_force": task.get("insertion_force_N", 0.),
                      "lever_arm": abs(task.get("load_x_m", 0.)),
                      "connection_moment_limit": problem.physics["connection"]["moment_limit_Nm"],
                      "grasps": [{"id": "declared_grasp"}]})
        for arm in ARMS:
            supports = []
            for cid, contact in problem.physics.get("contacts", {}).items():
                helper = contact["arm"]
                if helper != arm:
                    supports.append({"arm": helper,
                                     "feasible": capability.check(helper, "support", cid)["status"] == "VALID",
                                     "cost": problem.durations["acquire_support"] + problem.durations["release_support"],
                                     "tcp_position": [contact["x_m"], 0., contact.get("z_m", 0.)],
                                     "quaternion": [0., 0., 0., 1.]})
            candidates.append({"part": pid, "arm": arm, "grasp": "declared_grasp",
                               "feasible": capability.check(arm, "install", tid)["status"] == "VALID",
                               "cost": task["duration"], "support_candidates": supports})
    answer = solve_allocation({"arms": list(ARMS), "parts": parts, "candidates": candidates,
                               "cost_unit": "abstract_skill_seconds", "load_balance_weight": .15})
    answer.update(hint_only=True, qualifications="declared capabilities, not IK/path evidence")
    return answer


@dataclass(frozen=True)
class Running:
    action: Action
    remaining: float


@dataclass(frozen=True)
class SearchState:
    physical: State
    running: tuple[Running, ...] = ()


def _running_key(item: Running):
    a = item.action
    return (a.kind, a.arm, a.target, a.part or "", a.destination or "", a.receiver or "")


def _trace(parent: dict, node: SearchState) -> list[dict]:
    edges = []
    while node in parent:
        previous, event = parent[node]
        edges.append(event)
        node = previous
    edges.reverse()
    active, result = {}, []
    clock = 0.
    for event in edges:
        if event["event"] == "start":
            action = event["action"]
            row = {"id": f"r{len(result):04d}", **action.to_dict(), "start": clock, "end": None}
            active[action] = row
            result.append(row)
        else:
            clock = round(clock + event["dt"], 9)
            for action in event["finished"]:
                active.pop(action)["end"] = clock
    if active:
        raise RuntimeError("only a quiescent safe incumbent may be returned")
    return result


def plan_recovery(problem: Problem, *, policy="recovery", cache: EvidenceCache | None = None,
                  max_expansions=20000, time_limit_s=15.) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, int) or max_expansions < 1:
        raise ValueError("max_expansions must be a positive integer")
    if number(time_limit_s, "time_limit_s") <= 0:
        raise ValueError("time_limit_s must be positive")
    started = time.perf_counter()
    cache = cache if cache is not None else EvidenceCache()
    counts_before = cache.counts.copy()
    capability, wrench = DeclaredCapabilities(problem, cache), PlanarWrenchOracle(problem, cache)
    initial_wrench = wrench.check(problem.initial)
    if initial_wrench["status"] != "VALID":
        return {"schema_version": 1, "case": problem.name, "policy": policy,
                "status": "invalid_snapshot" if initial_wrench["status"] == "INVALID" else "unknown",
                "solved": False, "actions": [], "reason": "initial structure has no validated static equilibrium",
                "initial_wrench": initial_wrench, "final_state": problem.initial.to_dict(),
                "search": {"expanded": 0, "complete": False}, "executable_in_coppeliasim": False}
    hint = allocation_hint(problem, capability)
    preferred = {c["part"]: c["arm"] for c in hint["choices"]}
    initial = SearchState(problem.initial)
    distances, parent = {initial: 0.}, {}
    serial = itertools.count()
    heap = [(0., next(serial), initial)]
    best, best_score = initial, (0, 0.)
    found, limit = False, False
    stats = Counter()
    rejection_counts = Counter()

    def push(previous, target, distance, event):
        if distance < distances.get(target, float("inf")) - 1e-9:
            distances[target] = distance
            parent[target] = (previous, event)
            heapq.heappush(heap, (distance, next(serial), target))

    while heap:
        if stats["expanded"] >= max_expansions or time.perf_counter()-started >= time_limit_s:
            limit = True
            break
        elapsed, _, current = heapq.heappop(heap)
        if elapsed > distances[current]+1e-9:
            continue
        stats["expanded"] += 1
        state = current.physical
        if not current.running:
            completed = len((state.completed - problem.initial.completed) & problem.tasks.keys())
            score = (completed, -elapsed)
            if score > best_score:
                best, best_score = current, score
            if problem.finished(state):
                best, found = current, True
                break
        busy = {arm for run in current.running for arm in run.action.actors}
        active_tasks = {run.action.target for run in current.running if run.action.kind in {"install", "work"}}
        assembly_busy = any(run.action.kind != "work" for run in current.running)
        choices = actions(problem, state, policy)
        # MILP only breaks equal search-cost ties; it cannot erase a candidate.
        choices.sort(key=lambda a: (a.part in preferred and preferred[a.part] != a.arm,
                                    a.kind, a.arm, a.target))
        for action in choices:
            if busy.intersection(action.actors) or action.target in active_tasks:
                continue
            if action.kind != "work" and assembly_busy:
                continue
            stats["generated"] += 1
            statuses = [capability.check(*spec)["status"] for spec in qualifications(action)]
            if any(s != "VALID" for s in statuses):
                rejection_counts["capability_unknown" if "UNKNOWN" in statuses else "capability_invalid"] += 1
                continue
            after = transition(problem, state, action)
            # Check loading while old contacts still exist, then the completed
            # contact/connection state. acquire_support has no early effect.
            during, final = wrench.check(state, action), wrench.check(after)
            if during["status"] != "VALID" or final["status"] != "VALID":
                rejection_counts["wrench_unknown" if "UNKNOWN" in {during["status"], final["status"]}
                                 else "wrench_invalid"] += 1
                continue
            running = tuple(sorted(current.running+(Running(action, action.duration),), key=_running_key))
            push(current, SearchState(state, running), elapsed, {"event": "start", "action": action})
        if current.running:
            dt = min(run.remaining for run in current.running)
            finished = [run.action for run in current.running if run.remaining <= dt+1e-9]
            after = state
            for action in finished:
                after = transition(problem, after, action)
            remaining = tuple(Running(run.action, round(run.remaining-dt, 9)) for run in current.running
                              if run.remaining > dt+1e-9)
            # All independent work has no assembly effect. Revalidate the
            # completed state rather than trusting a stored predecessor flag.
            if wrench.check(after)["status"] == "VALID":
                push(current, SearchState(after, remaining), round(elapsed+dt, 9),
                     {"event": "finish", "dt": dt, "finished": finished})
    trace = _trace(parent, best)
    uncertainty = bool(capability.unknown_queries or wrench.unknown_queries)
    status = "solved" if found else ("partial" if trace else ("unknown" if limit or uncertainty else "no_plan"))
    return {"schema_version": 1, "case": problem.name, "policy": policy, "status": status,
            "solved": found, "actions": trace, "final_state": best.physical.to_dict(),
            "unfinished_tasks": sorted(problem.tasks.keys()-best.physical.completed),
            "predicted_makespan_s": distances[best], "allocation_hint": hint,
            "search": {**dict(stats), "visited_states": len(distances),
                       "complete": found or (not limit and not uncertainty),
                       "exhausted_finite_frontier": not heap and not found,
                       "budget_exhausted": limit, "unknown_capabilities": sorted(capability.unknown_queries),
                       "unknown_wrench_queries": wrench.unknown_queries, "rejections": dict(rejection_counts),
                       "wall_time_s": time.perf_counter()-started},
            "evidence_queries": dict(cache.counts-counts_before),
            "execution_model": "abstract deterministic event simulation with a global assembly-skill mutex",
            "objective": "remaining abstract makespan; no original D-ITAGS optimality bound claimed",
            "executable_in_coppeliasim": False,
            "limitations": ["declared capabilities, no IK or path validation", "planar quasi-static structure LP",
                            "finite candidates and bounded search", "no live state capture or controller resume"]}


def affected_tasks(problem: Problem, changed_facts: list[str], previous_plan: dict | None = None) -> dict:
    """Task-level impact report; the remaining bounded window is searched afresh."""
    pending = set(problem.tasks)-problem.initial.completed
    seeds, reasons, resource_arms = set(), {}, set()
    for fact in changed_facts:
        path = fact.split("/")
        touched = set()
        if path[0] == "physics":
            touched = {tid for tid in pending if problem.tasks[tid]["kind"] == "install"}
        elif path[0] == "capabilities":
            arm, operation, target = path[1:4]
            resource_arms.add(arm)
            touched = {tid for tid in pending if tid == target and operation in {"install", "work"}}
            for row in (previous_plan or {}).get("actions", []):
                action = Action.from_dict(row)
                if (arm, operation, target) in qualifications(action):
                    if action.kind in {"install", "work"}:
                        touched.add(action.target)
                    else:
                        touched.update(tid for tid in pending if problem.tasks[tid]["kind"] == "install")
        seeds.update(touched)
        for tid in touched:
            reasons.setdefault(tid, []).append(fact)
    affected = set(seeds)
    while True:
        extra = {tid for tid in pending if set(problem.tasks[tid].get("dependencies", [])) & affected}
        if extra <= affected:
            break
        affected.update(extra)
    return {"seed_tasks": sorted(seeds), "affected_tasks": sorted(affected), "seed_reasons": reasons,
            "other_pending_tasks": sorted(pending-affected), "resource_recheck_arms": sorted(resource_arms),
            "preserved_completed": sorted(problem.initial.completed),
            "scope": "evidence invalidation and task impact only; remaining-window search is rebuilt"}
