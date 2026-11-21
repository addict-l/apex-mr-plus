"""Offline checkpoint alternatives and deterministic revalidation after a fault."""

from math import isfinite
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.cost.model import DistanceCost
from skillnet.planning.bounds import HMaxBound
from skillnet.planning.search import SearchResult, SearchStep, _advance, _search, _time


def serialize_steps(steps):
    return [{"action": str(step.action), "start": step.start, "end": step.end,
             "state_before": [str(fact) for fact in sorted(step.state)]} for step in steps]


def _route(steps):
    return tuple(step.action.key for step in steps)


def prepare_checkpoints(engine, nominal, graph, *, checkpoint_limit, backups, seconds, max_expansions):
    indices, last_end, seen_states = [], 0., set()
    for index, step in enumerate(nominal.steps):
        if step.start >= last_end and step.state not in seen_states:
            indices.append(index)
            seen_states.add(step.state)
        last_end = max(last_end, step.end)
    if len(indices) > checkpoint_limit:
        if checkpoint_limit == 1:
            indices = indices[:1]
        else:
            indices = [indices[i*(len(indices)-1)//(checkpoint_limit-1)] for i in range(checkpoint_limit)]

    all_actions = frozenset(engine.actions)
    checkpoints = []
    for index in indices:
        suffix = nominal.steps[index:]
        start, upcoming = suffix[0].state, suffix[0].action
        plans = [{"kind": "nominal_suffix", "schedule": serialize_steps(suffix)}]
        seen = {_route(suffix)}
        # Backups are generated under failure scenarios, so two similar routes
        # needing the same failed arm/skill do not count as fault coverage.
        masks = []
        for arm in upcoming.occupies:
            disabled = frozenset(action.key for action in engine.actions.values()
                                 if action.name == upcoming.name and arm in action.occupies)
            masks.append((f"{arm}/{upcoming.name}", disabled))
        masks.append((str(upcoming), frozenset({upcoming.key})))
        expired = monotonic()+seconds
        attempts = []
        for label, excluded in masks:
            remaining = expired-monotonic()
            if len(plans)-1 >= backups or remaining <= 0:
                break
            allowed = all_actions-excluded
            bound = HMaxBound(engine, coarse=True, allowed=allowed)
            result = _search(engine, start=start, allowed=allowed, bound=bound, weight=3.,
                             max_expansions=max_expansions, time_limit=remaining,
                             deadline=expired, candidate_graph=graph)
            attempts.append({"assumed_failure": label, "status": result.status})
            if result.status == "solved" and _route(result.steps) not in seen:
                plans.append({"kind": "backup", "assumed_failure": label,
                              "schedule": serialize_steps(result.steps)})
                seen.add(_route(result.steps))
        checkpoints.append({"id": index, "state": [str(fact) for fact in sorted(start)],
                            "plans": plans, "preparation": attempts})
    return checkpoints


def _read_route(engine, rows):
    result, last_start = [], 0.
    for row in rows:
        start, end = row["start"], row["end"]
        if (any(type(value) not in (int, float) or not isfinite(value) for value in (start, end))
                or start < last_start or end < start):
            raise ValueError("备份时间表的开始或结束时间无效")
        result.append((engine.parse_action(row["action"]), start, end))
        last_start = start
    return result


def replay(engine, rows, *, allowed, protected, cuts, deadline):
    """Recompute timing and check a saved action sequence from measured facts."""
    began = monotonic()
    state, running, elapsed = engine.initial, (), 0.
    if engine.hooks.invariant(state):
        return None, "invariant"
    cost = engine.cost if engine.cost is not None else DistanceCost(engine.hooks)
    saved_ends, steps = {}, []

    def advance():
        nonlocal state, running, elapsed
        state, running, delta = _advance(engine, state, running)
        elapsed = _time(elapsed+delta)
        return engine.hooks.invariant(state)

    for action, planned_start, planned_end in _read_route(engine, rows):
        if monotonic() >= deadline:
            return None, "budget"
        if action.key not in allowed or protected.intersection(action.delete):
            return None, "excluded"
        status = engine.oracle.evaluate(action.capability)[0]
        if status != Status.VALID:
            return None, status.value.lower()
        while True:
            active = [engine.actions[key] for key, _ in running]
            must_finish = any(saved_ends[current.key] <= planned_start for current in active)
            conflicts = any(set(current.occupies).intersection(action.occupies)
                            or (current.structural and action.structural) for current in active)
            if (not must_finish and not conflicts and engine.satisfied(action.pre, state)
                    and (state, action.key) not in cuts):
                break
            if not running:
                return None, "precondition_or_cut"
            if advance():
                return None, "invariant"
        duration = _time(cost.estimate(action, state))
        if duration <= 0:
            raise ValueError(f"动作 {action} 的耗时必须大于零")
        steps.append(SearchStep(action, elapsed, _time(elapsed+duration), state))
        state = engine.start(action, state)
        if engine.hooks.invariant(state):
            return None, "invariant"
        running = tuple(sorted(running+((action.key, duration),)))
        saved_ends[action.key] = planned_end
    while running:
        if advance():
            return None, "invariant"
    if not engine.satisfied(engine.goal, state):
        return None, "goal"
    if monotonic() >= deadline:
        return None, "budget"
    return SearchResult("solved", steps, elapsed, 0, 0, {}, frozenset(), monotonic()-began), "valid"


def try_backups(engine, checkpoints, *, allowed, protected, cuts, deadline):
    checkpoints = sorted(checkpoints, key=lambda item: len(
        engine.initial.symmetric_difference(map(parse_fact, item["state"]))))
    best, chosen, attempts = None, None, []
    for checkpoint in checkpoints:
        for plan in checkpoint["plans"]:
            if monotonic() >= deadline:
                return best, chosen, attempts, True
            result, status = replay(engine, plan["schedule"], allowed=allowed,
                                    protected=protected, cuts=cuts, deadline=deadline)
            detail = {"checkpoint": checkpoint["id"], "kind": plan["kind"], "status": status,
                      "assumed_failure": plan.get("assumed_failure"),
                      "makespan": result.makespan if result is not None else None}
            attempts.append(detail)
            if result is not None and (best is None or result.makespan < best.makespan):
                best, chosen = result, detail
    return best, chosen, attempts, any(attempt["status"] == "budget" for attempt in attempts)
