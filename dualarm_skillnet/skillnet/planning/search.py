"""Optimal event-state A* and its independent zero-heuristic UCS baseline."""
from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import floor, isfinite, log10
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.cost.model import DistanceCost


def _time(value):
    """Canonicalize ordinary times while retaining positive subnanosecond values."""
    if value == 0 or abs(value) >= 1e-9:
        return round(value, 9)
    return round(value, max(9, 12 - floor(log10(abs(value)))))


@dataclass(frozen=True)
class SearchStep:
    action: object
    start: float
    end: float
    state: frozenset

    @property
    def state_before(self):
        return self.state


@dataclass
class SearchResult:
    status: str
    steps: list[SearchStep]
    makespan: float
    expanded: int
    generated: int
    rejected: dict[str, int]
    unknown: frozenset
    elapsed: float
    suboptimality_bound: float | None = None
    lower_bound: float | None = None
    weight: float = 1.
    stages: tuple[dict, ...] = ()
    first_solution_seconds: float | None = None
    first_solution_expanded: int | None = None
    early_stop_enabled: bool = False
    early_stop_triggered: bool = False
    early_stop_reason: str | None = None
    post_first_solution_limit: float | None = None


@dataclass(slots=True)
class _Node:
    state: frozenset
    running: tuple
    time: float
    deviation: int
    parent: object = None
    step: SearchStep | None = None
    estimate: float = 0.


def _early_reachability(engine, *, start, goal, allowed, protected, running=(), began):
    from .heuristic import relaxed_reachability
    status, unknown, rejected = relaxed_reachability(engine, start=start, goal=goal,
        allowed=allowed, protected=protected, initial_running=running)
    if status is None:
        return None
    return SearchResult(status, [], 0., 0, 1, rejected, unknown, monotonic()-began)


def _advance(engine, state, running):
    delta = min(remaining for _, remaining in running)
    pending = []
    for key, remaining in running:
        if remaining <= delta:
            state = engine.finish(engine.actions[key], state)
        else:
            pending.append((key, _time(remaining - delta)))
    return state, tuple(pending), delta


def _search(engine, *, start=None, goal=None, hint=None, cuts=(), protected=(),
            allowed=None, max_expansions=200_000, time_limit=60.0,
            bound=None, upper=None, observer=None, initial_running=(), weight=1.,
            prune_infinite=False, deadline=None, candidate_graph=None):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not isfinite(weight) or weight < 1:
        raise ValueError("搜索权重必须是大于等于1的有限数")
    began = monotonic()
    initial = engine.initial if start is None else frozenset(start)
    targets = engine.goal if goal is None else tuple(goal)
    hint_keys = {getattr(a, "key", a) for a in (hint or ())}
    allowed = None if allowed is None else frozenset(allowed)
    protected = frozenset(protected)
    cuts = frozenset(cuts)
    cost = engine.cost if engine.cost is not None else DistanceCost(engine.hooks)
    rejected = Counter({name: 0 for name in ("capability_invalid", "capability_unknown",
                                           "invariant", "cut", "protected")})
    unknown = set()
    expanded = 0
    generated = 1
    sequence = count()
    root_running = tuple(sorted(initial_running))
    root = _Node(initial, root_running, 0.0, 0)
    root_token = next(sequence)
    queue = [(0.0, 0, root_token, root)]
    frontier = [(0., root_token, root)]
    active = {root_token}
    best = {(initial, root_running): (0.0, 0)}
    capability_cache = {}

    def frontier_bound():
        while frontier:
            estimate, token, node = frontier[0]
            if token not in active or best[(node.state, node.running)] != (node.time, node.deviation):
                heappop(frontier)
                continue
            return estimate
        return None

    def result(status, node=None):
        steps = []
        makespan = node.time if node is not None else 0.0
        while node is not None:
            if node.step is not None:
                steps.append(node.step)
            node = node.parent
        steps.reverse()
        lower = frontier_bound()
        certificate = None
        if status == "solved":
            lower = min(makespan, lower) if lower is not None else makespan
            if makespan == 0:
                certificate = 1.
            elif lower > 0:
                certificate = max(1., makespan/lower)
        lower = lower if lower is not None and isfinite(lower) else None
        return SearchResult(status, steps, makespan, expanded, generated,
                            dict(rejected), frozenset(unknown), monotonic() - began,
                            certificate, lower, float(weight))

    def window_expired(token=None):
        if monotonic() < deadline:
            return False
        if token is not None:
            # A partially expanded node still bounds its unvisited successors.
            active.add(token)
        return True

    def enqueue(node):
        nonlocal generated
        key = (node.state, node.running)
        rank = (node.time, node.deviation)
        previous = best.get(key)
        if previous is None or rank < previous:
            estimate = bound(node.state, node.running) if bound is not None else 0.0
            if prune_infinite and not isfinite(estimate):
                return
            if upper is not None and node.time + estimate > upper + 1e-8:
                return
            best[key] = rank
            node.estimate = estimate
            token = next(sequence)
            priority = node.time + estimate if weight == 1 else node.time + weight*estimate
            heappush(queue, (priority, node.deviation, token, node))
            heappush(frontier, (node.time + estimate, token, node))
            active.add(token)
            generated += 1

    if deadline is not None and window_expired():
        return result("budget")
    if engine.hooks.invariant(initial):
        rejected["invariant"] += 1
        return result("no_plan")
    if max_expansions > 0 and time_limit > 0:
        early = _early_reachability(engine, start=initial, goal=targets, allowed=allowed,
                                   protected=protected, running=root_running, began=began)
        if deadline is not None and window_expired():
            return result("budget")
        if early is not None:
            early.weight = float(weight)
            return early
    if bound is not None:
        root.estimate = bound(initial, root_running)
        if deadline is not None and window_expired():
            return result("budget")
        if prune_infinite and not isfinite(root.estimate):
            return result('no_plan')
        active.remove(root_token)
        token = next(sequence)
        active.add(token)
        queue[0] = (root.estimate if weight == 1 else weight*root.estimate, 0, token, root)
        heappush(frontier, (root.estimate, token, root))
    while queue:
        if expanded >= max_expansions or (deadline is not None and window_expired()) or monotonic() - began >= time_limit:
            return result("budget")
        _, _, token, node = heappop(queue)
        active.discard(token)
        if best[(node.state, node.running)] != (node.time, node.deviation):
            continue
        expanded += 1
        if observer is not None:
            observer(node.state, node.running, node.time)
        if deadline is not None and window_expired(token):
            return result("budget")
        if not node.running and engine.satisfied(targets, node.state):
            return result("budget") if deadline is not None and window_expired(token) else result("solved", node)

        busy = set()
        structural = False
        running_keys = set()
        for key, remaining in node.running:
            action = engine.actions[key]
            busy.update(action.occupies)
            structural |= action.structural
            running_keys.add(key)
        candidates = ((action, None) for action in engine.candidates(node.state)) if candidate_graph is None else (
            candidate_graph.starts(node.state))
        for action, cached_duration in candidates:
            if deadline is not None and window_expired(token):
                return result("budget")
            if allowed is not None and action.key not in allowed:
                continue
            if action.key in running_keys or busy.intersection(action.occupies):
                continue
            if action.structural and structural:
                continue
            if candidate_graph is None and not engine.satisfied(action.pre, node.state):
                continue
            if action.key not in capability_cache:
                capability_cache[action.key] = (
                    engine.oracle.evaluate(action.capability) if candidate_graph is None
                    else candidate_graph.evaluate(action))
            status, evidence = capability_cache[action.key]
            if status != Status.VALID:
                rejected["capability_invalid" if status == Status.INVALID else "capability_unknown"] += 1
                unknown.update(e.fact for e in evidence if e.status == Status.UNKNOWN)
                continue
            if (node.state, action.key) in cuts:
                rejected["cut"] += 1
                continue
            if protected.intersection(action.delete):
                rejected["protected"] += 1
                continue
            state = engine.start(action, node.state)
            if (engine.hooks.invariant(state) if candidate_graph is None else cached_duration is None):
                rejected["invariant"] += 1
                continue
            duration = _time(cost.estimate(action, node.state)) if candidate_graph is None else cached_duration
            if duration <= 0:
                raise ValueError(f"动作 {action} 的耗时必须大于零")
            running = tuple(sorted(node.running + ((action.key, duration),)))
            deviation = node.deviation + int(bool(hint_keys) and action.key not in hint_keys)
            step = SearchStep(action, node.time, _time(node.time + duration), node.state)
            enqueue(_Node(state, running, node.time, deviation, node, step))

        if deadline is not None and window_expired(token):
            return result("budget")
        if node.running:
            state, running, delta = _advance(engine, node.state, node.running)
            if engine.hooks.invariant(state):
                rejected["invariant"] += 1
            else:
                enqueue(_Node(state, running, _time(node.time + delta),
                              node.deviation, node))
        if deadline is not None and window_expired(token):
            return result("budget")
    return result("unknown" if rejected["capability_unknown"] else "no_plan")


def weighted_anytime(engine, *, start=None, goal=None, hint=None, cuts=(), protected=(),
                     allowed=None, max_expansions=200_000, time_limit=60.0,
                     bound=None, observer=None, initial_running=(),
                     post_first_solution_limit: float | None = None, candidate_graph=None):
    """Restart weighted A* under one budget, retaining a feasible incumbent.

    Weights change only OPEN ordering. Every certificate uses the unweighted
    live frontier of the most recent full-action search; the reference probe
    can provide an incumbent, never a full-action optimality certificate.
    An optional improvement window starts with the first feasible incumbent.
    """
    enabled = post_first_solution_limit is not None
    if enabled and (isinstance(post_first_solution_limit, bool)
            or not isinstance(post_first_solution_limit, (int, float))
            or not isfinite(post_first_solution_limit) or post_first_solution_limit < 0):
        raise ValueError("首解后的时间上限必须是非负有限秒数")

    def annotate(answer):
        answer.early_stop_enabled = enabled
        answer.post_first_solution_limit = post_first_solution_limit
        return answer

    start = None if start is None else frozenset(start)
    goal = None if goal is None else tuple(goal)
    hint = None if hint is None else tuple(hint)
    cuts = frozenset(cuts)
    protected = frozenset(protected)
    allowed = None if allowed is None else frozenset(allowed)
    initial_running = tuple(initial_running)
    options = dict(start=start, goal=goal, hint=hint, cuts=cuts, protected=protected,
                   allowed=allowed, max_expansions=max_expansions, time_limit=time_limit,
                   observer=observer, initial_running=initial_running)
    if candidate_graph is not None:
        options["candidate_graph"] = candidate_graph
    if max_expansions <= 0 or time_limit <= 0:
        return annotate(_search(engine, **options))
    began = monotonic()
    original_deadline = deadline = began + time_limit
    if engine.hooks.invariant(engine.initial if start is None else start):
        return annotate(_search(engine, **options))
    early = _early_reachability(engine, start=start, goal=goal, allowed=allowed,
                               protected=protected, running=initial_running, began=began)
    if early is not None:
        return annotate(early)
    from .heuristic import build_heuristic, _operation_floor_view
    from .bounds import HMaxBound
    from skillnet.domains.blocks import BlocksHooks

    remaining = max(0., deadline-monotonic())
    if remaining <= 0:
        return annotate(SearchResult("budget", [], 0., 0, 1,
            {name: 0 for name in ("capability_invalid", "capability_unknown", "invariant", "cut", "protected")},
            frozenset(), monotonic()-began))
    provided_bound = bound is not None
    initial = engine.initial if start is None else start
    targets = engine.goal if goal is None else goal
    # Allocation constrains filled goals; h_add also guides location goals.
    # Generic domains retain
    # their exact v133 ordering and reference-probe contract.
    guide_eligible = (type(engine.hooks) is BlocksHooks and max_expansions >= 1000
        and any(not getattr(c, 'negated', False) for c in targets))
    floor_view = _operation_floor_view(engine) if guide_eligible else None
    if bound is None and guide_eligible:
        # A cheap admissible root survives even if guidance consumes the time
        # before the full HMax/PDB builder can run. Never substitute h_add.
        bound = HMaxBound(floor_view, goal=goal, coarse=True, allowed=allowed)
    elif bound is None:
        bound = build_heuristic(engine, start=start, goal=goal,
                               time_limit=min(8.0, time_limit / 5, remaining), allowed=allowed)
    total_expanded, total_generated = 0, 0
    total_rejected, total_unknown = Counter(), set()
    stages = []
    incumbent, last = None, None
    lower = None
    first_solution_seconds, first_solution_expanded = None, None

    def active_deadline():
        return deadline if enabled and first_solution_seconds is not None else None

    def account(answer, phase, weight, *, action_scope=None, allowed_actions=None,
                time_budget=None, expansion_budget=None):
        nonlocal total_expanded, total_generated, first_solution_seconds, first_solution_expanded, deadline
        if answer.status == 'solved' and first_solution_seconds is None:
            first_at = monotonic()
            first_solution_seconds = first_at-began
            first_solution_expanded = total_expanded+answer.expanded
            if enabled:
                deadline = min(deadline, first_at+post_first_solution_limit)
        total_expanded += answer.expanded
        total_generated += answer.generated
        total_rejected.update(answer.rejected)
        total_unknown.update(answer.unknown)
        stages.append({"phase": phase, "weight": weight, "status": answer.status,
            "makespan": answer.makespan if answer.status == "solved" else None,
            "expanded": answer.expanded, "generated": answer.generated, "elapsed": answer.elapsed,
            "lower_bound": answer.lower_bound if phase == "weighted" else None,
            "suboptimality_bound": answer.suboptimality_bound if phase == "weighted" else None,
            "action_scope": action_scope or ("all_actions" if allowed is None else "caller_allowed"),
            "allowed_actions": allowed_actions if phase in {"reference", "incumbent"} else
                (len(allowed) if allowed is not None else None),
            "incumbent_only": phase in {"reference", "incumbent"}, "time_budget": time_budget,
            "expansion_budget": expansion_budget, "non_admissible_ordering": phase=='incumbent',
            "first_solution_seconds": first_solution_seconds,
            "first_solution_expanded": first_solution_expanded})

    if guide_eligible:
        from .allocation import Allocation
        # Construction and both attempts share this allowance and the original
        # global wall/expansion budget. Failed restricted probes prove nothing.
        guide_deadline = min(deadline, monotonic()+min(15., time_limit/4))
        allocation = Allocation(engine, targets, start=initial)
        goal_domain = frozenset(engine.hooks.reference_actions(engine.actions.values(), initial, targets))
        if allowed is not None:
            goal_domain &= allowed
        if hint:
            goal_domain &= frozenset(getattr(a, 'key', a) for a in hint)
        scopes = [('goal_domain', goal_domain), ('all_actions' if allowed is None else 'caller_allowed', allowed)]
        for scope, subset in scopes:
            remaining_guide = guide_deadline-monotonic()
            if remaining_guide <= 0 or total_expanded >= max_expansions:
                break
            relaxed = HMaxBound(floor_view, goal=goal, coarse=True, allowed=subset)
            def guide(state, running=()):
                if allocation.deficiency(state, running, relaxed.effects)>0:
                    return float('inf')
                return relaxed.hadd(state, running)
            allowance = max(0., guide_deadline-monotonic())
            if scope == 'goal_domain':
                allowance /= 2.  # Reserve a real all-action attempt on budget failure.
            if allowance <= 0:
                break
            expansion_budget = max_expansions-total_expanded
            answer = _search(engine, **{**options, 'allowed':subset,
                'time_limit':allowance, 'max_expansions':expansion_budget},
                bound=guide, weight=3., prune_infinite=True)
            account(answer, 'incumbent', 3., action_scope=scope,
                allowed_actions=len(subset) if subset is not None else len(engine.actions),
                time_budget=allowance, expansion_budget=expansion_budget)
            if answer.status=='solved':
                incumbent = answer
                if not enabled or monotonic() < deadline:
                    lower = bound(initial, initial_running)
                break

        # Only the admissible full-action graph can supply a certificate.
        remaining = max(0., deadline-monotonic())
        if not provided_bound and remaining>0:
            bound = build_heuristic(engine, start=start, goal=goal, allowed=allowed,
                time_limit=min(8., time_limit/5, remaining))

    probe_options = dict(options)
    probe_options.update(max_expansions=min(20_000, max_expansions // 5, max_expansions-total_expanded),
                         time_limit=min(10.0, time_limit / 4))
    reference_filter = getattr(engine.hooks, "reference_actions", None)
    if reference_filter is not None and max_expansions >= 1000 and monotonic() < deadline:
        # This optional domain/goal filter supplies only an incumbent probe.
        # It is independent of proposals or generated witnesses; an explicit
        # caller hint retains its original intersection semantics.
        reference_allowed = frozenset(reference_filter(engine.actions.values(),
            engine.initial if start is None else start, engine.goal if goal is None else goal))
        if allowed is not None:
            reference_allowed &= allowed
        if hint:
            reference_allowed &= frozenset(getattr(a, "key", a) for a in hint)
        # This legacy probe reuses the caller/full admissible ordering exactly.
        # Restricted graph construction belongs to the new incumbent stages;
        # changing this old ranking can worsen a budgeted v133 reference plan.
        probe_bound = bound
        probe_options["allowed"] = reference_allowed
        probe_options["time_limit"] = max(0.0, min(probe_options["time_limit"],
            deadline - monotonic()))
        if probe_options['time_limit']>0 and probe_options['max_expansions']>0:
            reference = _search(engine, **probe_options, bound=probe_bound, weight=3.,
                                upper=incumbent.makespan if incumbent is not None else None,
                                deadline=active_deadline())
            account(reference, "reference", 3.,
                action_scope="goal_domain_and_hint" if hint else "goal_domain",
                allowed_actions=len(reference_allowed), time_budget=probe_options["time_limit"],
                expansion_budget=probe_options["max_expansions"])
            # A failed restricted probe is not a full-action conclusion.
            if reference.status == "solved" and (incumbent is None or reference.makespan < incumbent.makespan):
                incumbent = reference
                if not enabled or monotonic() < deadline:
                    lower = bound(initial, initial_running)
    for weight in (3., 2., 1.5, 1.):
        remaining_time = max(0., deadline-monotonic())
        remaining_expansions = max_expansions-total_expanded
        if remaining_time <= 0 or remaining_expansions <= 0:
            break
        stage_options = {**options, "time_limit": remaining_time,
                         "max_expansions": remaining_expansions}
        answer = _search(engine, **stage_options, bound=bound, weight=weight,
                         upper=incumbent.makespan if incumbent is not None else None,
                         deadline=active_deadline())
        account(answer, "weighted", weight, time_budget=remaining_time,
                expansion_budget=remaining_expansions)
        last, lower = answer, answer.lower_bound
        if answer.status == "solved":
            if incumbent is None or answer.makespan < incumbent.makespan or weight == 1.:
                incumbent = answer
            # A bound of 1 proves optimal cost, not the exact hint/tie choice.
            # Always let the w=1 restart return its own original exact result.
            if weight == 1.:
                break
        else:
            break  # A full-stage budget/closure cannot improve by resetting it.
    chosen = incumbent or last
    if chosen is None:
        chosen = SearchResult("budget", [], 0., 0, 1,
            {name: 0 for name in ("capability_invalid", "capability_unknown", "invariant", "cut", "protected")},
            frozenset(), 0.)
        total_generated = total_generated or 1
    if incumbent is not None:
        chosen.status = "solved"
        if lower is None and (last is None or last.status=='budget'):
            if not enabled or monotonic() < deadline:
                lower = bound(initial, initial_running)
        if lower is not None:
            lower = min(chosen.makespan, lower)
        elif not enabled or chosen.makespan == 0.:
            lower = chosen.makespan
        chosen.suboptimality_bound = (1. if chosen.makespan == 0 else
            max(1., chosen.makespan/lower) if lower is not None and lower > 0 else None)
        chosen.lower_bound = lower if lower is not None and isfinite(lower) else None
    chosen.expanded, chosen.generated = total_expanded, total_generated
    chosen.rejected = dict(total_rejected) if stages else chosen.rejected
    chosen.unknown = frozenset(total_unknown)
    finished = monotonic()
    chosen.elapsed = finished-began
    if enabled and last is chosen and chosen.status == "solved" and chosen.weight == 1. and chosen.suboptimality_bound == 1.:
        chosen.early_stop_reason = "optimality_proven"
    elif enabled and deadline < original_deadline and finished >= deadline:
        chosen.early_stop_triggered = True
        chosen.early_stop_reason = "post_first_solution_limit"
    chosen.stages = tuple(stages)
    chosen.first_solution_seconds = first_solution_seconds
    chosen.first_solution_expanded = first_solution_expanded
    return annotate(chosen)


def search(engine, *, start=None, goal=None, hint=None, cuts=(), protected=(),
           allowed=None, max_expansions=200_000, time_limit=60.0, initial_running=(),
           post_first_solution_limit: float | None = None, candidate_graph=None):
    return weighted_anytime(engine, start=start, goal=goal, hint=hint, cuts=cuts,
        protected=protected, allowed=allowed, max_expansions=max_expansions,
        time_limit=time_limit, initial_running=initial_running,
         post_first_solution_limit=post_first_solution_limit, candidate_graph=candidate_graph)
