"""Build admissible remaining-makespan estimates independently of proposals."""
from copy import copy
from dataclasses import replace
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.logic import Condition, Fact
from skillnet.cost.model import DistanceCost, _calibration
from skillnet.domains.blocks import BlocksHooks

from .bounds import HMaxBound
from .abstraction import PatternDatabase, EventPatternDatabase


def relaxed_reachability(engine, *, start=None, goal=None, allowed=None,
                         protected=(), initial_running=()):
    """A necessary-condition graph, with goal-support UNKNOWN provenance.

    Positive and negative facts are independent rails: both may become true.
    Deletes therefore add a negative rail without removing the positive one.
    Resources, ordering, invariants and arbitrary derived conditions are fully
    relaxed. State-specific cuts must not remove a producer in other states.
    Already running effects are unconditional, including implicit arm moves.
    """
    initial = engine.initial if start is None else frozenset(start)
    goals = tuple(Condition(item) if isinstance(item, Fact) else item
                  for item in (engine.goal if goal is None else goal))
    allowed = None if allowed is None else frozenset(allowed)
    protected = frozenset(protected)
    actions = tuple(engine.actions.values())
    facts = set(initial | engine.static)
    facts.update(condition.fact for condition in goals)
    for action in actions:
        facts.update(condition.fact for condition in action.pre)
        facts.update(action.add)
        facts.update(action.delete)
        facts.update(Fact("arm_at", (arm, place)) for arm, place in action.move_to.items())
    arm_facts = [fact for fact in facts if fact.predicate == "arm_at" and len(fact.args) == 2]
    effects = {}
    for action in actions:
        positive = set(action.add) | {Fact("arm_at", (arm, place)) for arm, place in action.move_to.items()}
        negative = set(action.delete) | {fact for fact in arm_facts
            if fact.args[0] in action.move_to and fact.args[1] != action.move_to[fact.args[0]]}
        effects[action.key] = tuple(Condition(fact) for fact in positive) + tuple(Condition(fact, True) for fact in negative)
    proofs = {}
    for fact in facts:
        if fact.predicate in engine.hooks.derived_predicates:
            # Nonmonotone/custom derivations cannot be evaluated on a union of
            # mutually inconsistent relaxed facts. Both values are possible.
            proofs[Condition(fact)] = proofs[Condition(fact, True)] = frozenset()
        else:
            present = fact in (engine.static if engine.is_static(fact.predicate) else initial)
            proofs[Condition(fact, not present)] = frozenset()
    for key, _ in initial_running:
        for effect in effects[key]:
            proofs[effect] = frozenset()
    rules = []
    rejected = {name: 0 for name in ("capability_invalid", "capability_unknown", "invariant", "cut", "protected")}
    for action in actions:
        if allowed is not None and action.key not in allowed:
            continue
        if protected.intersection(action.delete):
            rejected["protected"] += 1
            continue
        status, evidence = engine.oracle.evaluate(action.capability)
        if status == Status.INVALID:
            rejected["capability_invalid"] += 1
            continue
        unknown = frozenset(item.fact for item in evidence if item.status == Status.UNKNOWN)
        rules.append((action.pre, effects[action.key], unknown))

    def close(include_unknown):
        changed = True
        while changed:
            changed = False
            for prerequisites, additions, unknown in rules:
                if unknown and not include_unknown or not all(pre in proofs for pre in prerequisites):
                    continue
                support = unknown.union(*(proofs[pre] for pre in prerequisites))
                for effect in additions:
                    if effect not in proofs:
                        proofs[effect] = support
                        changed = True

    close(False)
    if all(goal in proofs for goal in goals):
        return None, frozenset(), rejected
    close(True)
    if not all(goal in proofs for goal in goals):
        return "no_plan", frozenset(), rejected
    unknown = frozenset().union(*(proofs[goal] for goal in goals))
    rejected["capability_unknown"] = len(unknown)
    return "unknown", unknown, rejected


class _OriginalActionHooks:
    """Keep domain callbacks on original actions while raising cost floors."""
    def __init__(self, hooks, actions):
        self.hooks, self.actions = hooks, actions

    def __getattr__(self, name):
        target = getattr(self.hooks, name)
        if name == "invariant_needs":
            return lambda action: target(self.actions[action.key])
        if name == "optimistic_effect_time":
            return lambda effect, earliest, times, action: target(effect, earliest, times, self.actions[action.key])
        return target


def _operation_floor_view(engine):
    if engine.cost is not None and (type(engine.cost) is not DistanceCost
                                    or engine.cost.hooks is not engine.hooks):
        return engine
    actions = {}
    for key, action in engine.actions.items():
        delta = _calibration(action).get("fixed_delta", 0.)
        actions[key] = (replace(action, skill=replace(action.skill, op_time=action.op_time+delta))
                        if delta else action)
    if all(actions[key] is action for key, action in engine.actions.items()):
        return engine
    view = copy(engine)
    view.actions = actions
    view.hooks = _OriginalActionHooks(engine.hooks, engine.actions)
    return view


def build_heuristic(engine, *, start=None, goal=None, time_limit=8.0, allowed=None):
    """Build a bound for the given action space (None means the full space).

    A restricted relaxation is valid for that subproblem only; final search
    certificates must use a full-space instance. PDB remains a full-space
    lower bound, including unconditional effects of already running work.
    """
    began = monotonic()
    floor_view = _operation_floor_view(engine)
    hmax = HMaxBound(floor_view, goal=goal, coarse=True, allowed=allowed)
    # PDB transitions use raw_op plus the old optimistic d/speed, never the
    # concrete DistanceCost estimator. Thus fixed_delta appears once and the
    # travel multiplier is conservatively omitted. Keep exact domain hooks for
    # PDB's compatibility check; HMax alone needs the original-action proxy.
    pattern_view = engine
    if floor_view is not engine and type(engine.hooks) is BlocksHooks:
        pattern_view = copy(floor_view)
        pattern_view.hooks = engine.hooks
    database = PatternDatabase(pattern_view, start=start, goal=goal,
                               time_limit=max(0.0, min(3.0, time_limit / 3)))
    if database.enabled and database.states <= 1000:
        database = EventPatternDatabase(pattern_view, start=start, goal=goal, quantum=0.5,
            max_states=200_000, time_limit=max(0.0, time_limit - (monotonic() - began)))

    def estimate(state, running=()):
        return max(hmax(state, running), database(state, running))

    return estimate
