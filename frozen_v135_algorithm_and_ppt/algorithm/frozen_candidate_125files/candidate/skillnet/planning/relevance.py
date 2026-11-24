"""Backward relevance over positive effects, deletions and domain needs."""

from skillnet.core.logic import Condition, Fact
from skillnet.core.skill import ActionKey


def relevant_actions(engine, goal=None) -> tuple[frozenset[ActionKey], dict[ActionKey, list[Fact]]]:
    targets = engine.goal if goal is None else tuple(goal)
    needed = set()
    needed_deleted = set()

    def base_facts(fact):
        pending = [fact]
        visited = set()
        result = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current.predicate in engine.hooks.derived_predicates:
                pending.extend(engine.hooks.expand(current))
            else:
                result.add(current)
        return result

    for target in targets:
        condition = Condition(target) if isinstance(target, Fact) else target
        if condition.negated:
            needed_deleted.add(condition.fact)
        else:
            needed.update(base_facts(condition.fact))

    relevant = set()
    changed = True
    while changed:
        changed = False
        for action in engine.actions.values():
            hits = needed.intersection(action.add) | needed_deleted.intersection(action.delete)
            if not hits or action.key in relevant:
                continue
            relevant.add(action.key)
            changed = True
            for condition in action.pre:
                if not condition.negated:
                    needed.update(base_facts(condition.fact))
            for fact in engine.hooks.invariant_needs(action):
                needed.update(base_facts(fact))

    reasons = {
        key: sorted(needed.intersection(engine.actions[key].add)
                    | needed_deleted.intersection(engine.actions[key].delete))
        for key in sorted(relevant)
    }
    return frozenset(relevant), reasons
