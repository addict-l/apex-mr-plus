"""Reuse grounded actions and repair a nominal schedule after a capability change."""

from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.skill import ActionKey
from skillnet.planning.search import SearchResult, search
from skillnet.planning.taskgraph import TaskGraph, build_task_graph


def _without_arm_binding(action):
    return action.name, tuple(value for value, kind in zip(action.args, action.skill.params.values())
                              if kind != "arm")


@dataclass
class RecoveryResult:
    schedule: SearchResult
    graph: TaskGraph | None
    elapsed: float
    setup_seconds: float
    affected_actions: int
    excluded: frozenset[ActionKey]
    repair_scope: int


class RecoveryPlanner:
    """Prepare once for a fixed set of task objects, goals and skill rules."""

    def __init__(self, problem, nominal_actions):
        began = monotonic()
        self.engine = Engine(problem)
        self.nominal = tuple(self.engine.parse_action(action) for action in nominal_actions)
        self._objects = frozenset(value for group in problem.objects.values() for value in group)
        self._objects |= frozenset(value for fact in problem.initial for value in fact.args)
        self._capability_users = defaultdict(set)
        self._skill_users = defaultdict(set)
        self._bindings = defaultdict(set)
        self._producers = defaultdict(set)
        for action in self.engine.actions.values():
            for fact in action.capability:
                self._capability_users[fact].add(action.key)
            for arm in action.occupies:
                self._skill_users[arm, action.name].add(action.key)
            self._bindings[_without_arm_binding(action)].add(action.key)
            for fact in action.add:
                self._producers[fact].add(action.key)
        self.preparation_seconds = monotonic() - began

    def _repair_scope(self, engine, remaining, allowed):
        keys = {key for action in remaining for key in self._bindings[_without_arm_binding(action)]}
        if allowed is not None:
            keys &= allowed
        # Include one layer of producers, such as a handover that gives a
        # substitute installer its part. The unrestricted search handles
        # longer chains, changes of part, and other scheduling alternatives.
        producers = {
            key
            for action_key in keys
            for condition in engine.actions[action_key].pre
            if not condition.negated and not engine.holds(condition, engine.initial)
            for key in self._producers.get(condition.fact, ())
        }
        scope = keys | producers
        return scope if allowed is None else scope & allowed

    def replan(self, state, *, capabilities=None, disabled_skills=(), completed_steps=(),
               protected=(), cuts=(), time_limit=60., max_expansions=200_000, incremental=True):
        """Return a symbolic plan from a quiescent, measured state.

        Capability updates are exact grounded facts. A disabled skill applies
        to actions of that skill that occupy the specified arm, including joint
        actions. The original task and capability table are safe for reuse.
        """
        began = monotonic()
        if not isfinite(time_limit) or time_limit <= 0 or max_expansions <= 0:
            raise ValueError("恢复搜索的时间和扩展预算必须为正")
        problem = copy(self.engine.problem)
        problem.initial = frozenset(state)
        invalid_facts = {fact for fact in problem.initial
                         if fact.predicate not in self.engine.dynamic_predicates}
        if invalid_facts:
            raise ValueError("恢复快照包含非动态事实：" + ", ".join(map(str, sorted(invalid_facts))))
        unknown_objects = {value for fact in problem.initial
                           for value in fact.args if value not in self._objects}
        if unknown_objects:
            raise ValueError("恢复快照包含未登记的对象或位置：" + ", ".join(sorted(unknown_objects)))

        disabled = frozenset(disabled_skills)
        for arm, skill in disabled:
            if arm not in problem.objects["arm"] or skill not in problem.library.by_name:
                raise ValueError(f"未知的臂或技能：{arm}, {skill}")
        completed = frozenset(completed_steps)
        if any(type(index) is not int or not 0 <= index < len(self.nominal) for index in completed):
            raise ValueError("completed_steps 必须使用原计划中有效的 0 基步骤编号")
        remaining = tuple(action for index, action in enumerate(self.nominal) if index not in completed)

        problem.oracle = deepcopy(self.engine.oracle)
        changed = set()
        for fact, status in (capabilities or {}).items():
            status = Status(status)
            if problem.oracle.check(fact).status != status:
                changed.add(fact)
                problem.oracle.update(fact, status, "恢复故障观测")

        if incremental:
            engine = copy(self.engine)
            engine.problem = problem
            engine.initial = problem.initial
            engine.oracle = problem.oracle
            affected = {key for fact in changed for key in self._capability_users.get(fact, ())}
            excluded = {key for arm_skill in disabled for key in self._skill_users.get(arm_skill, ())}
            affected |= excluded
            # UNKNOWN remains in the domain so an unverified capability is
            # available for diagnostics and does not become a no-plan proof.
            excluded.update(key for key in affected
                            if engine.oracle.evaluate(engine.actions[key].capability)[0] == Status.INVALID)
        else:
            engine = Engine(problem, cost=self.engine.cost)
            affected = {action.key for action in engine.actions.values()
                        if changed.intersection(action.capability)}
            excluded = {action.key for action in engine.actions.values()
                        if any((arm, action.name) in disabled for arm in action.occupies)}
            affected |= excluded

        allowed = frozenset(engine.actions) - excluded if excluded else None
        repair_scope = self._repair_scope(engine, remaining, allowed) if incremental else None
        setup = monotonic() - began
        schedule = search(engine, allowed=allowed, protected=protected, cuts=cuts,
                          hint=tuple(action.key for action in remaining) if incremental else None,
                          repair_actions=repair_scope, time_limit=time_limit-setup,
                          max_expansions=max_expansions)
        graph = build_task_graph(engine, schedule.steps) if schedule.status == "solved" else None
        return RecoveryResult(schedule, graph, monotonic()-began, setup, len(affected),
                              frozenset(excluded), len(repair_scope or ()))
