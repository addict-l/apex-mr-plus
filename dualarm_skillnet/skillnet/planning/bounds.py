"""Admissible temporal hmax bounds for optional UCS branch-and-bound."""

from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from math import inf

from skillnet.capability.oracle import Status
from skillnet.core.logic import Condition, Fact
from skillnet.cost.model import DistanceCost
from skillnet.domains.blocks import BlocksHooks


_BLOCKS_EFFECT_TIME = BlocksHooks.optimistic_effect_time


class HMaxBound:
    """Freeze capability evidence for one search and relax deletes/resources.

    Each positive fact receives its earliest relaxed completion time. Actions
    combine prerequisites with max and alternative producers with min. Custom
    cost models have no declared operation-time floor, so their cost floor is
    zero. Optional domain optimistic_time callbacks must be admissible.
    """

    def __init__(self, engine, goal=None, cache_limit=20_000, coarse=False,
                 allowed=None):
        self.engine = engine
        self.cache_limit = cache_limit
        self.coarse = coarse
        self.cache = {}
        self.allowed = None if allowed is None else frozenset(allowed)
        self.targets = tuple(Condition(item) if isinstance(item, Fact) else item
                             for item in (engine.goal if goal is None else goal))
        self.positive_goals = tuple(item.fact for item in self.targets if not item.negated)
        self.negative_goals = tuple(item.fact for item in self.targets if item.negated
                                    and item.fact.predicate not in engine.hooks.derived_predicates)
        self.optimistic = getattr(engine.hooks, "optimistic_time", None)
        self.effect_time = getattr(engine.hooks, "optimistic_effect_time", None)
        # The standard block callback is the identity unless an added filled
        # fact denotes an overhang. Only bypass this proven identity for the
        # exact unmodified implementation, including its floor-view proxy.
        raw_hooks = engine.hooks
        if type(raw_hooks) is not BlocksHooks:
            from .heuristic import _OriginalActionHooks
            if type(raw_hooks) is _OriginalActionHooks:
                raw_hooks = raw_hooks.hooks
        identity_blocks = (type(raw_hooks) is BlocksHooks
            and getattr(raw_hooks.optimistic_effect_time, "__func__", None)
                is _BLOCKS_EFFECT_TIME)
        self.effects = {
            action.key: tuple(fact for fact in action.add
                              if not (fact.predicate == "arm_at" and fact.args[0] in action.move_to))
            + tuple(Fact("arm_at", (arm, place)) for arm, place in action.move_to.items())
            for action in engine.actions.values()
        }
        self.actions = []
        self.rule_actions = []
        self.rule_clocks = []
        valid_actions = []
        self.action_dependents = defaultdict(list)
        self.derived_dependents = defaultdict(list)
        self.derived = set()
        distance_cost = engine.cost is None or type(engine.cost) is DistanceCost

        def operation_floor(action):
            # UCS rounds event times to nanoseconds, so its floor must allow
            # downward rounding before accumulating long relaxed paths.
            return max(0.0, action.op_time - 1e-9) if distance_cost else 0.0

        for action in engine.actions.values():
            if self.allowed is not None and action.key not in self.allowed:
                continue
            if engine.oracle.evaluate(action.capability)[0] != Status.VALID:
                continue
            valid_actions.append(action)
            prerequisites = tuple(condition.fact for condition in action.pre
                                  if not condition.negated and not engine.is_static(condition.fact.predicate))
            rule = (prerequisites, self.effects[action.key], operation_floor(action))
            index = len(self.actions)
            self.actions.append(rule)
            self.rule_actions.append(action)
            needs_clock = not identity_blocks or any(
                fact.predicate == "filled" and len(fact.args) == 1
                and raw_hooks.slots.get(fact.args[0], {}).get("overhang", False)
                for fact in rule[1])
            self.rule_clocks.append(self.effect_time if needs_clock else None)
            for fact in prerequisites:
                self.action_dependents[fact].append(index)
                self._collect_derived(fact)
            if self.effect_time:
                for fact in engine.hooks.invariant_needs(action):
                    self.action_dependents[fact].append(index)
                    self._collect_derived(fact)
                    if fact.predicate in engine.hooks.derived_predicates:
                        for dependency in engine.hooks.expand(fact):
                            if index not in self.action_dependents[dependency]:
                                self.action_dependents[dependency].append(index)
                            self._collect_derived(dependency)
        # Preconditions settle once in the ordinary nonnegative relaxation.
        # Invariant/effect dependencies do not gate firing: they may change an
        # already enabled rule's effect time (support/anchors in BlocksHooks).
        self.pre_dependents = defaultdict(list)
        self.extra_dependents = defaultdict(list)
        self.pre_count = []
        pre_sets = []
        for index, (prerequisites, _, _) in enumerate(self.actions):
            distinct = frozenset(prerequisites)
            pre_sets.append(distinct)
            self.pre_count.append(len(distinct))
            for fact in distinct:
                self.pre_dependents[fact].append(index)
        for fact, indices in self.action_dependents.items():
            for index in dict.fromkeys(indices):
                if fact not in pre_sets[index]:
                    self.extra_dependents[fact].append(index)
        for fact in self.positive_goals:
            self._collect_derived(fact)
        self.deletors = {}
        self.negative_tail = {}
        for fact in self.negative_goals:
            choices = []
            for action in valid_actions:
                at_end = (fact.predicate == "arm_at" and len(fact.args) == 2
                          and fact.args[0] in action.move_to and action.move_to[fact.args[0]] != fact.args[1])
                if (fact not in action.delete and not at_end) or fact in self.effects[action.key]:
                    continue
                prerequisites = tuple(condition.fact for condition in action.pre
                                      if not condition.negated and not engine.is_static(condition.fact.predicate))
                duration = operation_floor(action)
                restoration = 0.0
                for target in self.positive_goals:
                    if target not in action.delete or target in self.effects[action.key]:
                        continue
                    restorers = [other for other in valid_actions
                                 if target in self.effects[other.key] and fact not in self.effects[other.key]]
                    if not restorers:
                        restoration = inf
                        break
                    if all(set(action.occupies).intersection(other.occupies) for other in restorers):
                        restoration = max(restoration, min(operation_floor(other) for other in restorers))
                choices.append((prerequisites, duration, restoration, at_end))
            self.deletors[fact] = choices
            self.negative_tail[fact] = min(((0.0 if at_end else duration) + restoration
                                            for _, duration, restoration, at_end in choices), default=inf)

    def _collect_derived(self, fact):
        if fact.predicate not in self.engine.hooks.derived_predicates or fact in self.derived:
            return
        self.derived.add(fact)
        for dependency in self.engine.hooks.expand(fact):
            self.derived_dependents[dependency].append(fact)
            self._collect_derived(dependency)

    def evaluate(self, state, running=()):
        """Return (admissible max, non-admissible goal sum) from one closure.

        The sum is only a first-solution ordering signal. Running work and
        negative-goal cleanup keep the max semantics used by the bound.
        """
        state = frozenset(state)
        running = tuple(running)
        remaining_bound = max((remaining for _, remaining in running), default=0.0)
        future_adds = {}
        future_deletes = {}
        for action_key, remaining in running:
            action = self.engine.actions[action_key]
            for fact in self.negative_goals:
                if fact in self.effects[action_key]:
                    future_adds[fact] = max(future_adds.get(fact, 0.0), remaining)
                if (fact.predicate == "arm_at" and len(fact.args) == 2
                        and fact.args[0] in action.move_to
                        and action.move_to[fact.args[0]] != fact.args[1]):
                    future_deletes[fact] = min(future_deletes.get(fact, inf), remaining)
        future_cleanup = max((remaining + self.negative_tail[fact]
                              for fact, remaining in future_adds.items()), default=0.0)
        running_key = tuple(sorted(action_key for action_key, _ in running)) if self.coarse else running
        key = (state, running_key)
        if key in self.cache:
            maximum, additive = self.cache[key]
            return (max(maximum, remaining_bound, future_cleanup),
                    max(additive, remaining_bound, future_cleanup))
        times = {}
        queue = []
        order = count()
        counts = list(self.pre_count)
        fired = [None] * len(counts)
        settled = set()
        rule_clocks = self.rule_clocks
        actions = self.actions
        rule_actions = self.rule_actions

        def offer(fact, time):
            if time < times.get(fact, inf):
                times[fact] = time
                heappush(queue, (time, next(order), fact))

        def fire(index, earliest):
            _, effects, duration = actions[index]
            completion = earliest + duration
            effect_time = rule_clocks[index]
            for effect in effects:
                offer(effect, effect_time(effect, completion, times, rule_actions[index])
                      if effect_time else completion)

        def update_derived(fact):
            # The base protocol allows nonmonotone derivations, hence zero is
            # the only universal fallback without a domain-supplied bound.
            earliest = self.optimistic(fact, times) if self.optimistic else 0.0
            offer(fact, earliest)

        for fact in state | self.engine.static:
            offer(fact, 0.0)
        for action_key, remaining in running:
            for effect in self.effects[action_key]:
                offer(effect, 0.0 if self.coarse else remaining)
        for fact in self.derived:
            if self.engine.hooks.evaluate(fact.predicate, fact.args, state):
                offer(fact, 0.0)
            update_derived(fact)
        for index, remaining in enumerate(counts):
            if not remaining:
                fired[index] = 0.0
                fire(index, 0.0)
        while queue:
            time, _, fact = heappop(queue)
            if times[fact] != time:
                continue
            first = fact not in settled
            settled.add(fact)
            for index in self.pre_dependents.get(fact, ()):
                if first:
                    counts[index] -= 1
                    if counts[index] == 0:
                        fired[index] = time
                        fire(index, time)
                elif fired[index] is not None:
                    # Custom optimistic callbacks may lower a fact after its
                    # first pop. Preserve the previous fixed-point semantics
                    # without decrementing its prerequisite counter twice.
                    earliest = max((times.get(pre, inf) for pre in actions[index][0]), default=0.0)
                    fired[index] = earliest
                    fire(index, earliest)
            for index in self.extra_dependents.get(fact, ()):
                if fired[index] is not None:
                    fire(index, fired[index])
            for derived in self.derived_dependents.get(fact, ()):
                update_derived(derived)
        relaxed_goal_bound = max((times.get(fact, inf) for fact in self.positive_goals), default=0.0)
        additive = sum(times.get(fact, inf) for fact in self.positive_goals)
        negative_bound = 0.0
        for fact in self.negative_goals:
            if fact not in state and fact not in self.engine.static and fact not in future_adds:
                continue
            released = 0.0 if self.coarse else future_adds.get(fact, 0.0)
            completion_times = []
            if fact in future_deletes:
                completion_times.append(0.0 if self.coarse else future_deletes[fact])
            for prerequisites, duration, restoration, at_end in self.deletors[fact]:
                ready = max((times.get(pre, inf) for pre in prerequisites), default=0.0)
                completion = max(ready + duration, released) if at_end else max(ready, released) + duration
                completion_times.append(completion + restoration)
            deletion_time = min(completion_times, default=inf)
            relaxed_goal_bound = max(relaxed_goal_bound, deletion_time)
            negative_bound = max(negative_bound, deletion_time)
        additive = max(additive, negative_bound)
        if self.cache_limit > 0:
            if len(self.cache) >= self.cache_limit:
                self.cache.clear()
            self.cache[key] = (relaxed_goal_bound, additive)
        return (max(relaxed_goal_bound, remaining_bound, future_cleanup),
                max(additive, remaining_bound, future_cleanup))

    def __call__(self, state, running=()):
        return self.evaluate(state, running)[0]

    def hadd(self, state, running=()):
        """Non-admissible ordering value; never use for lower-bound proofs."""
        return self.evaluate(state, running)[1]

    estimate = __call__


def lower_bound(engine, state, running=()):
    """Convenience entry point which refreshes after capability updates."""
    snapshot = (tuple(sorted(engine.oracle.defaults.items())),
                tuple(sorted((fact, evidence.status) for fact, evidence in engine.oracle.entries.items())),
                tuple(engine.goal), id(engine.cost))
    previous = getattr(engine, "_hmax_bound_context", None)
    if previous is None or previous[0] != snapshot:
        previous = (snapshot, HMaxBound(engine))
        engine._hmax_bound_context = previous
    return previous[1](state, running)
