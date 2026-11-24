"""Optional admissible block-domain work bounds from a finite abstraction."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import dist, floor, isfinite
from itertools import combinations
from time import monotonic

from skillnet.capability.oracle import Status
from skillnet.core.logic import Condition, Fact
from skillnet.domains.blocks import BlocksHooks
from skillnet.cost.model import DistanceCost


# These templates delimit the serializable block semantics supported below.
_EXPECTED = {
    "pick": ("free(?arm)|at(?block, ?loc)|not two_arm_part(?block)", "held(?arm, ?block)|empty(?loc)", "free(?arm)|at(?block, ?loc)", False),
    "place": ("held(?arm, ?block)|empty(?loc)|placeable(?loc)", "at(?block, ?loc)|free(?arm)", "held(?arm, ?block)|empty(?loc)", False),
    "handover": ("held(?giver, ?block)|free(?taker)|handover_point(?loc)", "held(?taker, ?block)|free(?giver)", "held(?giver, ?block)|free(?taker)", False),
    "insert": ("held(?arm, ?block)|fits(?block, ?slot)|not filled(?slot)|slot_ready(?slot)|insert_supported(?slot)", "installed(?block, ?slot)|filled(?slot)|free(?arm)", "held(?arm, ?block)", True),
    "detach": ("free(?arm)|installed(?block, ?slot)|slot_clear(?slot)", "held(?arm, ?block)", "free(?arm)|installed(?block, ?slot)|filled(?slot)", True),
    "support_on": ("free(?arm)|support_point(?slot)", "supporting(?arm, ?slot)", "free(?arm)", False),
    "support_off": ("supporting(?arm, ?slot)", "free(?arm)", "supporting(?arm, ?slot)", True),
    "co_pick": ("arm_pair(?arm1, ?arm2)|free(?arm1)|free(?arm2)|at(?block, ?loc)|two_arm_part(?block)", "coheld(?arm1, ?arm2, ?block)|empty(?loc)", "free(?arm1)|free(?arm2)|at(?block, ?loc)", False),
    "co_insert": ("coheld(?arm1, ?arm2, ?block)|fits(?block, ?slot)|not filled(?slot)|slot_ready(?slot)|insert_supported(?slot)", "installed(?block, ?slot)|filled(?slot)|free(?arm1)|free(?arm2)", "coheld(?arm1, ?arm2, ?block)", True),
}
_IGNORED = frozenset({"empty", "arm_at"})
_PARAMETER_TYPES = {"arm": "arm", "arm1": "arm", "arm2": "arm", "giver": "arm", "taker": "arm", "block": "block", "loc": "loc", "slot": "slot"}


@dataclass(frozen=True)
class _Action:
    positive: frozenset
    negative: frozenset
    derived: tuple
    add: frozenset
    delete: frozenset
    costs: tuple[float, ...]
    moves: tuple
    structural: bool
    occupies: frozenset
    raw_op: float


class PatternDatabase:
    """Return zero when the bounded abstraction cannot establish a lower bound."""

    def __init__(self, engine, *, start=None, goal=None, max_states=30_000, time_limit=20.0, include_positions=True):
        self.engine = engine
        self.enabled = False
        self.states = 0
        self.arms = tuple(engine.problem.objects["arm"])
        self.regions = {}
        self.position_classes = {}
        self.class_positions = defaultdict(list)
        self.travel_cache = {}
        self.include_positions = include_positions
        self.distances = []
        self.state_ids = {}
        self.reason = "不适用"
        if not self._compatible():
            return
        for place in engine.problem.objects["loc"]:
            mask = sum(1 << index for index, arm in enumerate(self.arms)
                       if engine.oracle.check(Fact("reach", (arm, place))).status == Status.VALID)
            self.regions[place] = f"__region_{mask}"
        for place, position in engine.hooks.positions.items():
            region = self.regions.get(place, f"__position_{place}")
            self.position_classes[place] = region
            self.class_positions[region].append(position)
        targets = engine.goal if goal is None else tuple(goal)
        targets = tuple(Condition(target) if isinstance(target, Fact) else target for target in targets)
        self.goal = tuple(self._condition(condition) for condition in targets
                          if condition.fact.predicate not in _IGNORED
                          and not (condition.negated and condition.fact.predicate == "at"))
        initial = engine.initial if start is None else frozenset(start)
        actions = self._actions()
        self._build(self.project(initial), actions, max_states, time_limit)

    def _compatible(self):
        cost = self.engine.cost
        if cost is not None and (type(cost) is not DistanceCost or cost.hooks is not self.engine.hooks):
            return False
        if type(self.engine.hooks) is not BlocksHooks:
            return False
        skills = self.engine.library.skills
        if {skill.name for skill in skills} != set(_EXPECTED) or len(skills) != 9:
            return False
        for skill in skills:
            expected = _EXPECTED[skill.name]
            actual = ("|".join(map(str, skill.pre)), "|".join(map(str, skill.add)),
                      "|".join(map(str, skill.delete)), skill.structural)
            if actual != expected:
                return False
            if any(_PARAMETER_TYPES.get(name) != kind for name, kind in skill.params.items()):
                return False
            arm_parameters = {"?" + name for name, kind in skill.params.items() if kind == "arm"}
            if set(skill.occupies) != arm_parameters:
                return False
        return True

    def _fact(self, fact):
        if fact.predicate == "arm_at" and self.include_positions:
            return Fact("arm_at", (fact.args[0], self.position_classes[fact.args[1]]))
        if fact.predicate == "at" and len(fact.args) == 2 and fact.args[1] in self.regions:
            return Fact("at", (fact.args[0], self.regions[fact.args[1]]))
        return fact

    def _condition(self, condition):
        return Condition(self._fact(condition.fact), condition.negated)

    def project(self, state):
        ignored = {"empty"} if self.include_positions else _IGNORED
        return frozenset(self._fact(fact) for fact in state if fact.predicate not in ignored)

    def _actions(self):
        actions = set()
        for action in self.engine.actions.values():
            if self.engine.oracle.evaluate(action.capability)[0] != Status.VALID:
                continue
            positive = set()
            negative = set()
            derived = []
            for condition in action.pre:
                fact = condition.fact
                if self.engine.is_static(fact.predicate) or fact.predicate in _IGNORED:
                    continue
                if fact.predicate in self.engine.hooks.derived_predicates:
                    derived.append(condition)
                elif condition.negated:
                    negative.add(self._fact(fact))
                else:
                    positive.add(self._fact(fact))
            cost = max(0.0, action.op_time - 1e-9)
            actions.add(_Action(frozenset(positive), frozenset(negative), tuple(derived),
                                self.project(action.add), self.project(action.delete),
                                tuple(cost if arm in action.occupies else 0.0 for arm in self.arms)
                                + (cost if action.structural else 0.0,),
                                tuple(sorted((arm, self.position_classes[place]) for arm, place in action.move_to.items()))
                                if self.include_positions else (), action.structural, frozenset(action.occupies), action.op_time))
        return tuple(actions)

    def _transition(self, action, state):
        successor = (state - action.delete) | action.add
        if not self.include_positions:
            return successor, action.costs
        origins = {fact.args[0]: fact.args[1] for fact in state if fact.predicate == "arm_at"}
        costs = list(action.costs)
        times = []
        for arm, target in action.moves:
            source = origins[arm]
            key = (arm, source, target)
            if key not in self.travel_cache:
                self.travel_cache[key] = min(dist(left, right) for left in self.class_positions[source]
                                             for right in self.class_positions[target]) / self.engine.hooks.speed(arm)
            travel = self.travel_cache[key]
            times.append(travel)
            costs[self.arms.index(arm)] += travel
            successor = frozenset(fact for fact in successor
                                  if not (fact.predicate == "arm_at" and fact.args[0] == arm))
            successor = successor | {Fact("arm_at", (arm, target))}
        if action.structural:
            costs[-1] += max(times, default=0.0)
        return successor, tuple(costs)

    def _build(self, initial, actions, max_states, time_limit):
        began = monotonic()
        frequency = Counter(fact for action in actions for fact in action.positive)
        triggers = defaultdict(list)
        unconditional = []
        for action in actions:
            if action.positive:
                trigger = min(action.positive, key=lambda fact: (frequency[fact], fact))
                triggers[trigger].append(action)
            else:
                unconditional.append(action)
        state_ids = {initial: 0}
        queue = deque([initial])
        reverse = [[]]
        goals = []
        while queue:
            if monotonic() - began >= time_limit:
                self.reason = "抽象构建超时"
                self.states = len(state_ids)
                return
            state = queue.popleft()
            state_id = state_ids[state]
            if self.engine.satisfied(self.goal, state):
                goals.append(state_id)
            candidates = list(unconditional)
            for fact in state:
                candidates.extend(triggers.get(fact, ()))
            for action in candidates:
                if not action.positive <= state or action.negative & state:
                    continue
                if not self.engine.satisfied(action.derived, state):
                    continue
                successor, costs = self._transition(action, state)
                if self.engine.hooks.invariant(successor):
                    continue
                target_id = state_ids.get(successor)
                if target_id is None:
                    if len(state_ids) >= max_states:
                        self.reason = "抽象状态数超限"
                        self.states = len(state_ids)
                        return
                    target_id = len(state_ids)
                    state_ids[successor] = target_id
                    reverse.append([])
                    queue.append(successor)
                reverse[target_id].append((state_id, costs))
        self.states = len(state_ids)
        self.state_ids = state_ids
        for arm_index in range(len(self.arms) + 1):
            distances = [float("inf")] * self.states
            heap = []
            for goal in goals:
                distances[goal] = 0.0
                heappush(heap, (0.0, goal))
            while heap:
                distance, target = heappop(heap)
                if distance != distances[target]:
                    continue
                for source, costs in reverse[target]:
                    candidate = distance + costs[arm_index]
                    if candidate < distances[source]:
                        distances[source] = candidate
                        heappush(heap, (candidate, source))
            self.distances.append(distances)
        self.enabled = True
        self.reason = "已构建"

    def estimate(self, state, running=()):
        if not self.enabled:
            return 0.0
        finished = state
        for key, _ in running:
            finished = self.engine.finish(self.engine.actions[key], finished)
        state_id = self.state_ids.get(self.project(finished))
        if state_id is None:
            return 0.0
        bound = max((distances[state_id] for distances in self.distances), default=0.0)
        if bound == float("inf"):
            return 0.0
        return max(bound, max((remaining for _, remaining in running), default=0.0))

    __call__ = estimate


class EventPatternDatabase(PatternDatabase):
    """A tick event abstraction where ready actions may delay their completion.

    Mapping real event times with floor(time / quantum) retains at least
    floor(duration / quantum) ticks per action. Zero-remaining actions can
    remain busy, and ready subsets may finish together. This preserves the
    original event order and every state invariant after location projection.
    """

    def __init__(self, engine, *, start=None, goal=None, max_states=100_000,
                 time_limit=5.0, quantum=1.0, include_positions=True):
        if not isfinite(quantum) or quantum <= 0:
            raise ValueError("抽象时间步长必须是大于零的有限数值")
        began = monotonic()
        super().__init__(engine, start=start, goal=goal, max_states=min(max_states, 30_000),
                         time_limit=time_limit, include_positions=include_positions)
        self.quantum = quantum
        self.event_enabled = False
        self.event_states = 0
        self.event_reason = "不适用"
        self.event_ids = {}
        self.event_distances = []
        if not self.enabled or len(self.arms) > 2:
            return
        initial = engine.initial if start is None else frozenset(start)
        self._build_events(self.project(initial), self._actions(), max_states,
                           max(0.0, time_limit - (monotonic() - began)))

    def _ticks(self, action, state):
        travel = []
        if action.moves:
            self._transition(action, state)
            origins = {fact.args[0]: fact.args[1] for fact in state if fact.predicate == "arm_at"}
            travel = [self.travel_cache[(arm, origins[arm], target)] for arm, target in action.moves]
        duration = round(action.raw_op + max(travel, default=0.0), 9)
        return floor(duration / self.quantum)

    def _finish_event(self, action, state):
        finished = state | action.add
        for arm, target in action.moves:
            finished = frozenset(fact for fact in finished
                                 if not (fact.predicate == "arm_at" and fact.args[0] == arm))
            finished = finished | {Fact("arm_at", (arm, target))}
        return finished

    def _build_events(self, initial, actions, max_states, time_limit):
        began = monotonic()
        frequency = Counter(fact for action in actions for fact in action.positive)
        triggers = defaultdict(list)
        unconditional = []
        for index, action in enumerate(actions):
            if action.positive:
                trigger = min(action.positive, key=lambda fact: (frequency[fact], fact))
                triggers[trigger].append(index)
            else:
                unconditional.append(index)
        initial_node = (initial, ())
        state_ids = {initial_node: 0}
        queue = deque([initial_node])
        reverse = [[]]
        goals = []

        def edge(source, target, duration):
            target_id = state_ids.get(target)
            if target_id is None:
                if len(state_ids) >= max_states:
                    return False
                target_id = len(state_ids)
                state_ids[target] = target_id
                reverse.append([])
                queue.append(target)
            reverse[target_id].append((source, duration))
            return True

        while queue:
            self.event_states = len(state_ids)
            if monotonic() - began >= time_limit:
                self.event_reason = "事件抽象构建超时"
                return
            state, running = queue.popleft()
            source_id = state_ids[(state, running)]
            if not running and self.engine.satisfied(self.goal, state):
                goals.append(source_id)
            busy = frozenset(arm for index, _ in running for arm in actions[index].occupies)
            structural_busy = any(actions[index].structural for index, _ in running)
            candidates = list(unconditional)
            for fact in state:
                candidates.extend(triggers.get(fact, ()))
            for index in candidates:
                action = actions[index]
                if action.occupies & busy or (action.structural and structural_busy):
                    continue
                if not action.positive <= state or action.negative & state:
                    continue
                if not self.engine.satisfied(action.derived, state):
                    continue
                started = state - action.delete
                if self.engine.hooks.invariant(started):
                    continue
                ticks = self._ticks(action, state)
                if ticks < 1:
                    self.event_reason = "动作时长小于一个抽象时间步"
                    return
                target = (started, tuple(sorted(running + ((index, ticks),))))
                if not edge(source_id, target, 0):
                    self.event_reason = "事件抽象状态数超限"
                    return
            ready = [index for index, remaining in running if remaining == 0]
            for amount in range(1, len(ready) + 1):
                for chosen in combinations(ready, amount):
                    finished = state
                    for index in chosen:
                        finished = self._finish_event(actions[index], finished)
                    if self.engine.hooks.invariant(finished):
                        continue
                    rest = tuple((index, remaining) for index, remaining in running if index not in chosen)
                    if not edge(source_id, (finished, rest), 0):
                        self.event_reason = "事件抽象状态数超限"
                        return
            if any(remaining > 0 for _, remaining in running):
                advanced = tuple((index, max(0, remaining - 1)) for index, remaining in running)
                if not edge(source_id, (state, advanced), 1):
                    self.event_reason = "事件抽象状态数超限"
                    return
        self.event_states = len(state_ids)
        distances = [float("inf")] * self.event_states
        heap = []
        for goal in goals:
            distances[goal] = 0
            heappush(heap, (0, goal))
        while heap:
            if monotonic() - began >= time_limit:
                self.event_reason = "事件抽象反向搜索超时"
                return
            distance, target = heappop(heap)
            if distance != distances[target]:
                continue
            for source, cost in reverse[target]:
                candidate = distance + cost
                if candidate < distances[source]:
                    distances[source] = candidate
                    heappush(heap, (candidate, source))
        self.event_ids = state_ids
        self.event_distances = distances
        self.event_enabled = True
        self.event_reason = "已构建"

    def estimate(self, state, running=()):
        work_bound = super().estimate(state, running)
        if not self.event_enabled:
            return work_bound
        finished = state
        for key, _ in running:
            finished = self.engine.finish(self.engine.actions[key], finished)
        node_id = self.event_ids.get((self.project(finished), ()))
        if node_id is None or self.event_distances[node_id] == float("inf"):
            return work_bound
        event_bound = max(0.0, self.event_distances[node_id] * self.quantum - 1e-9)
        return max(work_bound, event_bound, max((remaining for _, remaining in running), default=0.0))

    __call__ = estimate
