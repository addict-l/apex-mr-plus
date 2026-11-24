"""Generic grounding and start/end state transitions."""
from collections import Counter, defaultdict
from itertools import product
from .logic import Fact, Condition, parse_fact


class Engine:
    def __init__(self, problem, cost=None):
        self.problem = problem
        self.library = problem.library
        self.hooks = problem.hooks
        self.oracle = problem.oracle
        self.static = problem.static
        self.initial = problem.initial
        self.goal = problem.goal
        self.dynamic_predicates = self.library.dynamic_predicates | {"arm_at"}
        conflict = self.dynamic_predicates & self.hooks.derived_predicates
        if conflict:
            raise ValueError("谓词不能同时是动态和推导条件：" + ", ".join(sorted(conflict)))
        self.cost = cost
        self.actions = {}
        for skill in self.library.skills:
            types = tuple(skill.params.values())
            for args in product(*(problem.objects[t] for t in types)):
                if any(types[i] == types[j] and args[i] == args[j]
                       for i in range(len(args)) for j in range(i)):
                    continue
                action = skill.ground(args)
                if all(self.holds(c, self.initial) for c in action.pre
                       if self.is_static(c.fact.predicate)):
                    self.actions[action.key] = action
        counts = Counter(c.fact for a in self.actions.values() for c in a.pre
                         if not c.negated and c.fact.predicate in self.dynamic_predicates)
        self.trigger_index = defaultdict(list)
        self.untriggered = []
        for a in self.actions.values():
            triggers = [c.fact for c in a.pre if not c.negated
                        and c.fact.predicate in self.dynamic_predicates]
            if triggers:
                self.trigger_index[min(triggers, key=lambda f: (counts[f], str(f)))].append(a)
            else:
                self.untriggered.append(a)

    def is_static(self, predicate):
        return predicate not in self.dynamic_predicates and predicate not in self.hooks.derived_predicates

    def holds(self, condition, state):
        if isinstance(condition, Fact):
            condition = Condition(condition)
        fact = condition.fact
        if fact.predicate in self.hooks.derived_predicates:
            result = self.hooks.evaluate(fact.predicate, fact.args, state)
        elif self.is_static(fact.predicate):
            result = fact in self.static
        else:
            result = fact in state
        return not result if condition.negated else result

    def satisfied(self, conditions, state):
        return all(self.holds(c, state) for c in conditions)

    def failed(self, action, state):
        return [c for c in action.pre if not self.holds(c, state)]

    def candidates(self, state):
        yield from self.untriggered
        for fact in sorted(state):
            yield from self.trigger_index.get(fact, ())

    def parse_action(self, text):
        fact = parse_fact(text)
        skill = self.library.by_name.get(fact.predicate)
        if skill is None:
            raise ValueError(f"未知技能：{fact.predicate}")
        if len(fact.args) != len(skill.params):
            raise ValueError(f"{fact.predicate} 参数个数应为 {len(skill.params)}")
        for arg, kind in zip(fact.args, skill.params.values()):
            if arg not in self.problem.objects[kind]:
                raise ValueError(f"对象 {arg} 不存在或类型不是 {kind}")
        action = self.actions.get((fact.predicate, fact.args))
        if action is None:
            raise ValueError(f"{text} 不满足静态条件或同类型参数必须不同")
        return action

    def start(self, action, state):
        return state.difference(action.delete)

    def finish(self, action, state):
        result = set(state)
        result.update(action.add)
        for arm, place in action.move_to.items():
            result = {f for f in result if not (f.predicate == "arm_at" and f.args[0] == arm)}
            result.add(Fact("arm_at", (arm, place)))
        return frozenset(result)

    def execute(self, action, state):
        started = self.start(action, state)
        violation = self.hooks.invariant(started)
        if violation:
            raise ValueError(violation)
        ended = self.finish(action, started)
        violation = self.hooks.invariant(ended)
        if violation:
            raise ValueError(violation)
        return ended
