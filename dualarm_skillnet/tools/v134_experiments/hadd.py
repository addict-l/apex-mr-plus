"""Experiment: additive (non-admissible) relaxed heuristic for the first-solution stage."""
import os, sys, time, json
os.chdir(r"D:\APEX_MR_Plus\dualarm_skillnet"); sys.path.insert(0, os.getcwd())
from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from math import inf
from skillnet.domains.blocks import load_problem
from skillnet.core.engine import Engine
from skillnet.planning import heuristic as H, search as S
from skillnet.planning.bounds import HMaxBound


class Relax(HMaxBound):
    """Counter-based delete relaxation exposing the fact times (coarse, running ignored for sum)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pre_dependents = defaultdict(list); self.extra_dependents = defaultdict(list); self.pre_count = []
        for index, (prerequisites, _, _) in enumerate(self.actions):
            self.pre_count.append(len(prerequisites))
            for fact in prerequisites: self.pre_dependents[fact].append(index)
        pre_sets = [set(p) for p, _, _ in self.actions]
        for fact, indices in self.action_dependents.items():
            seen = set()
            for index in indices:
                if fact not in pre_sets[index] and index not in seen:
                    seen.add(index); self.extra_dependents[fact].append(index)

    def times(self, state, running=()):
        state = frozenset(state)
        times = {}; queue = []; order = count()
        counts = list(self.pre_count); fired = [None]*len(counts)
        effect_time, rule_actions, actions = self.effect_time, self.rule_actions, self.actions
        def offer(fact, t):
            if t < times.get(fact, inf):
                times[fact] = t; heappush(queue, (t, next(order), fact))
        def fire(index, earliest):
            _, effects, duration = actions[index]
            completion = earliest + duration
            for effect in effects:
                offer(effect, effect_time(effect, completion, times, rule_actions[index]) if effect_time else completion)
        def update_derived(fact):
            offer(fact, self.optimistic(fact, times) if self.optimistic else 0.0)
        for fact in state | self.engine.static: offer(fact, 0.0)
        for action_key, remaining in running:
            for effect in self.effects[action_key]: offer(effect, 0.0)
        for fact in self.derived:
            if self.engine.hooks.evaluate(fact.predicate, fact.args, state): offer(fact, 0.0)
            update_derived(fact)
        for index, c in enumerate(counts):
            if c == 0: fired[index] = 0.0; fire(index, 0.0)
        while queue:
            t, _, fact = heappop(queue)
            if times[fact] != t: continue
            for index in self.pre_dependents.get(fact, ()):
                counts[index] -= 1
                if counts[index] == 0: fired[index] = t; fire(index, t)
            for index in self.extra_dependents.get(fact, ()):
                if fired[index] is not None: fire(index, fired[index])
            for derived in self.derived_dependents.get(fact, ()): update_derived(derived)
        return times

    def hadd(self, state, running=()):
        times = self.times(state, running)
        total = 0.0
        for fact in self.positive_goals:
            t = times.get(fact, inf)
            if t == inf: return inf
            total += t
        return total


mode, limit = sys.argv[1], float(sys.argv[2])
for task in sys.argv[3:]:
    problem = load_problem(task, "skills/blocks_v1.toml"); engine = Engine(problem)

    def build(engine, *, start=None, goal=None, time_limit=8.0):
        fv = H._operation_floor_view(engine)
        relax = Relax(fv, goal=goal, coarse=True, cache_limit=0)
        if mode == "hadd":
            return relax.hadd
        if mode == "goalcount":
            goals = relax.positive_goals
            return lambda state, running=(): 10.0*sum(1 for g in goals if g not in state)
        raise SystemExit("mode?")
    H.build_heuristic = build
    t = time.monotonic(); res = S.search(engine, max_expansions=200_000, time_limit=limit); el = time.monotonic() - t
    print(json.dumps({"mode": mode, "task": os.path.basename(task)[:22], "status": res.status, "makespan": round(res.makespan or 0, 1),
        "bound": res.suboptimality_bound, "expanded": res.expanded, "seconds": round(el, 1),
        "stages": [(s["phase"], s["weight"], s["status"], s["expanded"], round(s["elapsed"], 1)) for s in (getattr(res, "stages", None) or [])]}), flush=True)
