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
from skillnet.planning.abstraction import PatternDatabase, EventPatternDatabase


class FastHMax(HMaxBound):
    """Precondition-counter hmax: each rule fires once, when its last prerequisite pops."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pre_dependents = defaultdict(list); self.extra_dependents = defaultdict(list)
        self.pre_count = []
        for index, (prerequisites, _, _) in enumerate(self.actions):
            self.pre_count.append(len(prerequisites))
            for fact in prerequisites: self.pre_dependents[fact].append(index)
        pre_sets = [set(p) for p, _, _ in self.actions]
        for fact, indices in self.action_dependents.items():
            seen = set()
            for index in indices:
                if fact not in pre_sets[index] and index not in seen:
                    seen.add(index); self.extra_dependents[fact].append(index)

    def __call__(self, state, running=()):
        state = frozenset(state); running = tuple(running)
        remaining_bound = max((remaining for _, remaining in running), default=0.0)
        future_adds, future_deletes = {}, {}
        for action_key, remaining in running:
            action = self.engine.actions[action_key]
            for fact in self.negative_goals:
                if fact in self.effects[action_key]:
                    future_adds[fact] = max(future_adds.get(fact, 0.0), remaining)
                if (fact.predicate == "arm_at" and len(fact.args) == 2 and fact.args[0] in action.move_to
                        and action.move_to[fact.args[0]] != fact.args[1]):
                    future_deletes[fact] = min(future_deletes.get(fact, inf), remaining)
        future_cleanup = max((remaining + self.negative_tail[fact] for fact, remaining in future_adds.items()), default=0.0)
        running_key = tuple(sorted(k for k, _ in running)) if self.coarse else running
        key = (state, running_key)
        if key in self.cache: return max(self.cache[key], remaining_bound, future_cleanup)
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
            for effect in self.effects[action_key]: offer(effect, 0.0 if self.coarse else remaining)
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
        bound = max((times.get(fact, inf) for fact in self.positive_goals), default=0.0)
        for fact in self.negative_goals:
            if fact not in state and fact not in self.engine.static and fact not in future_adds: continue
            released = 0.0 if self.coarse else future_adds.get(fact, 0.0)
            cts = []
            if fact in future_deletes: cts.append(0.0 if self.coarse else future_deletes[fact])
            for prerequisites, duration, restoration, at_end in self.deletors[fact]:
                ready = max((times.get(pre, inf) for pre in prerequisites), default=0.0)
                completion = max(ready + duration, released) if at_end else max(ready, released) + duration
                cts.append(completion + restoration)
            bound = max(bound, min(cts, default=inf))
        if self.cache_limit > 0:
            if len(self.cache) >= self.cache_limit: self.cache.clear()
            self.cache[key] = bound
        return max(bound, remaining_bound, future_cleanup)


def make_db(engine, fv, start, goal):
    pv = engine
    if fv is not engine and type(engine.hooks) is H.BlocksHooks:
        pv = H.copy(fv); pv.hooks = engine.hooks
    db = PatternDatabase(pv, start=start, goal=goal, time_limit=3.0)
    if db.enabled and db.states <= 1000:
        db = EventPatternDatabase(pv, start=start, goal=goal, quantum=0.5, max_states=200_000, time_limit=5.0)
    return db


mode, limit = sys.argv[1], float(sys.argv[2])
ORIG_BUILD = H.build_heuristic
for task in sys.argv[3:]:
    problem = load_problem(task, "skills/blocks_v1.toml"); engine = Engine(problem)
    samples = []; holder = {}
    orig_build = ORIG_BUILD

    def build(engine, *, start=None, goal=None, time_limit=8.0):
        fv = H._operation_floor_view(engine)
        if mode == "record":
            est = orig_build(engine, start=start, goal=goal, time_limit=time_limit)
            holder["slow"] = HMaxBound(fv, goal=goal, coarse=True); holder["fast"] = FastHMax(fv, goal=goal, coarse=True)
            def wrapped(state, running=()):
                if len(samples) < 300: samples.append((frozenset(state), tuple(running)))
                return est(state, running)
            return wrapped
        db = make_db(engine, fv, start, goal)
        if mode == "pdb":
            return lambda state, running=(): db(state, running)
        fast = FastHMax(fv, goal=goal, coarse=True)
        if mode == "fasthmax":
            return lambda state, running=(): fast(state, running)
        return lambda state, running=(): max(fast(state, running), db(state, running))
    H.build_heuristic = build
    t = time.monotonic(); res = S.search(engine, max_expansions=200_000, time_limit=limit); el = time.monotonic() - t
    out = {"mode": mode, "task": os.path.basename(task)[:22], "status": res.status, "makespan": round(res.makespan or 0, 1),
           "bound": res.suboptimality_bound, "expanded": res.expanded, "seconds": round(el, 1)}
    if mode == "record":
        slow, fast = holder["slow"], holder["fast"]
        slow.cache_limit = 0; fast.cache_limit = 0; slow.cache.clear(); fast.cache.clear()
        t = time.monotonic(); a = [slow(s, r) for s, r in samples]; ts = time.monotonic() - t
        t = time.monotonic(); b = [fast(s, r) for s, r in samples]; tf = time.monotonic() - t
        diffs = [abs(x - y) for x, y in zip(a, b) if not (x == y)]
        out.update(samples=len(samples), slow_ms=round(1000*ts/len(samples), 2), fast_ms=round(1000*tf/len(samples), 2),
                   mismatches=len(diffs), max_diff=max(diffs, default=0.0))
    print(json.dumps(out), flush=True)
