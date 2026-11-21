"""Experiment: h_add plus an allocation-feasibility check (bipartite matching blocks -> unfilled slots)."""
import os, sys, time, json, runpy
os.chdir(r"D:\APEX_MR_Plus\dualarm_skillnet"); sys.path.insert(0, os.getcwd())
from math import inf
SP = os.path.dirname(os.path.abspath(__file__))
saved = sys.argv; sys.argv = ["hadd.py", "hadd", "0"]
ns = runpy.run_path(os.path.join(SP, "hadd.py")); sys.argv = saved
Relax = ns["Relax"]
from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from skillnet.core.engine import Engine
from skillnet.planning import heuristic as H, search as S


class Allocation:
    """Max matching between free/held blocks and unfilled goal slots under VALID reach."""
    def __init__(self, engine, goals):
        self.engine = engine
        self.goal_slots = [g.args[0] for g in goals if g.predicate == "filled" and len(g.args) == 1]
        hooks = engine.hooks
        self.arms = tuple(hooks.arms)
        self.handover_points = [place for place, row in hooks.locations.items() if row.get("kind") == "handover"]
        self.valid = {}
        for arm in self.arms:
            for place in list(hooks.locations) + list(hooks.slots):
                self.valid[(arm, place)] = engine.oracle.check(Fact("reach", (arm, place))).status == Status.VALID
        self.fits = {(f.args[0], f.args[1]) for f in engine.static if f.predicate == "fits"}
        self.cross = any(self.valid[(a, hp)] and self.valid[(b, hp)] for hp in self.handover_points
                         for a in self.arms for b in self.arms if a != b)
        self.calls = 0; self.pruned = 0

    def arms_holding_or_reaching(self, state):
        where = {}
        for fact in state:
            if fact.predicate == "at" and len(fact.args) == 2:
                where[fact.args[0]] = ("at", fact.args[1])
            elif fact.predicate == "held" and len(fact.args) == 2:
                where[fact.args[1]] = ("held", fact.args[0])
        return where

    def deficiency(self, state, running=(), effects=None):
        self.calls += 1
        if running and effects is not None:
            state = set(state)
            for action_key, _ in running:
                state.update(effects[action_key])
        unfilled = [s for s in self.goal_slots if Fact("filled", (s,)) not in state]
        if not unfilled: return 0
        where = self.arms_holding_or_reaching(state)
        # arms that can bring block b somewhere: directly reachable arms, or any arm if a handover is possible
        def arms_for(block):
            kind, place = where[block]
            direct = {place} if kind == "held" else {arm for arm in self.arms if self.valid[(arm, place)]}
            if self.cross and direct: return set(self.arms)
            return direct
        edges = {}
        for block in where:
            arms = arms_for(block)
            edges[block] = [s for s in unfilled if (block, s) in self.fits and any(self.valid[(arm, s)] for arm in arms)]
        match = {}
        def augment(block, seen):
            for s in edges[block]:
                if s in seen: continue
                seen.add(s)
                if s not in match or augment(match[s], seen):
                    match[s] = block; return True
            return False
        size = sum(1 for block in edges if augment(block, set()))
        return len(unfilled) - size


mode, limit = sys.argv[1], float(sys.argv[2])
for task in sys.argv[3:]:
    problem = load_problem(task, "skills/blocks_v1.toml"); engine = Engine(problem)
    holder = {}

    def build(engine, *, start=None, goal=None, time_limit=8.0):
        fv = H._operation_floor_view(engine)
        relax = Relax(fv, goal=goal, coarse=True, cache_limit=0)
        alloc = Allocation(engine, relax.positive_goals); holder["alloc"] = alloc
        def est(state, running=()):
            if alloc.deficiency(state, running, relax.effects) > 0:
                alloc.pruned += 1
                return inf
            return relax.hadd(state, running)
        return est
    H.build_heuristic = build
    t = time.monotonic(); res = S.search(engine, max_expansions=200_000, time_limit=limit); el = time.monotonic() - t
    alloc = holder.get("alloc")
    print(json.dumps({"mode": mode, "task": os.path.basename(task)[:22], "status": res.status, "makespan": round(res.makespan or 0, 1),
        "expanded": res.expanded, "seconds": round(el, 1), "alloc_calls": alloc.calls if alloc else None, "alloc_pruned": alloc.pruned if alloc else None,
        "cross_handover": alloc.cross if alloc else None,
        "stages": [(s["phase"], s["weight"], s["status"], s["expanded"], round(s["elapsed"], 1)) for s in (getattr(res, "stages", None) or [])],
        "steps": [str(s.action) for s in (res.steps or [])][:30]}, ensure_ascii=False), flush=True)
