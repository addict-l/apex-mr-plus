"""Diagnose where greedy h_add gets stuck: track the best h seen and the unachieved goals there."""
import os, sys, time, json, runpy
os.chdir(r"D:\APEX_MR_Plus\dualarm_skillnet"); sys.path.insert(0, os.getcwd())
from math import inf
SP = os.path.dirname(os.path.abspath(__file__))
saved = sys.argv; sys.argv = ["hadd.py", "hadd", "0"]
ns = runpy.run_path(os.path.join(SP, "hadd.py")); sys.argv = saved
Relax = ns["Relax"]
from skillnet.domains.blocks import load_problem
from skillnet.core.engine import Engine
from skillnet.planning import heuristic as H, search as S

limit = float(sys.argv[1])
for task in sys.argv[2:]:
    problem = load_problem(task, "skills/blocks_v1.toml"); engine = Engine(problem)
    track = {"best": inf, "calls": 0, "best_state": None, "history": []}

    def build(engine, *, start=None, goal=None, time_limit=8.0):
        fv = H._operation_floor_view(engine)
        relax = Relax(fv, goal=goal, coarse=True, cache_limit=0)
        goals = relax.positive_goals
        def est(state, running=()):
            value = relax.hadd(state, running)
            track["calls"] += 1
            if value < track["best"]:
                track["best"] = value; track["best_state"] = (frozenset(state), tuple(running))
                track["history"].append((track["calls"], round(value, 1), sum(1 for g in goals if g not in state)))
            return value
        track["goals"] = goals
        return est
    H.build_heuristic = build
    t = time.monotonic(); res = S.search(engine, max_expansions=200_000, time_limit=limit); el = time.monotonic() - t
    state, running = track["best_state"]
    unachieved = [str(g) for g in track["goals"] if g not in state]
    holding = sorted(str(f) for f in state if f.predicate in ("holding", "arm_at", "free", "installed"))
    print(json.dumps({"task": os.path.basename(task)[:22], "status": res.status, "expanded": res.expanded, "seconds": round(el, 1),
        "best_h": round(track["best"], 1), "unachieved_goals_at_best": unachieved, "running_at_best": [str(r) for r in running],
        "arm_facts_at_best": holding, "history_tail": track["history"][-12:],
        "steps": [str(s.action) for s in (res.steps or [])][:40]}, ensure_ascii=False), flush=True)
