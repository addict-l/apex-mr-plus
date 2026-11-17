"""Experiment: LIFO tie-breaking among equal-f nodes (dive on plateaus), default heuristic."""
import os, sys, time, json
os.chdir(r"D:\APEX_MR_Plus\dualarm_skillnet"); sys.path.insert(0, os.getcwd())
from heapq import heappush as real_heappush
from skillnet.domains.blocks import load_problem
from skillnet.core.engine import Engine
from skillnet.planning import search as S


def lifo_heappush(heap, item):
    if len(item) == 4 and isinstance(item[2], int):
        item = (item[0], item[1], -item[2], item[3])
    real_heappush(heap, item)

mode, limit = sys.argv[1], float(sys.argv[2])
if mode == "lifo":
    S.heappush = lifo_heappush
for task in sys.argv[3:]:
    problem = load_problem(task, "skills/blocks_v1.toml"); engine = Engine(problem)
    t = time.monotonic(); res = S.search(engine, max_expansions=200_000, time_limit=limit); el = time.monotonic() - t
    print(json.dumps({"mode": mode, "task": os.path.basename(task)[:22], "status": res.status, "makespan": round(res.makespan or 0, 1),
        "bound": res.suboptimality_bound, "expanded": res.expanded, "seconds": round(el, 1),
        "stages": [(s["phase"], s["weight"], s["status"], s["expanded"], round(s["elapsed"], 1)) for s in (getattr(res, "stages", None) or [])]}), flush=True)
