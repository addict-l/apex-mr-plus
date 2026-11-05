"""Run: python -m recovery --case support_swap, or --suite --policy all."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys

import numpy
import scipy

from .model import Problem, apply_event
from .oracles import EvidenceCache
from .planner import POLICIES, affected_tasks, plan_recovery
from .scenarios import cases
from .schedule import action_schedule, build_schedule, validate_schedule


def run_case(before: Problem, *, policy="recovery", reuse=True, max_expansions=20000, time_limit_s=15.) -> dict:
    cache = EvidenceCache()
    options = {"policy": policy, "max_expansions": max_expansions, "time_limit_s": time_limit_s}
    # Warm evidence using a nominal PLAN from the SAME snapshot. It has not
    # been executed, and applying an event must never reset the snapshot.
    old_plan = plan_recovery(before, cache=cache, **options)
    after, changed = apply_event(before, before.data.get("event"))
    invalidated = cache.invalidate(changed)
    if not reuse:
        cache = EvidenceCache()
    result = plan_recovery(after, cache=cache, **options)
    result["event"] = deepcopy(before.data.get("event"))
    result["repair"] = {**affected_tasks(before, changed, old_plan), "changed_facts": changed,
                        "invalidated_evidence": invalidated, "reuse_evidence": reuse,
                        "nominal_plan_status": old_plan["status"]}
    graph = build_schedule(after, result)
    rows = action_schedule(result, graph)
    audit = validate_schedule(after, rows, policy=policy, expected_final=result["final_state"])
    result["schedule"] = rows
    result["validation"] = audit
    result["tpg"] = graph
    result["predicted_makespan_s"] = audit["predicted_makespan_s"]
    return result


def summary(result: dict) -> dict:
    return {"case": result["case"], "policy": result["policy"], "status": result["status"],
            "abstract_audit_passed": result["validation"]["valid"],
            "completed": result["final_state"]["completed"],
            "predicted_makespan_s": result["predicted_makespan_s"],
            "action_count": len(result["actions"]), "expanded": result["search"]["expanded"],
            "wall_time_s": result["search"].get("wall_time_s"),
            "evidence_queries": result.get("evidence_queries", {}),
            "search_budget_exhausted": result["search"].get("budget_exhausted", False)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=list(cases()), default=None)
    group.add_argument("--scenario", type=Path, help="complete JSON scenario with optional event updates")
    group.add_argument("--suite", action="store_true", help="run all eight synthetic cases")
    parser.add_argument("--policy", choices=(*POLICIES, "all"), default="recovery")
    parser.add_argument("--output", type=Path, default=Path("results/recovery"))
    parser.add_argument("--no-reuse", action="store_true", help="discard all nominal evidence after the event")
    parser.add_argument("--max-expansions", type=int, default=20000)
    parser.add_argument("--time-limit-s", type=float, default=15.)
    parser.add_argument("--export-scenario", type=Path, help="write selected complete scenario JSON")
    args = parser.parse_args(argv)
    if args.export_scenario and args.suite:
        parser.error("--export-scenario needs one case or --scenario")
    selected = ([json.loads(args.scenario.read_text(encoding="utf-8"))] if args.scenario else
                list(cases().values()) if args.suite else [cases()[args.case or "support_swap"]])
    if args.export_scenario:
        args.export_scenario.parent.mkdir(parents=True, exist_ok=True)
        args.export_scenario.write_text(json.dumps(selected[0], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    records, valid = [], True
    for data in selected:
        for policy in POLICIES if args.policy == "all" else [args.policy]:
            result = run_case(Problem(data), policy=policy, reuse=not args.no_reuse,
                              max_expansions=args.max_expansions, time_limit_s=args.time_limit_s)
            output = args.output/result["case"]/policy
            output.mkdir(parents=True, exist_ok=True)
            for name, value in (("result", {k: v for k, v in result.items() if k != "tpg"}),
                                ("tpg", result["tpg"]), ("scenario", data)):
                (output/f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                                                  encoding="utf-8")
            record = summary(result)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            valid = valid and result["validation"]["valid"]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": "synthetic offline discrete recovery, not CoppeliaSim execution",
              "python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
              "platform": platform.platform(), "evidence_reuse": not args.no_reuse, "cases": records}
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # Expected no-plan/unknown/partial cases are legitimate outcomes. Exit 2
    # indicates an audit failure, not inability to solve an impossible case.
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
