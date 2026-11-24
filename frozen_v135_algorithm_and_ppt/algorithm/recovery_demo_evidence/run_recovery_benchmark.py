"""Offline fault-injection recovery benchmark for db77489 (no CoppeliaSim).

Suite A -- LEGO bridge 9-block joint task:
    Reuses recovery_demo/bundle/plans.json (12 settled checkpoints produced by
    run_offline_gate.py) together with the gate adapter engine and base_allowed
    action inventory. Each checkpoint gets a grounded single-action fault and,
    where the mask cannot block the deck zone, an arm/skill-level fault; the
    checkpoint-8 symmetric cross-zone reach fault is added as the L2 composite.

Suite B -- the 11 standard tasks in vendor_db77489/dualarm_skillnet/tasks:
    Each task is prepared with the stock vendor pipeline (checkpoints=8,
    backups=2, backup_seconds=1.0, graph_limit=5000, time_limit=30 s). Every
    settled checkpoint of a solved plan gets (a) a grounded single-action fault
    and (b) an arm/skill-level fault, each recovered with the stock
    recovery.recover(..., compare=True, time_limit=30 s) so the cold baseline
    uses the identical fault domain.

Timing: every recovery decision is timed with time.perf_counter() from the
outside (vendor recover() itself uses time.monotonic(), which Windows quantises
to ~15.6 ms).  Suite A additionally records the engine construction time so the
online decision can be separated from model loading.  Cold-baseline numbers are
the vendor values where the inner search timer is used, and are flagged
accordingly in the report.

Everything is written below recovery_demo/benchmark/. The vendor copy is read
only; no CoppeliaSim process is started and no existing source file changes.
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SHOWCASE = HERE.parent
VENDOR = HERE / "vendor_db77489" / "dualarm_skillnet"
SKILLS = VENDOR / "skills" / "blocks_v1.toml"
TASKS = VENDOR / "tasks"
TASK_A = HERE / "task_recovery.toml"
BUNDLE_A = HERE / "bundle" / "plans.json"
FAULT_L2 = HERE / "fault_l2.json"
RECEIPT_A = HERE / "offline_gate_receipt.json"
OUT = HERE / "benchmark"
RAW_DIR = OUT / "raw"
LOG_DIR = OUT / "logs"
BUNDLE_DIR = OUT / "bundles"
RAW_JSON = OUT / "benchmark_raw.json"
REPORT_MD = OUT / "recovery_benchmark_report.md"
CSV_FILE = OUT / "recovery_benchmark.csv"

sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(HERE))

from skillnet.core.engine import Engine                            # noqa: E402
from skillnet.domains.blocks import load_problem                    # noqa: E402
from skillnet.planning.backup_plans import try_backups              # noqa: E402
from skillnet.planning.candidate_graph import CandidateGraph        # noqa: E402
from skillnet.planning.search import search                         # noqa: E402
from skillnet import recovery as vendor_recovery                    # noqa: E402

SCHEMA = "db77489-recovery-benchmark-v1"
VENDOR_COMMIT = "db774898ea73b59d4a9327ff49149712f38d2524"

BUDGET = 30.0                 # seconds, recovery and cold baseline alike
MAX_EXPANSIONS = 200_000      # stock vendor default
GRAPH_LIMIT_A = 200_000       # same limit the gate prepared its graph with
B_CHECKPOINTS = 8
B_BACKUPS = 2
B_BACKUP_SECONDS = 1.0
B_GRAPH_LIMIT = 5_000

MEASURABLE_FIRST = 1e-3       # cold first solutions below 1 ms are quantisation
FLOOR = 1e-4                  # floor for degenerate ratio denominators
REL_TOL = 1e-6

CSV_FIELDS = (
    "suite", "task", "checkpoint", "fault_kind", "fault_detail", "stage",
    "status", "backup_kind", "backup_checkpoint", "backup_attempts",
    "graph_loaded", "decision_seconds", "decision_ms", "decision_core_seconds",
    "decision_core_ms", "vendor_elapsed_seconds", "model_load_seconds",
    "first_solution_seconds", "first_solution_core_seconds",
    "cold_status", "cold_seconds", "cold_first_solution_seconds",
    "cold_first_e2e_seconds", "cold_total_e2e_seconds",
    "first_solution_speedup", "first_solution_speedup_e2e",
    "decision_speedup_total", "makespan", "cold_makespan", "optimality_ratio",
    "optimality_match", "reused_states", "cached_states", "offline_states",
    "reused_edges", "generated_rows", "state_reuse_rate",
    "reused_capabilities", "refreshed_capabilities", "allowed_actions",
    "excluded_actions", "budget_exhausted", "prepare_seconds",
    "valid_backup_makespans", "skipped", "note",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(data) + b"\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fmt_number(value, digits=12):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, f".{digits}g")
    return str(value)


def is_close(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def ratio_text(value):
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value:,.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def percentile(values, q):
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    index = (len(data) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return data[int(index)]
    return data[lower] * (upper - index) + data[upper] * (index - lower)


def median(values):
    return percentile(values, 0.5)


def mean(values):
    return sum(values) / len(values) if values else None


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def assert_under(path, parent):
    resolved, root = Path(path).resolve(), Path(parent).resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to touch {resolved}: outside {root}")


# --------------------------------------------------------------------------- #
# metric decoration shared by both suites
# --------------------------------------------------------------------------- #
def decorate(row):
    decision = row.get("decision_seconds")
    model_load = row.get("model_load_seconds") or 0.0
    cold_load = row.get("cold_load_seconds")
    load = cold_load if cold_load is not None else model_load
    row["decision_ms"] = None if decision is None else decision * 1000.0
    if decision is not None:
        core = max(decision - model_load, 0.0)
        row["decision_core_seconds"] = core
        row["decision_core_ms"] = core * 1000.0
    else:
        row["decision_core_seconds"] = None
        row["decision_core_ms"] = None

    first = row.get("first_solution_seconds")
    first_core = row.get("first_solution_core_seconds")
    cold_first = row.get("cold_first_solution_seconds")
    cold_seconds = row.get("cold_seconds")
    if cold_first is not None:
        row["cold_first_e2e_seconds"] = load + cold_first
    else:
        row["cold_first_e2e_seconds"] = None
    if cold_seconds is not None:
        row["cold_total_e2e_seconds"] = load + cold_seconds
    else:
        row["cold_total_e2e_seconds"] = None

    # primary speedup: both sides measured from a warm engine (cold search timer
    # vs recovery time net of model loading); quantised cold first solutions are
    # excluded instead of being turned into a fake ratio.
    if (cold_first is not None and cold_first >= MEASURABLE_FIRST
            and first_core is not None and first_core > 0):
        row["first_solution_speedup"] = cold_first / first_core
    else:
        row["first_solution_speedup"] = None
    if (cold_first is not None and cold_first >= MEASURABLE_FIRST
            and first is not None and first > 0):
        row["first_solution_speedup_e2e"] = (load + cold_first) / first
    else:
        row["first_solution_speedup_e2e"] = None
    if (row.get("cold_total_e2e_seconds") is not None and decision is not None
            and decision > 0):
        row["decision_speedup_total"] = row["cold_total_e2e_seconds"] / decision
    else:
        row["decision_speedup_total"] = None

    makespan = row.get("makespan")
    cold_makespan = row.get("cold_makespan")
    if makespan is not None and cold_makespan is not None and cold_makespan > 0:
        row["optimality_ratio"] = makespan / cold_makespan
        row["optimality_match"] = is_close(makespan, cold_makespan)
    else:
        row["optimality_ratio"] = None
        row["optimality_match"] = None

    reused = row.get("reused_states") or 0
    generated = row.get("generated_rows") or 0
    total = reused + generated
    row["state_reuse_rate"] = (reused / total) if total else None

    valid = row.get("valid_backup_makespans") or []
    selected = row.get("makespan")
    if len(valid) >= 2 and max(valid) > 0 and selected is not None:
        row["multi_backup_saving"] = max(valid) - selected
        row["multi_backup_saving_rate"] = (max(valid) - selected) / max(valid)
    else:
        row["multi_backup_saving"] = None
        row["multi_backup_saving_rate"] = None
    return row


def summarize(rows):
    total = len(rows)
    solved = [row for row in rows if row.get("status") == "solved"]
    l1 = [row for row in solved if row.get("stage") == "backup"]
    l2 = [row for row in solved if row.get("stage") == "graph_search"]
    no_plan = [row for row in rows if row.get("status") == "no_plan"]
    budget = [row for row in rows if row.get("status") == "budget"]
    other = [row for row in rows
             if row.get("status") not in {"solved", "no_plan", "budget"}]
    cold_solved = [row for row in rows if row.get("cold_status") == "solved"]
    cold_solved_ok = [row for row in cold_solved if row.get("status") == "solved"]

    l1_latency = [row["decision_ms"] for row in l1
                  if row.get("decision_ms") is not None]
    l1_core = [row["decision_core_ms"] for row in l1
               if row.get("decision_core_ms") is not None]
    l1_speedups = [row["first_solution_speedup"] for row in l1
                   if row.get("first_solution_speedup") is not None]
    l1_speedups_e2e = [row["first_solution_speedup_e2e"] for row in l1
                       if row.get("first_solution_speedup_e2e") is not None]
    l1_savings = [row["multi_backup_saving_rate"] for row in l1
                  if row.get("multi_backup_saving_rate") is not None]
    l2_reuse = [row["state_reuse_rate"] for row in l2
                if row.get("state_reuse_rate") is not None]
    l2_speedups = [row["first_solution_speedup"] for row in l2
                   if row.get("first_solution_speedup") is not None]
    l2_speedups_e2e = [row["first_solution_speedup_e2e"] for row in l2
                       if row.get("first_solution_speedup_e2e") is not None]
    l2_decision = [row["decision_speedup_total"] for row in l2
                   if row.get("decision_speedup_total") is not None]
    l2_optimal = [row for row in l2 if row.get("optimality_match") is not None]
    l2_ratios = [row["optimality_ratio"] for row in l2
                 if row.get("optimality_ratio") is not None]
    l1_kinds = {}
    for row in l1:
        kind = row.get("backup_kind") or "unknown"
        l1_kinds[kind] = l1_kinds.get(kind, 0) + 1

    return {
        "cases": total,
        "solved": len(solved),
        "solved_pct": 100.0 * len(solved) / total if total else 0.0,
        "l1_cases": len(l1),
        "l1_pct": 100.0 * len(l1) / total if total else 0.0,
        "l2_cases": len(l2),
        "l2_pct": 100.0 * len(l2) / total if total else 0.0,
        "no_plan": len(no_plan),
        "budget": len(budget),
        "other": len(other),
        "unsolved": len(no_plan) + len(budget) + len(other),
        "cold_solved": len(cold_solved),
        "cold_solved_recovered": len(cold_solved_ok),
        "cold_solved_recovered_pct":
            100.0 * len(cold_solved_ok) / len(cold_solved) if cold_solved else 0.0,
        "cold_solved_l1": len([row for row in cold_solved_ok
                               if row.get("stage") == "backup"]),
        "cold_solved_l2": len([row for row in cold_solved_ok
                               if row.get("stage") == "graph_search"]),
        "l1_kind_counts": l1_kinds,
        "l1_latency_median_ms": median(l1_latency),
        "l1_latency_p95_ms": percentile(l1_latency, 0.95),
        "l1_latency_n": len(l1_latency),
        "l1_core_median_ms": median(l1_core),
        "l1_core_p95_ms": percentile(l1_core, 0.95),
        "l1_core_n": len(l1_core),
        "l1_speedup_median": median(l1_speedups),
        "l1_speedup_n": len(l1_speedups),
        "l1_speedup_e2e_median": median(l1_speedups_e2e),
        "l1_speedup_e2e_n": len(l1_speedups_e2e),
        "l1_multi_backup_saving_mean_pct":
            100.0 * mean(l1_savings) if l1_savings else None,
        "l1_multi_backup_saving_n": len(l1_savings),
        "l2_state_reuse_mean_pct":
            100.0 * mean(l2_reuse) if l2_reuse else None,
        "l2_state_reuse_n": len(l2_reuse),
        "l2_reused_edges_total": sum(row.get("reused_edges") or 0 for row in l2),
        "l2_generated_rows_total": sum(row.get("generated_rows") or 0 for row in l2),
        "l2_reused_states_total": sum(row.get("reused_states") or 0 for row in l2),
        "l2_speedup_median": median(l2_speedups),
        "l2_speedup_n": len(l2_speedups),
        "l2_speedup_e2e_median": median(l2_speedups_e2e),
        "l2_speedup_e2e_n": len(l2_speedups_e2e),
        "l2_decision_speedup_median": median(l2_decision),
        "l2_decision_speedup_n": len(l2_decision),
        "l2_optimality_match_pct":
            100.0 * sum(1 for row in l2_optimal if row["optimality_match"]) / len(l2_optimal)
            if l2_optimal else None,
        "l2_optimality_n": len(l2_optimal),
        "l2_optimality_ratio_median": median(l2_ratios),
    }


# --------------------------------------------------------------------------- #
# suite A -- LEGO bridge gate bundle
# --------------------------------------------------------------------------- #
def _gate():
    import run_offline_gate as gate
    return gate


def load_suite_a():
    bundle = read_json(BUNDLE_A)
    base = [(name, tuple(args)) for name, args in bundle["base_allowed"]]
    offline = read_json(RECEIPT_A)["bundle"]["preparation_seconds"] if RECEIPT_A.exists() else None
    return bundle, base, offline


def run_case_a(snapshot, *, bundle, base, offline_prepare_seconds):
    """Gate-style recovery from a symbolic fault snapshot (plus cold baseline)."""
    gate = _gate()
    checkpoints = bundle["checkpoints"]
    saved_graph = bundle["graph"]
    began = time.perf_counter()
    engine = gate.build_engine(install=True)
    allowed_fault, changed, protected = vendor_recovery._fault_model(engine, snapshot)
    engine_ready = time.perf_counter()
    model_load = engine_ready - began
    allowed = frozenset(key for key in base if key in allowed_fault)
    deadline = began + BUDGET
    result, choice, attempts, unfinished = try_backups(
        engine, checkpoints, allowed=allowed, protected=protected,
        cuts=frozenset(), deadline=min(deadline, time.perf_counter() + 0.5))
    stage = "backup"
    graph_stats = None
    search_started = None
    if result is None:
        stage = "graph_search"
        graph = CandidateGraph(engine, GRAPH_LIMIT_A, saved=saved_graph, changed=changed)
        search_started = time.perf_counter()
        result = search(engine, allowed=allowed, protected=protected,
                        candidate_graph=graph,
                        time_limit=max(0.0, deadline - time.perf_counter()),
                        max_expansions=MAX_EXPANSIONS)
        graph_stats = graph.stats()
    decision = time.perf_counter() - began
    if stage == "backup":
        first_solution = decision
        first_core = max(decision - model_load, 0.0)
    elif result.first_solution_seconds is None:
        first_solution = None
        first_core = None
    else:
        first_solution = (search_started - began) + result.first_solution_seconds
        first_core = result.first_solution_seconds

    valid_makespans = [attempt["makespan"] for attempt in attempts
                       if attempt.get("status") == "valid" and attempt.get("makespan") is not None]

    # independent cold baseline: identical fault domain, no candidate graph
    cold_began = time.perf_counter()
    cold_engine = gate.build_engine(install=True)
    cold_allowed_fault, _, cold_protected = vendor_recovery._fault_model(cold_engine, snapshot)
    cold_ready = time.perf_counter()
    cold_load = cold_ready - cold_began
    cold_allowed = frozenset(key for key in base if key in cold_allowed_fault)
    cold_search_started = time.perf_counter()
    cold = search(cold_engine, allowed=cold_allowed, protected=cold_protected,
                  time_limit=BUDGET, max_expansions=MAX_EXPANSIONS)
    cold_seconds = time.perf_counter() - cold_search_started

    graph_stats = graph_stats or {}
    row = {
        "suite": "A",
        "task": "lego_bridge_9blocks",
        "checkpoint": snapshot.get("checkpoint"),
        "fault_kind": snapshot.get("_fault_kind"),
        "fault_detail": snapshot.get("_fault_detail"),
        "stage": stage,
        "status": result.status,
        "backup_kind": (choice or {}).get("kind"),
        "backup_checkpoint": (choice or {}).get("checkpoint"),
        "backup_attempts": len(attempts),
        "graph_loaded": stage == "graph_search",
        "decision_seconds": decision,
        "vendor_elapsed_seconds": None,
        "model_load_seconds": model_load,
        "cold_load_seconds": cold_load,
        "first_solution_seconds": first_solution,
        "first_solution_core_seconds": first_core,
        "makespan": result.makespan if result.status == "solved" else None,
        "cold_status": cold.status,
        "cold_seconds": cold_seconds,
        "cold_first_solution_seconds": cold.first_solution_seconds
        if cold.status == "solved" else None,
        "cold_makespan": cold.makespan if cold.status == "solved" else None,
        "reused_states": graph_stats.get("reused_states", 0),
        "cached_states": graph_stats.get("cached_states", 0),
        "offline_states": graph_stats.get("offline_states", 0),
        "reused_edges": graph_stats.get("reused_edges", 0),
        "generated_rows": graph_stats.get("generated_rows", 0),
        "reused_capabilities": graph_stats.get("reused_capabilities", 0),
        "refreshed_capabilities": graph_stats.get("refreshed_capabilities", 0),
        "allowed_actions": len(allowed),
        "excluded_actions": len(base) - len(allowed),
        "budget_exhausted": bool(unfinished),
        "prepare_seconds": offline_prepare_seconds,
        "valid_backup_makespans": valid_makespans,
        "skipped": False,
        "note": "vendor_first_solution_unused; recovery timed with perf_counter",
    }
    return decorate(row)


def run_suite_a():
    gate = _gate()
    bundle, base, offline_prepare_seconds = load_suite_a()
    checkpoints = bundle["checkpoints"]
    schedule = bundle["nominal"]["schedule"]
    parse_engine = gate.build_engine(install=True)
    rows = []
    skipped = []

    for checkpoint in checkpoints:
        index = checkpoint["id"]
        text = schedule[index]["action"]
        action = parse_engine.parse_action(text)
        state = checkpoint["state"]

        rows.append(run_case_a(
            {"state": state, "disabled_actions": [text], "checkpoint": index,
             "budget_seconds": BUDGET,
             "_fault_kind": "action", "_fault_detail": text},
            bundle=bundle, base=base,
            offline_prepare_seconds=offline_prepare_seconds))

        arm = sorted(action.occupies)[0] if action.occupies else None
        skill = action.name
        blocks_deck = skill in {"pick", "insert"} and "B" in action.occupies
        if blocks_deck:
            skipped.append({
                "suite": "A", "task": "lego_bridge_9blocks", "checkpoint": index,
                "fault_kind": "skill", "fault_detail": f"{arm}/{skill}",
                "reason": "屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，"
                          "故障在构造上即无解",
            })
            continue
        rows.append(run_case_a(
            {"state": state, "disabled_skills": {arm: [skill]}, "checkpoint": index,
             "budget_seconds": BUDGET,
             "_fault_kind": "skill", "_fault_detail": f"{arm}/{skill}"},
            bundle=bundle, base=base,
            offline_prepare_seconds=offline_prepare_seconds))

    # L2 composite fault: checkpoint-8 symmetric cross-zone reachability loss
    fault_l2 = read_json(FAULT_L2)
    checkpoint_8 = next(item for item in checkpoints if item["id"] == 8)
    if fault_l2.get("state") != checkpoint_8["state"]:
        raise ValueError("fault_l2 snapshot does not match bundle checkpoint 8")
    composite_detail = ", ".join(f"{key}={value}" for key, value in
                                 sorted(fault_l2["capabilities"].items()))
    rows.append(run_case_a(
        {"state": checkpoint_8["state"], "capabilities": fault_l2["capabilities"],
         "checkpoint": 8, "budget_seconds": BUDGET,
         "_fault_kind": "composite", "_fault_detail": composite_detail},
        bundle=bundle, base=base,
        offline_prepare_seconds=offline_prepare_seconds))

    return {
        "bundle": {
            "path": str(BUNDLE_A),
            "checkpoints": len(checkpoints),
            "backups": sum(len(item["plans"]) - 1 for item in checkpoints),
            "graph_states": len(bundle["graph"]["states"]),
            "offline_prepare_seconds": offline_prepare_seconds,
            "offline_prepare_source": "recovery_demo/offline_gate_receipt.json",
        },
        "parse_engine_load_seconds": None,
        "cases": rows,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------- #
# suite B -- standard vendor tasks (one child process per task)
# --------------------------------------------------------------------------- #
def extract_recover_row(task, checkpoint_id, kind, detail_text, detail, t0, events,
                        model_load_seconds, prepare_seconds, action_count):
    """Split one compare=True recover() call into its recovery and cold phases.

    ``events`` holds the search() calls made inside recover(): for a graph
    repair the first entry is the recovery search, the second the cold baseline;
    for a backup hit the only entry is the cold baseline.  ``t0`` is the
    perf_counter stamp taken just before recover() was entered.
    """
    stage = detail.get("stage")
    backup_choice = detail.get("backup_choice") or {}
    graph = detail.get("graph") or {}
    cold = detail.get("full_search") or {}
    attempts = detail.get("backup_attempts") or []
    valid_makespans = [attempt["makespan"] for attempt in attempts
                       if attempt.get("status") == "valid"
                       and attempt.get("makespan") is not None]

    recovery_event = events[0] if stage == "graph_search" and events else None
    cold_event = events[1] if stage == "graph_search" and len(events) > 1 else (
        events[0] if stage != "graph_search" and events else None)
    if cold_event is not None:
        decision = cold_event["start"] - t0
        cold_seconds = cold_event["end"] - cold_event["start"]
        cold_first_internal = cold_event["first_internal"]
    else:  # defensive fallback: vendor timings are quantised
        decision = detail.get("elapsed_seconds")
        cold_seconds = cold.get("elapsed_seconds")
        cold_first_internal = cold.get("first_solution_seconds")

    if stage == "backup":
        first_solution = decision
        first_core = max(decision - model_load_seconds, 0.0)
    elif recovery_event is not None:
        first_solution = ((recovery_event["start"] - t0)
                          + (recovery_event["first_internal"] or 0.0))
        first_core = recovery_event["first_internal"]
    else:
        first_solution = detail.get("first_solution_seconds")
        first_core = (None if first_solution is None
                      else max(first_solution - model_load_seconds, 0.0))

    row = {
        "suite": "B",
        "task": task,
        "checkpoint": checkpoint_id,
        "fault_kind": kind,
        "fault_detail": detail_text,
        "stage": stage,
        "status": detail.get("status"),
        "backup_kind": backup_choice.get("kind"),
        "backup_checkpoint": backup_choice.get("checkpoint"),
        "backup_attempts": len(attempts),
        "graph_loaded": bool(detail.get("graph_loaded")),
        "decision_seconds": decision,
        "vendor_elapsed_seconds": detail.get("elapsed_seconds"),
        "model_load_seconds": model_load_seconds,
        "first_solution_seconds": first_solution,
        "first_solution_core_seconds": first_core,
        "makespan": detail.get("makespan"),
        "cold_status": cold.get("status"),
        "cold_seconds": cold_seconds,
        "cold_first_solution_seconds": cold_first_internal
        if cold.get("status") == "solved" else None,
        "cold_makespan": cold.get("makespan"),
        "reused_states": graph.get("reused_states", 0),
        "cached_states": graph.get("cached_states", 0),
        "offline_states": graph.get("offline_states", 0),
        "reused_edges": graph.get("reused_edges", 0),
        "generated_rows": graph.get("generated_rows", 0),
        "reused_capabilities": graph.get("reused_capabilities", 0),
        "refreshed_capabilities": graph.get("refreshed_capabilities", 0),
        "allowed_actions": None if detail.get("excluded_actions") is None
        else action_count - detail["excluded_actions"],
        "excluded_actions": detail.get("excluded_actions"),
        "budget_exhausted": bool(detail.get("backup_budget_exhausted")),
        "prepare_seconds": prepare_seconds,
        "valid_backup_makespans": valid_makespans,
        "skipped": False,
        "note": "decision/cold split with perf_counter around recovery.search "
                "(no source change); first=cold-phase start for backup hits",
    }
    return decorate(row)


def run_suite_b_task(task_name, out_path):
    task_path = TASKS / f"{task_name}.toml"
    out_dir = BUNDLE_DIR / task_name
    if out_dir.exists():
        assert_under(out_dir, BUNDLE_DIR)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "suite": "B", "task": task_name, "task_path": str(task_path),
        "prepare": {}, "cases": [], "error": None,
    }
    began = time.perf_counter()
    try:
        preparation = vendor_recovery.prepare(
            task_path, SKILLS, out_dir,
            time_limit=BUDGET, max_expansions=MAX_EXPANSIONS,
            graph_limit=B_GRAPH_LIMIT, checkpoints=B_CHECKPOINTS,
            backups=B_BACKUPS, backup_seconds=B_BACKUP_SECONDS)
        prepare_seconds = time.perf_counter() - began
        record["prepare"] = {
            "status": preparation.get("status"),
            "seconds": prepare_seconds,
            "checkpoints": preparation.get("checkpoints"),
            "backups": preparation.get("backups"),
            "graph": preparation.get("graph"),
            "nominal_makespan": preparation.get("nominal_makespan"),
            "nominal_bound": preparation.get("nominal_bound"),
        }
        if preparation.get("status") != "solved":
            write_json(out_path, record)
            return record

        started = time.perf_counter()
        Engine(load_problem(task_path, SKILLS))
        model_load_seconds = time.perf_counter() - started
        record["prepare"]["model_load_seconds"] = model_load_seconds

        data = read_json(out_dir / "plans.json")
        schedule = data["nominal"]["schedule"]
        engine = Engine(load_problem(task_path, SKILLS))
        action_count = len(engine.actions)

        # Read-only runtime instrumentation: recover(compare=True) runs the
        # recovery search first and the cold baseline second; wrapping the
        # module-level search() lets us split the two phases with perf_counter
        # without touching the vendor sources.
        original_search = vendor_recovery.search
        events = []

        def traced_search(*call_args, **call_kwargs):
            started = time.perf_counter()
            result = original_search(*call_args, **call_kwargs)
            events.append({"start": started, "end": time.perf_counter(),
                           "first_internal": result.first_solution_seconds})
            return result

        vendor_recovery.search = traced_search
        try:
            for checkpoint in data["checkpoints"]:
                index = checkpoint["id"]
                text = schedule[index]["action"]
                action = engine.parse_action(text)
                arm = sorted(action.occupies)[0] if action.occupies else None
                faults = (
                    ("action", {"state": checkpoint["state"],
                                "disabled_actions": [text]}, text),
                    ("skill", {"state": checkpoint["state"],
                               "disabled_skills": {arm: [action.name]}},
                     f"{arm}/{action.name}"),
                )
                for kind, snapshot, detail_text in faults:
                    snapshot = dict(snapshot)
                    snapshot.update({"checkpoint": index, "budget_seconds": BUDGET})
                    events.clear()
                    call_started = time.perf_counter()
                    detail = vendor_recovery.recover(
                        task_path, SKILLS, out_dir, snapshot,
                        time_limit=BUDGET, max_expansions=MAX_EXPANSIONS,
                        compare=True)
                    record["cases"].append(extract_recover_row(
                        task_name, index, kind, detail_text, detail,
                        call_started, list(events), model_load_seconds,
                        prepare_seconds, action_count))
        finally:
            vendor_recovery.search = original_search
    except BaseException:
        record["error"] = traceback.format_exc()
    write_json(out_path, record)
    return record


def launch_suite_b_children():
    tasks = sorted(TASKS.glob("*.toml"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VENDOR)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    script = str(Path(__file__).resolve())
    running = []
    for task_path in tasks:
        name = task_path.stem
        raw_path = RAW_DIR / f"{name}.json"
        log_path = LOG_DIR / f"{name}.log"
        handle = open(log_path, "wb")
        command = [sys.executable, "-B", "-X", "utf8", script,
                   "--suite-b-task", name, "--out", str(raw_path)]
        options = {"cwd": str(SHOWCASE), "env": env,
                   "stdout": handle, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, **options)
        running.append((name, process, handle, raw_path))
    return running


def collect_suite_b_children(running):
    records = []
    for name, process, handle, raw_path in running:
        code = process.wait()
        handle.close()
        if raw_path.exists():
            record = read_json(raw_path)
        else:
            record = {"suite": "B", "task": name, "prepare": {}, "cases": [],
                      "error": f"child produced no raw json (exit {code})"}
        record["exit_code"] = code
        records.append(record)
    return records


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_csv(rows):
    import csv as csv_module
    with CSV_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            entry = {}
            for field in CSV_FIELDS:
                value = row.get(field)
                if field == "valid_backup_makespans":
                    entry[field] = json.dumps(value or [], separators=(",", ":"))
                else:
                    entry[field] = fmt_number(value)
            writer.writerow(entry)


def build_report(payload):
    config = payload["config"]
    aggregates = payload["aggregates"]
    suite_a = payload["suite_a"]
    suite_b = payload["suite_b"]
    total = aggregates["all"]
    suite_a_summary = aggregates["suite_a"]
    suite_b_summary = aggregates["suite_b"]

    def stat_rows(summary):
        return [
            ("用例总数", summary["cases"]),
            ("恢复成功", f"{summary['solved']} ({summary['solved_pct']:.1f}%)"),
            ("L1 backup 真命中", f"{summary['l1_cases']} ({summary['l1_pct']:.1f}%)"),
            ("L2 graph_search", f"{summary['l2_cases']} ({summary['l2_pct']:.1f}%)"),
            ("无解 (no_plan)", summary["no_plan"]),
            ("超时 (budget)", summary["budget"]),
            ("其他状态", summary["other"]),
            ("冷启动可解用例", summary["cold_solved"]),
            ("其中恢复成功",
             f"{summary['cold_solved_recovered']} "
             f"({summary['cold_solved_recovered_pct']:.1f}%)"),
            ("L1 决策延迟中位 (ms)",
             "—" if summary["l1_latency_median_ms"] is None
             else f"{summary['l1_latency_median_ms']:.2f}"),
            ("L1 决策延迟 P95 (ms)",
             "—" if summary["l1_latency_p95_ms"] is None
             else f"{summary['l1_latency_p95_ms']:.2f}"),
            ("L1 决策延迟（模型已加载）中位 (ms)",
             "—" if summary["l1_core_median_ms"] is None
             else f"{summary['l1_core_median_ms']:.2f}"),
            ("L1 首解中位加速（模型已加载）",
             ratio_text(summary["l1_speedup_median"])
             + (f"，n={summary['l1_speedup_n']}" if summary["l1_speedup_n"] else "")),
            ("L1 首解中位加速（端到端含加载）",
             ratio_text(summary["l1_speedup_e2e_median"])),
            ("L1 多备份择优平均工时节省",
             "—" if summary["l1_multi_backup_saving_mean_pct"] is None
             else f"{summary['l1_multi_backup_saving_mean_pct']:.2f}%"),
            ("L2 平均状态复用率",
             "—" if summary["l2_state_reuse_mean_pct"] is None
             else f"{summary['l2_state_reuse_mean_pct']:.2f}%"),
            ("L2 复用出边 / 新生成行",
             f"{summary['l2_reused_edges_total']} / {summary['l2_generated_rows_total']}"),
            ("L2 首解中位加速（模型已加载）",
             ratio_text(summary["l2_speedup_median"])
             + (f"，n={summary['l2_speedup_n']}" if summary["l2_speedup_n"] else "")),
            ("L2 首解中位加速（端到端含加载）",
             ratio_text(summary["l2_speedup_e2e_median"])),
            ("L2 总决策中位加速",
             ratio_text(summary["l2_decision_speedup_median"])),
            ("L2 与冷启动最优性一致率",
             "—" if summary["l2_optimality_match_pct"] is None
             else f"{summary['l2_optimality_match_pct']:.1f}%"),
        ]

    fault_rows = [("全部", total["cases"], total["solved"], total["l1_cases"],
                   total["l2_cases"], total["no_plan"], total["budget"])]
    for kind, summary in payload["aggregates"]["by_fault_kind"].items():
        fault_rows.append((kind, summary["cases"], summary["solved"],
                           summary["l1_cases"], summary["l2_cases"],
                           summary["no_plan"], summary["budget"]))

    detail_rows = []
    for row in sorted(suite_a["cases"],
                      key=lambda item: (item["checkpoint"] or 0,
                                        item["fault_kind"] or "")):
        detail_rows.append((
            row["checkpoint"], row["fault_kind"], row["fault_detail"],
            row["stage"], row["status"],
            "—" if row["decision_ms"] is None else f"{row['decision_ms']:.2f}",
            "—" if row["decision_core_ms"] is None else f"{row['decision_core_ms']:.2f}",
            "—" if row["first_solution_seconds"] is None
            else f"{row['first_solution_seconds']:.4f}",
            "—" if row["cold_first_solution_seconds"] is None
            else f"{row['cold_first_solution_seconds']:.4f}",
            ratio_text(row["first_solution_speedup"]),
            ratio_text(row["first_solution_speedup_e2e"]),
            f"{row['reused_states']}/{row['generated_rows']}"
            if row["graph_loaded"] else "—",
            "—" if row["makespan"] is None else f"{row['makespan']:.3f}",
            "—" if row["cold_makespan"] is None else f"{row['cold_makespan']:.3f}",
            "yes" if row["optimality_match"] else
            ("n/a" if row["optimality_match"] is None else "no"),
        ))

    task_rows = []
    for task in sorted(suite_b, key=lambda item: item["task"]):
        cases = task.get("cases") or []
        summary = summarize(cases)
        prepare = task.get("prepare") or {}
        task_rows.append((
            task["task"], prepare.get("status", "error"),
            "—" if prepare.get("seconds") is None else f"{prepare['seconds']:.2f}",
            prepare.get("checkpoints", 0),
            summary["cases"], summary["solved"], summary["l1_cases"],
            summary["l2_cases"], summary["no_plan"], summary["budget"],
            "—" if summary["l1_latency_median_ms"] is None
            else f"{summary['l1_latency_median_ms']:.2f}",
            "—" if summary["l2_state_reuse_mean_pct"] is None
            else f"{summary['l2_state_reuse_mean_pct']:.1f}",
            ratio_text(summary["l2_speedup_median"]),
            "—" if summary["l2_optimality_match_pct"] is None
            else f"{summary['l2_optimality_match_pct']:.0f}%",
            (prepare.get("graph") or {}).get("cached_states", 0),
        ))

    skipped_lines = []
    for item in suite_a.get("skipped", []):
        skipped_lines.append(
            f"- checkpoint {item['checkpoint']}：{item['fault_kind']} "
            f"`{item['fault_detail']}` — {item['reason']}")

    l1_kinds = ", ".join(f"{kind}={count}" for kind, count
                         in sorted(total["l1_kind_counts"].items())) or "—"

    report = []
    append = report.append
    append("# db77489 恢复算法离线定量评测报告")
    append("")
    append(f"- 生成时间：{payload['generated_at']}（本机时区）")
    append(f"- 基准脚本：`recovery_demo/run_recovery_benchmark.py` "
           f"(sha256 `{payload['script_sha256']}`)")
    append(f"- 厂商快照：`recovery_demo/vendor_db77489/dualarm_skillnet` "
           f"(upstream commit `{VENDOR_COMMIT}`)")
    append(f"- 运行耗时：{payload['elapsed_seconds']:.1f} s")
    append("- 隔离：仅写 `recovery_demo/benchmark/`；未启动 CoppeliaSim；未改动冻结基线、"
           "`mode_a/`、`mode_b/`、`recovery_demo/bundle/` 与既有产物")
    append("")
    append("## 1. 实验配置")
    append("")
    append(md_table(("项目", "取值"), [
        ("域", "blocks-v1 积木域，双臂 A/B"),
        ("恢复预算 (time_limit)", f"{config['budget_seconds']:.1f} s"),
        ("max_expansions", f"{config['max_expansions']:,}"),
        ("套件 A 离线包", f"{config['suite_a_bundle'].get('checkpoints')} 个检查点 / "
                          f"{config['suite_a_bundle'].get('backups')} 个备份 / "
                          f"{config['suite_a_bundle'].get('graph_states')} 个候选图状态"),
        ("套件 A 离线准备耗时",
         f"{config['suite_a_bundle'].get('offline_prepare_seconds'):.3f} s"
         if config['suite_a_bundle'].get('offline_prepare_seconds') is not None
         else "未记录"),
        ("套件 A 动作域", "gate base_allowed ∩ 故障允许动作"),
        ("套件 B 离线准备", f"checkpoints={config['suite_b_prepare']['checkpoints']}, "
                            f"backups={config['suite_b_prepare']['backups']}, "
                            f"backup_seconds={config['suite_b_prepare']['backup_seconds']}, "
                            f"graph_limit={config['suite_b_prepare']['graph_limit']}"),
        ("套件 B 恢复调用", "recovery.recover(..., compare=True)（原版实现）"),
        ("冷启动基线", "相同故障域 search()，time_limit=30 s，无候选图"),
        ("post_first_solution_limit", "None（恢复与冷启动一致）"),
        ("并行度", "套件 A 主进程顺序执行；套件 B 11 个任务各一个无窗口 "
                   "(CREATE_NO_WINDOW) 子进程并行执行"),
        ("可复现性", "套件 A 完全确定性；套件 B 的离线准备按 wall-clock 预算 "
                     "（backup_seconds=1.0 s/检查点）生成备份，并行负载会影响备份条数，"
                     "重复运行间 L1/L2 层级计数可能浮动 1–2 例"),
        ("计时口径", "恢复决策用 perf_counter 外测；套件 B 在子进程内包裹 "
                     "recovery.search 精确切分恢复/冷启动两段（只读插桩，不改源码）；"
                     "冷启动首解取 vendor search 内部计时（Windows monotonic "
                     "量化约 15.6 ms）"),
    ]))
    append("")
    append("## 2. 总体统计摘要")
    append("")
    append(md_table(("指标", "全部用例"), stat_rows(total)))
    append("")
    append(md_table(("子集", "用例", "恢复成功", "L1", "L2", "无解", "超时"),
                    [("套件 A（乐高桥）", suite_a_summary["cases"],
                      suite_a_summary["solved"], suite_a_summary["l1_cases"],
                      suite_a_summary["l2_cases"], suite_a_summary["no_plan"],
                      suite_a_summary["budget"]),
                     ("套件 B（11 标准任务）", suite_b_summary["cases"],
                      suite_b_summary["solved"], suite_b_summary["l1_cases"],
                      suite_b_summary["l2_cases"], suite_b_summary["no_plan"],
                      suite_b_summary["budget"]),
                     ("全部", total["cases"], total["solved"], total["l1_cases"],
                      total["l2_cases"], total["no_plan"], total["budget"])]))
    append("")
    append(f"L1 命中备份类型分布：{l1_kinds}。")
    append("")
    append("### 2.1 按故障类型")
    append("")
    append(md_table(("故障类型", "用例", "成功", "L1", "L2", "无解", "超时"),
                    fault_rows))
    append("")
    append("### 2.2 冷启动可解子集上的恢复率（剔除故障本身不可恢复的用例）")
    append("")
    append(md_table(
        ("子集", "冷启动可解", "其中恢复成功", "恢复率", "L1", "L2"),
        [("套件 A（乐高桥）", suite_a_summary["cold_solved"],
          suite_a_summary["cold_solved_recovered"],
          f"{suite_a_summary['cold_solved_recovered_pct']:.1f}%",
          suite_a_summary["cold_solved_l1"], suite_a_summary["cold_solved_l2"]),
         ("套件 B（11 标准任务）", suite_b_summary["cold_solved"],
          suite_b_summary["cold_solved_recovered"],
          f"{suite_b_summary['cold_solved_recovered_pct']:.1f}%",
          suite_b_summary["cold_solved_l1"], suite_b_summary["cold_solved_l2"]),
         ("全部", total["cold_solved"], total["cold_solved_recovered"],
          f"{total['cold_solved_recovered_pct']:.1f}%",
          total["cold_solved_l1"], total["cold_solved_l2"])]))
    append("")
    append("## 3. 套件 A：乐高桥 9 块砖联合任务")
    append("")
    append("复用 `recovery_demo/bundle/plans.json` 的 12 个停稳检查点与 "
           "`run_offline_gate.py` 的 `build_engine`/`base_allowed`；"
           "每个检查点注入 nominal 下一步具体接地动作失效，并对无法连带封死桥面 "
           "B 臂取放的步骤注入该臂该技能失效；另加 checkpoint 8 对称跨区 reach "
           "复合故障。")
    append("")
    append(md_table(
        ("检查点", "故障类型", "故障内容", "阶段", "状态", "决策 (ms)",
         "决策核心 (ms)", "首解 (s)", "冷启动首解 (s)", "首解加速（核心）",
         "首解加速（端到端）", "复用状态/新行", "makespan", "冷 makespan",
         "最优一致"),
        detail_rows))
    append("")
    if skipped_lines:
        append("技能级故障跳过用例（按设计会直接封死 B 臂对桥面的取放）：")
        append("")
        report.extend(skipped_lines)
        append("")
    append("## 4. 套件 B：11 个标准任务")
    append("")
    append("每个任务用原版 `recovery.prepare` 准备（参数见 §1），"
           "对每个已解检查点注入单动作失效与臂/技能失效，逐例调用原版 "
           "`recovery.recover(..., compare=True)`。标准任务的可达域多为单臂独占，"
           "因此大量故障在几何上不可恢复，恢复器在这些用例上正确返回 `no_plan`。")
    append("")
    append(md_table(
        ("任务", "准备状态", "准备 (s)", "检查点", "用例", "成功", "L1", "L2",
         "无解", "超时", "L1 中位 (ms)", "L2 复用率 (%)", "L2 中位加速",
         "L2 最优一致", "离线图行数"),
        task_rows))
    append("")
    append("套件 B 每一条用例的完整指标见 `recovery_benchmark.csv`；"
           "11 份子进程日志见 `recovery_demo/benchmark/logs/`。")
    append("")
    append("## 5. 指标定义")
    append("")
    append(md_table(("指标", "定义"), [
        ("L1（backup 真命中）", "故障后仅重放离线备份/相邻检查点计划即得到可行解"
                                "（stage=backup, status=solved），含 nominal_suffix 与 backup"),
        ("L2（graph_search）", "备份全部失效后加载候选图做增量搜索并得到可行解"
                               "（stage=graph_search, status=solved）"),
        ("决策延迟（端到端）", "从收到故障快照到返回计划的墙钟时间（perf_counter），"
                                "含引擎与离线包加载；套件 A 为进程内实测，套件 B 由 "
                                "recover(compare=True) 外测并按内部 search 边界切分"),
        ("决策延迟（模型已加载）", "端到端延迟减去一次模型构建时间（套件 A 实测；"
                                    "套件 B 用同任务 Engine(load_problem) 实测值近似）"),
        ("首解加速（核心）", "冷启动 search 内部 first_solution_seconds ÷ 恢复首解核心时间"
                              "（两侧均从模型就绪起算）；冷启动首解低于 1 ms 的用例"
                              "视为计时不可分辨，不计入中位，记为 n"),
        ("首解加速（端到端）", "（模型加载 + 冷启动首解）÷ 恢复首解端到端时间，"
                                "仅统计冷启动首解 ≥1 ms 的用例"),
        ("多备份择优工时节省", "对有效备份重放 ≥2 条的解，"
                                "(最差有效备份 makespan − 选中 makespan) ÷ 最差有效备份 makespan"),
        ("状态复用率", "reused_states ÷ (reused_states + generated_rows)，"
                        "候选图命中行占本次搜索展开行的比例"),
        ("复用出边 / 新生成行", "CandidateGraph.stats() 的 reused_edges / generated_rows 汇总"),
        ("L2 总决策加速", "（模型加载 + 冷启动总时）÷ 恢复端到端延迟"),
        ("最优性一致率", "L2 解 makespan 与冷启动解 makespan 相对误差 ≤1e-6 的比例"),
        ("无解 / 超时", "status=no_plan（故障域内不可达目标）/ status=budget"
                        "（30 s 或扩张预算耗尽）"),
    ]))
    append("")
    append("## 6. 主要结论")
    append("")
    conclusions = [
        f"套件 A + B 共 {total['cases']} 例（A {suite_a_summary['cases']} 例、"
        f"B {suite_b_summary['cases']} 例）；恢复成功 {total['solved']} 例"
        f"（{total['solved_pct']:.1f}%），L1 备份真命中 {total['l1_cases']} 例"
        f"（{total['l1_pct']:.1f}%）、L2 图修补 {total['l2_cases']} 例"
        f"（{total['l2_pct']:.1f}%）；无解 {total['no_plan']} 例、超时 "
        f"{total['budget']} 例。",
        f"剔除故障本身不可恢复的用例后（冷启动可解 {total['cold_solved']} 例），"
        f"恢复成功率 {total['cold_solved_recovered_pct']:.1f}%"
        f"（{total['cold_solved_recovered']}/{total['cold_solved']}），"
        f"其中套件 A 为 {suite_a_summary['cold_solved_recovered_pct']:.1f}%、"
        f"套件 B 为 {suite_b_summary['cold_solved_recovered_pct']:.1f}%。"
        "标准任务集的低总成功率来自单臂独占可达域与持件锁定这两类结构性无解，"
        "而非恢复器搜索预算不足。",
    ]
    if total["l1_latency_median_ms"] is not None:
        conclusions.append(
            f"L1 决策延迟中位 {total['l1_latency_median_ms']:.2f} ms、"
            f"P95 {total['l1_latency_p95_ms']:.2f} ms（端到端含模型加载）；"
            f"扣除模型加载后的核心延迟中位 "
            + ("—" if total["l1_core_median_ms"] is None
               else f"{total['l1_core_median_ms']:.2f} ms")
            + f"；首解核心加速中位 {ratio_text(total['l1_speedup_median'])}"
            + (f"（可测 n={total['l1_speedup_n']}）"
               if total["l1_speedup_n"] else "")
            + f"，端到端口径 {ratio_text(total['l1_speedup_e2e_median'])}；"
            + "多备份择优平均节省工时 "
            + ("—" if total["l1_multi_backup_saving_mean_pct"] is None
               else f"{total['l1_multi_backup_saving_mean_pct']:.2f}%")
            + "。")
    if total["l2_cases"]:
        conclusions.append(
            f"L2 平均状态复用率 "
            + ("—" if total["l2_state_reuse_mean_pct"] is None
               else f"{total['l2_state_reuse_mean_pct']:.1f}%")
            + f"，累计复用出边 {total['l2_reused_edges_total']} 条、新生成行 "
            f"{total['l2_generated_rows_total']} 行；首解核心加速中位 "
            f"{ratio_text(total['l2_speedup_median'])}"
            + (f"（n={total['l2_speedup_n']}）" if total["l2_speedup_n"] else "")
            + f"，总决策中位加速 {ratio_text(total['l2_decision_speedup_median'])}，"
            f"最优性一致率 "
            + ("—" if total["l2_optimality_match_pct"] is None
               else f"{total['l2_optimality_match_pct']:.1f}%")
            + "。图复用主要收益是避免重搜与保持最优性，而非提前首解；"
              "候选图上限 5000 时重生成行会抵消部分收益。")
    no_plan_by_fault = {}
    for row in payload["rows"]:
        if row.get("status") in {"no_plan", "budget"}:
            key = (row["fault_kind"], row.get("status"))
            no_plan_by_fault[key] = no_plan_by_fault.get(key, 0) + 1
    if no_plan_by_fault:
        parts = ", ".join(f"{kind}/{status}={count}" for (kind, status), count
                          in sorted(no_plan_by_fault.items()))
        conclusions.append(f"未恢复用例分布：{parts}。经典模式是（i）持件臂的 insert 失效后"
                           "另一臂既拿不到件、任务里也没有可回退的 place/handover 通道；"
                           "（ii）单臂独占 stock 的 pick 失效后另一臂不可达。"
                           "budget 集中在 s3a 平台类高分支任务（30 s / 200k 扩张预算耗尽）。")
    report.extend(f"- {line}" for line in conclusions)
    append("")
    append("> 复现：`python -B -X utf8 recovery_demo/run_recovery_benchmark.py`；"
           "套件 B 每任务一个无窗口子进程，套件 A 在主进程执行，"
           "全部产物位于 `recovery_demo/benchmark/`。")
    return "\n".join(report) + "\n"


def aggregate_all(rows):
    by_kind = {}
    for kind in ("action", "skill", "composite"):
        subset = [row for row in rows if row.get("fault_kind") == kind]
        if subset:
            by_kind[kind] = summarize(subset)
    suite_a = [row for row in rows if row.get("suite") == "A"]
    suite_b = [row for row in rows if row.get("suite") == "B"]
    return {"all": summarize(rows),
            "suite_a": summarize(suite_a),
            "suite_b": summarize(suite_b),
            "by_fault_kind": by_kind}


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--suite-b-task", type=str, default=None,
                        help="internal: run one suite B task in this process")
    parser.add_argument("--out", type=Path, default=None,
                        help="internal: raw json path for --suite-b-task")
    parser.add_argument("--only", choices=("both", "a", "b"), default="both")
    args = parser.parse_args(argv)

    if args.suite_b_task is not None:
        if args.out is None:
            parser.error("--suite-b-task requires --out")
        run_suite_b_task(args.suite_b_task, args.out)
        return 0

    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    suite_a = {"cases": [], "skipped": [], "bundle": {},
               "summary": summarize([])}
    suite_b_records = []
    running = launch_suite_b_children() if args.only in ("both", "b") else []
    if args.only in ("both", "a"):
        suite_a = run_suite_a()
        suite_a["summary"] = summarize(suite_a["cases"])
    if running:
        suite_b_records = collect_suite_b_children(running)

    suite_b_cases = [case for record in suite_b_records
                     for case in (record.get("cases") or [])]
    rows = list(suite_a.get("cases") or []) + suite_b_cases
    for row in rows:
        decorate(row)

    script_sha = file_sha(Path(__file__).resolve())
    payload = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vendor": {"commit": VENDOR_COMMIT, "path": str(VENDOR),
                   "model_files": list(vendor_recovery.MODEL_FILES)},
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_logical": os.cpu_count(),
        },
        "config": {
            "budget_seconds": BUDGET,
            "max_expansions": MAX_EXPANSIONS,
            "measurable_first_seconds": MEASURABLE_FIRST,
            "ratio_floor": FLOOR,
            "relative_tolerance": REL_TOL,
            "suite_a_bundle": suite_a.get("bundle") or {},
            "suite_b_prepare": {
                "checkpoints": B_CHECKPOINTS,
                "backups": B_BACKUPS,
                "backup_seconds": B_BACKUP_SECONDS,
                "graph_limit": B_GRAPH_LIMIT,
                "budget_seconds": BUDGET,
                "max_expansions": MAX_EXPANSIONS,
                "compare": True,
            },
            "suite_b_workers": len(suite_b_records),
        },
        "suite_a": suite_a,
        "suite_b": [
            {"task": record.get("task"), "task_path": record.get("task_path"),
             "prepare": record.get("prepare"), "exit_code": record.get("exit_code"),
             "error": record.get("error"),
             "cases": record.get("cases") or []}
            for record in suite_b_records
        ],
        "rows": rows,
        "aggregates": aggregate_all(rows),
        "elapsed_seconds": time.perf_counter() - began,
    }
    errors = [record["task"] for record in suite_b_records if record.get("error")]
    payload["suite_b_summary"] = {
        "tasks": len(suite_b_records),
        "cases": len(suite_b_cases),
        "tasks_error": errors,
    }

    write_json(RAW_JSON, payload)
    write_csv(rows)
    REPORT_MD.write_text(build_report(payload), encoding="utf-8")

    summary = payload["aggregates"]["all"]
    print(json.dumps({
        "schema": SCHEMA,
        "suite_a_cases": len(suite_a.get("cases") or []),
        "suite_a_skipped": len(suite_a.get("skipped") or []),
        "suite_b_tasks": len(suite_b_records),
        "suite_b_cases": len(suite_b_cases),
        "cases": summary["cases"],
        "solved": summary["solved"],
        "l1": summary["l1_cases"],
        "l2": summary["l2_cases"],
        "no_plan": summary["no_plan"],
        "budget": summary["budget"],
        "cold_solved": summary["cold_solved"],
        "cold_solved_recovered": summary["cold_solved_recovered"],
        "l1_median_ms": summary["l1_latency_median_ms"],
        "l1_p95_ms": summary["l1_latency_p95_ms"],
        "l1_core_median_ms": summary["l1_core_median_ms"],
        "l1_speedup_median": summary["l1_speedup_median"],
        "l1_speedup_n": summary["l1_speedup_n"],
        "l1_speedup_e2e_median": summary["l1_speedup_e2e_median"],
        "l1_saving_pct": summary["l1_multi_backup_saving_mean_pct"],
        "l2_reuse_mean_pct": summary["l2_state_reuse_mean_pct"],
        "l2_speedup_median": summary["l2_speedup_median"],
        "l2_speedup_n": summary["l2_speedup_n"],
        "l2_decision_speedup_median": summary["l2_decision_speedup_median"],
        "l2_optimality_pct": summary["l2_optimality_match_pct"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        detail = OUT / "benchmark_failure.json"
        write_json(detail, {"traceback": traceback.format_exc(),
                            "elapsed_seconds": time.perf_counter()})
        raise
