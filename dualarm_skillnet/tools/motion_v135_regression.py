"""Owned, serial M6 regressions with all new evidence inside motion_v135.

Run this supervisor directly, not inside another control.run_job (it acquires
that same serial lock). --default performs one ordinary default collection;
--hashseed0 explicitly adds one separate default run. --original11 independently
solves the immutable top-level eleven via pipeline.run, without proposals or a
motion backend. Importing this module never runs a test or planner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import motion_v135_control as control
from tools import overnight_regression as timed
from tools import overnight_batch as old
from tools.v134_originals import comparison, replay as rule_replay
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run as run_problem
from skillnet.report import plan_to_dict

ORIGINAL11 = ("s1_far_handover", "s1_nearest_arm", "s1_parallel_place", "s1_reach_handover",
              "s2_tower", "s2_tower_same_type", "s3a_cantilever", "s3a_cantilever_with_proposal",
              "s3a_platform", "s3a_platform_demo_needs_support", "s3a_platform_with_proposal")
TASK_SECONDS = 60.


def _portable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _save(path, value):
    old.atomic_json(path, _portable(value))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _output(path):
    path = old.confined(path)
    if not (path.is_relative_to(control.BASE) or path.is_relative_to(ROOT/"out")):
        raise ValueError("回归输出仅允许本轮独立结果目录或out，拒绝写历史结果")
    return path


def original_manifest():
    """Read immutable inputs and formal calibrated expectations; never solve."""
    baseline_path = control.BASE/"baseline.json"
    baseline = old.read_json(baseline_path)
    source = ROOT/"results/sim_v1_3/v134/original11_four_fixes_runtime_fix2.json"
    formal = old.read_json(source)
    v121_source = ROOT/"results/sim_v1_3/v133/original11_v121_reference.json"
    v121 = {row["file"]: row for row in old.read_json(v121_source)}
    calibration_path = ROOT/"results/sim_v1_3/v133/calibration.json"
    refs = {row["file"]: row for row in formal["rows"]}
    expected = {row["path"]: row["sha256"] for row in baseline["source_files"]}
    tasks = []
    actual_top_level = {path.stem for path in (ROOT/"tasks").glob("*.toml")}
    if actual_top_level != set(ORIGINAL11) or set(refs) != {f"tasks/{name}.toml" for name in ORIGINAL11}:
        raise ValueError("原始11任务集合或正式对照身份不一致")
    for name in ORIGINAL11:
        relative = f"tasks/{name}.toml"
        snapshot = control.BASE/"baseline_source"/relative
        current_sha, snapshot_sha = old.sha256(ROOT/relative), old.sha256(snapshot)
        reference = refs[relative]
        if not (current_sha == snapshot_sha == expected.get(relative) == reference.get("task_sha256")):
            raise ValueError("原任务/冻结快照/正式对照SHA不一致："+relative)
        tasks.append({"task": name, "file": relative, "task_sha256": current_sha,
            "snapshot_source": str(snapshot), "snapshot_sha256": snapshot_sha,
            "formal_task_sha256": reference["task_sha256"],
            "formal_reference": deepcopy(reference), "v121_reference": deepcopy(v121[relative])})
    return {"tasks": tasks, "baseline_source": str(source), "baseline_sha256": old.sha256(source),
        "snapshot_manifest": str(baseline_path), "snapshot_manifest_sha256": old.sha256(baseline_path),
        "v121_reference": str(v121_source), "v121_reference_sha256": old.sha256(v121_source),
        "calibration_source": str(calibration_path), "calibration_sha256": old.sha256(calibration_path),
        "coefficients": old.read_json(calibration_path)["skills"]}


def default_worker(label, output):
    output = _output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reuse its exact discovery, TimedResult and support-task timing; do not call
    # the old supervise(), which performs three runs and appends historical files.
    code = timed.worker(SimpleNamespace(label=label, json=str(output), module=None))
    if output.is_file():
        record = old.read_json(output)
        record.update(wrapper_tool=str(Path(__file__)), wrapper_tool_sha256=old.sha256(Path(__file__)))
        _save(output, record)
    return code


def _certificate(schedule, reference):
    cost = schedule.makespan
    bound = getattr(schedule, "suboptimality_bound", None)
    lower = getattr(schedule, "lower_bound", None)
    if not comparison(reference, schedule.status, cost, bound):
        return False
    if not (isinstance(lower, (int, float)) and math.isfinite(lower) and lower >= 0
            and lower <= cost+1e-7 and bound >= 1.-1e-8):
        return False
    if cost > 0 and (lower <= 0 or cost/lower > bound+1e-7):
        return False
    return not math.isclose(reference["bound"], 1., abs_tol=1e-8) or bound <= 1.+1e-8


def original_worker(label, output, *, manifest=None):
    output = _output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = original_manifest() if manifest is None else manifest
    version = control.version()
    receipt = {"label": label, "kind": "original11", "started_at": old.utc_now(),
        "status": "running", "passed": False, "rows": [], "task_seconds": TASK_SECONDS,
        "proposal_enabled": False, "witness_loaded": False, "motion_backend": False,
        "manifest_sha256": _hash(manifest), "manifest": deepcopy(manifest), "code_version": version,
        "tool_sha256": old.sha256(Path(__file__)), "hashseed": os.environ.get("PYTHONHASHSEED", "random/default")}
    if output.exists():
        saved = old.read_json(output)
        if (saved.get("manifest_sha256") != receipt["manifest_sha256"]
                or saved.get("code_version") != version or saved.get("tool_sha256") != receipt["tool_sha256"]):
            raise ValueError("已有原11回归的输入/代码版本不同，保留现场并使用新job名称")
        receipt = saved
        files = [row["file"] for row in receipt["rows"]]
        if files != [row["file"] for row in manifest["tasks"][:len(files)]]:
            raise ValueError("已有断点任务身份/顺序不一致，拒绝跳过未经核验的样本")
        for saved_row, task in zip(receipt["rows"], manifest["tasks"]):
            if saved_row.get("task_sha256") != task["task_sha256"]:
                raise ValueError("已有断点任务SHA不一致")
    completed = {row["file"] for row in receipt["rows"]}
    began = time.monotonic()
    _save(output, receipt)
    for task in manifest["tasks"]:
        if task["file"] in completed:
            continue
        row = {"task": task["task"], "file": task["file"], "task_sha256": task["task_sha256"],
            "baseline_source": task["snapshot_source"], "baseline_task_sha256": task["snapshot_sha256"],
            "budget_seconds": TASK_SECONDS, "proposal_enabled": False, "witness_loaded": False,
            "status": "running", "passed": False, "rule_replay": {"passed": False},
            "formal_reference": {key: task["formal_reference"].get(key) for key in ("status", "makespan", "bound")},
            "v121_expected_status": task["v121_reference"]["expected_status"],
            "v121_uncalibrated_makespan": task["v121_reference"]["v121_makespan"]}
        started = time.monotonic()
        if old.sha256(ROOT/task["file"]) != task["task_sha256"]:
            reason = {"status": "global_stop", "at": old.utc_now(),
                      "reason": "求解前原任务SHA变化", "task": task["file"]}
            receipt.update(**reason)
            _save(output, receipt)
            _save(control.BASE/"global_stop.json", reason)
            return 2
        try:
            problem = load_problem(ROOT/task["file"], ROOT/"skills/blocks_v1.toml")
            problem.proposal = None
            # No backend argument: this is the normal pipeline's independent
            # symbolic behavior. Deadline includes loading/grounding and search.
            result = run_problem(problem, deadline=started+TASK_SECONDS)
            schedule = result.schedule
            row.update(status=result.status, schedule_status=schedule.status,
                makespan=schedule.makespan, bound=getattr(schedule, "suboptimality_bound", None),
                lower_bound=getattr(schedule, "lower_bound", None), weight=getattr(schedule, "weight", None),
                expanded=schedule.expanded, generated=schedule.generated, search_seconds=schedule.elapsed,
                first_solution_seconds=getattr(schedule, "first_solution_seconds", None),
                first_solution_expanded=getattr(schedule, "first_solution_expanded", None),
                stages=list(getattr(schedule, "stages", ())), feedback=list(result.feedback))
            plan = output.parent/(task["task"]+".plan.json")
            _save(plan, plan_to_dict(result))
            row["plan"] = str(plan)
            if result.status != "solved" or schedule.status != result.status:
                raise ValueError("未求解或pipeline/schedule状态不一致："+str(result.status))
            row["certificate_passed"] = _certificate(schedule, task["formal_reference"])
            if not row["certificate_passed"]:
                raise ValueError("正式工时/最优性/bound/lower-bound证书条件未通过")
            costs = rule_replay(Engine(problem), schedule, manifest["coefficients"])
            row.update(passed=True, rule_replay={"passed": True, "steps": len(costs),
                "scope": "event effects, capabilities, arm resources, structural mutex, goal and independently calibrated costs",
                "independent_step_costs": costs})
        except Exception as exc:
            row.update(status=row["status"] if row["status"] != "running" else "error",
                       reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        row["elapsed_seconds"] = time.monotonic()-started
        receipt["rows"].append(row)
        receipt.update(elapsed_seconds=time.monotonic()-began, updated_at=old.utc_now())
        _save(output, receipt)  # durable after every task, including budgets/failures
        print(json.dumps(_portable(row), ensure_ascii=False), flush=True)
    receipt.update(status="passed" if len(receipt["rows"]) == 11 and all(row["passed"] for row in receipt["rows"]) else "failed",
                   finished_at=old.utc_now(), elapsed_seconds=time.monotonic()-began)
    receipt["passed"] = receipt["status"] == "passed"
    _save(output, receipt)
    return 0 if receipt["passed"] else 1


@contextmanager
def _hashseed(value):
    before = os.environ.get("PYTHONHASHSEED")
    try:
        if value is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = value
        yield
    finally:
        if before is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = before


def dispatch(name, kind, *, hashseed0=False, seconds=1200., base=None):
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError("name必须是单一安全job标识")
    if kind not in {"default", "original11"} or not math.isfinite(seconds) or not 0 < seconds <= 1200:
        raise ValueError("回归类型或任务总预算无效")
    if hashseed0 and kind != "default":
        raise ValueError("hashseed0仅用于显式第二遍默认集合")
    base = _output(base or control.BASE)
    records = []
    for seed in ([None, "0"] if hashseed0 else [None]):
        label = kind+("_seed0" if seed == "0" else "")
        folder = base/"regressions"/name/label
        folder.mkdir(parents=True, exist_ok=True)
        output = folder/"worker.json"
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                   "--worker-kind", kind, "--label", name+"_"+label, "--json", str(output)]
        request = {"name": name, "kind": kind, "hashseed": seed, "seconds": seconds,
                   "command": command, "output": str(output), "tool_sha256": old.sha256(Path(__file__))}
        request_path = folder/"request.json"
        if request_path.exists() and old.read_json(request_path) != request:
            raise ValueError("已有请求与当前request不同，使用新名称，保留旧结果")
        _save(request_path, request)
        with _hashseed(seed):
            job = control.run_job(name+"_"+label, command, seconds, live=False)
        worker = old.read_json(output) if output.is_file() else None
        reason = None
        if worker is None:
            reason = "missing_worker_json"
        elif worker.get("status") != "passed":
            reason = "worker_not_passed"
        elif kind == "default" and (not worker.get("tests_discovered")
                or worker.get("tests_run") != worker.get("tests_discovered")
                or worker.get("failure_count") != 0 or worker.get("error_count") != 0):
            reason = "incomplete_default_collection"
        elif kind == "original11" and (worker.get("passed") is not True or len(worker.get("rows", [])) != 11
                or not all(row.get("passed") is True for row in worker["rows"])):
            reason = "incomplete_original_eleven"
        records.append({"label": label, "job": job, "worker_json": str(output),
                        "worker_evidence_reason": reason,
                        "passed": job.get("status") == "passed"
                            and job.get("code_unchanged_during_job") is True and reason is None})
    answer = {"name": name, "kind": kind, "runs": records,
              "passed": bool(records) and all(row["passed"] for row in records)}
    _save(base/"regressions"/name/"summary.json", answer)
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description="本轮独立默认集合或原11任务串行回归")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--default", action="store_true")
    mode.add_argument("--original11", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--hashseed0", action="store_true")
    parser.add_argument("--seconds", type=float, default=1200.)
    parser.add_argument("--worker-kind", choices=("default", "original11"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_kind:
        if not args.label or not args.json:
            parser.error("worker需要label/json")
        return (default_worker if args.worker_kind == "default" else original_worker)(args.label, args.json)
    if not args.name or not (args.default or args.original11):
        parser.error("需要--name和--default或--original11")
    result = dispatch(args.name, "default" if args.default else "original11",
                      hashseed0=args.hashseed0, seconds=args.seconds)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
