"""Rebuild evidence-only Chinese reports; never run planning or simulation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tomllib

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from tools import overnight_batch as batch


ROOT = batch.ROOT
BASE = ROOT / "results/sim_v1_3"
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
FIXTURES = ("s2_tower_sim", "s3a_cantilever_sim", "s3a_platform_sim")


def read(path, default=None):
    return batch.read_json(path) if Path(path).is_file() else deepcopy(default)


def path_text(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def finite(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def real_passed(summary):
    return (summary.get("passed") is True and summary.get("status") == "passed"
            and summary.get("profile") == "skillnet_v1_3" and summary.get("simulator_closed") is True
            and finite(summary.get("actual_makespan_seconds"), positive=True)
            and isinstance(summary.get("simulator_pid"), int) and summary["simulator_pid"] > 0)


def distribution(values):
    values = sorted(float(x) for x in values if finite(x))
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    def quantile(q):
        index = q * (len(values)-1)
        lower = int(index)
        return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)
    return {"n": len(values), "min": values[0], "p25": quantile(.25),
            "median": quantile(.5), "p75": quantile(.75), "max": values[-1]}


def failure_categories(rows):
    categories = Counter()
    for row in rows:
        status = row.get("status")
        if row.get("passed") is True and status in {"passed", "solved"}:
            continue
        phase = row.get("failure_phase") or ("selection" if status == "skipped" else "unknown")
        if status == "timeout" or phase in {"task_wall_timeout", "phase_budget"}:
            category = "timeout"
        elif status == "skipped" or row.get("executed") is False:
            category = "skipped"
        elif row.get("motion_status") in {"INVALID", "UNKNOWN"}:
            category = "motion_" + row["motion_status"].lower()
        else:
            category = status or "failed"
        categories[(phase, category)] += 1
    return [{"failure_phase": phase, "category": category, "count": count}
            for (phase, category), count in sorted(categories.items())]


def execution_wall(jobs, records):
    """Use committed worker results once, never add their summary references."""
    seen, sources, duplicates, missing = set(), [], [], []
    for job in jobs:
        row = records.get(job["id"])
        if row is None or row.get("status") == "skipped" or row.get("executed") is False:
            continue
        key = ("simulator_log", str(Path(row["simulator_log"]).resolve())) if row.get("simulator_log") else ("job", job["id"])
        if key in seen:
            duplicates.append(job["id"])
            continue
        seen.add(key)
        field = "wall_seconds" if finite(row.get("wall_seconds")) else "worker_wall_seconds"
        if not finite(row.get(field)):
            missing.append(job["id"])
            continue
        sources.append({"id": job["id"], "seconds": row[field], "field": field,
                        "source": "committed_journal_row", "output": row.get("output", job.get("output"))})
    return {"seconds": sum(row["seconds"] for row in sources), "measured_runs": len(sources),
            "sources": sources, "duplicate_ids_excluded": duplicates, "missing_wall_ids": missing,
            "scope": "完成日志的自有 worker 结果；每个模拟器启动日志标识仅一次，未再相加 summary 扫描引用"}


def counts(jobs, records, *, planning=False):
    selected = [records[job["id"]] for job in jobs if job["id"] in records]
    skipped = sum(row.get("status") == "skipped" or row.get("executed") is False for row in selected)
    executed = len(selected)-skipped
    passed = sum((row.get("passed") is True and row.get("status") == "solved") if planning
                 else real_passed(row) for row in selected)
    auxiliary = sum(row.get("auxiliary") is True for row in selected)
    auxiliary_passed = sum(row.get("auxiliary") is True and
        ((row.get("passed") is True and row.get("status") == "solved") if planning else real_passed(row)) for row in selected)
    compile_failed = sum(row.get("compile_failed") is True or
                         row.get("failure_phase") in {"real_planning", "compile"} for row in selected)
    return {"requested": len(jobs), "recorded": len(selected), "executed": executed,
            "skipped": skipped, "pending": len(jobs)-len(selected), "passed": passed,
            "failed": executed-passed, "success_rate_requested": passed/len(jobs) if jobs else None,
            "success_rate_executed": passed/executed if executed else None,
            "auxiliary": auxiliary, "auxiliary_rate_executed": auxiliary/executed if executed else None,
            "auxiliary_passed": auxiliary_passed, "auxiliary_rate_passed": auxiliary_passed/passed if passed else None,
            "compilation_failed": compile_failed,
            "compilation_failure_rate_executed": compile_failed/executed if executed else None,
            "failure_categories": failure_categories(selected),
            "wall_seconds": sum(row.get("wall_seconds", 0.) for row in selected if finite(row.get("wall_seconds"))),
            "search_seconds": sum(row.get("search_elapsed_seconds", 0.) for row in selected
                                  if finite(row.get("search_elapsed_seconds")))}


def planning_diagnostics(jobs, records):
    reasons, unknown, sizes = Counter(), Counter(), defaultdict(Counter)
    for job in jobs:
        row = records.get(job["id"])
        if row is None:
            reasons["pending"] += 1
            continue
        status = row.get("status", "unknown_record")
        if status == "budget":
            if row.get("expanded", 0) >= 200_000:
                reason = "budget_expanded_ge_200000"
            elif row.get("search_elapsed_seconds", 0) >= 60:
                reason = "budget_last_search_ge_60s"
            else:
                reason = "budget_other"
        else:
            reason = status
        reasons[reason] += 1
        unknown.update(row.get("unknown", []))
        task_path = batch.confined(job["task"]) if job.get("task") else None
        if task_path and task_path.is_file():
            task = tomllib.loads(task_path.read_text(encoding="utf-8-sig"))
            sizes[(len(task.get("block", [])), len(task.get("slot", [])))][status] += 1
    return {"reason_counts": dict(reasons), "unknown_facts": dict(unknown.most_common()),
            "structure_status": [{"blocks": b, "slots": s, "statuses": dict(count)} for (b, s), count in sorted(sizes.items())],
            "scope": "budget分类只描述末次搜索的实测expanded/elapsed；不是全部pipeline累计耗时或无解证明"}


def precheck_comparisons(jobs, records):
    rows = []
    for job in jobs:
        row = records.get(job["id"], {})
        estimate = job.get("witness_estimate_seconds")
        actual = row.get("makespan") if row.get("status") == "solved" and row.get("passed") is True else None
        ratio = actual/estimate if finite(actual) and finite(estimate, positive=True) else None
        rows.append({"id": job["id"], "family": job.get("family"), "status": row.get("status", "pending"),
            "witness_estimate_seconds": estimate, "independent_makespan_seconds": actual,
            "independent_to_witness_ratio": ratio, "reason": row.get("reason"), "source": job.get("output")})
    return rows


def final_plan_estimate(job, row):
    directory = row.get("evidence_directory")
    if directory:
        directory = batch.confined(directory)
    elif job.get("output") and (row.get("task") or job.get("task")):
        task = Path(row.get("task") or job["task"]).stem
        directory = batch.confined(job["output"]) / "simulation" / task
    else:
        return None
    plan = read(directory / "plan.json", {})
    value = plan.get("makespan")
    return value if plan.get("status") == "solved" and finite(value, positive=True) else None


def summary_evidence(value):
    if isinstance(value, dict):
        value = value.get("summary") or value.get("path") or value.get("evidence_directory")
    if not isinstance(value, str):
        return {}, None
    path = batch.confined(value)
    if path.is_dir():
        path /= "summary.json"
    return read(path, {}), path


def backend_evidence(path):
    raw = read(path.parent / "motion_backend.json", {}) if path else {}
    return {"validations": [{k: row[k] for k in ("action", "status", "detail", "failed_segment") if k in row}
                            for row in raw.get("validations", [])],
            "qualifications": [{k: row[k] for k in ("fact", "status", "detail", "target_pose") if k in row}
                               for row in raw.get("qualifications", [])]}


def m4_evidence(outcome):
    fixture_rows = outcome.get("fixtures", [])
    if isinstance(fixture_rows, dict):
        fixture_rows = [{"task": key, "summary": value} for key, value in fixture_rows.items()]
    fixtures = {}
    for name in FIXTURES:
        evidence = next((row for row in fixture_rows if row.get("task") == name), {})
        summary, path = summary_evidence(evidence)
        fixtures[name] = {"passed": real_passed(summary) and summary.get("task") == name,
                          "source": path_text(path) if path else None}
    invalid, invalid_path = summary_evidence(outcome.get("invalid_feedback"))
    invalid_backend = backend_evidence(invalid_path)
    invalid_records = [row for row in invalid_backend.get("validations", []) if row.get("status") == "INVALID"]
    feedback = invalid.get("feedback", [])
    cut = any("运动失败" in str(line) and ("禁止" in str(line) or "cut" in str(line)) for line in feedback)
    invalid_ok = (real_passed(invalid) and invalid.get("motion_feedback_count", 0) > 0
                  and bool(invalid_records) and cut)
    unknown, unknown_path = summary_evidence(outcome.get("unknown_qualification"))
    unknown_backend = backend_evidence(unknown_path)
    qualified = [row for row in unknown_backend.get("qualifications", [])
                 if row.get("status") == "VALID" and str(row.get("fact", "")).startswith("reach(")]
    unknown_ok = (real_passed(unknown) and unknown.get("qualification_calls", 0) > 0
                  and bool(qualified) and any("按需验证" in str(line) for line in unknown.get("feedback", [])))
    return {"fixtures": fixtures, "notes": list(outcome.get("notes", [])),
            "invalid_cut_execution": {"passed": invalid_ok,
            "invalid_records": len(invalid_records), "cut_feedback_present": cut,
            "source": path_text(invalid_path) if invalid_path else None},
            "unknown_qualify_execution": {"passed": unknown_ok, "valid_reach_qualifications": len(qualified),
            "source": path_text(unknown_path) if unknown_path else None},
            "notes": outcome.get("notes", []),
            "passed": all(row["passed"] for row in fixtures.values()) and invalid_ok and unknown_ok}


def real_runs(base):
    """Count copied references once by the simulator's per-launch log nonce."""
    paths = set()
    for directory in (base / "overnight", base / "m3_attempt_5"):
        if directory.exists():
            paths.update(directory.rglob("summary.json"))
    unique = {}
    for path in sorted(paths):
        data = read(path, {})
        if data.get("profile") != "skillnet_v1_3" or not data.get("task_file") or not finite(data.get("wall_seconds")):
            continue
        identity = ("simulator_log", str(Path(data["simulator_log"]).resolve())) if data.get("simulator_log") else ("source", str(path.resolve()))
        if identity in unique:
            unique[identity]["duplicate_sources"].append(path_text(path))
            continue
        initial = read(path.parent / "initial_plan.json", {})
        final = read(path.parent / "plan.json", {})
        estimate, actual = initial.get("makespan"), data.get("actual_makespan_seconds")
        ratio = actual/estimate if finite(actual, positive=True) and finite(estimate, positive=True) else None
        final_estimate = final.get("makespan") if final.get("status") == "solved" else None
        final_ratio = actual/final_estimate if finite(actual, positive=True) and finite(final_estimate, positive=True) else None
        unique[identity] = {"source": path_text(path), "duplicate_sources": [], "task": data.get("task"),
            "status": data.get("status"), "passed": data.get("passed") is True,
            "failure_phase": data.get("failure_phase"), "motion_status": data.get("motion_status"),
            "reason": data.get("reason"),
            "simulator_pid": data.get("simulator_pid"), "simulator_log": data.get("simulator_log"),
            "wall_seconds": data["wall_seconds"], "phase_seconds": data.get("phase_seconds", {}),
            "estimated_seconds": estimate, "actual_seconds": actual,
            "actual_to_estimated": ratio, "estimated_to_actual": 1/ratio if ratio else None,
            "final_plan_estimated_seconds": final_estimate,
            "actual_to_final_plan_estimated": final_ratio, "final_plan_estimated_to_actual": 1/final_ratio if final_ratio else None,
            "step_durations": data.get("step_durations", []), "feedback": data.get("feedback", []),
            "feedback_count": data.get("motion_feedback_count", 0),
            "qualification_calls": data.get("qualification_calls", 0),
            "backend": backend_evidence(path)}
    return list(unique.values())


def collect(base=BASE):
    base = batch.confined(base)
    manifest_path = base / "overnight/manifest.json"
    manifest = read(manifest_path, {"jobs": []})
    jobs = manifest.get("jobs", [])
    state_dir = batch.confined(manifest.get("state_dir", base / "overnight/batch_state"))
    journal = batch.Journal(state_dir) if state_dir.exists() else None
    records = journal.load() if journal else {}
    planning_jobs = [j for j in jobs if j["phase"] == "plan" and j.get("cohort") == "planning_200"]
    precheck_jobs = [j for j in jobs if j["phase"] == "plan" and j.get("cohort") == "precheck"]
    planning = counts(planning_jobs, records, planning=True)
    planning["families"] = {}
    for family in FAMILIES:
        family_jobs = [j for j in planning_jobs if j.get("family") == family]
        item = counts(family_jobs, records, planning=True)
        item["diagnostics"] = planning_diagnostics(family_jobs, records)
        threshold = 1. if family in {"F1", "F2", "F3"} else .9
        item.update(required=40, threshold=threshold,
                    threshold_passed=item["requested"] == 40 and item["executed"] == 40 and item["passed"]/40 >= threshold)
        planning["families"][family] = item
    planning["thresholds_passed"] = planning["requested"] == 200 and all(x["threshold_passed"] for x in planning["families"].values())
    planning["unsolved"] = [{"id": j["id"], "family": j.get("family"),
        "status": records.get(j["id"], {}).get("status", "pending"),
        "reason": records.get(j["id"], {}).get("reason") or "尚未记录独立规划结论",
        "source": records.get(j["id"], {}).get("output", j.get("output"))}
        for j in planning_jobs if records.get(j["id"], {}).get("status") != "solved" or records.get(j["id"], {}).get("passed") is not True]
    simulations = {}
    for label, selection in (("sim20", 1), ("sim40", None)):
        subset = [j for j in jobs if j["phase"] == "sim" and (selection is None or j.get("selection", 1) == selection)]
        item = counts(subset, records)
        item["execution_wall"] = execution_wall(subset, records)
        measured_ratios, final_ratios = [], []
        for job in subset:
            row = records.get(job["id"], {})
            if row.get("status") == "skipped" or row.get("executed") is False:
                continue
            actual, estimate = row.get("actual_makespan_seconds"), row.get("makespan")
            ratio = (actual/estimate if finite(actual, positive=True) and finite(estimate, positive=True)
                     else row.get("actual_to_estimate_ratio"))
            if finite(ratio, positive=True):
                measured_ratios.append(ratio)
            final_estimate = final_plan_estimate(job, row)
            if finite(actual, positive=True) and finite(final_estimate, positive=True):
                final_ratios.append(actual/final_estimate)
        item["ratios"] = {"actual_to_estimated": distribution(measured_ratios),
                          "estimated_to_actual": distribution(1/value for value in measured_ratios),
                          "actual_to_final_plan_estimated": distribution(final_ratios),
                          "final_plan_estimated_to_actual": distribution(1/value for value in final_ratios)}
        item["families"] = {f: counts([j for j in subset if j.get("family") == f], records) for f in FAMILIES}
        item["outcomes"] = [{"id": j["id"], "status": records.get(j["id"], {}).get("status", "pending"),
            "failure_phase": records.get(j["id"], {}).get("failure_phase"),
            "reason": records.get(j["id"], {}).get("reason"), "source": records.get(j["id"], {}).get("output", j.get("output"))} for j in subset]
        simulations[label] = item
    gates = []
    gate_jobs = [job for job in jobs if job["phase"] == "gates" and job.get("test_name") in batch.LIVE_TESTS]
    for test in batch.LIVE_TESTS:
        requested = [job for job in gate_jobs if job["test_name"] == test]
        matching = [records[job["id"]] for job in requested if job["id"] in records]
        row = next((r for r in matching if real_passed(r) and r.get("tests_run") == 1
                    and r.get("test_skipped") == 0 and r.get("test_failures") == 0 and r.get("test_errors") == 0), None)
        evidence_row = row if row is not None else next(iter(matching), None)
        gates.append({"test": test, "passed": row is not None and len(requested) == 1,
                      "requested": len(requested),
                      "source": evidence_row.get("output") if evidence_row else None, "recorded": len(matching)})
    regressions = []
    for path in sorted(base.glob("overnight_regression_*.json")):
        report = read(path, {})
        for run in report.get("runs", []):
            if run.get("label") in {"default_1", "default_2", "seed0"}:
                regressions.append({key: run.get(key) for key in ("label", "status", "tests_run", "skipped", "total_seconds", "log", "json", "support_tasks")})
    initial_latest = {row["label"]: row for row in regressions}
    extra_out = ROOT / "out" if base == BASE else base / "out"
    extra_regression_paths = ((extra_out / "overnight_post_m3_generator_regression.json", "post_m3_generator"),
                              (extra_out / "overnight_final_regression.json", "final_after_sim"))
    for path, label in extra_regression_paths:
        try:
            report = read(path, {})
        except (ValueError, UnicodeDecodeError):
            report = {"status": "incomplete_artifact", "reason": "附加回归JSON未完整提交；不计为通过"}
        if report:
            regressions.append({**{key: report.get(key) for key in ("status", "tests_run", "skipped", "total_seconds", "support_tasks", "reason")},
                "label": label, "json": path_text(path),
                "log": report.get("log") or path_text(path.with_suffix(".log") if path.with_suffix(".log").is_file() else path)})
    latest = initial_latest
    regression_passed = all(latest.get(label, {}).get("status") == "passed" and latest[label].get("skipped") == 6
                            and latest[label].get("tests_run", 0) > 6
                            for label in ("default_1", "default_2", "seed0"))
    diversity_path = next((p for p in (base / "overnight/generator_diversity.json", base / "generator_diversity.json") if p.is_file()), None)
    diversity = read(diversity_path, {}) if diversity_path else {}
    m3_outcome = read(base / "overnight/m3_outcome.json", {})
    handover, handover_path = summary_evidence(m3_outcome.get("handover"))
    parallel, parallel_path = summary_evidence(m3_outcome.get("parallel"))
    m3_passed = (m3_outcome.get("M3_passed") is True and regression_passed
                 and real_passed(handover) and real_passed(parallel)
                 and parallel.get("motion_task_count") == 18)
    m4 = m4_evidence(read(base / "overnight/m4_outcome.json", {}))
    sim = simulations["sim20"]
    m5_checks = {"planning_200_thresholds": planning["thresholds_passed"],
        "sim20_executed": sim["requested"] == 20 and sim["executed"] == 20
            and all(row["requested"] == 4 and row["executed"] == 4 for row in sim["families"].values()),
        "sim20_success_ge_80_percent": sim["executed"] == 20 and sim["passed"]/20 >= .8,
        "generator_exact_contract": diversity.get("exact_contract_passed") is True}
    runs = real_runs(base)
    phases = defaultdict(float)
    steps, feedback, qualifications = [], [], []
    for run in runs:
        for name, seconds in run["phase_seconds"].items():
            if finite(seconds):
                phases[name] += seconds
        steps.extend({**step, "source": run["source"], "task": run["task"]} for step in run["step_durations"])
        feedback.extend({"source": run["source"], "reason": str(reason)} for reason in run["feedback"])
        qualifications.extend({**record, "source": run["source"]} for record in run["backend"].get("qualifications", []))
    return {"updated_at": batch.utc_now(), "manifest": path_text(manifest_path),
        "journal_sources": [path_text(p) for p in journal.paths()] if journal else [],
        "journal_warnings": journal.warnings if journal else [], "batch_state": read(state_dir / "state.json", {}),
        "planning": planning, "precheck": {**counts(precheck_jobs, records, planning=True),
            "comparisons": precheck_comparisons(precheck_jobs, records)}, "simulations": simulations,
        "gates": {"all_five_passed": all(r["passed"] for r in gates), "tests": gates,
                  "execution_wall": execution_wall(gate_jobs, records)},
        "regressions": regressions, "regressions_passed": regression_passed,
        "generator": {"source": path_text(diversity_path) if diversity_path else None, **diversity},
        "M3": {"evidence_version": "overnight-v1", "status": "passed" if m3_passed else "not_met", "accepted": m3_passed,
            "regression_passed": regression_passed, "pitch_degrees": m3_outcome.get("pitch_degrees"),
            "handover_summary": path_text(handover_path) if handover_path else None,
            "parallel_summary": path_text(parallel_path) if parallel_path else None},
        "M4": {"evidence_version": "overnight-v1", "status": "passed" if m4["passed"] else "not_met", **m4},
        "M5": {"evidence_version": "overnight-v1", "status": "passed" if all(m5_checks.values()) else "not_met",
            "passed": all(m5_checks.values()), "checks": m5_checks},
        "real_runs": runs, "real_run_count": len(runs), "real_attempt_wall_seconds": sum(r["wall_seconds"] for r in runs),
        "real_failure_categories": failure_categories(runs),
        "phase_seconds": dict(phases), "step_durations": steps,
        "ratios": {"actual_to_estimated": distribution(r["actual_to_estimated"] for r in runs),
                   "estimated_to_actual": distribution(r["estimated_to_actual"] for r in runs),
                   "actual_to_final_plan_estimated": distribution(r["actual_to_final_plan_estimated"] for r in runs),
                   "final_plan_estimated_to_actual": distribution(r["final_plan_estimated_to_actual"] for r in runs),
                   "step_actual_to_estimated": distribution(s["actual_seconds"]/s["estimated_seconds"] for s in steps
                       if finite(s.get("actual_seconds"), positive=True) and finite(s.get("estimated_seconds"), positive=True)),
                   "step_estimated_to_actual": distribution(s["estimated_seconds"]/s["actual_seconds"] for s in steps
                       if finite(s.get("actual_seconds"), positive=True) and finite(s.get("estimated_seconds"), positive=True))},
        "feedback_count": sum(r["feedback_count"] for r in runs), "feedback_reasons": feedback,
        "qualification_calls": sum(r["qualification_calls"] for r in runs), "qualifications": qualifications}


def markdown(metrics):
    def number(value):
        return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    def percent(value):
        return "—" if value is None else f"{100*value:.1f}%"
    lines = ["# v1.3 夜间验收证据汇总", "", "更新：" + metrics["updated_at"], "",
        "结论仅来自已提交完成日志及真实输出；pending、跳过、构造见证均不计为通过。", "",
        f"M3：**{metrics['M3']['status']}**；M4：**{metrics['M4']['status']}**；M5：**{metrics['M5']['status']}**。", "",
        "## 200 项独立规划", "", "| 任务族 | 请求 | 执行 | solved | 跳过 | 待处理 | 请求成功率 | 执行成功率 | 阈值 | 观测辅助/尝试 | solved中辅助比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family, row in metrics["planning"]["families"].items():
        lines.append(f"| {family} | {row['requested']} | {row['executed']} | {row['passed']} | {row['skipped']} | {row['pending']} | "
            f"{percent(row['success_rate_requested'])} | {percent(row['success_rate_executed'])} | {percent(row['threshold'])} | {percent(row['auxiliary_rate_executed'])} | {percent(row['auxiliary_rate_passed'])} |")
    lines.append("\n辅助步骤只统计已观测方案；未 solved 不能证明不需要辅助。无 solved 的族，其 solved 中辅助比例为空。")
    lines.extend(("", "逐族结论与预算观测分类：", "",
                  "| 族 | solved | UNKNOWN | budget扩展≥200000 | budget末次搜索≥60s | 中断 | 待处理 | 常见UNKNOWN事实 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"))
    structure_notes = []
    for family, row in metrics["planning"]["families"].items():
        diagnostic = row["diagnostics"]
        reasons = diagnostic["reason_counts"]
        common = "; ".join(f"{fact} ×{count}" for fact, count in list(diagnostic["unknown_facts"].items())[:3]) or "—"
        lines.append(f"| {family} | {reasons.get('solved', 0)} | {reasons.get('unknown', 0)} | {reasons.get('budget_expanded_ge_200000', 0)} | "
                     f"{reasons.get('budget_last_search_ge_60s', 0)} | {reasons.get('interrupted', 0)} | {reasons.get('pending', 0)} | {common} |")
        sizes = diagnostic["structure_status"]
        if sizes:
            structure_notes.append(f"- {family} 已记录输入规模：{min(s['blocks'] for s in sizes)}–{max(s['blocks'] for s in sizes)} 块积木，"
                                   f"{min(s['slots'] for s in sizes)}–{max(s['slots'] for s in sizes)} 个槽位；详细规模×结论见 metrics。")
    if structure_notes:
        lines.extend(("", *structure_notes))
    lines.append("\n扩展/秒数分类只描述最后一次搜索的实测值。pipeline 还可能先做剪枝搜索再回退完整搜索，末次 search_elapsed_seconds 不是整条规划耗时；UNKNOWN 不是不可达或无解证明。")
    precheck = metrics["precheck"]
    lines.extend(("", "预检见证与独立规划事后对照：", "",
        f"请求 {precheck['requested']} 项，独立 solved {precheck['passed']} 项，未完成 {precheck['pending']} 项；构造见证未输入独立搜索，也不代替 solved 或证明最优性。", "",
        "| 原始仿真任务预检 | 结论 | 串行见证估算秒数 | 独立makespan秒数 | 独立/见证 |",
        "|---|---|---:|---:|---:|"))
    for row in precheck["comparisons"]:
        if row["family"] == "fixture":
            lines.append(f"| {row['id']} | {row['status']} | {number(row['witness_estimate_seconds'])} | "
                         f"{number(row['independent_makespan_seconds'])} | {number(row['independent_to_witness_ratio'])} |")
    old_precheck = defaultdict(Counter)
    for row in precheck["comparisons"]:
        if row["family"] != "fixture":
            old_precheck[row["family"]][row["status"]] += 1
    if old_precheck:
        lines.append("\n旧生成样例预检：" + "；".join(f"{family} " + ", ".join(f"{status}={count}" for status, count in statuses.items())
                                                      for family, statuses in sorted(old_precheck.items())) + "。")
    lines.extend(("", "未 solved 的全部任务：", ""))
    lines.extend(f"- `{r['id']}`：{r['status']}；{r['reason']}；证据 `{r['source']}`。" for r in metrics["planning"]["unsolved"])
    if not metrics["planning"]["unsolved"]:
        lines.append("尚无规划请求。" if not metrics["planning"]["requested"] else "无。")
    lines.extend(("", "## 仿真抽样", "", "| 集合 | 请求槽位 | 实际执行 | 通过 | 跳过 | 待处理 | 通过/请求 | 通过/执行 | 编译失败/执行 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for name, row in metrics["simulations"].items():
        lines.append(f"| {name} | {row['requested']} | {row['executed']} | {row['passed']} | {row['skipped']} | {row['pending']} | "
            f"{percent(row['success_rate_requested'])} | {percent(row['success_rate_executed'])} | {percent(row['compilation_failure_rate_executed'])} |")
    lines.extend(("", "实际执行/尝试表示启动完整任务链路，包含规划或编译失败，不等于动力学完成；编译失败口径包含real_planning中的compile_action失败及最终compile，阶段详见CSV；成功必须真实动力学验收passed。", "",
                  "每族分母与每个抽样任务的状态/跳过原因保存在 `overnight/summary_metrics.json`；sim40 为两轮累计，未准备第二轮时不代表已执行 40 项。", "",
                  "抽样中未通过的全部任务（两轮累计，逐 ID 保留原因）：", ""))
    for row in metrics["simulations"]["sim40"]["outcomes"]:
        if row["status"] != "passed":
            lines.append(f"- `{row['id']}`：{row['status']} / {row['failure_phase'] or '未执行'}；{row['reason'] or '尚无完成结论'}；证据 `{row['source']}`。")
    lines.extend(("", "失败、超时与跳过分类（pending 不列为执行失败）：", "",
                  "| 范围 | 失败阶段 | 类别 | 数量 |", "|---|---|---|---:|"))
    categories = {"整夜去重运行": metrics["real_failure_categories"],
                  **{name: row["failure_categories"] for name, row in metrics["simulations"].items()}}
    for scope, entries in categories.items():
        if not entries:
            lines.append(f"| {scope} | — | 无已记录失败/跳过 | 0 |")
        for entry in entries:
            lines.append(f"| {scope} | {entry['failure_phase']} | {entry['category']} | {entry['count']} |")
    lines.extend(("", "独立范围墙钟合计（从完成日志取值；这些范围不再加到整夜去重合计）：", "",
                  "| 范围 | 请求 | 有时长的执行记录 | 已记录墙钟秒数 |", "|---|---:|---:|---:|"))
    for name, row in metrics["simulations"].items():
        wall = row["execution_wall"]
        lines.append(f"| {name} | {row['requested']} | {wall['measured_runs']} | {wall['seconds']:.6f} |")
    wall = metrics["gates"]["execution_wall"]
    lines.append(f"| 精确五项真实门控 | 5 | {wall['measured_runs']} | {wall['seconds']:.6f} |")
    lines.extend(("", "只累计已执行且有时长的任务；缺项、跳过或时长缺失均不表示整组已完成。逐条时长依据和去重排除记录保存在 metrics JSON。", "",
                  "## 里程碑证据", "", f"完整回归三轮：{metrics['regressions_passed']}。"))
    for row in metrics["regressions"]:
        lines.append(f"- {row['label']}：{row['status']}；{row['tests_run']} 项、跳过 {row['skipped']}，总计 {number(row['total_seconds'])} 秒；证据 `{row['log']}`。"
                     + ("原因：" + str(row["reason"]) if row.get("reason") else ""))
    lines.append(f"\nM3：交接 `{metrics['M3']['handover_summary']}`；18 任务目录顶抓 `{metrics['M3']['parallel_summary']}`。")
    for name, row in metrics["M4"]["fixtures"].items():
        lines.append(f"- M4 `{name}`：{row['passed']}；证据 `{row['source']}`。")
    for key in ("invalid_cut_execution", "unknown_qualify_execution"):
        row = metrics["M4"][key]
        lines.append(f"- M4 {key}：{row['passed']}；证据 `{row['source']}`。")
    lines.extend(("", "M5 必要条件：", ""))
    lines.extend(f"- {key}：{value}。" for key, value in metrics["M5"]["checks"].items())
    lines.append(f"\n生成器证据：`{metrics['generator'].get('source')}`；严格合约通过：{metrics['generator'].get('exact_contract_passed', False)}。")
    lines.extend("- " + str(reason) for reason in metrics["generator"].get("exact_contract_failure_reasons", []))
    lines.extend(("", "五项真实门控测试（必须各实际执行一项且无 skip）：", ""))
    lines.extend(f"- `{row['test']}`：{row['passed']}；证据 `{row['source']}`。" for row in metrics["gates"]["tests"])
    lines.extend(("", "## 实测耗时与反馈", "", f"去重端到端尝试 {metrics['real_run_count']} 次，累计墙钟 **{metrics['real_attempt_wall_seconds']:.6f} 秒**，含规划和失败尝试。按 simulator_log 的唯一启动标识去重；无启动标识时按 summary 路径计数，不相加聚合 outcome/worker 引用。", "",
                  "| 比值 | n | min | p25 | median | p75 | max |", "|---|---:|---:|---:|---:|---:|---:|"))
    scopes = {"整夜去重运行": metrics["ratios"], **{name: row["ratios"] for name, row in metrics["simulations"].items()}}
    for scope, ratios in scopes.items():
        for name, row in ratios.items():
            lines.append("| " + scope + " / " + name + " | " + " | ".join(number(row[k]) for k in ("n", "min", "p25", "median", "p75", "max")) + " |")
    lines.extend(("", "actual_to_estimated / estimated_to_actual 保留初始 initial_plan 距离代价口径；actual_to_final_plan_estimated / final_plan_estimated_to_actual 来自最终 solved 的 plan.json，与实际执行路线对应。初始 UNKNOWN 空计划的零 makespan 不参与初始比值。以最终路线的逐步 step_durations 为主要估算偏差证据；分位数采用端点内线性插值。", "", "阶段墙钟累计：", ""))
    lines.extend(f"- {phase}：{seconds:.6f} 秒。" for phase, seconds in sorted(metrics["phase_seconds"].items()))
    lines.append(f"\n真实 INVALID 反馈记录总数 {metrics['feedback_count']}；qualify 调用 {metrics['qualification_calls']}。")
    lines.extend(f"- `{row['source']}`：{row['reason']}" for row in metrics["feedback_reasons"])
    lines.extend(("", "所有逐步耗时、资格结果、运行标识、去重来源、族分母、完成日志路径及当前预算状态见 `overnight/summary_metrics.json`。", "", "证据来源：", "", "- 清单：`" + metrics["manifest"] + "`。"))
    lines.extend("- 完成日志：`" + source + "`。" for source in metrics["journal_sources"])
    lines.extend(f"- 实测运行：`{run['source']}`；{run['task']} / {run['status']}；{run['wall_seconds']:.6f} 秒；PID {run['simulator_pid']}；"
                 f"阶段 {run['failure_phase'] or '完成'}；原因 {run['reason'] or '无'}。" for run in metrics["real_runs"])
    lines.extend(("", "## 发现的问题与建议", ""))
    lines.extend("- " + finding for finding in findings(metrics))
    return "\n".join(lines) + "\n"


def findings(metrics):
    result = []
    failed_fixtures = [name for name, row in metrics["M4"]["fixtures"].items() if not row["passed"]]
    mechanical = next((str(note) for note in metrics["M4"].get("notes", []) if "24mm" in str(note) or "24 mm" in str(note)), None)
    if mechanical:
        result.append("机械约束：" + mechanical + " 建议先明确托板机构或工具感知抓取设计，再恢复这些真实任务；不要增加碰撞豁免或重复无效种子。")
    elif failed_fixtures:
        result.append("M4 尚缺真实通过证据：" + "、".join(failed_fixtures) + "；应依据各目录原始失败证据继续处理，不把规则见证视为动力学通过。")
    empty = [family for family, row in metrics["planning"]["families"].items() if row["requested"] and not row["passed"]]
    if empty:
        result.append("独立规划零 solved 的任务族：" + "、".join(empty) + "。结合上表逐族 UNKNOWN 事实与预算分类诊断；UNKNOWN 应先按需资格确认，不能把未验证能力直接改成 VALID，也不能据此宣称无解。")
    budgets = Counter()
    for row in metrics["planning"]["families"].values():
        budgets.update(row["diagnostics"]["reason_counts"])
    nbudget = sum(count for reason, count in budgets.items() if reason.startswith("budget"))
    if nbudget:
        result.append(f"预算观测：{nbudget} 项 budget，其中末次 expanded≥200000 为 {budgets['budget_expanded_ge_200000']} 项，"
                      f"其余末次搜索≥60秒为 {budgets['budget_last_search_ge_60s']} 项。后续应独立研究规模增长、启发式及可证明的搜索优化；本轮不放宽时间或扩展预算，不把 budget 当作无解。")
    if budgets["interrupted"]:
        result.append(f"{budgets['interrupted']} 项规划记录为 interrupted；保留未提交证据，不能归为 solved 或直接重试后覆盖原记录。")
    if metrics["generator"].get("exact_contract_passed") is not True:
        reasons = metrics["generator"].get("exact_contract_failure_reasons", [])
        result.append("生成器严格合约尚未通过。" + ("；".join(map(str, reasons)) if reasons else "尚缺严格合约通过证据。")
                      + " 建议先解决所要求分布指标与几何可达交集的兼容性，特别核对 F4 每轴标准差；不能只看合成向量标准差或改写阈值冒充通过。")
    sim = metrics["simulations"]["sim20"]
    if sim["requested"] and (sim["executed"] < sim["requested"] or sim["skipped"]):
        result.append(f"首轮抽样请求 {sim['requested']} 槽，实际尝试 {sim['executed']}，跳过 {sim['skipped']}，待处理 {sim['pending']}。"
                      "样本缺额应单列，不能把请求槽位或少数任务族的成功率称作20项泛化通过。")
    steps = metrics["ratios"]["step_actual_to_estimated"]
    if steps["n"]:
        result.append(f"最终路线逐步实际/估算比值中位数为 {steps['median']:.6f}（n={steps['n']}）。"
                      "距离代价没有覆盖全部动力学执行开销；反馈换路会改变初始计划分母，初始 UNKNOWN 空计划不提供有效分母。"
                      "建议用已执行最终路线和逐步实测校准对比，保留原始代价和两套比值，不据少量样本更改核心 cost。")
    return result or ["尚无足够完成证据形成专项建议；继续保留未完成状态。"]


def rebuild(base=BASE):
    base = batch.confined(base)
    metrics = collect(base)
    milestones = read(base / "milestones.json", {})
    old = milestones.get("M3")
    if old is not None and not old.get("evidence_version"):
        history = milestones.setdefault("history", {})
        if not isinstance(history, dict):
            history = milestones["history"] = {"previous": history}
        records = history.setdefault("M3", [])
        if old not in records:
            records.append(deepcopy(old))
    for name in ("M3", "M4", "M5"):
        milestones[name] = deepcopy(metrics[name])
    milestones["overnight_summary"] = {"updated_at": metrics["updated_at"],
        "metrics": path_text(base / "overnight/summary_metrics.json"), "gates_passed": metrics["gates"]["all_five_passed"]}
    batch.atomic_json(base / "overnight/summary_metrics.json", metrics)
    batch.atomic_text(base / "summary.md", markdown(metrics))
    batch.atomic_json(base / "milestones.json", milestones)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="只从事实证据重建中文 summary.md 与里程碑；不运行任何实验")
    parser.add_argument("--rebuild", action="store_true", help="重建报告（默认行为相同）")
    parser.parse_args(argv)
    metrics = rebuild()
    print(json.dumps({"summary": path_text(BASE / "summary.md"),
        "milestones": {key: metrics[key]["status"] for key in ("M3", "M4", "M5")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
