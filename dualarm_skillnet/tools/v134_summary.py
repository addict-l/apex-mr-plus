"""Journal-only v134 CSV comparison/rebuild, preserving every v133 artifact."""
from collections import Counter
import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import v134_batch as batch


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path, issues, label):
    try:
        return old.read_json(path) if Path(path).is_file() else {}
    except (OSError, ValueError) as exc:
        issues.append({"source": str(path), "reason": label+": "+str(exc)})
        return {}


def _committed(path, issues):
    """Read frozen definitions and append-only records, never create a journal."""
    if not Path(path).is_file():
        return [], {}
    try:
        manifest = old.load_manifest(path)
        records = batch.read_records(manifest["state_dir"])
    except (OSError, ValueError) as exc:
        issues.append({"source": str(path), "reason": "manifest_or_journal_invalid: "+str(exc)})
        return [], {}
    selected = {}
    for job in manifest["jobs"]:
        row = records.get(job["id"])
        if row is None:
            continue
        if row.get("definition_sha256") != job["definition_sha256"]:
            issues.append({"id": job["id"], "source": str(path), "reason": "definition_sha_mismatch"})
        else:
            selected[job["id"]] = row
    return manifest["jobs"], selected


def feedback_metrics(backend, plan, summary):
    """Count structured C receipts; legacy text supports direct-exit evidence only."""
    rounds = backend.get("feedback_rounds", [])
    rounds = [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []
    motions = [m for r in rounds for m in r.get("motion", []) if isinstance(m, dict)]
    cuts = [m for m in motions if m.get("cut") is True]
    upgrades = [m for m in motions if m.get("escalated") is True
        and m.get("initial_context", {}).get("ik_seed_limit") == 8
        and m.get("escalation_context", {}).get("ik_seed_limit") == 24
        and not m.get("escalation_context", {}).get("escalation_skipped")]
    reasons = Counter(m.get("escalation_context", m.get("initial_context", {})).get("classification", "unclassified")
                      for m in cuts)
    text = plan.get("feedback", summary.get("feedback", []))
    text = [s for s in text if isinstance(s, str)] if isinstance(text, list) else []
    explicit_avoidance = any("重新搜索" in s and ("已规避" in s or "已禁止" in s) for s in text)
    validations = backend.get("validations", [])
    last = validations[-1] if validations and isinstance(validations[-1], dict) else {}
    direct = (not rounds and summary.get("real_planning_status") == "unknown"
        and last.get("status") == "UNKNOWN" and plan.get("status") == "unknown"
        and plan.get("search", {}).get("rejected", {}).get("cut") == 0
        and any("运动验证未知" in s for s in text) and not explicit_avoidance)
    terminal = rounds[-1].get("terminal_status") if rounds else summary.get("real_planning_status")
    return {"structured_rounds": len(rounds), "search_rounds": sum("search_status" in r for r in rounds),
        "cut_events": len(cuts), "unique_state_action_cuts": len({_json_key([m.get('state'), m.get('action')]) for m in cuts}),
        "actual_8_to_24_escalations": len(upgrades), "escalation_requests": sum(m.get("escalated") is True for m in motions),
        "downgraded_capabilities": sorted({f for m in cuts for f in m.get("downgraded_capabilities", []) if isinstance(f, str)}),
        "cut_reasons": dict(reasons), "terminal_status": terminal,
        "initial_statuses": dict(Counter(str(m.get("initial_status", "missing")) for m in motions)),
        "final_statuses": dict(Counter(str(m.get("status", "missing")) for m in motions)),
        "round_receipts": rounds,
        "direct_exit_evidence_status": ("proven_direct" if direct else "structured_rounds" if rounds else
            "insufficient" if summary.get("real_planning_status") == "unknown" and not last else "not_proven_direct"),
        "unknown_direct_terminal": bool(direct), "unknown_after_cut": terminal == "unknown" and bool(cuts),
        "auxiliary_explicit_avoidance_text": explicit_avoidance,
        "scope": "cut/escalation/rounds only from motion_backend.feedback_rounds; qualification_calls and runner counters excluded"}


def _json_key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _csv(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _json_key(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    old.atomic_text(path, stream.getvalue())


def failure_category(summary, backend):
    """Separate terminal IK/RRT candidate rejection from executed failures."""
    if summary.get("passed") is True and summary.get("status") == "passed":
        return "passed"
    phase = summary.get("failure_phase") or "unclassified"
    reason = summary.get("reason")
    reason = reason if isinstance(reason, str) else _json_key(reason)
    if phase == "real_planning":
        records = backend.get("validations", [])
        last = records[-1] if records and isinstance(records[-1], dict) else {}
        evidence = last.get("failure_evidence")
        while isinstance(evidence, dict) and "recovery_evidence" in evidence:
            evidence = evidence["recovery_evidence"]
        if isinstance(evidence, dict) and ("attempts" in evidence or "search_stages" in evidence):
            return "real_planning:rrt_unknown"
        if isinstance(evidence, list) and evidence:
            return "real_planning:ik_exhausted"
        kind = last.get("classification", {}).get("kind")
        if kind == "rrt_exhausted":
            return "real_planning:rrt_unknown"
        if kind in {"ik_collision", "ik_branch_discontinuity", "sampling_unknown"}:
            return "real_planning:"+kind
        return "real_planning:unclassified"
    if phase in {"acceptance", "execute", "validation"} and ("tracking" in reason.lower() or "跟踪" in reason):
        return "execution:tracking_failure"
    return str(phase)+":unclassified"


def native_case(job, row, root, issues):
    result = {"id": job.get("id"), "family": job.get("family"), "committed": row is not None,
              "pending": row is None, "real_passed": False, "source": None, "failure_category": "pending"}
    if row is None:
        return result
    result.update(status=row.get("status"), reason=row.get("reason"), failure_phase=row.get("failure_phase"))
    directory = row.get("evidence_directory")
    if not directory:
        result.update(failure_category="evidence_missing", evidence_issues=["native_directory_missing"])
        return result
    directory = Path(directory).resolve()
    output = Path(job["output"]).resolve()
    if not directory.is_relative_to(Path(root).resolve()) or not directory.is_relative_to(output):
        result.update(failure_category="evidence_invalid", evidence_issues=["native_directory_outside_job_output"])
        return result
    summary_path = directory / "summary.json"
    summary = _read(summary_path, issues, "native_summary_invalid")
    if not summary:
        result.update(failure_category="evidence_missing", evidence_issues=["summary_missing"])
        return result
    plan = _read(directory / "plan.json", issues, "final_plan_invalid")
    backend = _read(directory / "motion_backend.json", issues, "backend_evidence_invalid")
    evidence_issues = []
    if summary.get("task_sha256") != job.get("task_sha256"):
        evidence_issues.append("native_task_sha_mismatch_or_missing")
    if not summary.get("task_file") or Path(summary["task_file"]).resolve() != Path(job["task"]).resolve():
        evidence_issues.append("native_task_path_mismatch_or_missing")
    actual, pid = summary.get("actual_makespan_seconds"), summary.get("simulator_pid")
    passed = (not evidence_issues and row.get("passed") is True and row.get("status") == "passed"
        and summary.get("passed") is True and summary.get("status") == "passed"
        and row.get("simulator_closed") is True and summary.get("simulator_closed") is True
        and isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 and row.get("simulator_pid") == pid
        and _number(actual) and actual > 0 and row.get("actual_makespan_seconds") == actual)
    estimate = plan.get("makespan")
    ratio = estimate/actual if passed and plan.get("status") == "solved" and _number(estimate) and estimate > 0 else None
    result.update(source=str(summary_path), status=summary.get("status"), reason=summary.get("reason"),
        failure_phase=summary.get("failure_phase"), failure_category=failure_category(summary, backend),
        real_passed=bool(passed), final_plan_to_actual=ratio, evidence_issues=evidence_issues,
        feedback_metrics=feedback_metrics(backend, plan, summary), phase_seconds=summary.get("phase_seconds"),
        task_sha256=summary.get("task_sha256"), attempted=isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
        motion_backend_source=str(directory / "motion_backend.json"), final_plan_source=str(directory / "plan.json"))
    result["real_planning_status"] = summary.get("real_planning_status")
    if result["failure_category"] == "passed" and not passed:
        result["failure_category"] = "evidence:uncredited_native_pass"
    if not (directory / "motion_backend.json").is_file():
        evidence_issues.append("motion_backend_missing")
    if not (directory / "plan.json").is_file():
        evidence_issues.append("final_plan_missing")
    return result


def sim20_comparison(base, project, issues):
    previous = project / "results/sim_v1_3/v133"
    old_path, new_path = previous / "manifests/sim20.json", base / "manifests/sim20.json"
    old_jobs, old_rows = _committed(old_path, issues)
    new_jobs, new_rows = _committed(new_path, issues)
    sources = {}
    for job in new_jobs:
        sources.setdefault(job.get("source_job_id"), []).append(job)
    manifest_sha = old.sha256(old_path) if old_path.is_file() else None
    source_cohort_valid = len(old_jobs) == 20 and Counter(j["family"] for j in old_jobs) == Counter({f: 4 for f in batch.FAMILIES})
    current_cohort_valid = len(new_jobs) == 20 and Counter(j["family"] for j in new_jobs) == Counter({f: 4 for f in batch.FAMILIES})
    cases = []
    for old_job in old_jobs:
        candidates = sources.get(old_job["id"], [])
        job = candidates[0] if len(candidates) == 1 else None
        faults = []
        if not source_cohort_valid:
            faults.append("v133_manifest_cohort_invalid")
        if new_jobs and not current_cohort_valid:
            faults.append("v134_manifest_cohort_invalid")
        if len(candidates) != 1:
            faults.append("v134_job_missing" if not candidates else "duplicate_source_job_id")
        if job:
            for key, expected in [("source_task_sha256", old_job["task_sha256"]), ("task_sha256", old_job["task_sha256"]), ("source_manifest_sha256", manifest_sha),
                                  ("family", old_job["family"])]:
                if job.get(key) != expected:
                    faults.append(key+"_mismatch")
        before = native_case(old_job, old_rows.get(old_job["id"]), previous, issues)
        after = (native_case(job, new_rows.get(job["id"]), base, issues) if job else
                 {"id": None, "pending": True, "committed": False, "real_passed": False, "failure_category": "pending"})
        cases.append({"source_job_id": old_job["id"], "source_task_sha256": old_job["task_sha256"],
            "source_task": old_job["task"], "family": old_job["family"], "v134_job_id": job["id"] if job else None,
            "source_chain_valid": not faults, "source_chain_issues": faults, "v133": before, "v134": after})
    old_ids = {j["id"] for j in old_jobs}
    unmatched = [{"id": j["id"], "source_job_id": j.get("source_job_id"), "reason": "source_job_not_in_v133_manifest"}
                 for j in new_jobs if j.get("source_job_id") not in old_ids]
    metrics = {}
    for version in ("v133", "v134"):
        included = [c[version] for c in cases if source_cohort_valid and (version == "v133" or c["source_chain_valid"])]
        passed = sum(r["real_passed"] for r in included)
        metrics[version] = {"denominator": 20, "real_passed": passed, "rate": passed/20,
            "pending": 20-sum(r["committed"] for r in included),
            "failure_categories": dict(Counter(r["failure_category"] for r in included if r["committed"] and not r["real_passed"])),
            "unknown_direct_terminal": sum(r.get("feedback_metrics", {}).get("unknown_direct_terminal", False) for r in included),
            "unknown_after_cut": sum(r.get("feedback_metrics", {}).get("unknown_after_cut", False) for r in included),
            "cut_events": sum(r.get("feedback_metrics", {}).get("cut_events", 0) for r in included),
            "actual_8_to_24_escalations": sum(r.get("feedback_metrics", {}).get("actual_8_to_24_escalations", 0) for r in included),
            "structured_rounds": sum(r.get("feedback_metrics", {}).get("structured_rounds", 0) for r in included),
            "search_rounds": sum(r.get("feedback_metrics", {}).get("search_rounds", 0) for r in included),
            "cut_reasons": dict(sum((Counter(r.get("feedback_metrics", {}).get("cut_reasons", {})) for r in included), Counter()))}
        metrics[version]["direct_exit_evidence_missing"] = sum(r.get("feedback_metrics", {}).get("direct_exit_evidence_status") == "insufficient" for r in included)
    return {"denominator": 20, "v133_manifest": str(old_path), "v134_manifest": str(new_path),
        "source_manifest_slots": len(old_jobs), "v134_manifest_slots": len(new_jobs),
        "source_cohort_valid": source_cohort_valid, "current_cohort_valid": current_cohort_valid,
        "source_chain_invalid": sum(not c["source_chain_valid"] for c in cases), "unmatched_current_jobs": unmatched,
        "cases": cases, **metrics,
        "scope": "same original20 IDs and SHA; missing/failing samples kept; walls3 excluded; no resampling"}


def rebuild(*, base=batch.BASE, project=ROOT):
    base, project = batch.version_base(base, project)
    stages, planning, native, invalid_records = {}, [], {}, []
    for stage in ("motion_gate", "qualify5", "planning200", "sim20", "walls3", "gates"):
        path = base / "manifests" / (stage + ".json")
        if not path.exists():
            continue
        manifest = old.load_manifest(path)
        records = old.Journal(manifest["state_dir"]).load()
        selected = []
        for job in manifest["jobs"]:
            row = records.get(job["id"])
            if row is None:
                continue
            if row.get("definition_sha256") != job["definition_sha256"]:
                invalid_records.append({"stage": stage, "id": job["id"], "reason": "definition_sha_mismatch"})
                continue
            selected.append({**job, **row, "task_sha256": job.get("task_sha256"),
                "phase": job["phase"], "family": job.get("family", "fixture"),
                "index": job.get("index"), "d_requalified": job.get("d_requalified", False)})
        stages[stage] = {"requested": len(manifest["jobs"]), "committed": len(selected),
                         "statuses": dict(Counter(r.get("status") for r in selected))}
        if stage == "planning200":
            planning = selected
        elif stage in {"sim20", "walls3", "gates"}:
            native[stage] = selected
    source = project / "results/sim_v1_3/v133"
    old_selection = old.read_json(source / "experiment_selection.json") if (source / "experiment_selection.json").exists() else {}
    old_stage = old_selection.get("stages", {}).get("planning200", "planning200")
    old_manifest = source / "manifests" / (old_stage + ".json")
    historical = {}
    historical_jobs = {}
    if old_manifest.exists():
        jobs, rows = _committed(old_manifest, invalid_records)
        historical_jobs = {(j["family"], j["index"]): j for j in jobs if j.get("table") == "qualified"}
        historical = {key: rows.get(j["id"], {}) for key, j in historical_jobs.items()}
    comparisons = []
    for row in planning:
        previous = historical.get((row["family"], row["index"]), {})
        comparisons.append({**row, "v133_status": previous.get("status"), "v133_bound": previous.get("bound"),
            "v133_expanded": previous.get("expanded"), "v133_elapsed_seconds": previous.get("elapsed_seconds"),
            "v133_source_job_id": historical_jobs.get((row["family"], row["index"]), {}).get("id"),
            "v133_task_sha256": historical_jobs.get((row["family"], row["index"]), {}).get("task_sha256")})
    columns = ("id", "family", "index", "status", "passed", "makespan", "bound", "first_solution_seconds",
        "first_solution_expanded", "expanded", "elapsed_seconds", "search_elapsed_seconds", "physical_infeasible",
        "physical_infeasible_goal_proof", "unknown", "reason", "v133_status", "v133_bound", "v133_expanded", "v133_elapsed_seconds", "output",
        "v133_source_job_id", "task_sha256", "d_requalified")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in comparisons:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    old.atomic_text(base / "planning_200.csv", stream.getvalue())
    old.atomic_json(base / "planning_200.json", comparisons)
    families = {}
    rejected = []
    for family in batch.FAMILIES:
        rows = [r for r in planning if r["family"] == family]
        infeasible = [r for r in rows if r.get("d_requalified") is True and r.get("physical_infeasible") is True
                      and r.get("status") == "no_plan" and r.get("physical_infeasible_goal_proof")]
        rejected += infeasible
        solved = sum(r.get("status") == "solved" for r in rows)
        denominator = 40-len(infeasible)
        families[family] = {"committed": len(rows), "solved": solved, "original_denominator": 40,
            "explicit_physical_rejected": len(infeasible), "feasible_denominator": denominator,
            "raw_rate": solved/40, "feasible_rate": solved/denominator if denominator else None}
        previous = [r for (f, _), r in historical.items() if f == family and r]
        families[family].update(v133_committed=len(previous),
            v133_solved=sum(r.get("status") == "solved" for r in previous),
            v133_raw_rate=sum(r.get("status") == "solved" for r in previous)/40,
            status_transitions=dict(Counter(str(r.get("v133_status", "missing"))+"→"+str(r.get("status"))
                for r in comparisons if r["family"] == family)))
    slow_early = [{"id": r["id"], "status": r.get("status"), "seconds": r.get("elapsed_seconds")} for r in planning
                  if r.get("status") in {"unknown", "no_plan"} and (not isinstance(r.get("elapsed_seconds"), (int, float)) or r["elapsed_seconds"] > 5)]
    late_first = [{"id": r["id"], "seconds": r.get("first_solution_seconds")} for r in planning
                  if r.get("status") == "solved" and (not isinstance(r.get("first_solution_seconds"), (int, float)) or r["first_solution_seconds"] > 15)]
    native_metrics, ratios = {}, []
    for stage, rows in native.items():
        details = []
        for row in rows:
            actual = row.get("actual_makespan_seconds")
            pid = row.get("simulator_pid")
            passed = (row.get("passed") is True and row.get("status") == "passed" and row.get("simulator_closed") is True
                and isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
                and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and actual > 0)
            if stage == "gates":
                passed = passed and row.get("tests_run") == 1 and all(row.get(k) == 0 for k in ("test_skipped", "test_failures", "test_errors"))
            plan_path = Path(row.get("evidence_directory", "")) / "plan.json"
            plan = old.read_json(plan_path) if row.get("evidence_directory") and plan_path.is_file() else {}
            estimate = plan.get("makespan")
            ratio = estimate/actual if passed and isinstance(estimate, (int, float)) and math.isfinite(estimate) and estimate > 0 else None
            if stage == "sim20" and ratio is not None:
                ratios.append(ratio)
            details.append({"id": row["id"], "status": row.get("status"), "real_passed": bool(passed),
                "reason": row.get("reason"), "failure_phase": row.get("failure_phase"), "feedback": row.get("feedback"),
                "final_plan_to_actual": ratio, "phase_seconds": row.get("phase_seconds"), "step_durations": row.get("step_durations")})
        native_metrics[stage] = {"requested": stages[stage]["requested"], "committed": len(rows),
            "real_passed": sum(d["real_passed"] for d in details), "cases": details}
    comparison = sim20_comparison(base, project, invalid_records)
    # The native summary must agree with its committed row. Source-chain-invalid
    # executions remain visible but cannot fill one of the fixed twenty slots.
    if "sim20" in native_metrics:
        native_metrics["sim20"].update(real_passed=comparison["v134"]["real_passed"],
            cases=[{**c["v134"], "source_job_id": c["source_job_id"],
                    "real_passed": c["source_chain_valid"] and c["v134"]["real_passed"],
                    "source_chain_issues": c["source_chain_issues"]} for c in comparison["cases"]])
    if "walls3" in native_metrics:
        wall_jobs, wall_rows = _committed(base / "manifests/walls3.json", invalid_records)
        details = [native_case(j, wall_rows.get(j["id"]), base, invalid_records) for j in wall_jobs]
        native_metrics["walls3"].update(real_passed=sum(r["real_passed"] for r in details), cases=details)
    ratios = [c["v134"]["final_plan_to_actual"] for c in comparison["cases"]
              if c["source_chain_valid"] and c["v134"].get("final_plan_to_actual") is not None]
    old.atomic_json(base / "sim20_comparison.json", comparison)
    flat = [{"source_job_id": c["source_job_id"], "family": c["family"], "source_task": c["source_task"],
        "source_task_sha256": c["source_task_sha256"], "v134_job_id": c["v134_job_id"],
        "source_chain_valid": c["source_chain_valid"], "source_chain_issues": c["source_chain_issues"],
        **{version+"_"+key: c[version].get(key) for version in ("v133", "v134")
           for key in ("status", "real_passed", "failure_phase", "failure_category", "reason", "source", "feedback_metrics", "final_plan_to_actual")}}
        for c in comparison["cases"]]
    _csv(base / "sim20_comparison.csv", flat,
         ("source_job_id", "family", "source_task", "source_task_sha256", "v134_job_id", "source_chain_valid", "source_chain_issues",
          *[version+"_"+key for version in ("v133", "v134") for key in ("status", "real_passed", "failure_phase", "failure_category", "reason", "source", "feedback_metrics", "final_plan_to_actual")]))
    failures = [{"version": version, "category": category, "count": count, "denominator": 20}
                for version in ("v133", "v134") for category, count in comparison[version]["failure_categories"].items()]
    _csv(base / "sim20_failure_categories.csv", failures, ("version", "category", "count", "denominator"))
    value = {"experiment_namespace": batch.NAMESPACE, "updated_at": old.utc_now(), "stages": stages,
        "planning_families": families, "explicit_physical_rejections": rejected,
        "unknown_no_plan_over_five_seconds_or_missing": slow_early, "solved_first_solution_over_fifteen_or_missing": late_first,
        "native": native, "native_metrics": native_metrics, "invalid_records": invalid_records,
        "sim20_comparison": comparison,
        "sim20_final_plan_to_actual": {"n": len(ratios), "median": median(ratios) if ratios else None},
        "scope": "只从当前manifest+ID/definition/journal重建；v133仅作只读同任务IK对照；固定20不重抽、另列3墙；不改标定"}
    old.atomic_json(base / "summary_metrics.json", value)
    lines = ["# v1.3.4 批次比较", "", f"更新：{value['updated_at']}。命名空间 `{batch.NAMESPACE}`。", "",
        "|族|已提交|solved|原分母|明确物理拒绝|可行分母|可行求解率|", "|---|---:|---:|---:|---:|---:|---:|"]
    for family, row in families.items():
        lines.append(f"|{family}|{row['committed']}|{row['solved']}|40|{row['explicit_physical_rejected']}|{row['feasible_denominator']}|{row['feasible_rate']}|")
    lines += ["", "## 同任务 IK 规划：v133 → v134", "",
        "|族|v133 solved/40|v134 solved/40|v134明确物理拒绝|状态转移|", "|---|---:|---:|---:|---|"]
    for family, row in families.items():
        lines.append(f"|{family}|{row['v133_solved']}/40|{row['solved']}/40|{row['explicit_physical_rejected']}|{row['status_transitions']}|")
    lines += ["", "物理不可行不静默删除：仅D五任务有明确目标reach全臂INVALID证明且搜索no_plan时，从可行分母单列扣除。所有其它失败仍保留。", "",
        f"UNKNOWN/no_plan超5秒或缺耗时：{len(slow_early)}；solved首解超15秒或缺耗时：{len(late_first)}。", "",
        "|阶段|请求|已提交|状态|", "|---|---:|---:|---|"]
    for stage, row in stages.items():
        lines.append(f"|{stage}|{row['requested']}|{row['committed']}|{row['statuses']}|")
    for row in rejected:
        lines.append(f"\n- 已正确拒绝 `{row['id']}`：{row['physical_infeasible_goal_proof']}，{row.get('elapsed_seconds')}秒。")
    lines += ["", "逐任务v133/当前状态、bound、扩展、耗时及两项首解字段见planning_200.csv；缺失数据保持空值。真实阶段逐任务CSV由状态journal重建，固定样本/墙样本分开记录。"]
    lines += ["", "## 固定同一组20项真实任务：v133 → v134", "",
        "按原manifest source_job_id、原任务SHA及原清单SHA配对；缺失和失败不抽样补齐，固定分母20。",
        "", "|版本|真实通过/20|待证据|UNKNOWN直接终止|cut后UNKNOWN|结构化cut|实际8→24升级|反馈轮次|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for version in ("v133", "v134"):
        r = comparison[version]
        lines.append(f"|{version}|{r['real_passed']}/20|{r['pending']}|{r['unknown_direct_terminal']}|{r['unknown_after_cut']}|{r['cut_events']}|{r['actual_8_to_24_escalations']}|{r['structured_rounds']}|")
    for version in ("v133", "v134"):
        r = comparison[version]
        lines.append(f"\n{version} 结构化 cut 原因：{r['cut_reasons']}；直接终止证据不足 {r['direct_exit_evidence_missing']} 项。")
        lines.append(f"\n{version} 结构化cut原因：{r['cut_reasons']}。\n")
    lines += ["", f"源链异常 {comparison['source_chain_invalid']} 项；无法映射的新任务 {len(comparison['unmatched_current_jobs'])} 项。",
        "逐样本状态、失败原因及native链接：sim20_comparison.csv / sim20_comparison.json。",
        "", "|版本|直接终止分类|数量（分母20）|", "|---|---|---:|"]
    for r in failures:
        lines.append(f"|{r['version']}|{r['category']}|{r['count']}|")
    lines += ["", "IK/RRT 候选碰撞不等于执行碰撞，有限种子UNKNOWN不等于绝对不可达。旧15项失败应按终端证据区分IK耗尽、RRT未知和真实跟踪失败。",
        "cut/升级/轮次只取 motion_backend.json.feedback_rounds；runner motion_feedback_count 与全量预资格调用不当作cut。旧报告无结构化轮次时不补造次数，plan.feedback只作辅助。",
        "", f"仅成功且源链有效样本的最终plan.makespan/actual：n={len(ratios)}，median={median(ratios) if ratios else None}。",
        "", "## walls3 单列（不补入固定20分母）", "",
        "|任务|提交|真实通过|失败阶段|分类|证据|", "|---|---|---|---|---|---|"]
    for row in native_metrics.get("walls3", {}).get("cases", []):
        lines.append(f"|{row['id']}|{row.get('committed')}|{row['real_passed']}|{row.get('failure_phase')}|{row.get('failure_category')}|{row.get('source')}|")
    if "walls3" not in native_metrics:
        lines.append("|固定3项|0|0|pending|pending|清单尚未准备|")
    old.atomic_text(base / "summary.md", "\n".join(lines) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅重建v134比较CSV与summary，不运行实验")
    parser.add_argument("--rebuild", action="store_true")
    parser.parse_args(argv)
    value = rebuild()
    print(json.dumps(value["stages"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
