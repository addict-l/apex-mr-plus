"""Read-only reconstruction of the fixed motion cohort and its evidence.

Only this attempt's reports are written; --publish additionally updates the
motion_v135 overview. Importing/running this tool never launches work, creates
journals, updates milestones, or repairs historical evidence.

Optional --evidence JSON maps full_suite/original11/optimality/archive/
environment/selected_cases to proof/result JSON paths (or {path, code_sha256}).
Missing, unstructured or differently versioned proofs remain pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STAGES = {"planning200": ("planning200", 200), "fixed20": ("sim20", 20),
          "walls3": ("walls3", 3), "gates": ("gates", 5)}
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
GATE_TASKS = {"test_s1_parallel_place": "s1_parallel_place", "test_s1_reach_handover": "s1_reach_handover",
              "test_s2_tower": "sim/s2_tower_sim", "test_s3a_cantilever": "sim/s3a_cantilever_sim",
              "test_s3a_platform": "sim/s3a_platform_sim"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, issues=None):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        if issues is not None:
            issues.append({"path": str(path), "reason": "invalid_json: "+str(exc)})
        return {}


def _last_motion_boundary(directory):
    """Read the last complete event, preserving any interrupted final line."""
    path = directory/"motion_live.jsonl"
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position, data = stream.tell(), b""
        while position > 0 and data.count(b"\n") < 2:
            size = min(position, 64*1024)
            position -= size
            stream.seek(position)
            data = stream.read(size)+data
    for line in reversed(data.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict) and isinstance(row.get("event"), dict):
            return {**row, "source": str(path), "partial_boundary_only": True}
    return {}


def _latest_native_progress(directory, issues):
    paths = [directory/name for name in ("run_progress.json", "executor_progress.json", "motion_progress.json")
             if (directory/name).is_file()]
    for path in sorted(paths, key=lambda value: value.stat().st_mtime_ns, reverse=True):
        data = read_json(path, issues)
        if isinstance(data, dict) and data:
            return data, str(path)
    return {}, None


def _path(value, project):
    return (Path(project)/value).resolve()


def _task(job, project):
    path = job.get("task")
    if not path and job.get("test_name"):
        name = GATE_TASKS.get(job["test_name"].rsplit(".", 1)[-1])
        path = str(Path(project)/"tasks"/(name+".toml")) if name else None
    return _path(path, project) if path else None


def _identity(job, project):
    task = _task(job, project)
    digest = job.get("prepared_task_sha256", job.get("task_sha256", job.get("source_task_sha256")))
    if digest is None and task is not None and task.is_file():
        digest = sha256(task)
    return job.get("source_job_id", job.get("id")), digest or job.get("test_name")


def _normalized_job(job, project):
    result = deepcopy(job)
    result["timeout_seconds"] = float(result.get("timeout_seconds", 1200.))
    result["output"] = str(_path(result["output"], project))
    if result.get("task"):
        task = _task(result, project)
        result["task"] = str(task)
        if task.is_file():
            result["task_sha256"] = sha256(task)
    phase = result.get("phase")
    result["family"] = str(result.get("family", "fixture"))
    result["cohort"] = str(result.get("cohort", "planning_200" if phase == "plan" else phase))
    result["selection"] = int(result.get("selection", 1))
    result.pop("definition_sha256", None)
    return result


def read_journal(directory, issues):
    """Read committed lines without Journal's mkdir/repair side effects."""
    directory, rows = Path(directory), {}
    paths = ([directory/"completed.jsonl"] if (directory/"completed.jsonl").is_file() else [])
    paths += sorted(directory.glob("completed.recovery_*.jsonl")) if directory.is_dir() else []
    for path in paths:
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines, 1):
            if index == len(lines) and not line.endswith(b"\n"):
                issues.append({"path": str(path), "line": index, "reason": "uncommitted_tail_preserved"})
                continue
            try:
                row = json.loads(line)
                key = row["id"]
            except (ValueError, KeyError, TypeError):
                issues.append({"path": str(path), "line": index, "reason": "invalid_journal_record"})
                continue
            if key in rows and rows[key] != row:
                issues.append({"path": str(path), "id": key, "reason": "conflicting_committed_id"})
                rows[key] = None
            elif key not in rows:
                rows[key] = row
    return [row for row in rows.values() if row is not None]


def _selected(manifest, project, attempt_root, stage, issues):
    rows = read_journal(_path(manifest.get("state_dir", str(attempt_root/"states"/stage)), project), issues)
    selected, jobs = {}, manifest.get("jobs", [])
    for job in jobs:
        logical = job["id"]
        candidates = [row for row in rows if row.get("logical_job_id") == logical or row.get("id") == logical
                      or re.fullmatch(re.escape(logical)+r"-resume[1-9][0-9]*", str(row.get("id", "")))]
        completed = [row for row in candidates if row.get("status") != "interrupted"]
        if len(completed) > 1:
            issues.append({"id": logical, "reason": "multiple_formal_commits"})
            continue
        if completed:
            row = completed[0]
            actual_job = job
            if row["id"] != logical:
                recovery = read_json(attempt_root/"recovery"/stage/(row["id"]+".json"), issues)
                actual_job = recovery.get("job", {})
                if (recovery.get("logical_job_id") != logical
                        or _identity(actual_job, project) != _identity(job, project)):
                    issues.append({"id": logical, "reason": "unverified_recovery_definition"})
                    continue
            expected = fingerprint(_normalized_job(actual_job, project))
            faults = []
            if row.get("definition_sha256") != expected:
                faults.append("definition_sha_mismatch_or_missing")
            selected[logical] = (row, actual_job, faults)
    return selected, rows


def feedback_metrics(backend):
    rounds = [row for row in backend.get("feedback_rounds", []) if isinstance(row, dict)]
    motions = [row for group in rounds for row in group.get("motion", []) if isinstance(row, dict)]
    cuts = [row for row in motions if row.get("cut") is True]
    validations = [row for row in backend.get("validations", []) if isinstance(row, dict)]
    references = [row for row in backend.get("reference_validations", []) if isinstance(row, dict)]
    return {"rounds": len(rounds), "cuts": len(cuts),
        "cut_reasons": dict(Counter(row.get("escalation_context", row.get("initial_context", {})).get(
            "classification", "unclassified") for row in cuts)),
        "unknown_validations": sum(row.get("status") == "UNKNOWN" for row in validations),
        "candidate_deduplicated": sum(row.get("deduplicated") is True for row in validations),
        "candidate_attempts": len(backend.get("candidate_attempts", [])),
        "geometry_contexts": len(backend.get("geometry_contexts", {})),
        "global_downgrades": sum(len(row.get("downgraded_capabilities", [])) for row in motions),
        "temporary_reopen_events": sum(len(row.get("reopened_cuts", [])) for row in rounds),
        "terminal_reference_failures": sum(row.get("status") != "VALID" for row in references),
        "reference_failure_reasons": dict(Counter(row.get("classification", "unclassified")
            for row in references if row.get("status") != "VALID"))}


def failure_category(row, summary, backend, progress):
    status = row.get("status", "pending")
    phase = summary.get("failure_phase") or row.get("failure_phase") or progress.get("phase") or "unclassified"
    if status == "timeout":
        observed = progress.get("phase", phase) if phase == "task_wall_timeout" else phase
        return "timeout:"+str(observed)
    references = [r for r in backend.get("reference_validations", []) if r.get("status") == "UNKNOWN"]
    if phase == "real_planning" and references:
        return "final_reference:"+str(references[-1].get("classification", "unknown"))
    if phase in {"real_planning", "compile"}:
        records = [r for r in backend.get("validations", []) if r.get("status") in {"UNKNOWN", "INVALID"}]
        return str(phase)+":"+str(records[-1].get("classification", {}).get("kind", "unknown") if records else "unknown")
    reason = json.dumps(summary.get("reason", row.get("reason")), ensure_ascii=False)
    if phase in {"execution", "execute", "acceptance", "validation"} and ("tracking" in reason.lower() or "跟踪" in reason):
        return "execution:tracking_failure"
    return str(phase)+":"+str(summary.get("real_planning_status", status))


def _native(job, selected, project, allowed_root, issues, *, strict=True):
    result = {"committed": selected is not None, "status": "pending", "real_passed": False,
              "evidence_issues": [], "failure_category": "pending", "feedback": {},
              "native_evidence_contract": "motion_v135_complete_native_v1" if strict else "v134_archived_contract"}
    if selected is None:
        return result
    row, actual_job, faults = selected
    result.update(status=row.get("status", "missing"), reason=row.get("reason"), evidence_issues=list(faults))
    output = _path(actual_job.get("output", job["output"]), project)
    directory = _path(row.get("evidence_directory") or str(output/"simulation"/(_task(job, project).stem.removesuffix(".qualified")
                       if _task(job, project) else "unknown_task")), project)
    if not directory.is_relative_to(Path(allowed_root).resolve()) or not directory.is_relative_to(output):
        result["evidence_issues"].append("native_directory_outside_job_output")
        result["failure_category"] = "evidence:invalid_directory"
        return result
    summary = read_json(directory/"summary.json", issues)
    progress, progress_source = _latest_native_progress(directory, issues)
    boundary = _last_motion_boundary(directory)
    outer_phase = str(progress.get("phase", "")).split("/", 1)[0]
    if (outer_phase in {"real_planning", "prequalification", "compile"}
            and number(boundary.get("elapsed_seconds")) and number(progress.get("elapsed_seconds"))
            and boundary["elapsed_seconds"] > progress["elapsed_seconds"]):
        # The JSONL event is fsynced before its atomic pointer is replaced.
        # A hard kill between those writes must retain the newer boundary.
        # Execution checkpoints use a different elapsed-time origin, so never
        # compare their seconds with this runner-clock journal.
        event = boundary["event"]
        progress = {**progress, "phase": outer_phase+"/"+str(event.get("query", "motion"))
                    +":"+str(event.get("phase", "event")), "event": event,
                    "elapsed_seconds": boundary["elapsed_seconds"]}
        progress_source = boundary["source"]
    if not boundary and isinstance(progress.get("event"), dict):
        boundary = {"event": progress["event"], "source": progress_source, "partial_boundary_only": True}
    backend = read_json(directory/"motion_backend.json", issues)
    plan = read_json(directory/"plan.json", issues)
    execution = read_json(directory/"execution.json", issues)
    validation = read_json(directory/"validation.json", issues)
    rollout = read_json(directory/"geometry/rollout_verification.json", issues)
    geometry = read_json(directory/"geometry/geometry_statistics.json", issues)
    receipt = read_json(directory/"reference_receipt.json", issues)
    task, expected_sha = _task(job, project), _identity(job, project)[1]
    if not summary:
        result["evidence_issues"].append("native_summary_missing")
    if not backend:
        result["evidence_issues"].append("motion_backend_missing")
    if not plan:
        result["evidence_issues"].append("final_plan_missing")
    if summary.get("task_sha256") != expected_sha:
        result["evidence_issues"].append("native_task_sha_mismatch_or_missing")
    if task is not None and (not summary.get("task_file") or _path(summary["task_file"], project) != task):
        result["evidence_issues"].append("native_task_path_mismatch_or_missing")
    pid, actual = summary.get("simulator_pid"), summary.get("actual_makespan_seconds")
    if strict and (row.get("passed") is True or summary.get("passed") is True or row.get("status") == "passed"):
        required = ("summary.json", "plan.json", "motion_backend.json", "execution.json", "validation.json",
                    "execution_validation.json", "reference_receipt.json", "tasks.json", "tpg.json",
                    "geometry/rollout_verification.json")
        already_read = {"summary.json": summary, "plan.json": plan, "motion_backend.json": backend,
                        "execution.json": execution, "validation.json": validation,
                        "reference_receipt.json": receipt, "geometry/rollout_verification.json": rollout}
        for name in required:
            document = already_read[name] if name in already_read else read_json(directory/name, issues)
            if not (directory/name).is_file() or not document:
                result["evidence_issues"].append("required_native_evidence_missing:"+name)
        checks = {
            "complete_native_summary": (summary.get("success") is True
                and summary.get("real_planning_status") == "solved"
                and not summary.get("fatal_environment") and not summary.get("remote_interface_failure")),
            "independent_node_coverage": (validation.get("violations") == []
                and number(validation.get("expected_nodes")) and validation["expected_nodes"] > 0
                and validation.get("actual_node_intervals") == validation["expected_nodes"]),
            "complete_native_execution": (execution.get("done") is True
                and execution.get("independent_validation_passed") is True
                and number(execution.get("total")) and execution["total"] > 0
                and execution.get("completed") == execution["total"]
                and execution.get("tracking_failures") == execution.get("collision_failures") == []
                and not execution.get("timeout") and not execution.get("remote_interface_failure")
                and bool(execution.get("task_schedule"))),
            "complete_nominal_sampled_rollout": (rollout.get("coverage_complete") is True
                and rollout.get("no_sampled_collision") is True
                and rollout.get("stationary_waiting_poses_included") is True
                and number(rollout.get("planned_samples")) and rollout["planned_samples"] > 0
                and rollout.get("tested_samples") == rollout["planned_samples"])}
        result["evidence_issues"].extend(name for name, passed in checks.items() if not passed)
    passed = (not result["evidence_issues"] and row.get("status") == "passed" and row.get("passed") is True
        and summary.get("status") == "passed" and summary.get("passed") is True
        and summary.get("simulator_closed") is True and row.get("simulator_closed") is True
        and isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 and row.get("simulator_pid") == pid
        and execution.get("success") is True and validation.get("success") is True
        and rollout.get("valid") is True and number(actual) and actual > 0
        and row.get("actual_makespan_seconds") == actual)
    if row.get("passed") is True and not passed:
        for label, valid in (("execution_success", execution.get("success") is True),
                             ("independent_validation", validation.get("success") is True),
                             ("rollout", rollout.get("valid") is True)):
            if not valid:
                result["evidence_issues"].append(label+"_missing_or_failed")
    estimate = plan.get("makespan")
    nominal = summary.get("nominal_makespan_seconds", summary.get("backend_nominal_seconds"))
    result.update(real_passed=bool(passed), native_status=summary.get("status"),
        symbolic_status=plan.get("status", summary.get("real_planning_status")),
        failure_phase=summary.get("failure_phase", row.get("failure_phase")),
        failure_category="passed" if passed else failure_category(row, summary, backend, progress),
        reason=summary.get("reason", row.get("reason")), elapsed_seconds=summary.get("wall_seconds", row.get("wall_seconds")),
        actual_seconds=actual, estimated_seconds=estimate, nominal_seconds=nominal,
        estimate_actual_ratio=estimate/actual if passed and number(estimate) and estimate > 0 else None,
        nominal_actual_ratio=nominal/actual if passed and number(nominal) and nominal > 0 else None,
        phase_seconds=summary.get("phase_seconds", progress.get("phase_seconds", {})),
        stage_seconds=summary.get("stage_seconds", progress.get("stage_seconds", {})),
        tpg_seconds=summary.get("tpg_seconds"), tpg_nodes=summary.get("tpg_nodes"), tpg_edges=summary.get("tpg_edges"),
        geometry=geometry, feedback=feedback_metrics(backend), step_durations=summary.get("step_durations", []),
        reference={key: receipt.get(key) for key in ("context_sha256", "schedule_sha256", "tasks_sha256", "home_order")},
        symbolic_certificate={"lower_bound": summary.get("symbolic_lower_bound", plan.get("lower_bound")),
            "suboptimality_bound": summary.get("symbolic_suboptimality_bound", plan.get("bound")),
            "feedback_scopes": [group["certificate_scope"] for group in backend.get("feedback_rounds", [])
                                if isinstance(group, dict) and isinstance(group.get("certificate_scope"), dict)],
            "scope": "symbolic cost only; restricted motion cuts are conditional, never executed-trajectory optimality"},
        last_motion_event=boundary, progress_source=progress_source,
        evidence_directory=str(directory), summary_source=str(directory/"summary.json"),
        backend_source=str(directory/"motion_backend.json"), geometry_source=str(directory/"geometry/geometry_statistics.json"))
    if row.get("status") == "passed" and not passed:
        result["failure_category"] = "evidence:uncredited_native_pass"
    return result


def _planning(job, selected, project, issues):
    result = {"committed": selected is not None, "status": "pending", "solved": False, "evidence_issues": []}
    if selected is None:
        return result
    row, actual_job, faults = selected
    plan = read_json(_path(actual_job["output"], project)/"plan.json", issues)
    result.update({key: row.get(key, plan.get(key)) for key in ("status", "makespan", "bound", "expanded",
        "elapsed_seconds", "search_elapsed_seconds", "first_solution_seconds", "first_solution_expanded",
        "physical_infeasible", "physical_infeasible_goal_proof", "unknown", "reason", "stages")})
    result["evidence_issues"] = list(faults)
    if row.get("task_sha256") != _identity(job, project)[1]:
        result["evidence_issues"].append("planning_task_sha_mismatch_or_missing")
    result["solved"] = not result["evidence_issues"] and result["status"] == "solved"
    result["source"] = str(_path(actual_job["output"], project)/"plan.json")
    result["certificate_scope"] = "symbolic DistanceCost makespan only; excludes motion candidates, homes, TPG and actual execution"
    result["certificate_bound_present"] = result["solved"] and number(result.get("bound")) and result["bound"] >= 1.
    stages = result.get("stages") or []
    result["inadmissible_certificate_claims"] = [stage for stage in stages
        if stage.get("non_admissible_ordering") is True and stage.get("suboptimality_bound") is not None]
    return result


def _versions(project, root, selection, issues):
    maps = {**selection.get("code_sha256", {}), **selection.get("worker_code_sha256", {})}
    changed, missing_snapshots = [], []
    for name, expected in maps.items():
        path = _path(name, project)
        if not path.is_relative_to(project) or not path.is_file() or sha256(path) != expected:
            changed.append(name)
        snapshot = root/"code_snapshot"/name
        if not snapshot.is_file() or sha256(snapshot) != expected:
            missing_snapshots.append(name)
    report_files = ("tools/motion_v135_report.py", "tests/test_motion_v135_report.py")
    tool_sha = {name: sha256(project/name) if (project/name).is_file() else None for name in report_files}
    unregistered = [name for name in report_files if name not in maps]
    return {"prepared": bool(selection), "expected_code_sha256": selection.get("code_sha256", {}),
        "current_changed_files": changed, "snapshot_missing_or_changed": missing_snapshots,
        "report_tool_sha256": tool_sha, "report_tool_unregistered": unregistered,
        "matched": bool(maps) and not changed and not missing_snapshots and not unregistered}


def _proofs(project, root, selection, path, issues):
    registry_path = _path(path, project) if path else root/"acceptance_evidence.json"
    registry = read_json(registry_path, issues)
    result = {}
    for name in ("full_suite", "original11", "optimality", "archive", "environment", "selected_cases"):
        item = registry.get(name)
        source = item if isinstance(item, str) else item.get("path") if isinstance(item, dict) else None
        proof_path = _path(source, project) if source else None
        data = read_json(proof_path, issues) if proof_path else {}
        code = (item.get("code_sha256") if isinstance(item, dict) else None) or data.get("code_sha256")
        if not code and proof_path:
            code = read_json(proof_path.parent/"code_version.json", issues)
        code_match = bool(code) and bool(selection.get("code_sha256")) and all(
            code.get(key) == value for key, value in selection.get("code_sha256", {}).items())
        passed = data.get("passed")
        if passed is None and "returncode" in data:
            passed = data["returncode"] == 0 and data.get("status") not in {"timeout", "interrupted", "global_stop"}
        own_evidence = proof_path is not None and proof_path.is_relative_to(project/"results/sim_v1_3/motion_v135")
        state = "failed" if passed is False else "passed" if passed is True and own_evidence and (code_match or name in {"archive", "environment"}) else "pending"
        result[name] = {"status": state, "path": str(proof_path) if proof_path else None,
                        "code_match": code_match, "data": data}
    return result


def _distribution(values):
    values = sorted(value for value in values if number(value))
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "min": values[0], "median": median(values),
            "p90": values[math.ceil(.9*len(values))-1], "max": values[-1]}


def collect(attempt="formal_a1", *, project=ROOT, evidence=None):
    """Pure file reads: no solver, simulator, writes or process inspection."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt):
        raise ValueError("attempt只允许安全的字母数字标识")
    project = Path(project).resolve()
    base = project/"results/sim_v1_3/motion_v135"
    root = base/"experiments"/attempt
    issues = []
    selection = read_json(root/"selection.json", issues)
    versions = _versions(project, root, selection, issues)
    result = {"schema_version": 1, "attempt": attempt, "updated_at": datetime.now(timezone.utc).isoformat(),
        "formal_baseline": "v1.3.4 c_wall_v134/runtime_fix2", "selection_source": str(root/"selection.json"),
        "selection_sha256": sha256(root/"selection.json") if (root/"selection.json").is_file() else None,
        "versions": versions, "workflow": read_json(root/"workflow_state.json", issues),
        "progress": read_json(base/"progress_state.json", issues), "stages": {}, "issues": issues,
        "budget_policy": "original input/cohort/task and phase budgets; no extension or resampling",
        "randomness": "original task seeds fixed by immutable task identity; motion stochasticity belongs to each retained attempt"}
    for stage, (old_name, denominator) in STAGES.items():
        source_path = project/"results/sim_v1_3/v134/manifests"/(old_name+".json")
        baseline = read_json(source_path, issues)
        current_path = root/"manifests"/(stage+".json")
        current = read_json(current_path, issues)
        old_selected, old_rows = _selected(baseline, project, project/"results/sim_v1_3/v134", old_name, issues)
        selected, rows = _selected(current, project, root, stage, issues)
        expected_code = fingerprint(selection["code_sha256"]) if selection.get("code_sha256") else None
        expected_workers = fingerprint(selection["worker_code_sha256"]) if selection.get("worker_code_sha256") else None
        for row, actual_job, faults in selected.values():
            if not expected_code or row.get("code_version_sha256") != expected_code:
                faults.append("formal_code_version_mismatch_or_missing")
            if actual_job.get("worker_code_version_sha256") is not None and (
                    actual_job["worker_code_version_sha256"] != expected_workers
                    or row.get("worker_code_version_sha256") != expected_workers):
                faults.append("formal_worker_version_mismatch_or_missing")
        for label, manifest in (("baseline", baseline), ("current", current)):
            if manifest and len(manifest.get("jobs", [])) != denominator:
                issues.append({"stage": stage, "reason": label+"_manifest_count_mismatch",
                               "expected": denominator, "observed": len(manifest.get("jobs", []))})
        if current and baseline and current.get("phase_budgets_seconds") != baseline.get("phase_budgets_seconds"):
            issues.append({"stage": stage, "reason": "phase_budget_changed"})
        definition = selection.get("manifests", {}).get(stage, {})
        if current_path.is_file() and sha256(current_path) != definition.get("sha256"):
            issues.append({"path": str(current_path), "reason": "selection_manifest_sha_mismatch"})
        if definition and (not source_path.is_file() or sha256(source_path) != definition.get("baseline_manifest_sha256")):
            issues.append({"path": str(source_path), "reason": "baseline_manifest_sha_mismatch"})
        by_identity = {}
        for job in current.get("jobs", []):
            by_identity.setdefault(_identity(job, project), []).append(job)
        cases, used = [], set()
        for index in range(denominator):
            old_jobs = baseline.get("jobs", [])
            old_job = old_jobs[index] if index < len(old_jobs) else {"id": "missing-baseline-"+str(index), "family": "missing"}
            identity = _identity(old_job, project)
            candidates = by_identity.get(identity, [])
            job = candidates[0] if len(candidates) == 1 else None
            faults = []
            if len(candidates) != 1:
                faults.append("current_identity_missing" if not candidates else "duplicate_original_identity")
            if job:
                used.add(job["id"])
                if job.get("timeout_seconds", 1200.) != old_job.get("timeout_seconds", 1200.):
                    faults.append("task_budget_changed")
                task = _task(job, project)
                if task is not None and (not task.is_file() or sha256(task) != identity[1]):
                    faults.append("prepared_input_sha_mismatch")
            reader = _planning if stage == "planning200" else _native
            before = reader(old_job, old_selected.get(old_job["id"]), project, issues) if stage == "planning200" else reader(
                old_job, old_selected.get(old_job["id"]), project, project/"results/sim_v1_3/v134", issues, strict=False)
            after = (reader(job, selected.get(job["id"]), project, issues) if stage == "planning200" else reader(
                job, selected.get(job["id"]), project, root, issues)) if job else {
                    "committed": False, "status": "pending", "real_passed": False, "solved": False,
                    "failure_category": "pending", "feedback": {}, "evidence_issues": []}
            if faults:
                after["real_passed"] = after["solved"] = False
            cases.append({"source_job_id": identity[0], "task_sha256": identity[1],
                "family": old_job.get("family", "fixture"), "baseline_job_id": old_job["id"],
                "job_id": job["id"] if job else None, "task": str(_task(old_job, project)),
                "input_budget_seconds": old_job.get("timeout_seconds", 1200.), "identity_issues": faults,
                "baseline": before, "current": after,
                "cost_delta_seconds": (after["makespan"]-before["makespan"] if number(after.get("makespan"))
                                       and number(before.get("makespan")) else None)})
        committed = sum(case["current"]["committed"] for case in cases)
        statuses = dict(Counter(case["current"]["status"] for case in cases))
        passed_key = "solved" if stage == "planning200" else "real_passed"
        metrics = {"denominator": denominator, "committed": committed, "pending": denominator-committed,
            "passed": sum(case["current"].get(passed_key, False) for case in cases), "statuses": statuses,
            "baseline_passed": sum(case["baseline"].get(passed_key, False) for case in cases),
            "cases": cases, "interrupted_attempts": sum(row.get("status") == "interrupted" for row in rows),
            "unmatched_current_jobs": [job["id"] for job in current.get("jobs", []) if job["id"] not in used],
            "manifest_source": str(current_path), "baseline_manifest_source": str(source_path),
            "phase_budgets_seconds": current.get("phase_budgets_seconds"),
            "journal_source": str(_path(current.get("state_dir", str(root/"states"/stage)), project)/"completed.jsonl")}
        if stage == "planning200":
            families = {}
            for family in FAMILIES:
                subset = [case for case in cases if case["family"] == family]
                physical = sum(case["current"].get("physical_infeasible") is True
                    and bool(case["current"].get("physical_infeasible_goal_proof")) for case in subset)
                families[family] = {"denominator": 40, "solved": sum(c["current"].get("solved", False) for c in subset),
                    "baseline_solved": sum(c["baseline"].get("solved", False) for c in subset),
                    "statuses": dict(Counter(c["current"]["status"] for c in subset)),
                    "physical_proven_rejected": physical, "feasible_denominator": 40-physical,
                    "cost_seconds": _distribution(c["current"].get("makespan") for c in subset if c["current"].get("solved")),
                    "baseline_cost_seconds": _distribution(c["baseline"].get("makespan") for c in subset if c["baseline"].get("solved")),
                    "bound": _distribution(c["current"].get("bound") for c in subset if c["current"].get("solved")),
                    "first_solution_seconds": _distribution(c["current"].get("first_solution_seconds") for c in subset if c["current"].get("solved"))}
            metrics["families"] = families
            metrics["strict_section7"] = {
                "family_rate_failures": [family for family, row in families.items()
                    if row["solved"]/(row["feasible_denominator"] or 40) < (.95 if family in {"F1", "F3"} else 1.)],
                "walls_not_solved_within60": [case["source_job_id"] for case in cases if case["family"] == "F3"
                    and not (case["current"].get("physical_infeasible") is True
                             and case["current"].get("physical_infeasible_goal_proof"))
                    and (not case["current"].get("solved") or not number(case["current"].get("elapsed_seconds"))
                         or case["current"]["elapsed_seconds"] > 60.)],
                "solved_first_over15_or_missing": [case["source_job_id"] for case in cases
                    if case["current"].get("solved") and (not number(case["current"].get("first_solution_seconds"))
                        or case["current"]["first_solution_seconds"] > 15.)],
                "unknown_no_plan_over5_or_missing": [case["source_job_id"] for case in cases
                    if case["current"].get("status") in {"unknown", "no_plan"}
                    and (not number(case["current"].get("elapsed_seconds")) or case["current"]["elapsed_seconds"] > 5.)],
                "ruling": "FIX v1.3.4 §8 explicitly retains _005/_012 budget and restricted-HMax limitations; report deviations without changing frozen planning"}
        else:
            metrics["failures"] = dict(Counter(c["current"].get("failure_category", "missing") for c in cases
                if c["current"]["committed"] and not c["current"].get("real_passed")))
            metrics["baseline_pass_regressions"] = [c["source_job_id"] for c in cases
                if c["baseline"].get("real_passed") and c["current"]["committed"] and not c["current"].get("real_passed")]
            metrics["baseline_pass_pending"] = [c["source_job_id"] for c in cases
                if c["baseline"].get("real_passed") and not c["current"]["committed"]]
            metrics["estimate_actual_ratio"] = _distribution(c["current"].get("estimate_actual_ratio") for c in cases)
            metrics["nominal_actual_ratio"] = _distribution(c["current"].get("nominal_actual_ratio") for c in cases)
            fields = sorted({key for case in cases for key, value in case["current"].get("geometry", {}).items() if number(value)})
            metrics["tpg_geometry_distributions"] = {key: _distribution(c["current"].get("geometry", {}).get(key) for c in cases)
                                                    for key in fields}
            metrics["tpg_seconds_distribution"] = _distribution(c["current"].get("tpg_seconds") for c in cases)
            metrics["feedback_totals"] = {key: sum(c["current"].get("feedback", {}).get(key, 0) for c in cases)
                for key in ("rounds", "cuts", "unknown_validations", "candidate_deduplicated", "candidate_attempts",
                            "global_downgrades", "terminal_reference_failures")}
            metrics["cut_reasons"] = dict(sum((Counter(c["current"].get("feedback", {}).get("cut_reasons", {})) for c in cases), Counter()))
            metrics["reference_failure_reasons"] = dict(sum((Counter(c["current"].get("feedback", {}).get("reference_failure_reasons", {})) for c in cases), Counter()))
            metrics["phase_time_distributions"] = {scope: {key: _distribution(c["current"].get(scope, {}).get(key) for c in cases)
                for key in sorted({key for c in cases for key in c["current"].get(scope, {})})}
                for scope in ("phase_seconds", "stage_seconds")}
        result["stages"][stage] = metrics
    result["proofs"] = _proofs(project, root, selection, evidence, issues)
    result["acceptance"] = acceptance(result)
    result["pending"] = [f"{name}: {row['pending']}/{row['denominator']}未提交" for name, row in result["stages"].items() if row["pending"]]
    result["pending"] += ["验收证明 "+name for name, row in result["proofs"].items() if row["status"] == "pending"]
    if not versions["matched"]:
        result["pending"].append("代码/工具登记或快照一致性核对")
    result["complete"] = not result["pending"] and not issues and all(row["status"] == "passed" for row in result["acceptance"].values())
    return result


def acceptance(report):
    stages, checks = report["stages"], {}
    for stage, threshold in (("fixed20", 14), ("walls3", 2), ("gates", 5)):
        row = stages[stage]
        checks[stage] = {"status": "pending" if row["pending"] else "passed" if row["passed"] >= threshold else "failed",
                         "required": f"{threshold}/{row['denominator']}", "observed": f"{row['passed']}/{row['denominator']}"}
    row = stages["fixed20"]
    checks["original_eight_regressions"] = {"status": "failed" if row["baseline_pass_regressions"] else
        "pending" if row["baseline_pass_pending"] or row["baseline_passed"] != 8 else "passed",
        "baseline_passed": row["baseline_passed"], "regressions": row["baseline_pass_regressions"], "pending": row["baseline_pass_pending"]}
    ratio = row["estimate_actual_ratio"]
    checks["estimated_actual_ratio"] = {"status": "pending" if row["pending"] or not ratio["count"] else
        "passed" if .7 <= ratio["median"] <= 1.3 else "failed", "required": "median 0.7–1.3", **ratio}
    plan = stages["planning200"]
    regression = any(row["solved"] < row["baseline_solved"] for row in plan["families"].values())
    checks["planning_regression_v134_ruling"] = {"status": "pending" if plan["pending"] else "failed" if regression else "passed",
        "scope": "FIX v1.3.4 §8 retains frozen planning and known _005/_012 budgets; no certificate extension to motion"}
    solved = [case["current"] for case in plan["cases"] if case["current"].get("solved")]
    bad_certificate = [case["source_job_id"] for case in plan["cases"] if case["current"].get("inadmissible_certificate_claims")]
    checks["symbolic_certificates"] = {"status": "failed" if bad_certificate else "pending" if plan["pending"]
        or any(not row.get("certificate_bound_present") for row in solved) else "passed",
        "invalid_sources": bad_certificate, "scope": "admissible symbolic cost only; optimality proof additionally required"}
    all_sim = [c["current"] for name in ("fixed20", "walls3", "gates") for c in stages[name]["cases"]]
    downgrade = sum(row.get("feedback", {}).get("global_downgrades", 0) for row in all_sim)
    checks["global_reach_downgrades"] = {"status": "failed" if downgrade else "pending" if any(stages[name]["pending"] for name in ("fixed20", "walls3", "gates")) else "passed", "observed": downgrade, "required": 0}
    checks["delivered_code_matches_formal"] = {"status": "passed" if report["versions"]["matched"] else
        "failed" if report["versions"]["prepared"] and (report["versions"]["current_changed_files"] or report["versions"]["snapshot_missing_or_changed"]) else "pending",
        "changed": report["versions"]["current_changed_files"], "missing_snapshots": report["versions"]["snapshot_missing_or_changed"],
        "unregistered_tools": report["versions"]["report_tool_unregistered"]}
    faults = [{"stage": name, "source_job_id": case["source_job_id"], "issues": case["identity_issues"]}
        for name, stage in stages.items() for case in stage["cases"] if case["identity_issues"]]
    definition_faults = [{"stage": name, "source_job_id": case["source_job_id"], "issues": case["current"]["evidence_issues"]}
        for name, stage in stages.items() for case in stage["cases"]
        if any(issue in {"definition_sha_mismatch_or_missing", "formal_code_version_mismatch_or_missing",
                         "formal_worker_version_mismatch_or_missing"}
               for issue in case["current"].get("evidence_issues", []))]
    checks["original_inputs_and_budgets"] = {"status": "pending" if not report["versions"]["prepared"] else
        "failed" if faults or definition_faults or report["issues"] else "passed",
        "identity_faults": faults, "definition_faults": definition_faults}
    workflow = report.get("workflow", {})
    checks["workflow_finished"] = {"status": "passed" if workflow.get("status") == "finished" else
        "failed" if workflow.get("status") == "global_stop" else "pending", "observed": workflow.get("status", "not_prepared")}
    for name, item in report.get("proofs", {}).items():
        checks[name] = {"status": item["status"], "evidence": item["path"], "code_match": item["code_match"]}
    return checks


def _markdown(report):
    lines = ["# 运动可行性与仿真性能优化交付报告", "", "批次："+report["attempt"]+"；更新："+report["updated_at"], "",
        "正式基线为 v1.3.4 c_wall_v134/runtime_fix2。原生执行与独立验收共同通过才计成功；UNKNOWN、budget、timeout、缺证据均保留分母。", "",
        "|阶段|分母|基线通过/solved|本轮通过/solved|待处理|状态|", "|---|---:|---:|---:|---:|---|"]
    for name, row in report["stages"].items():
        lines.append(f"|{name}|{row['denominator']}|{row['baseline_passed']}|{row['passed']}|{row['pending']}|{json.dumps(row['statuses'], ensure_ascii=False)}|")
    lines += ["", "规划200按族：", "", "|族|本轮 solved/40|基线 solved/40|已证明物理拒绝|状态|", "|---|---:|---:|---:|---|"]
    for family, row in report["stages"]["planning200"]["families"].items():
        lines.append(f"|{family}|{row['solved']}/40|{row['baseline_solved']}/40|{row['physical_proven_rejected']}|{json.dumps(row['statuses'], ensure_ascii=False)}|")
    lines += ["", "bound 只解释符号 DistanceCost 工时；不能证明加入候选、停靠、回位及 TPG 后的运动或执行最优性。FIX v1.3.4 §8 的两项墙预算和受限 HMax 豁免仍须单列，不能称严格 §7 全通过。", "",
        "固定20逐项对比：", "", "|原身份|族|基线|本轮|基线/本轮原因|原通过回退|", "|---|---|---|---|---|---|"]
    for case in report["stages"]["fixed20"]["cases"]:
        old, new = case["baseline"], case["current"]
        regression = "待检查" if old.get("real_passed") and not new["committed"] else "是" if old.get("real_passed") and not new.get("real_passed") else "无"
        lines.append(f"|{case['source_job_id']}|{case['family']}|{old['status']}|{new['status']}|{old.get('failure_category')}/{new.get('failure_category')}|{regression}|")
    for name in ("walls3", "gates"):
        lines += ["", name+"逐项：", "", "|原身份|基线|本轮|本轮原因|", "|---|---|---|---|"]
        for case in report["stages"][name]["cases"]:
            lines.append(f"|{case['source_job_id']}|{case['baseline']['status']}|{case['current']['status']}|{case['current'].get('failure_category')}|")
    lines += ["", "状态/分类改善与实际通过率分别报告：", ""]
    classified = sum(c["current"]["committed"] and c["baseline"].get("failure_category") != c["current"].get("failure_category") for c in report["stages"]["fixed20"]["cases"])
    lines.append(f"- 固定20类别改变 {classified} 项；其中可以包含保守 UNKNOWN 分类修正，不等于执行成功。实际通过 {report['stages']['fixed20']['baseline_passed']} → {report['stages']['fixed20']['passed']}。")
    lines.append("- 各阶段耗时、逐步估算/实际值、反馈 cut、候选去重、终态参考失败与 TPG 查询统计见 metrics.json 和相邻 CSV；有限采样不构成连续碰撞证明。")
    lines += ["", "反馈与性能：", "", "|批次|cut|UNKNOWN验证|重复候选去重|全局降级|终态参考失败|TPG秒中位数|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("fixed20", "walls3", "gates"):
        stage = report["stages"][name]
        totals = stage["feedback_totals"]
        lines.append(f"|{name}|{totals['cuts']}|{totals['unknown_validations']}|{totals['candidate_deduplicated']}|{totals['global_downgrades']}|{totals['terminal_reference_failures']}|{stage['tpg_seconds_distribution']['median']}|")
    lines += ["", "严格 §7 规划偏差（未完成任务也保留）：", "", "```json",
        json.dumps(report["stages"]["planning200"]["strict_section7"], ensure_ascii=False, indent=2), "```", "",
        "完整来源、每项原任务 SHA/预算、代码与工具指纹、TPG 查询分布和阶段耗时见：", "",
        "[metrics.json]("+(Path(report["selection_source"]).parent/"reports/metrics.json").as_posix()+")。"]
    lines += ["", "验收：", "", "|条目|结论|证据/测量|", "|---|---|---|"]
    labels = {"passed": "通过", "failed": "未通过", "pending": "等待证据"}
    for name, row in report["acceptance"].items():
        lines.append(f"|{name}|{labels[row['status']]}|{json.dumps({k:v for k,v in row.items() if k != 'status'}, ensure_ascii=False)}|")
    lines += ["", "已确认剩余限制：", "",
        "- F3_021 的另一臂持物占用同一接近区域，空闲臂停靠条件不成立；不能靠重复同候选冒称已解决。", 
        "- wall037 固定 home 目标与 filled s3_3 装配几何不兼容；本轮保留回位要求，更多 IK 不会消除这项固定终态约束。",
        "- 相关原始确认记录：results/sim_v1_3/motion_v135/m3/interarm_followup.json；正式本轮结果仍按各任务独立证据计数。", "", "恢复与重建：", "", "```powershell",
        f"python -X utf8 tools/motion_v135_workflow.py run --attempt {report['attempt']}",
        f"python -X utf8 tools/motion_v135_report.py --attempt {report['attempt']}",
        "```", "", "若 workflow 标记 global_stop，先按其原因保存现场并修复环境，不由报告工具重启或清除。", "", "待完成：", ""]
    lines += ["- "+item for item in report["pending"]] or ["- 无待提交项；仍以逐条验收结论为准。"]
    return "\n".join(lines)+"\n"


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def rebuild(attempt="formal_a1", *, project=ROOT, publish=False, evidence=None):
    report = collect(attempt, project=project, evidence=evidence)
    project = Path(project).resolve()
    base = project/"results/sim_v1_3/motion_v135"
    destination = (base/"experiments"/attempt/"reports").resolve()
    if not destination.is_relative_to((base/"experiments").resolve()):
        raise ValueError("报告输出越界")
    _atomic(destination/"metrics.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    summary = _markdown(report)
    _atomic(destination/"summary.md", summary)
    for stage, data in report["stages"].items():
        rows = []
        for case in data["cases"]:
            rows.append({**{key: value for key, value in case.items() if key not in {"baseline", "current"}},
                **{"baseline_"+key: value for key, value in case["baseline"].items()}, **case["current"]})
        fields = list(dict.fromkeys(key for row in rows for key in row))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
        _atomic(destination/(stage+"_comparison.csv"), stream.getvalue())
    step_rows = [{"stage": stage, "source_job_id": case["source_job_id"], "task_sha256": case["task_sha256"], **step}
        for stage in ("fixed20", "walls3", "gates") for case in report["stages"][stage]["cases"]
        for step in case["current"].get("step_durations", [])]
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(["stage", "source_job_id", "task_sha256", *[key for row in step_rows for key in row]]))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(step_rows)
    _atomic(destination/"step_durations.csv", stream.getvalue())
    if publish:
        _atomic(base/"summary.md", summary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅重建本轮正式对比报告，不启动任何任务")
    parser.add_argument("--attempt", default="formal_a1")
    parser.add_argument("--evidence", help="结构化验收证明索引 JSON")
    parser.add_argument("--publish", action="store_true", help="同时更新motion_v135/summary.md")
    args = parser.parse_args(argv)
    report = rebuild(args.attempt, publish=args.publish, evidence=args.evidence)
    print(json.dumps({"attempt": args.attempt, "complete": report["complete"], "pending": report["pending"],
                      "acceptance": report["acceptance"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
