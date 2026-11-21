"""Rebuild v1.3.3 reports from committed evidence; never run experiments.

Manifest stage/ID is authoritative. Worker summaries can legitimately say
phase='complete'; that field never determines a report cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as batch
from tools.v133_experiments import load_selection, resolve_stage
from tools.overnight_summary import finite, distribution, real_passed

BASE = ROOT / "results/sim_v1_3/v133"
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
STAGES = ("d_gate", "d_e2e", "d_gates", "qualify200", "planning200", "sim20", "gates")
LIVE_TESTS = batch.LIVE_TESTS
FIXTURES = ("s2_tower_sim", "s3a_cantilever_sim", "s3a_platform_sim")
V11_MODULES = ("tests.test_v11_search", "tests.test_v11_admissibility", "tests.test_v11_ordering")


class Evidence:
    def __init__(self, base, project):
        self.base, self.project = Path(base), Path(project)
        self.issues = []
        self.selection = load_selection(self.base, project=self.project)

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.project / path

    def source(self, path):
        if path is None:
            return None
        path = Path(path).resolve()
        try:
            return path.relative_to(self.project.resolve()).as_posix()
        except ValueError:
            return str(path)

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("object expected")
            return data
        except (OSError, ValueError, UnicodeError) as error:
            self.issues.append({"source": self.source(path), "reason": "incomplete_artifact: " + str(error)})
            return {}

    def task(self, path):
        if not path:
            return {}
        try:
            return tomllib.loads(self.path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError) as error:
            self.issues.append({"source": str(path), "reason": "task_unavailable: " + str(error)})
            return {}

    def stage(self, name, *, use_selection=True):
        physical = resolve_stage(name, self.base, selection=self.selection, project=self.project) if use_selection else name
        manifest_path = self.base / "manifests" / (physical + ".json")
        manifest = self.read(manifest_path)
        directory = self.base / "states" / physical
        records, conflicts = {}, set()
        paths = ([directory / "completed.jsonl"] if (directory / "completed.jsonl").is_file() else [])
        paths += sorted(directory.glob("completed.recovery_*.jsonl"))
        for path in paths:
            try:
                lines = path.read_bytes().splitlines(keepends=True)
            except OSError as error:
                self.issues.append({"source": self.source(path), "reason": str(error)})
                continue
            for index, line in enumerate(lines, 1):
                if not line.endswith(b"\n"):
                    self.issues.append({"source": self.source(path), "line": index, "reason": "uncommitted_tail_ignored"})
                    continue
                try:
                    row = json.loads(line)
                    ident = row["id"]
                    if ident in records and records[ident] != row:
                        conflicts.add(ident)
                    records[ident] = row
                except (ValueError, KeyError, TypeError, UnicodeError) as error:
                    self.issues.append({"source": self.source(path), "line": index, "reason": "invalid_journal: " + str(error)})
        for ident in conflicts:
            records.pop(ident, None)
            self.issues.append({"source": self.source(directory), "id": ident, "reason": "conflicting_committed_records"})
        jobs = manifest.get("jobs", [])
        ids = Counter(j.get("id") for j in jobs)
        cases = []
        for job in jobs:
            row = records.get(job.get("id"), {}) if ids[job.get("id")] == 1 else {}
            cases.append({"stage": physical, "logical_stage": name, "job": job, "record": row,
                          "source": self.source(directory / "completed.jsonl")})
        return {"logical_stage": name, "physical_stage": physical,
                "manifest": self.source(manifest_path), "cases": cases,
                "requested": len(jobs), "recorded": sum(bool(c["record"]) for c in cases),
                "orphan_record_ids": sorted(set(records)-set(ids))}


def case_fields(case):
    job, row = case["job"], case["record"]
    return {"id": job.get("id"), "stage": case["stage"], "phase": job.get("phase"),
        "family": job.get("family"), "index": job.get("index"), "status": row.get("status", "pending"),
        "failure_phase": row.get("failure_phase"), "reason": row.get("reason"),
        "stages": deepcopy(row.get("stages")), "search_stages": deepcopy(row.get("search_stages")),
        "bound": row.get("bound", row.get("suboptimality_bound")), "expanded": row.get("expanded"),
        "elapsed_seconds": row.get("elapsed_seconds"), "wall_seconds": row.get("wall_seconds"),
        "auxiliary_skills": row.get("auxiliary_skills", row.get("skills")),
        "unknown": row.get("unknown"), "phase_seconds": row.get("phase_seconds"),
        "step_durations": row.get("step_durations"), "feedback": row.get("feedback"),
        "qualification_calls": row.get("qualification_calls"), "tpg_seconds": row.get("tpg_seconds"),
        "execution_wall_seconds": row.get("execution_wall_seconds"),
        "executor_wall_seconds": row.get("executor_wall_seconds"),
        "source": case["source"], "output": job.get("output")}


def reason_label(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def simulation_category(row):
    """Classify only explicit reported symptoms, never infer infeasibility."""
    if row.get("real_passed"):
        return "passed"
    status = row.get("status", "pending")
    if status in {"pending", "skipped"}:
        return status
    phase = row.get("failure_phase") or "unclassified"
    text = reason_label({"reason": row.get("reason"), "feedback": row.get("feedback")}).lower()
    if status == "timeout" or phase in {"task_wall_timeout", "phase_budget"}:
        detail = "timeout"
    elif any(word in text for word in ("table_collision", "self_collision", "interarm_collision", "actual collision", "碰撞")):
        detail = "reported_collision"
    elif "ik" in text and any(word in text for word in ("exhaust", "种子", "耗尽", "unknown", "未知")):
        detail = "ik_exhausted_or_unknown"
    elif "rrt" in text and any(word in text for word in ("budget", "timeout", "耗尽", "超时", "未找到")):
        detail = "rrt_budget_or_unknown"
    elif "tracking" in text or "跟踪" in text:
        detail = "tracking_failure"
    elif "guard timeout" in text or "did not settle" in text:
        detail = "physical_guard_timeout"
    elif phase == "acceptance" and '"code"' in text:
        detail = "reported_acceptance_violation"
    else:
        detail = "unclassified"
    return str(phase)+":"+detail


def feedback_statistics(rows):
    counts, messages = Counter(), []
    qualification_calls = 0
    for row in rows:
        if row.get("status") == "pending" or row.get("skipped"):
            continue
        calls = row.get("qualification_calls")
        if isinstance(calls, int) and not isinstance(calls, bool) and calls >= 0:
            qualification_calls += calls
        supplied = row.get("feedback") or []
        supplied = supplied if isinstance(supplied, list) else [supplied]
        for index, message in enumerate(supplied):
            kind = "other"
            if isinstance(message, str):
                if message.startswith("运动失败：") and "已禁止该状态下的此动作并重新搜索" in message:
                    kind = "invalid_cut"
                elif message.startswith("运动后端按需验证：") and re.search(r"→\s*(VALID|INVALID)(?:[。.\s]|$)", message):
                    kind = "qualification_update"
                elif message.startswith("运动验证未知："):
                    kind = "motion_unknown"
            counts[kind] += 1
            messages.append({"id": row.get("id"), "source": row.get("source"), "index": index,
                             "kind": kind, "message": message})
    return {"messages": len(messages), "invalid_cut_messages": counts["invalid_cut"],
        "qualification_update_messages": counts["qualification_update"],
        "motion_unknown_messages": counts["motion_unknown"], "other_messages": counts["other"],
        "explicit_loop_messages": counts["invalid_cut"]+counts["qualification_update"],
        "qualification_calls_observed": qualification_calls, "reasons": messages,
        "scope": "仅按pipeline明确文案计消息；按需回填每个事实一条，不能当作完整重规划轮次数。qualification_calls含全量预资格，不计为反馈次数。"}


def diagnostic_native_rows(evidence, stages):
    """Current-version native launches with an explicit committed manifest row."""
    cohorts = dict(stages)
    for path in sorted((evidence.base / "manifests").glob("*.json")):
        cohorts.setdefault(path.stem, evidence.stage(path.stem))
    rows, seen = [], set()
    for cohort in cohorts.values():
        for case in cohort["cases"]:
            record, job = case["record"], case["job"]
            if not record or job.get("kind") not in {"sim", "gate"} or record.get("status") == "skipped" or record.get("executed") is False:
                continue
            row = native_evidence(evidence, case)
            if not row.get("source"):
                continue
            source = evidence.path(row["source"]).resolve()
            if not source.is_relative_to(evidence.base.resolve()) or source.name != "summary.json":
                continue
            summary = evidence.read(source)
            pid = summary.get("simulator_pid")
            if summary.get("profile") != "skillnet_v1_3" or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                continue
            if job.get("task") and (not summary.get("task_file") or
                    evidence.path(job["task"]).resolve() != evidence.path(summary["task_file"]).resolve()):
                continue
            identity = str(evidence.path(summary["simulator_log"]).resolve()) if summary.get("simulator_log") else str(source)
            if identity in seen:
                continue
            seen.add(identity)
            row["failure_phase"] = row.get("failure_phase") or summary.get("failure_phase")
            rows.append(row)
    return rows


def diagnostics(evidence, stages, plan, sim_cases, simulation):
    plan_data = {}
    for table in ("distance_prior", "ik"):
        families = {}
        for family in FAMILIES:
            group = [row for row in plan[table]["cases"] if row.get("family") == family]
            unsolved = []
            for row in group:
                if row.get("status") == "solved":
                    continue
                path = evidence.path(row["output"])/"plan.json" if row.get("output") else None
                unsolved.append({**row, "plan_source": evidence.source(path) if path else None,
                                 "plan_present": bool(path and path.is_file())})
            families[family] = {"status_counts": dict(Counter(row.get("status", "unclassified") for row in group)),
                "requested": len(group), "denominator": 40, "missing_manifest_slots": max(0, 40-len(group)),
                "unsolved_cases": unsolved}
        plan_data[table] = {"families": families, "scope": "直接保留规划status；UNKNOWN未改写为确定无解"}
    failures = [{**row, "category": simulation_category(row)} for row in sim_cases if not row.get("real_passed")]
    attempted_failures = sum(row.get("attempted") is True for row in failures)
    native = diagnostic_native_rows(evidence, stages)

    def phase_totals(rows):
        totals = defaultdict(float)
        for row in rows:
            for name, seconds in (row.get("phase_seconds") or {}).items():
                if finite(seconds):
                    totals[name] += seconds
        return dict(totals)

    planning_seconds = {table: sum(row["elapsed_seconds"] for row in plan[table]["cases"]
                                  if finite(row.get("elapsed_seconds"))) for table in ("distance_prior", "ik")}
    return {"planning": plan_data,
        "simulation": {"denominator": 20, "attempted_failures": attempted_failures,
            "failure_rate": attempted_failures/20, "not_passed_slots": 20-simulation["passed"],
            "category_counts": dict(Counter(row["category"] for row in failures)), "cases": failures,
            "missing_manifest_slots": max(0, 20-len(sim_cases)),
            "scope": "失败率=已实际尝试但未通过/20；pending、跳过和清单缺额另列，不冒称动力学失败"},
        "feedback": {"sim20": feedback_statistics(sim_cases), "committed_native": feedback_statistics(native),
            "committed_native_launches": len(native),
            "scope": "sim20与全部当前v133已提交原生启动去重是两个重叠口径，不相加；不从v132补数"},
        "phase_seconds": {"sim20": phase_totals([row for row in sim_cases if row.get("attempted")]),
                          "committed_native": phase_totals(native), "paired_planning_elapsed": planning_seconds},
        "timing_scope": "phase_seconds按原始字段求和；execute等阶段可含TPG/执行子阶段，不把嵌套字段或重叠样本口径相加当总墙钟"}


def planning(stages):
    cases = stages["planning200"]["cases"]
    result = {}
    for table, label in (("raw", "distance_prior"), ("qualified", "ik")):
        selected = [c for c in cases if c["job"].get("kind") == "plan" and c["job"].get("table") == table]
        families = {}
        for family in FAMILIES:
            group = [c for c in selected if c["job"].get("family") == family]
            indices = [c["job"].get("index") for c in group]
            solved = sum(c["record"].get("status") == "solved" and c["record"].get("passed") is True for c in group)
            threshold = .95 if family in FAMILIES[:3] else .85
            complete = len(group) == 40 and len(set(indices)) == 40 and all(bool(c["record"]) for c in group)
            families[family] = {"requested": len(group), "recorded": sum(bool(c["record"]) for c in group),
                "solved": solved, "denominator": 40, "rate": solved/40, "threshold": threshold,
                "passed": complete and solved/40 >= threshold,
                "failure_reasons": dict(Counter(reason_label(c["record"].get("reason") or c["record"].get("status", "pending"))
                    for c in group if c["record"].get("status") != "solved")),
                "bound": distribution(c["record"].get("bound") for c in group),
                "expanded": distribution(c["record"].get("expanded") for c in group),
                "elapsed_seconds": distribution(c["record"].get("elapsed_seconds") for c in group)}
        result[label] = {"source_table": table, "families": families,
            "thresholds_passed": all(f["passed"] for f in families.values()),
            "cases": [{**case_fields(c), "table": label, "makespan": c["record"].get("makespan")} for c in selected]}
    unknown = [case_fields(c) for c in cases if c["record"].get("status") == "unknown"]
    result["unknown_cases"] = unknown
    result["unknown_within_5s"] = all(finite(c["elapsed_seconds"]) and c["elapsed_seconds"] <= 5 for c in unknown)
    return result


def reach_cells(data):
    arms = data.get("arms", [])
    places = [r["id"] for r in data.get("location", []) + data.get("slot", [])]
    table = data.get("capabilities", {}).get("reach", {})
    cells = {}
    for arm in arms:
        for place in places:
            status = str(table.get("default", "UNKNOWN")).upper()
            for candidate in ("VALID", "INVALID", "UNKNOWN"):
                if place in table.get(candidate.lower(), {}).get(arm, []):
                    status = candidate
            cells[arm, place] = status
    return cells


def qualification(evidence, stages):
    rows, transitions, prior_changes = [], Counter(), Counter()
    prior_cases = {(c["job"].get("family"), c["job"].get("index")): c for c in stages["planning200"]["cases"]
                   if c["job"].get("table") == "raw"}
    for case in stages["qualify200"]["cases"]:
        job, row = case["job"], case["record"]
        source = evidence.task(job.get("task"))
        expected = reach_cells(source)
        output = evidence.path(job["output"]) if job.get("output") else evidence.base / "missing"
        log_path = evidence.path(row["log_file"]) if row.get("log_file") else output / source.get("name", "unknown") / "qualify_log.json"
        log = evidence.read(log_path)
        updates = {(u.get("arm"), u.get("place")): u.get("status") for u in log.get("updates", [])}
        counts = Counter(updates.values())
        checked = {(r.get("arm"), r.get("place")) for r in log.get("records", [])}
        for key, after in updates.items():
            if key in expected:
                transitions[expected[key] + "→" + str(after)] += 1
        prior_case = prior_cases.get((job.get("family"), job.get("index")))
        prior = reach_cells(evidence.task(prior_case["job"].get("task"))) if prior_case else {}
        differences = []
        for key, after in updates.items():
            if key in prior:
                prior_changes[prior[key] + "→" + str(after)] += 1
                if prior[key] != after:
                    differences.append({"arm": key[0], "place": key[1], "distance_prior": prior[key], "ik": after})
        rows.append({**case_fields(case), "source_task": job.get("task"), "log_source": evidence.source(log_path),
            "expected_combinations": len(expected), "checked_combinations": len(checked & set(expected)),
            "all_combinations_recorded": bool(expected) and set(expected) == set(updates) == checked,
            "counts": dict(counts), "elapsed_seconds": log.get("elapsed_seconds", row.get("elapsed_seconds")),
            "records": log.get("records", []), "remaining_unchecked": log.get("remaining_unchecked"),
            "prior_to_ik_differences": differences})
    cohort_complete = len(rows) == 200 and all(sum(r["family"] == f for r in rows) == 40 for f in FAMILIES)
    return {"requested_tasks": len(rows), "all_200_qualified": cohort_complete and all(r["all_combinations_recorded"] for r in rows),
        "expected_combinations": sum(r["expected_combinations"] for r in rows),
        "checked_combinations": sum(r["checked_combinations"] for r in rows), "transitions": dict(transitions),
        "unknown_to_valid": transitions["UNKNOWN→VALID"], "unknown_to_invalid": transitions["UNKNOWN→INVALID"],
        "distance_prior_to_ik": dict(prior_changes), "cases": rows,
        "scope": "抓取与接近位的全部臂×位置/槽位组合；IK资格不证明轨迹或动力学成功"}


def native_evidence(evidence, case):
    job, record = case["job"], case["record"]
    directory = evidence.path(record["evidence_directory"]) if record.get("evidence_directory") else None
    if directory is None and job.get("output"):
        root = evidence.path(job["output"]) / "simulation"
        task = record.get("task") or (Path(job["task"]).stem if job.get("task") else None)
        if task:
            directory = root / Path(task).name
        else:
            found = list(root.glob("*/summary.json"))
            directory = found[0].parent if len(found) == 1 else root / "unknown_task"
    path = directory / "summary.json" if directory else None
    summary = evidence.read(path) if path else {}
    data = {**case_fields(case), **{k: summary.get(k, record.get(k)) for k in
        ("phase_seconds", "step_durations", "feedback", "qualification_calls", "tpg_seconds", "execution_wall_seconds", "executor_wall_seconds")}}
    committed_pass = record.get("passed") is True and record.get("status") == "passed" and record.get("executed") is not False
    data.update(source=evidence.source(path) or data["output"], real_passed=committed_pass and real_passed(summary),
        simulator_pid=summary.get("simulator_pid"), simulator_closed=summary.get("simulator_closed"),
        task=summary.get("task"), actual_makespan_seconds=summary.get("actual_makespan_seconds"))
    skipped = record.get("status") == "skipped" or record.get("executed") is False
    data["attempted"] = bool(record) and not skipped
    data["skipped"] = skipped
    if directory:
        initial, final = evidence.read(directory / "initial_plan.json"), evidence.read(directory / "plan.json")
        actual = summary.get("actual_makespan_seconds")
        for label, plan in (("initial", initial), ("final", final)):
            estimate = plan.get("makespan") if plan.get("status") == "solved" else None
            data[label + "_plan_makespan"] = estimate
            data[label + "_estimate_to_actual"] = estimate/actual if finite(estimate, positive=True) and finite(actual, positive=True) else None
        task_source = evidence.task(summary.get("task_file") or job.get("task"))
        prequalification = evidence.read(directory / "prequalification.json")
        data["no_reach_input"] = bool(task_source) and "reach" not in task_source.get("capabilities", {})
        expected = set(reach_cells(task_source))
        checked = {(r.get("arm"), r.get("place")) for r in prequalification.get("records", [])}
        data["no_reach_pipeline_passed"] = (data["real_passed"] and data["no_reach_input"] and bool(expected)
            and expected == checked and prequalification.get("checked") == prequalification.get("total") == len(expected))
    return data


def native_runs(evidence):
    """New v133 native launches only; copied references share their launch log."""
    unique = {}
    paths = sorted(evidence.base.rglob("summary.json"), key=lambda p: (not (p.parent / "plan.json").is_file(), str(p))) if evidence.base.exists() else []
    for path in paths:
        summary = evidence.read(path)
        if summary.get("profile") != "skillnet_v1_3" or not isinstance(summary.get("simulator_pid"), int):
            continue
        identity = str(evidence.path(summary["simulator_log"]).resolve()) if summary.get("simulator_log") else str(path.resolve())
        if identity in unique:
            unique[identity]["duplicate_sources"].append(evidence.source(path))
            continue
        case = {"stage": "all_new_native", "source": evidence.source(path),
                "job": {"id": evidence.source(path), "output": str(path.parent), "phase": "sim"},
                "record": {**summary, "evidence_directory": str(path.parent)}}
        row = native_evidence(evidence, case)
        row["duplicate_sources"] = []
        unique[identity] = row
    return list(unique.values())


def attempt_order(name):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", name))


def fixture_verified(evidence, row, name):
    """A physical fixture proof is a new v133 native artifact for that input."""
    if not row.get("source"):
        return False
    path = evidence.path(row["source"]).resolve()
    try:
        path.relative_to(evidence.base.resolve())
    except ValueError:
        return False
    summary = evidence.read(path)
    task_file = summary.get("task_file")
    expected = (evidence.project / "tasks/sim" / (name + ".toml")).resolve()
    return (path.name == "summary.json" and path.parent.name == name and row.get("task") == name
        and bool(task_file) and evidence.path(task_file).resolve() == expected
        and row.get("real_passed") is True and row.get("no_reach_pipeline_passed") is True)


def required_platform_coverage(evidence, outcome):
    """Only the explicitly selected extra stage can prove platform cleanup."""
    selection = outcome.get("required_platform", {})
    stage = selection.get("stage") if isinstance(selection, dict) else None
    result = {"status": "pending", "passed": False, "attempted": False, "committed": False,
        "stage": stage, "task": "s3a_platform_required_sim", "phase": None, "reason": "缺少明确专项阶段及已提交原生证据",
        "source": None, "manifest": None, "journal": None, "tpg_entered": None, "dynamics_entered": None,
        "real_planning_status": None, "feedback": [],
        "scope": "临时台搭建/承托/拆除的额外独立覆盖；不替代原平台臂托结果，不改变第8节六项验收口径"}
    if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
        return result
    cohort = evidence.stage(stage)
    result.update(manifest=cohort["manifest"],
                  journal=evidence.source(evidence.base / "states" / stage / "completed.jsonl"))
    expected = (evidence.project / "tasks/sim/s3a_platform_required_sim.toml").resolve()
    matches = [case for case in cohort["cases"] if case["job"].get("kind") == "sim"
               and case["job"].get("task") and evidence.path(case["job"]["task"]).resolve() == expected]
    if len(matches) != 1:
        result["reason"] = "所选专项清单缺少唯一的required平台任务；未采用其他阶段补位"
        return result
    case = matches[0]
    row = native_evidence(evidence, case)
    result.update(committed=bool(case["record"]), attempted=row["attempted"], source=row.get("source"))
    if not case["record"] or not row.get("source"):
        return result
    source = evidence.path(row["source"]).resolve()
    if (not source.is_relative_to(evidence.base.resolve()) or source.name != "summary.json"
            or source.parent.name != result["task"]):
        result["reason"] = "专项原生来源不在当前v133任务目录内；未采用旧来源补位"
        return result
    summary = evidence.read(source)
    if (not summary or summary.get("task") != result["task"] or not summary.get("task_file")
            or evidence.path(summary["task_file"]).resolve() != expected):
        result["reason"] = "缺少匹配专项输入的原生summary；保留pending"
        return result
    attempted = row["attempted"]
    record = case["record"]
    phase = summary.get("failure_phase") or summary.get("phase") or record.get("failure_phase")
    reason = record.get("reason") if row["skipped"] else summary.get("reason", record.get("reason"))
    tpg = evidence.read(source.parent / "tpg.json") if attempted else {}
    execution = evidence.read(source.parent / "execution.json") if attempted else {}
    tpg_entered = attempted and (bool(tpg.get("nodes")) or finite(summary.get("tpg_seconds"), positive=True)
        or finite(summary.get("tpg_nodes"), positive=True) or finite(summary.get("phase_seconds", {}).get("tpg"), positive=True))
    dynamics_entered = attempted and (bool(execution) or finite(summary.get("actual_makespan_seconds"), positive=True)
        or finite(summary.get("execution_wall_seconds"), positive=True)
        or finite(summary.get("phase_seconds", {}).get("execution"), positive=True))
    passed = bool(row["real_passed"] and tpg_entered and dynamics_entered)
    incomplete_pass = row["real_passed"] and not passed
    result.update(status="passed" if passed else "pending" if incomplete_pass else "not_met", passed=passed,
        result_status=record.get("status"), native_status=summary.get("status"), phase=phase,
        reason="缺少专项TPG/动力学阶段证据" if incomplete_pass else reason,
        tpg_entered=bool(tpg_entered), dynamics_entered=bool(dynamics_entered),
        real_planning_status=summary.get("real_planning_status"), feedback=summary.get("feedback", record.get("feedback", [])),
        simulator_pid=summary.get("simulator_pid"), simulator_closed=summary.get("simulator_closed"),
        actual_makespan_seconds=summary.get("actual_makespan_seconds"))
    return result


def physical(evidence, stages, all_runs):
    final_gates = []
    for test in LIVE_TESTS:
        cases = [c for c in stages["gates"]["cases"] if c["job"].get("test_name") == test]
        row = native_evidence(evidence, cases[0]) if cases else {"source": None}
        record = cases[0]["record"] if cases else {}
        passed = len(cases) == 1 and row.get("real_passed") and record.get("passed") is True and record.get("tests_run") == 1 and all(
            record.get(k) == 0 for k in ("test_skipped", "test_failures", "test_errors"))
        final_gates.append({**row, "test": test, "passed": bool(passed),
            **{k: record.get(k) for k in ("tests_run", "test_skipped", "test_failures", "test_errors")}})
    e2e_attempts = [native_evidence(evidence, c) for stage in sorted(stages, key=attempt_order)
        if stage == "d_e2e" or stage.startswith(("d_e2e_", "d_e2e-")) for c in stages[stage]["cases"]]
    outcome = evidence.read(evidence.base / "physical_stage_outcome.json")
    outcome_fixtures = outcome.get("fixtures", {}) if outcome.get("finished") is True else {}
    if isinstance(outcome_fixtures, list):
        outcome_fixtures = {r.get("task"): r for r in outcome_fixtures if isinstance(r, dict)}
    outcome_rows = {}
    for name in FIXTURES:
        value = outcome_fixtures.get(name, {}) if isinstance(outcome_fixtures, dict) else {}
        path_value = (value.get("summary") or value.get("path") or value.get("evidence_directory")) if isinstance(value, dict) else value
        if not isinstance(path_value, str):
            continue
        path = evidence.path(path_value)
        if path.is_dir():
            path /= "summary.json"
        try:
            path.resolve().relative_to(evidence.base.resolve())
        except ValueError:
            continue
        summary = evidence.read(path)
        case = {"stage": "physical_stage_outcome", "source": evidence.source(path),
            "job": {"id": "physical-outcome-" + name, "phase": "sim", "kind": "sim",
                "task": str(evidence.project / "tasks/sim" / (name + ".toml")), "output": str(path.parent)},
            "record": {**summary, "evidence_directory": str(path.parent)}}
        outcome_rows[name] = native_evidence(evidence, case)
    e2e = []
    for name in FIXTURES[1:]:
        matches = [r for r in e2e_attempts if r.get("task") == name or
                   Path(str(r.get("output", ""))).name == name]
        # Completed journal attempts take precedence over unrecorded pending
        # slots; a newer failed attempt must never be hidden by an old pass.
        completed = [r for r in matches if r.get("status") != "pending"]
        latest = (completed or matches or [None])[-1]
        selected = outcome_rows.get(name)
        if latest is not None:
            if selected is None or selected.get("source") != latest.get("source"):
                selected = latest
        if selected is not None:
            selected = {**selected, "fixture_verified": bool(fixture_verified(evidence, selected, name))}
            e2e.append(selected)
    fixtures = {}
    for name in FIXTURES:
        # Use explicit fixture/gate jobs or the root's selected current outcome,
        # never an arbitrary summary found by the historical launch scan.
        matches = [r for r in e2e + final_gates if r.get("task") == name]
        if not matches and name in outcome_rows:
            matches = [outcome_rows[name]]
        accepted = next((r for r in matches if fixture_verified(evidence, r, name)), None)
        fallback = next(iter(matches), {})
        fixtures[name] = {"passed": accepted is not None, "source": (accepted or fallback).get("source"),
            "reason": None if accepted else "缺少无reach输入、真实预验证和动力学通过的完整证据"}
    e2e_passed = all(sum(r.get("task") == name and r.get("fixture_verified") for r in e2e) == 1
                     for name in FIXTURES[1:]) and len(e2e) == 2
    required_platform = required_platform_coverage(evidence, outcome)
    return {"d_e2e": e2e, "d_e2e_attempts": e2e_attempts, "d_e2e_passed": e2e_passed, "gates": final_gates,
        "gates_passed": sum(g["passed"] for g in final_gates),
        "no_reach_fixtures": fixtures, "no_reach_pipeline_passed": all(r["passed"] for r in fixtures.values()),
        "physical_stage_outcome": outcome,
        "required_platform": required_platform,
        "temporary_platform_coverage_passed": required_platform["passed"],
        "d_gate_evidence": [native_evidence(evidence, c) for c in stages["d_gate"]["cases"]],
        "d_gates_evidence": [native_evidence(evidence, c) for c in stages["d_gates"]["cases"]]}


def regressions(evidence):
    full, originals = [], []
    labels = ("abc", "e", "final") if (evidence.project / "out/v133_final_full.json").is_file() else ("abc", "e")
    for label in labels:
        path = evidence.project / "out" / f"v133_{label}_full.json"
        data = evidence.read(path)
        modules = {}
        for module in V11_MODULES:
            tests = [t for t in data.get("per_test", []) if t.get("id", "").startswith(module + ".")]
            modules[module] = {"count": len(tests), "passed": bool(tests) and all(t.get("status") == "passed" for t in tests), "per_test": tests}
        full.append({"label": label, "source": evidence.source(path), "status": data.get("status", "pending"),
            "passed": data.get("status") == "passed" and data.get("tests_run", 0) > 0
                and data.get("failure_count") == 0 and data.get("error_count") == 0 and all(m["passed"] for m in modules.values()),
            "tests_run": data.get("tests_run"), "skipped": data.get("skipped"), "total_seconds": data.get("total_seconds"),
            "v11_modules": modules, "failures": data.get("failures")})
        if label == "final":
            continue
        path = evidence.base / f"original11_{label}.json"
        data = evidence.read(path)
        rows = data.get("rows", [])
        originals.append({"label": label, "source": evidence.source(path), "rows": rows,
            "passed": data.get("passed") is True and len(rows) == 11 and all(r.get("passed") is True for r in rows)})
    if evidence.selection["selection_present"]:
        selected = evidence.selection
        expected = selected.get("planning_code_sha256", {})
        label = "selected:"+selected["selection_id"]
        paths = selected.get("regressions", {})
        full_path = evidence.path(paths["full"]) if paths.get("full") else None
        data = evidence.read(full_path) if full_path else {}
        modules = {}
        for module in V11_MODULES:
            tests = [t for t in data.get("per_test", []) if t.get("id", "").startswith(module+".")]
            modules[module] = {"count": len(tests), "passed": bool(tests) and all(t.get("status") == "passed" for t in tests), "per_test": tests}
        fingerprint_passed = bool(expected) and data.get("planning_code_sha256", data.get("code_sha256")) == expected
        full.append({"label": label, "selected": True, "source": evidence.source(full_path),
            "status": data.get("status", "pending"), "fingerprint_passed": fingerprint_passed,
            "passed": fingerprint_passed and data.get("status") == "passed" and data.get("tests_run", 0) > 0
                and data.get("failure_count") == 0 and data.get("error_count") == 0 and all(m["passed"] for m in modules.values()),
            "tests_run": data.get("tests_run"), "skipped": data.get("skipped"), "total_seconds": data.get("total_seconds"),
            "v11_modules": modules, "failures": data.get("failures")})
        original_path = evidence.path(paths["original11"]) if paths.get("original11") else None
        data = evidence.read(original_path) if original_path else {}
        rows = data.get("rows", [])
        fingerprint_passed = bool(expected) and data.get("planning_code_sha256", data.get("code_sha256")) == expected
        originals.append({"label": label, "selected": True, "source": evidence.source(original_path), "rows": rows,
            "fingerprint_passed": fingerprint_passed,
            "passed": fingerprint_passed and data.get("passed") is True and len(rows) == 11 and all(r.get("passed") is True for r in rows)})
    return {"full": full, "original11": originals,
        "passed": all(r["passed"] for r in full + originals)}


def calibration_evidence(evidence):
    path = evidence.base / "calibration.json"
    data = evidence.read(path)
    valid, duplicates, seen = [], [], set()
    for sample in data.get("samples", []):
        source = sample.get("source_summary")
        key = (source, str(sample.get("step")))
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        if (source and sample.get("step") is not None and finite(sample.get("actual_seconds"), positive=True)
                and finite(sample.get("distance_m")) and real_passed(evidence.read(evidence.path(source)))):
            valid.append(sample)
    return {"source": evidence.source(path), "present": bool(data), "artifact": data,
        "verified_distinct_successful_steps": len(valid), "duplicate_sample_keys": duplicates,
        "passed": bool(data.get("skills")) and len(valid) >= 30}


def d_geometry_contract(evidence):
    path = evidence.base / "support_geometry.json"
    data = evidence.read(path)
    keys = ("strict_tip_centre_80mm_passed", "whole_tip_faces_in_beam_width", "exact_geometry_contract_passed")
    checks = {key: {"value": data.get(key), "passed": data.get(key) is True,
                    "status": "passed" if data.get(key) is True else "not_met"} for key in keys}
    passed = all(row["passed"] for row in checks.values())
    return {"status": "passed" if passed else "not_met", "passed": passed,
        "source": evidence.source(path), "checks": checks, "artifact": data,
        "scope": "D严格静态几何指标独立于第8节真实执行验收；最近实体面通过不能替代指尖中心80mm或完整端面宽度约束"}


def fixture_ik_differences(evidence, physical_data):
    """Compare frozen manual cells to selected real, fully checked IK caches."""
    result = {}
    for name in FIXTURES:
        baseline_path = evidence.base / "baseline_code/tasks/sim" / (name + ".toml")
        selected = ([r for r in physical_data["gates"] if r.get("task") == name] if name == FIXTURES[0]
                    else [r for r in physical_data["d_e2e"] if r.get("task") == name])
        row = selected[-1] if selected else {}
        source = evidence.path(row["source"]) if row.get("source") else None
        item = {"status": "pending", "baseline_source": evidence.source(baseline_path),
            "run_summary": evidence.source(source), "reason": "缺少所选夹具的完整真实IK资格证据", "changes": [], "transitions": {}}
        result[name] = item
        if source is None:
            continue
        try:
            source.resolve().relative_to(evidence.base.resolve())
        except ValueError:
            continue
        summary = evidence.read(source)
        expected_task = (evidence.project / "tasks/sim" / (name + ".toml")).resolve()
        pid = summary.get("simulator_pid")
        if (summary.get("task") != name or not summary.get("task_file")
                or evidence.path(summary["task_file"]).resolve() != expected_task
                or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
                or summary.get("simulator_closed") is not True):
            continue
        directory = source.parent
        progress = evidence.read(directory / "prequalification.json")
        qualified_path = directory / (name + ".qualified.toml")
        item.update(qualified_source=evidence.source(qualified_path),
                    prequalification_source=evidence.source(directory / "prequalification.json"))
        if not baseline_path.is_file() or not qualified_path.is_file():
            continue
        baseline, qualified = evidence.task(baseline_path), evidence.task(qualified_path)
        before, after = reach_cells(baseline), reach_cells(qualified)
        recorded = {(r.get("arm"), r.get("place")): r.get("status") for r in progress.get("records", [])}
        if (qualified.get("capability_source") != "ik_qualified" or not before or set(before) != set(after)
                or progress.get("checked") != progress.get("total") or progress.get("total") != len(after)
                or recorded != after):
            continue
        cells = [{"arm": key[0], "place": key[1], "manual": before[key], "ik": after[key]}
                 for key in sorted(before)]
        item.update(status="compared", reason=None, cells=cells, compared_cells=len(cells),
            changes=[cell for cell in cells if cell["manual"] != cell["ik"]],
            transitions=dict(Counter(cell["manual"] + "→" + cell["ik"] for cell in cells)))
    return result


def collect(base=BASE, *, project=ROOT):
    evidence = Evidence(base, project)
    stages = {name: evidence.stage(name) for name in STAGES}
    # D repairs use fresh stage/ID definitions instead of rerunning a failed
    # experiment. Discover these manifests while preserving all prior rows.
    for path in (evidence.base / "manifests").glob("d_e2e*.json"):
        if path.stem == "d_e2e" or path.stem.startswith(("d_e2e_", "d_e2e-")):
            stages.setdefault(path.stem, evidence.stage(path.stem))
    plan = planning(stages)
    qual = qualification(evidence, stages)
    runs = native_runs(evidence)
    phy = physical(evidence, stages, runs)
    sim_cases = [native_evidence(evidence, c) for c in stages["sim20"]["cases"]]
    families = {}
    for family in FAMILIES:
        rows = [r for r in sim_cases if r.get("family") == family]
        families[family] = {"requested": len(rows), "recorded": sum(r["status"] != "pending" for r in rows),
            "attempted": sum(r["attempted"] for r in rows), "passed": sum(r["real_passed"] for r in rows),
            "skipped": sum(r["skipped"] for r in rows), "denominator": 4,
            "failure_reasons": dict(Counter(reason_label(r.get("reason") or r["status"]) for r in rows if not r["real_passed"]))}
    sim = {"requested": len(sim_cases), "attempted": sum(r["attempted"] for r in sim_cases),
        "passed": sum(r["real_passed"] for r in sim_cases), "denominator": 20, "families": families, "cases": sim_cases}
    sim["success_rate"] = sim["passed"]/20
    sim["each_family_four_attempts"] = len(sim_cases) == 20 and all(f["requested"] == f["attempted"] == 4 for f in families.values())
    ratios = {"sim20_final_estimate_to_actual": distribution(r.get("final_estimate_to_actual") for r in sim_cases if r["real_passed"]),
        "sim20_initial_estimate_to_actual": distribution(r.get("initial_estimate_to_actual") for r in sim_cases if r["real_passed"]),
        "all_new_native_final_estimate_to_actual": distribution(r.get("final_estimate_to_actual") for r in runs if r["real_passed"]),
        "all_new_native_initial_estimate_to_actual": distribution(r.get("initial_estimate_to_actual") for r in runs if r["real_passed"])}
    calibration_path = evidence.base / "calibration.json"
    calibration = calibration_evidence(evidence)
    archive_path = evidence.base / "archive_check.json"
    archive = evidence.read(archive_path)
    median = ratios["sim20_final_estimate_to_actual"]["median"]
    ratio_complete = sim["passed"] > 0 and ratios["sim20_final_estimate_to_actual"]["n"] == sim["passed"]
    conditions = {
        "planning_200": (qual["all_200_qualified"] and plan["ik"]["thresholds_passed"] and plan["unknown_within_5s"],
            "IK表每族40：F1–F3≥95%、F4–F5≥85%；所有unknown含实测耗时且≤5s", stages["planning200"]["manifest"]),
        "sim_20": (sim["each_family_four_attempts"] and sim["passed"] >= 14,
            "五族各4次实际尝试，真实动力学通过≥14/20；缺额不改变分母", stages["sim20"]["manifest"]),
        "no_reach_pipeline": (phy["no_reach_pipeline_passed"],
            "任务不含reach表，塔/悬臂/平台三夹具真实qualify→规划→编译→执行完整链路通过", "v133/summary_metrics.json:physical.no_reach_fixtures"),
        "physical_fixtures_and_gates": (phy["d_e2e_passed"] and phy["gates_passed"] == 5,
            "悬臂/平台端到端通过；五门控各1测零跳过、原生实例已关闭，真实动力学通过5/5", stages["gates"]["manifest"]),
        "calibrated_estimate_ratio": (calibration["passed"] and ratio_complete and median is not None and .7 <= median <= 1.3,
            "≥30步真实成功源的标定产物；sim20成功样本最终规划makespan/实际中位数0.7–1.3，分母齐全", evidence.source(calibration_path)),
        "archive_unchanged": (archive.get("passed") is True,
            "v133_guard归档及保护校验passed", evidence.source(archive_path))}
    acceptance = {name: {"status": "passed" if passed else "not_met", "passed": bool(passed), "requirement": requirement, "evidence": source}
                  for name, (passed, requirement, source) in conditions.items()}
    timing = defaultdict(float)
    for case in sim_cases:
        if not case.get("attempted"):
            continue
        for key, seconds in (case.get("phase_seconds") or {}).items():
            if finite(seconds):
                timing[key] += seconds
    reg = regressions(evidence)
    geometry = d_geometry_contract(evidence)
    from tools import v133_scale_report
    scale = v133_scale_report.collect(base, project=project)
    fixture_differences = fixture_ik_differences(evidence, phy)
    diagnostic_data = diagnostics(evidence, stages, plan, sim_cases, sim)
    trials = []
    for name in evidence.selection.get("diagnostic_trials", []):
        trial = evidence.stage(name, use_selection=False)
        trials.append({"stage": name, "formal": False, "manifest": trial["manifest"],
            "requested": trial["requested"], "recorded": trial["recorded"],
            "status_counts": dict(Counter(c["record"].get("status", "pending") for c in trial["cases"])),
            "cases": [case_fields(c) for c in trial["cases"]]})
    acceptance_passed = all(c["passed"] for c in acceptance.values())
    return {"schema_version": 1, "evidence_version": "v1.3.3", "rebuilt_at": batch.utc_now(),
        "acceptance": acceptance, "acceptance_passed": acceptance_passed,
        "d_geometry_contract": geometry,
        "revision_passed": acceptance_passed and reg["passed"] and geometry["passed"] and scale['passed'],
        "fixture_ik_differences": fixture_differences,
        "planning": plan, "qualification": qual, "simulation": sim, "physical": phy,
        "ratios": ratios, "native_runs": runs, "regressions": reg,
        "calibration": calibration, "scale_acceptance": scale,
        "diagnostics": diagnostic_data, "project_root": str(evidence.project.resolve()),
        "experiment_selection": deepcopy(evidence.selection), "diagnostic_trials": trials,
        "archive": {"source": evidence.source(archive_path), **archive}, "sim20_phase_seconds": dict(timing),
        "stages": {name: {k: s[k] for k in ("logical_stage", "physical_stage", "manifest", "requested", "recorded", "orphan_record_ids")} for name, s in stages.items()},
        "artifact_issues": evidence.issues,
        "scope": "所有归类来自manifest阶段及ID；raw为显式distance_prior，qualified为IK；旧v132事实独立保留；重建不执行任务"}


def markdown(metrics):
    def cell(value):
        return reason_label(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")

    def link(source, label):
        if not source:
            return "缺失"
        path = Path(source)
        if not path.is_absolute():
            path = Path(metrics.get("project_root", ROOT))/path
        return "["+label+"](<"+path.as_posix()+">)"

    lines = ["# v1.3.3 事实验收报告", "", f"重建时间：{metrics['rebuilt_at']}。统计只读取已提交事实，不执行实验。", "",
        "历史：[v1.3.2 原报告](v133/previous_summary.md)；旧 M1/M2/M3 与历史见 [milestones.json](milestones.json)。", "",
        "## 第8节验收", "", "|验收项|结论|要求|证据|", "|---|---|---|---|"]
    for name, item in metrics["acceptance"].items():
        lines.append(f"|{name}|{item['status']}|{item['requirement']}|`{item['evidence']}`|")
    scale = metrics.get('scale_acceptance', {})
    lines += ['', '## 第4节规模验收', '',
        f"独立规模验收：{scale.get('status', 'pending')}。逐任务块数、60秒状态、bound和独立求解证据见 `v133/scale_acceptance.md` / `.json`。", '',
        '|规模组|请求|已提交|60秒内解出|结论|', '|---|---:|---:|---:|---|']
    for group in scale.get('groups', {}).values():
        lines.append(f"|{group['label']}|{group['requested']}|{group['recorded']}|{group['solved_within_60s']}|{group['status']}|")
    geometry = metrics["d_geometry_contract"]
    lines += ["", "## D 严格几何约束（独立口径）", "",
        "D 严格几何约束通过。" if geometry["passed"] else "D 严格几何约束未通过（缺证据同样不计通过）。",
        "真实动力学或门控通过不改变此几何结论；最近实体面80mm内不能替代指尖中心及完整端面宽度要求。", "",
        f"证据：`{geometry['source']}`。", "", "|严格几何项|原始结论|验收|", "|---|---|---|"]
    for name, row in geometry["checks"].items():
        lines.append(f"|{name}|{row['value']}|{row['status']}|")
    lines += ["", "## 三夹具：原手写能力表与真实 IK", "",
        "塔优先采用最终门控、悬臂/平台采用所选最新D端到端；仅比较全部组合已检查、ik_qualified标记的缓存。缺失记pending。", "",
        "|夹具|状态|全部单元|变化单元|原表→IK计数|", "|---|---|---:|---:|---|"]
    for name, row in metrics["fixture_ik_differences"].items():
        lines.append(f"|{name}|{row['status']}|{row.get('compared_cells')}|{len(row['changes'])}|{json.dumps(row['transitions'], ensure_ascii=False)}|")
    for name, row in metrics["fixture_ik_differences"].items():
        if row["changes"]:
            lines += ["", f"{name} 逐单元变化（证据 `{row.get('qualified_source')}`）：", "",
                      "|臂|位置/槽位|原手写表|真实IK|", "|---|---|---|---|"]
            lines += [f"|{cell['arm']}|{cell['place']}|{cell['manual']}|{cell['ik']}|" for cell in row["changes"]]
    choice = metrics["experiment_selection"]
    lines += ["", "## 当前实验选择与旧诊断批次", "",
        f"当前选择 `{choice['selection_id']}`；planning200 实际阶段 `{metrics['stages']['planning200']['physical_stage']}`，"
        f"选择SHA `{choice['selection_sha256']}`。只用该实际清单/日志计算正式规划率、规模验收和后续抽样；选中缺失不回退旧成功。",
        "", "|旧诊断批次（不计正式）|请求|已提交|状态|源清单|", "|---|---:|---:|---|---|"]
    for trial in metrics["diagnostic_trials"]:
        lines.append(f"|{trial['stage']}|{trial['requested']}|{trial['recorded']}|{cell(trial['status_counts'])}|{link(trial['manifest'], 'manifest')}|")
    lines += ["", "## 规划：距离先验与 IK", "", "|族|先验 solved/40|IK solved/40|IK阈值|", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        p, q = (metrics["planning"][table]["families"][family] for table in ("distance_prior", "ik"))
        lines.append(f"|{family}|{p['solved']}/40|{q['solved']}/40|{q['threshold']:.0%}|")
    qual = metrics["qualification"]
    lines += ["", f"资格组合检查 {qual['checked_combinations']}/{qual['expected_combinations']}；"
        f"UNKNOWN→VALID {qual['unknown_to_valid']}，UNKNOWN→INVALID {qual['unknown_to_invalid']}。",
        "全部组合、每个组合位姿/种子/结论/耗时、距离先验→IK差异和逐例未解原因、bound、扩展数、耗时及辅助步骤见 summary_metrics.json。"]
    diagnostic = metrics["diagnostics"]
    lines += ["", "## 规划状态分类及逐族未解清单", "",
        "每族每列固定分母40。直接保留solved/unknown/budget/no_plan等规划结论，UNKNOWN不推断为绝对不可达；pending和清单缺额单列。", "",
        "|列|族|状态计数|清单缺额/40|", "|---|---|---|---:|"]
    for table, data in diagnostic["planning"].items():
        for family, group in data["families"].items():
            lines.append(f"|{table}|{family}|{cell(group['status_counts'])}|{group['missing_manifest_slots']}|")
    for table, data in diagnostic["planning"].items():
        for family, group in data["families"].items():
            if not group["unsolved_cases"]:
                continue
            lines += ["", f"### {table} / {family} 逐族未解或待提交", "",
                      "|ID|状态|耗时(s)|原因/未知事实|证据|", "|---|---|---:|---|---|"]
            for row in group["unsolved_cases"]:
                reason = {"reason": row.get("reason"), "unknown": row.get("unknown")}
                sources = link(row.get("source"), "提交日志")+" / "+(
                    link(row.get("plan_source"), "plan.json") if row["plan_present"] else "plan.json缺失")
                lines.append(f"|{cell(row['id'])}|{row['status']}|{row.get('elapsed_seconds')}|{cell(reason)}|{sources}|")
    lines += ["", "## sim20", "", "|族|请求|实际尝试|真实动力学通过/4|跳过|", "|---|---:|---:|---:|---:|"]
    sim = metrics["simulation"]
    for family, row in sim["families"].items():
        lines.append(f"|{family}|{row['requested']}|{row['attempted']}|{row['passed']}/4|{row['skipped']}|")
    lines += ["", f"真实通过 {sim['passed']}/20（{sim['success_rate']:.1%}），实际尝试 {sim['attempted']}/20。",
        "实际执行/尝试表示启动完整任务链路，包含规划或编译失败，不等于动力学完成；编译失败口径包含real_planning中的compile_action失败及最终compile，阶段详见CSV；成功必须真实动力学验收passed。"]
    failure = diagnostic["simulation"]
    lines += ["", "## sim20 失败分类", "",
        f"实际尝试但未通过 {failure['attempted_failures']}/20（{failure['failure_rate']:.1%}）；"
        f"未通过槽位合计 {failure['not_passed_slots']}/20，包含pending/跳过/缺额，不能全称动力学失败。清单缺额 {failure['missing_manifest_slots']}。",
        "分类只使用failure_phase及明确报告原因；无法细分为unclassified，种子耗尽或UNKNOWN不等于数学不可达。", "",
        "|报告分类|条数|", "|---|---:|"]
    for category, count in failure["category_counts"].items():
        lines.append(f"|{cell(category)}|{count}|")
    lines += ["", "|ID/族|状态/阶段|分类|明确原因|原生证据|", "|---|---|---|---|---|"]
    for row in failure["cases"]:
        lines.append(f"|{cell(row.get('id'))}/{row.get('family')}|{row.get('status')}/{row.get('failure_phase')}|"
                     f"{cell(row['category'])}|{cell(row.get('reason'))}|{link(row.get('source'), 'summary.json')}|")
    lines += ["",
        "## 实体与门控证据", "", "|项目|通过|来源|", "|---|---|---|"]
    physical_data = metrics["physical"]
    for row in physical_data["d_e2e"]:
        lines.append(f"|端到端 {row.get('task') or row['id']}|{row['real_passed']}|`{row['source']}`|")
    for name, row in physical_data["no_reach_fixtures"].items():
        lines.append(f"|无reach完整链路 {name}|{row['passed']}|`{row['source']}`|")
    for row in physical_data["gates"]:
        lines.append(f"|{row['test']}|{row['passed']}|`{row['source']}`|")
    extra = physical_data["required_platform"]
    lines += ["", "## 临时台搭拆专项（额外独立覆盖）", "",
        "原平台任务允许臂托，其端到端成功不能作为临时台搭建/拆除通过。本专项仅采用physical_stage_outcome明确选中的阶段，缺证据保留pending；不改变第8节六项验收口径。", "",
        "|独立专项|结论|实际尝试|阶段|进入TPG|进入动力学|来源|", "|---|---|---|---|---|---|---|",
        f"|{extra['task']}|{extra['status']}|{extra['attempted']}|{extra['phase']}|{extra['tpg_entered']}|{extra['dynamics_entered']}|`{extra['source']}`|",
        "", f"所选阶段：`{extra['stage']}`；原因：{reason_label(extra['reason'])}。",
        "", "真实后端反馈："+reason_label(extra["feedback"])+"。", "",
        "有限IK种子失败或UNKNOWN只说明本次后端未验证成功，不能据此证明数学不可达。"]
    lines += ["", "## 估算与实际", "", "分子只用落盘的 initial_plan.json / plan.json 规划 makespan；不使用编译后的 nominal_makespan_seconds。", "",
        "|样本口径|n|min|p25|median|p75|max|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in metrics["ratios"].items():
        lines.append(f"|{name}|{row['n']}|{row['min']}|{row['p25']}|{row['median']}|{row['p75']}|{row['max']}|")
    lines += ["", "## 显式反馈消息统计", "", diagnostic["feedback"]["scope"]+"。", "",
        "只计明确pipeline文案的消息：运动失败且已禁止/重搜为cut，运动后端按需验证为能力回填，运动验证未知单列且不算cut。"
        "回填按事实计消息，不能冒充完整反馈轮次数。qualification_calls包含全量预资格，只并列给出，不计为反馈次数。", "",
        "|口径|cut消息|按需回填消息|验证未知消息|其他消息|qualification_calls（含全量预资格）|",
        "|---|---:|---:|---:|---:|---:|"]
    for scope in ("sim20", "committed_native"):
        stats = diagnostic["feedback"][scope]
        lines.append(f"|{scope}|{stats['invalid_cut_messages']}|{stats['qualification_update_messages']}|"
                     f"{stats['motion_unknown_messages']}|{stats['other_messages']}|{stats['qualification_calls_observed']}|")
    for scope in ("sim20", "committed_native"):
        stats = diagnostic["feedback"][scope]
        if stats["reasons"]:
            lines += ["", f"{scope} 明确消息与原因：", "", "|任务|消息类别|原消息|来源|", "|---|---|---|---|"]
            for row in stats["reasons"]:
                lines.append(f"|{cell(row.get('id'))}|{row['kind']}|{cell(row['message'])}|{link(row.get('source'), 'summary.json')}|")
    lines += ["", "## 阶段耗时汇总", "", diagnostic["timing_scope"]+"。", "",
              "|口径|阶段/字段|已记录秒数合计|", "|---|---|---:|"]
    for scope, phases in diagnostic["phase_seconds"].items():
        for phase, seconds in phases.items():
            lines.append(f"|{scope}|{phase}|{seconds:.6f}|")
    lines += ["", f"新版本原生任务启动去重后 {len(metrics['native_runs'])} 个；重复引用、两套路线、每阶段及逐步工时、反馈原因均保存在 summary_metrics.json。", "",
        "## 第7节独立回归", "", "|证据|通过|测试/任务数|", "|---|---|---:|"]
    for row in metrics["regressions"]["full"]:
        lines.append(f"|`{row['source']}`|{row['passed']}|{row['tests_run']}|")
        for name, module in row["v11_modules"].items():
            lines.append(f"|{name} per_test|{module['passed']}|{module['count']}|")
    for row in metrics["regressions"]["original11"]:
        lines.append(f"|`{row['source']}`|{row['passed']}|{len(row['rows'])}/11|")
    if metrics["artifact_issues"]:
        lines += ["", "## 缺失或不完整证据", "", *["- " + json.dumps(issue, ensure_ascii=False) for issue in metrics["artifact_issues"]]]
    return "\n".join(lines) + "\n"


def rebuild(base=BASE, *, project=ROOT):
    base, project = Path(base), Path(project)
    metrics = collect(base, project=project)
    text = markdown(metrics)
    evidence = Evidence(base, project)
    milestones_path = base.parent / "milestones.json"
    milestones = evidence.read(milestones_path)
    # Retain all prior M1..M5 and history. v133 is a separate evidence version.
    milestones["v1.3.3"] = {"status": "passed" if metrics["revision_passed"] else "not_met",
        "acceptance": deepcopy(metrics["acceptance"]), "regression_passed": metrics["regressions"]["passed"],
        "scale_status": metrics['scale_acceptance']['status'],
        "temporary_platform_coverage": deepcopy(metrics["physical"]["required_platform"]),
        "experiment_selection": deepcopy(metrics["experiment_selection"]),
        "d_geometry_contract": deepcopy(metrics["d_geometry_contract"]),
        "metrics": evidence.source(base / "summary_metrics.json"), "updated_at": metrics["rebuilt_at"]}
    batch.atomic_json(base / "summary_metrics.json", metrics)
    from tools import v133_scale_report
    batch.atomic_json(base / 'scale_acceptance.json', metrics['scale_acceptance'])
    batch.atomic_text(base / 'scale_acceptance.md', v133_scale_report.markdown(metrics['scale_acceptance']))
    batch.atomic_json(base / "fixture_ik_differences.json", metrics["fixture_ik_differences"])
    batch.atomic_text(base / "summary.md", text.replace("v133/previous_summary.md", "previous_summary.md").replace(
        "[milestones.json](milestones.json)", "[milestones.json](../milestones.json)"))
    batch.atomic_text(base.parent / "summary.md", text)
    batch.atomic_json(milestones_path, milestones)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅从事实重建v1.3.3报告，不运行规划或仿真")
    parser.add_argument("--rebuild", action="store_true")
    parser.parse_args(argv)
    metrics = rebuild()
    print(json.dumps({"acceptance": metrics["acceptance_passed"], "regressions": metrics["regressions"]["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
