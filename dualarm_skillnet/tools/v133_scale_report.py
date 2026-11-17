"""Read committed v1.3.3 planning evidence for FIX section 4; never solve."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/sim_v1_3/v133"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.v133_experiments import load_selection, resolve_stage
TARGET_FAMILIES = ("F1", "F3", "F5")
REQUIRED_GROUPS = ("F1_5_6", "F3_up_to_10", "F5_7")
LABELS = {"F1_5_6": "F1，5–6 块", "F3_up_to_10": "F3，不超过 10 块", "F5_7": "F5，7 块"}


def scale_group(family, blocks):
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        return None
    if family == "F1" and 5 <= blocks <= 6:
        return "F1_5_6"
    if family == "F3" and blocks <= 10:
        return "F3_up_to_10"
    if family == "F3" and 12 <= blocks <= 15:
        return "F3_12_15"
    if family == "F5" and blocks == 7:
        return "F5_7"
    return None


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class Evidence:
    def __init__(self, base, project):
        self.base, self.project = Path(base).resolve(), Path(project).resolve()
        self.issues = []
        self.selection = load_selection(self.base, project=self.project)
        self.planning_stage = resolve_stage("planning200", self.base, selection=self.selection, project=self.project)

    def source(self, path):
        path = Path(path).resolve()
        return path.relative_to(self.project).as_posix() if path.is_relative_to(self.project) else str(path)

    def confined(self, value):
        if not value:
            return None
        path = Path(value)
        path = (path if path.is_absolute() else self.project/path).resolve()
        return path if path.is_relative_to(self.base) else None

    def read(self, path):
        if path is None or not Path(path).is_file():
            return {}
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(result, dict):
                raise ValueError("JSON object required")
            return result
        except (OSError, ValueError, UnicodeError) as error:
            self.issues.append({"source": self.source(path), "reason": str(error)})
            return {}

    def journal(self):
        directory = self.base/"states"/self.planning_stage
        paths = [directory/"completed.jsonl", *sorted(directory.glob("completed.recovery_*.jsonl"))]
        records, conflicts = {}, set()
        for path in paths:
            if not path.is_file():
                continue
            for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
                if not line.endswith(b"\n"):
                    self.issues.append({"source": self.source(path), "line": line_number,
                                        "reason": "uncommitted_tail_ignored"})
                    continue
                try:
                    row = json.loads(line)
                    ident = row["id"]
                    if not isinstance(ident, str):
                        raise ValueError("id must be text")
                    if ident in records and records[ident] != row:
                        conflicts.add(ident)
                    records[ident] = row
                except (ValueError, KeyError, TypeError, UnicodeError) as error:
                    self.issues.append({"source": self.source(path), "line": line_number,
                                        "reason": "invalid_committed_record: "+str(error)})
        for ident in conflicts:
            records.pop(ident, None)
        return records, conflicts

    def case(self, job, row, *, conflict=False, duplicate=False):
        reasons = []
        path = self.confined(job.get("task"))
        task, digest, count = {}, None, None
        if path is None or not path.is_file():
            reasons.append("qualified_input_missing_or_outside_current_v133")
        else:
            try:
                task = tomllib.loads(path.read_text(encoding="utf-8-sig"))
                blocks = task.get("block")
                if isinstance(blocks, list) and blocks:
                    count = len(blocks)
                else:
                    reasons.append("input_block_list_missing_or_empty")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError, UnicodeError) as error:
                reasons.append("qualified_input_invalid")
                self.issues.append({"source": self.source(path), "reason": str(error)})
        if task and (task.get("capability_source") != "ik_qualified"
                     or "reach" not in task.get("capabilities", {})):
            reasons.append("input_not_ik_qualified")
        if digest is not None and job.get("prepared_task_sha256") != digest:
            reasons.append("manifest_input_sha_mismatch_or_missing")
        if duplicate:
            reasons.append("duplicate_manifest_id_or_family_index")
        if conflict:
            reasons.append("conflicting_committed_records")
        output = self.confined(job.get("output"))
        if output is None:
            reasons.append("output_outside_current_v133")
        proof_path = output/"plan.json" if output else None
        proof = self.read(proof_path) if row else {}
        bound_present = "bound" in proof or "suboptimality_bound" in proof
        bound = proof.get("bound", proof.get("suboptimality_bound"))
        elapsed = None
        if row:
            if row.get("task_sha256") != digest or digest is None:
                reasons.append("committed_input_sha_mismatch_or_missing")
            if row.get("table", "qualified") != "qualified":
                reasons.append("committed_table_is_not_qualified")
            if not proof:
                reasons.append("plan_artifact_missing_or_invalid")
            else:
                if proof.get("proposal_enabled") is not False or proof.get("witness_loaded") is not False:
                    reasons.append("independent_search_flags_missing_or_enabled")
                if not task.get("name") or proof.get("task") != task["name"]:
                    reasons.append("plan_input_task_identity_mismatch")
                if row.get("status") != proof.get("status"):
                    reasons.append("committed_status_differs_from_plan")
                if not bound_present:
                    reasons.append("bound_not_recorded")
                elif bound is not None and (not finite(bound) or bound < 1-1e-8):
                    reasons.append("invalid_suboptimality_bound")
                    bound = None
                if "bound" in row and row["bound"] != proof.get("bound", proof.get("suboptimality_bound")):
                    reasons.append("committed_bound_differs_from_plan")
                times = [row.get("elapsed_seconds"), proof.get("elapsed_seconds")]
                if all(finite(value) and value >= 0 for value in times):
                    elapsed = max(times)
                else:
                    reasons.append("elapsed_evidence_missing_or_invalid")
        group = scale_group(job.get("family"), count)
        required = group in REQUIRED_GROUPS
        solved = bool(row and row.get("status") == "solved" and row.get("passed") is True
                      and proof.get("status") == "solved" and proof.get("passed") is True)
        if row and required:
            if not solved:
                reasons.append("not_independently_solved")
            if elapsed is not None and elapsed > 60:
                reasons.append("elapsed_exceeds_60s")
        reasons = list(dict.fromkeys(reasons))
        status = ("not_met" if reasons else "pending" if not row else
                  "passed" if required else "reported")
        return {"id": job.get("id"), "family": job.get("family"), "index": job.get("index"),
                "table": "qualified", "blocks": count, "group": group,
                "status": row.get("status", "pending"), "plan_status": proof.get("status"),
                "acceptance_status": status, "reasons": reasons, "elapsed_seconds": elapsed,
                "journal_elapsed_seconds": row.get("elapsed_seconds") if finite(row.get("elapsed_seconds")) else None,
                "plan_elapsed_seconds": proof.get("elapsed_seconds") if finite(proof.get("elapsed_seconds")) else None, "bound": bound,
                "bound_recorded": bound_present, "independent_search_proved": bool(proof)
                    and proof.get("proposal_enabled") is False and proof.get("witness_loaded") is False,
                "committed": bool(row), "task_sha256": digest,
                "input": self.source(path) if path else job.get("task"),
                "plan": self.source(proof_path) if proof_path else job.get("output"),
                "result_reason": row.get("reason", job.get("skip_reason"))}


def collect(base=BASE, *, project=ROOT):
    evidence = Evidence(base, project)
    manifest_path = evidence.base/"manifests"/(evidence.planning_stage+".json")
    manifest = evidence.read(manifest_path)
    jobs = manifest.get("jobs", [])
    jobs = [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []
    records, conflicts = evidence.journal()
    selected = [job for job in jobs if job.get("kind") == "plan" and job.get("table") == "qualified"
                and job.get("family") in TARGET_FAMILIES]
    ids = Counter(job.get("id") for job in jobs)
    combinations = Counter((job.get("family"), job.get("index")) for job in selected)
    coverage = {}
    for family in TARGET_FAMILIES:
        group = [job for job in selected if job.get("family") == family]
        indices = [job.get("index") for job in group]
        complete = (len(indices) == 40 and all(isinstance(i, int) and not isinstance(i, bool) for i in indices)
                    and set(indices) == set(range(40)) and all(ids[job.get("id")] == 1 for job in group))
        coverage[family] = {"requested": len(group), "expected": 40,
                            "unique_indices": len(set(indices)), "complete": complete}
    cases = [evidence.case(job, records.get(job.get("id"), {}),
             conflict=job.get("id") in conflicts,
             duplicate=ids[job.get("id")] != 1 or combinations[job.get("family"), job.get("index")] != 1)
             for job in selected]
    groups = {}
    for name in REQUIRED_GROUPS:
        group = [case for case in cases if case["group"] == name]
        represented = sorted({case["blocks"] for case in group})
        size_coverage = set(represented) >= ({5, 6} if name == "F1_5_6" else {7} if name == "F5_7" else set())
        size_coverage = size_coverage and bool(group)
        states = Counter(case["acceptance_status"] for case in group)
        status = ("not_met" if states["not_met"] else "pending" if states["pending"] else
                  "passed" if size_coverage else "not_met" if selected else "pending")
        groups[name] = {"label": LABELS[name], "status": status, "passed": status == "passed",
            "requested": len(group), "recorded": sum(case["committed"] for case in group),
            "solved_within_60s": states["passed"], "pending": states["pending"], "not_met": states["not_met"],
            "represented_block_counts": represented, "size_coverage_passed": size_coverage, "cases": group}
    unclassified = [case for case in cases if case["blocks"] is None]
    large = [case for case in cases if case["group"] == "F3_12_15"]
    large_evidence_complete = bool(large) and all(case["acceptance_status"] == "reported" for case in large)
    large_failed = any(case["acceptance_status"] == "not_met" for case in large)
    large_status = ("reported" if large_evidence_complete else "not_met" if large_failed or (selected and not large)
                    else "pending")
    coverage_passed = all(row["complete"] for row in coverage.values())
    passed = (coverage_passed and not unclassified and large_evidence_complete
              and all(row["passed"] for row in groups.values()))
    any_committed = bool(records or conflicts)
    if passed:
        status = "passed"
    elif not manifest_path.exists() and not any_committed:
        status = "pending"
    elif not any_committed and selected and not any(case["reasons"] for case in cases):
        status = "pending"
    elif (any(row["status"] == "not_met" for row in groups.values()) or unclassified
          or not coverage_passed or large_status == "not_met"):
        status = "not_met"
    else:
        status = "pending"
    return {"schema_version": 1, "status": status, "passed": passed,
            "selected_stage": evidence.planning_stage, "experiment_selection": evidence.selection,
            "scope": "FIX_v1.3.3 第4节规模验收；不是200任务各族求解率或真实执行验收",
            "manifest": evidence.source(manifest_path),
            "journal": evidence.source(evidence.base/"states"/evidence.planning_stage/"completed.jsonl"),
            "coverage_requirement": "planning200清单的F1/F3/F5各40个qualified任务、索引0–39；仅统计当前清单的已提交证据",
            "coverage": coverage, "coverage_passed": coverage_passed,
            "groups": groups,
            "large_f3": {"label": "F3，12–15块（允许未解，逐项保留真实状态）", "requested": len(large),
                "recorded": sum(case["committed"] for case in large),
                "status": large_status, "evidence_complete": large_evidence_complete,
                "status_counts": dict(Counter(case["status"] for case in large)), "cases": large},
            "unclassified_input_cases": unclassified,
            "other_size_cases": [case for case in cases if case["blocks"] is not None and case["group"] is None],
            "ignored_raw_jobs": sum(job.get("kind") == "plan" and job.get("table") == "raw" for job in jobs),
            "orphan_record_ids": sorted(set(records)-set(ids)), "conflicting_record_ids": sorted(conflicts),
            "artifact_issues": evidence.issues,
            "elapsed_rule": "当前提交行及其plan.json都需提供非负有限elapsed_seconds，采用两者较大值且不超过60秒",
            "bound_rule": "记录plan.json的bound/suboptimality_bound；None如实表示没有有限界，不伪造为1",
            "search_performed": False, "qualification_performed": False, "simulation_performed": False}


def markdown(metrics):
    labels = {"passed": "通过", "not_met": "未达标", "pending": "等待结果", "reported": "已记录"}
    lines = ["# v1.3.3 第4节规模验收", "", "结论：**"+labels[metrics["status"]]+"**。", "",
             f"当前实际规划阶段 `{metrics['selected_stage']}`；选择 `{metrics['experiment_selection']['selection_id']}`。只读所选清单与提交日志，不回退旧试跑。", "",
             metrics["scope"]+"。", "", metrics["coverage_requirement"]+"。", "",
             "|规模组|样本数|已提交|独立solved且≤60秒|待提交|未达标|结论|",
             "|---|---:|---:|---:|---:|---:|---|"]
    for group in metrics["groups"].values():
        lines.append(f"|{group['label']}|{group['requested']}|{group['recorded']}|{group['solved_within_60s']}|"
                     f"{group['pending']}|{group['not_met']}|{labels[group['status']]}|")
    lines += ["", "F1 必须同时覆盖5块和6块；三组不得以空样本通过。清单覆盖：",
              "", "|族|清单qualified任务数|预期|索引/ID完整|", "|---|---:|---:|---|"]
    for family, row in metrics["coverage"].items():
        lines.append(f"|{family}|{row['requested']}|{row['expected']}|{row['complete']}|")
    source_base = Path(metrics["manifest"]).parent.parent

    def link(source, label):
        try:
            return "["+label+"]("+Path(source).relative_to(source_base).as_posix()+")"
        except (TypeError, ValueError):
            return "缺失" if not source else "`"+str(source)+"`"

    for name, group in [*metrics["groups"].items(), ("F3_12_15", metrics["large_f3"])]:
        lines += ["", "## "+group["label"], "", "|任务|块数|原始状态|秒|bound|独立证明|验收/证据问题|来源|",
                  "|---|---:|---|---:|---|---|---|---|"]
        for case in group["cases"]:
            bound = "null（未提供有限界）" if case["bound"] is None else str(case["bound"])
            reasons = "; ".join(case["reasons"]) or labels[case["acceptance_status"]]
            lines.append(f"|{case['id']}|{case['blocks']}|{case['status']}|{case['elapsed_seconds']}|{bound}|"
                         f"{case['independent_search_proved']}|{reasons}|"
                         +link(case["input"], "输入")+" / "+link(case["plan"], "plan.json")+"|")
        if not group["cases"]:
            lines.append("尚无当前清单对应的该规模样本；未用其他规模、距离先验列或旧结果补位。")
    if metrics["unclassified_input_cases"] or metrics["artifact_issues"] or metrics["conflicting_record_ids"]:
        lines += ["", "## 缺失或冲突证据", "", "```json", json.dumps({
            "unclassified_input_cases": metrics["unclassified_input_cases"],
            "artifact_issues": metrics["artifact_issues"], "conflicting_record_ids": metrics["conflicting_record_ids"]},
            ensure_ascii=False, indent=2), "```"]
    lines += ["", metrics["elapsed_rule"]+"。", "", metrics["bound_rule"]+"。", "",
              f"源清单 `{metrics['manifest']}`；提交日志 `{metrics['journal']}`。", "",
              "本工具只重建报告，不求解、不运行IK、不启动仿真、不修改输入或既有结果。"]
    return "\n".join(lines)+"\n"


def rebuild(base=BASE, *, project=ROOT):
    base, project = Path(base).resolve(), Path(project).resolve()
    if not base.is_relative_to(project):
        raise ValueError("规模报告只能写到项目根目录内")
    metrics = collect(base, project=project)
    base.mkdir(parents=True, exist_ok=True)
    (base/"scale_acceptance.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
        allow_nan=False)+"\n", encoding="utf-8")
    (base/"scale_acceptance.md").write_text(markdown(metrics), encoding="utf-8")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅从当前已提交规划证据重建第4节规模报告，不运行实验")
    parser.add_argument("--rebuild", action="store_true", help="默认行为；仅重建报告")
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args(argv)
    metrics = rebuild(args.base)
    print(json.dumps({"status": metrics["status"], "passed": metrics["passed"],
                      "output": str(args.base/"scale_acceptance.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
