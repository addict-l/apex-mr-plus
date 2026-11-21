"""Prepare immutable overnight manifests; importing this module starts no work.

``plan`` generates new, validated task files and schedules independent searches.
``select`` schedules only independently solved tasks, never witness successes.
``gates`` appends the five explicit acceptance tests. Execution belongs solely
to overnight_batch.py. The shared state directory retains the five-hour clock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as batch


OVERNIGHT = ROOT / "results/sim_v1_3/overnight"
DEFAULT_MANIFEST = OVERNIGHT / "manifest.json"
OLD_GENERATED = ROOT / "results/sim_v1_3/offline_v1_3_2/generated"
NEW_GENERATED = OVERNIGHT / "generated"
FIXTURES = tuple(ROOT / "tasks/sim" / (name + ".toml") for name in (
    "s2_tower_sim", "s3a_cantilever_sim", "s3a_platform_sim"))
FIXTURE_WITNESSES = ROOT / "results/sim_v1_3/offline_v1_3_2/layout_rule_witnesses.json"
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
PLANNING_SEED = 20261115
PER_FAMILY = 40
PER_SELECTION = 4


def relative(path):
    return batch.confined(path).relative_to(ROOT).as_posix()


def _default_manifest():
    return {"schema_version": 1, "session_id": "overnight_20261115", "port": batch.PORT,
        "state_dir": relative(OVERNIGHT / "batch_state"), "report_dir": "results/sim_v1_3",
        "session_budget_seconds": None,
        "phase_budgets_seconds": {"plan": None, "sim": batch.DEFAULT_SIM_BUDGET, "gates": None},
        "planning_seed": PLANNING_SEED, "planning_per_family": PER_FAMILY,
        "jobs": [], "preparation": {}}


def _read_manifest(path):
    path = batch.confined(path)
    data = batch.read_json(path) if path.exists() else _default_manifest()
    if data.get("session_id") != "overnight_20261115":
        raise ValueError("本准备器只维护 overnight_20261115 会话")
    data.setdefault("preparation", {})
    data.setdefault("jobs", [])
    return data


def _save_manifest(path, manifest):
    """Validate before replacement; preserve every earlier raw job definition."""
    path = batch.confined(path)
    temporary = path.with_name(path.stem + ".validate_" + uuid.uuid4().hex + ".json")
    try:
        batch.atomic_json(temporary, manifest)
        normalized = batch.load_manifest(temporary)
        journal = batch.Journal(normalized["state_dir"]).load()
        for job in normalized["jobs"]:
            previous = journal.get(job["id"])
            if previous is not None and previous.get("definition_sha256") != job["definition_sha256"]:
                raise ValueError("已完成任务定义发生变化，拒绝写清单：" + job["id"])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_job(manifest, job):
    for existing in manifest["jobs"]:
        if existing["id"] == job["id"]:
            if existing != job:
                raise ValueError("保留已有任务定义，拒绝覆盖：" + job["id"])
            return False
    manifest["jobs"].append(job)
    return True


def _estimate(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("见证估算必须为有限非负数")
    return value


def _planning_job(job_id, family, task, output, cohort, *, witness=None, witness_source=None, seed=None, index=None):
    path = batch.confined(task)
    job = {"id": job_id, "phase": "plan", "family": family, "task": relative(path),
        "output": relative(output), "cohort": cohort, "timeout_seconds": batch.HARD_TIMEOUT,
        "prepared_task_sha256": batch.sha256(path)}
    if witness is not None:
        job.update(witness_estimate_seconds=_estimate(witness), witness_source=relative(witness_source),
                   witness_scope="串行构造见证，仅用于事后对照；独立搜索不读取见证")
    if seed is not None:
        job.update(generation_seed=seed, generation_index=index)
    return job


def _existing_generated(directory, family, seed, index):
    name = f"gen_{family.lower()}_s{seed}_{index:03d}"
    task = directory / (name + ".toml")
    paths = {suffix: directory / (name + "." + suffix + ".json")
             for suffix in ("generation", "geometry", "witness")}
    if not task.exists() or not all(path.exists() for path in paths.values()):
        return None
    generation, layout, witness = (batch.read_json(paths[key]) for key in ("generation", "geometry", "witness"))
    if (generation.get("family"), generation.get("seed"), generation.get("index")) != (family, seed, index):
        raise ValueError("保留已有生成文件，但其身份不匹配：" + str(task))
    if not layout.get("valid") or not witness.get("rules_replayed") or not witness.get("goal_satisfied"):
        raise ValueError("已有生成任务的几何/规则证据未通过：" + str(task))
    return {"path": str(task), "witness": witness, "layout": layout,
            "attempts": generation.get("attempts"), "parameters": generation.get("parameters")}


def _precheck_jobs(fixtures, old_generated, fixture_witnesses):
    reports = batch.read_json(fixture_witnesses).get("reports", [])
    estimates = {row["task"]: row.get("serial_witness_estimate_seconds") for row in reports}
    jobs = []
    for task in fixtures:
        jobs.append(_planning_job("precheck-" + task.stem, "fixture", task,
            OVERNIGHT / "planning_precheck" / task.stem, "precheck",
            witness=estimates.get(task.stem), witness_source=fixture_witnesses))
    for family in FAMILIES:
        for index in range(2):
            task = old_generated / f"gen_{family.lower()}_s{PLANNING_SEED}_{index:03d}.toml"
            witness_path = task.with_suffix(".witness.json")
            witness = batch.read_json(witness_path)
            jobs.append(_planning_job(f"precheck-{family.lower()}-{index:03d}", family, task,
                OVERNIGHT / "planning_precheck" / task.stem, "precheck",
                witness=witness.get("serial_witness_estimate_seconds"), witness_source=witness_path))
    return jobs


def prepare_plan(path=DEFAULT_MANIFEST, *, generator=None, generated=NEW_GENERATED,
                 old_generated=OLD_GENERATED, fixtures=FIXTURES, fixture_witnesses=FIXTURE_WITNESSES):
    """Prepare 13 prechecks followed by 200 independent jobs; perform no search."""
    path, generated = batch.confined(path), batch.confined(generated)
    old_generated, fixture_witnesses = batch.confined(old_generated), batch.confined(fixture_witnesses)
    if generated == old_generated or generated.is_relative_to(old_generated):
        raise ValueError("新批次不得写入已验收的离线样例目录")
    if generator is None:
        from gen.generator import generate
        generator = generate
    with batch.execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        manifest = _read_manifest(path)
        if manifest.get("planning_seed", PLANNING_SEED) != PLANNING_SEED or manifest.get("planning_per_family", PER_FAMILY) != PER_FAMILY:
            raise ValueError("本轮固定使用 seed=20261115、每族 40 项")
        for job in _precheck_jobs(fixtures, old_generated, fixture_witnesses):
            _append_job(manifest, job)
        manifest["preparation"]["planning"] = {"status": "generating", "seed": PLANNING_SEED,
            "per_family": PER_FAMILY, "generated_directory": relative(generated),
            "independent_pipeline": True, "witness_hints": False}
        _save_manifest(path, manifest)
        existing = {job["id"]: job for job in manifest["jobs"]}
        for family in FAMILIES:
            for index in range(PER_FAMILY):
                job_id = f"plan-{family.lower()}-{index:03d}"
                if job_id in existing:
                    saved = existing[job_id]
                    if batch.sha256(batch.confined(saved["task"])) != saved.get("prepared_task_sha256"):
                        raise ValueError("已准备任务内容改变，拒绝重写：" + job_id)
                    continue
                report = _existing_generated(generated, family, PLANNING_SEED, index)
                if report is None:
                    report = generator(family, PLANNING_SEED, index, generated)
                if not report.get("path") or not report.get("layout", {}).get("valid"):
                    raise ValueError("生成器没有返回通过几何检查的任务文件")
                if not report.get("witness", {}).get("rules_replayed") or not report["witness"].get("goal_satisfied"):
                    raise ValueError("生成器没有返回通过规则检查的任务")
                task = batch.confined(report["path"])
                if not task.is_relative_to(generated):
                    raise ValueError("生成器返回的新任务不在本轮新目录内")
                job = _planning_job(job_id, family, task,
                    OVERNIGHT / "planning_200" / task.stem, "planning_200",
                    witness=report["witness"].get("serial_witness_estimate_seconds"),
                    witness_source=task.with_suffix(".witness.json"), seed=PLANNING_SEED, index=index)
                _append_job(manifest, job)
            # Generator sidecars retain each completed sample. A family-level
            # manifest checkpoint avoids repeatedly hashing all earlier inputs.
            _save_manifest(path, manifest)
        manifest["preparation"]["planning"]["status"] = "ready"
        manifest["preparation"]["planning"]["prepared_jobs"] = sum(job["phase"] == "plan" for job in manifest["jobs"])
        _save_manifest(path, manifest)
        rebuild_precheck_comparison(path, manifest=manifest)
        return manifest


def _truth(value):
    return value is True or str(value).lower() in {"true", "1"}


def _planning_rows(manifest):
    """The fsynced journal takes precedence; CSV is an artifact-checked fallback."""
    rows = batch.Journal(manifest["state_dir"]).load()
    csv_path = batch.confined(manifest["report_dir"]) / "planning_200.csv"
    csv_rows = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = {row["id"]: row for row in csv.DictReader(stream)}
    candidates, rejected = {family: [] for family in FAMILIES}, []
    normalized = batch.load_manifest(manifest["manifest_path"])
    normalized_jobs = {job["id"]: job for job in normalized["jobs"]}
    for job in manifest["jobs"]:
        if job["phase"] != "plan" or job.get("cohort") != "planning_200":
            continue
        row = rows.get(job["id"], csv_rows.get(job["id"]))
        if row is None or row.get("status") != "solved" or not _truth(row.get("passed")):
            continue
        reason = None
        if job["id"] in rows and row.get("definition_sha256") != normalized_jobs[job["id"]]["definition_sha256"]:
            reason = "完成记录与当前任务定义不符"
        if job["id"] not in rows:
            if (not row.get("task") or not row.get("output")
                    or batch.confined(row["task"]) != batch.confined(job["task"])
                    or batch.confined(row["output"]) != batch.confined(job["output"])):
                reason = "CSV 的任务或证据目录与清单不符"
        task = batch.confined(job["task"])
        if batch.sha256(task) != job.get("prepared_task_sha256"):
            reason = "规划输入已改变"
        artifact = batch.confined(job["output"]) / "plan.json"
        if not artifact.exists():
            reason = "独立规划 plan.json 缺失"
        else:
            plan = batch.read_json(artifact)
            independent = plan.get("batch_independent_planning", {})
            if (plan.get("status") != "solved" or plan.get("used_proposal") or plan.get("proposal") is not None
                    or independent.get("proposal_enabled") is not False or independent.get("witness_loaded") is not False):
                reason = "plan.json 未证明独立规划 solved"
        if reason:
            rejected.append({"id": job["id"], "reason": reason})
        else:
            candidates[job["family"]].append({"job": job, "row": row,
                                               "source": "journal" if job["id"] in rows else "csv"})
    for entries in candidates.values():
        entries.sort(key=lambda entry: entry["job"]["id"])
    return candidates, rejected, rows, set(rows) | set(csv_rows)


def _selection_rng(seed, selection, family):
    stream = hashlib.sha256(f"sim-selection:{seed}:{selection}:{family}".encode()).digest()
    return random.Random(int.from_bytes(stream, "big"))


def prepare_selection(selection, seed, path=DEFAULT_MANIFEST):
    if selection not in {1, 2} or isinstance(selection, bool):
        raise ValueError("selection 必须是 1 或 2")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("selection seed 必须为整数")
    path = batch.confined(path)
    with batch.execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        if not path.exists():
            raise ValueError("先准备并运行规划清单")
        manifest = _read_manifest(path)
        manifest["manifest_path"] = str(path)
        selections = manifest["preparation"].setdefault("selections", {})
        prior = selections.get(str(selection))
        if prior is not None:
            if prior.get("seed") != seed:
                raise ValueError("已有抽样的 seed 不同；不重写已有仿真任务")
            batch.atomic_json(path.parent / f"selection_{selection}.json", prior)
            rebuild_precheck_comparison(path, manifest=manifest)
            return manifest
        if selection == 2 and selections.get("1", {}).get("seed") == seed:
            raise ValueError("第二轮必须使用不同抽样 seed")
        candidates, rejected, completed, observed = _planning_rows(manifest)
        pending = [job["id"] for job in manifest["jobs"] if job["phase"] == "plan"
                   and job.get("cohort") == "planning_200" and job["id"] not in observed]
        if pending:
            raise ValueError(f"独立规划仍有 {len(pending)} 项没有结论，暂不固定抽样")
        first_jobs = [job for job in manifest["jobs"] if job["phase"] == "sim" and job.get("selection", 1) == 1]
        if selection == 2 and (len(first_jobs) != len(FAMILIES)*PER_SELECTION or any(job["id"] not in completed for job in first_jobs)):
            raise ValueError("第一轮 20 个槽位必须全部已有完成/跳过记录，再准备第二轮")
        if selection == 2:
            state_path = batch.confined(manifest["state_dir"]) / "state.json"
            if not state_path.exists() or batch.remaining_seconds(batch.read_json(state_path), "sim") <= 0:
                raise ValueError("仿真五小时持久预算已经用完或缺少时钟状态，不安排第二轮")
        first_by_task = {job["task"]: job for job in first_jobs if job.get("task")}
        selection_report = {"selection": selection, "seed": seed, "source": "actual_independent_solved_planning_200",
            "per_family": {}, "rejected_candidates": rejected,
            "failed_simulations_retried": False, "shared_sim_phase_budget_seconds": batch.DEFAULT_SIM_BUDGET}
        for family in FAMILIES:
            rng = _selection_rng(seed, selection, family)
            entries = candidates[family]
            fresh = [entry for entry in entries if entry["job"]["task"] not in first_by_task]
            reused = [entry for entry in entries if entry["job"]["task"] in first_by_task
                      and _truth(completed.get(first_by_task[entry["job"]["task"]]["id"], {}).get("passed"))]
            if selection == 1:
                fresh = entries
            chosen = rng.sample(fresh, min(PER_SELECTION, len(fresh)))
            if selection == 2 and len(chosen) < PER_SELECTION:
                chosen.extend(rng.sample(reused, min(PER_SELECTION-len(chosen), len(reused))))
            info = {"independently_solved": len(entries), "new_candidates": len(fresh),
                "eligible_successful_overlap": len(reused), "selected": [], "skipped_slots": 0}
            for slot in range(PER_SELECTION):
                job = {"id": f"sim-s{selection}-{family.lower()}-{slot:02d}", "phase": "sim", "family": family,
                    "cohort": "sim", "selection": selection, "selection_seed": seed,
                    "output": relative(OVERNIGHT / f"sim_selection_{selection}" / family.lower() / f"{slot:02d}"),
                    "timeout_seconds": batch.HARD_TIMEOUT}
                if slot < len(chosen):
                    source = chosen[slot]
                    task = source["job"]["task"]
                    overlap = task in first_by_task
                    job.update(task=task, source_plan_id=source["job"]["id"], source_plan_status="solved",
                        prepared_task_sha256=source["job"]["prepared_task_sha256"],
                        source_record=source["source"], overlap_with_selection_1=overlap,
                        overlap_previous_sim_id=first_by_task[task]["id"] if overlap else None)
                    info["selected"].append({"id": job["id"], "source_plan_id": job["source_plan_id"], "overlap": overlap})
                else:
                    job["skip_reason"] = (f"{family} 独立规划 solved 且证据完整的可用样本不足 4 项；"
                        f"本族 solved={len(entries)}，本轮可选={len(chosen)}；不重复失败仿真、不以见证结果补位")
                    info["skipped_slots"] += 1
                _append_job(manifest, job)
            selection_report["per_family"][family] = info
        selections[str(selection)] = selection_report
        manifest.pop("manifest_path", None)
        _save_manifest(path, manifest)
        batch.atomic_json(path.parent / f"selection_{selection}.json", selection_report)
        rebuild_precheck_comparison(path, manifest=manifest)
        return manifest


def prepare_gates(path=DEFAULT_MANIFEST):
    path = batch.confined(path)
    with batch.execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        manifest = _read_manifest(path)
        for test_name in batch.LIVE_TESTS:
            name = test_name.rsplit("test_", 1)[-1]
            _append_job(manifest, {"id": "gate-" + name, "phase": "gates", "family": "acceptance",
                "cohort": "gates", "test_name": test_name, "timeout_seconds": batch.HARD_TIMEOUT,
                "output": relative(OVERNIGHT / "gates" / name)})
        manifest["preparation"]["gates"] = {"tests": list(batch.LIVE_TESTS), "environment": {"SKILLNET_LIVE_SIM": "1"}}
        _save_manifest(path, manifest)
        return manifest


def rebuild_precheck_comparison(path=DEFAULT_MANIFEST, *, manifest=None):
    path = batch.confined(path)
    manifest = dict(manifest or _read_manifest(path))
    manifest["manifest_path"] = str(path)
    normalized = batch.load_manifest(path)
    rows = batch.Journal(normalized["state_dir"]).load()
    entries = []
    for job in manifest["jobs"]:
        if job["phase"] != "plan" or job.get("cohort") != "precheck":
            continue
        actual = rows.get(job["id"], {})
        solved = actual.get("status") == "solved" and _truth(actual.get("passed"))
        estimate = job.get("witness_estimate_seconds")
        makespan = actual.get("makespan") if solved else None
        valid = isinstance(estimate, (int, float)) and estimate > 0 and isinstance(makespan, (int, float))
        entries.append({"id": job["id"], "family": job["family"], "task": job["task"],
            "status": actual.get("status", "pending"), "witness_estimate_seconds": estimate,
            "independent_makespan_seconds": makespan,
            "independent_minus_witness_seconds": makespan-estimate if valid else None,
            "independent_to_witness_ratio": makespan/estimate if valid else None,
            "expanded": actual.get("expanded"), "planning_elapsed_seconds": actual.get("elapsed_seconds"),
            "reason": actual.get("reason"), "witness_source": job.get("witness_source"),
            "scope": "串行构造见证与独立搜索结果的事后对照；见证不参与搜索，不证明最优性"})
    destination = batch.confined(normalized["report_dir"]) / "precheck_comparison.csv"
    columns = ("id", "family", "task", "status", "witness_estimate_seconds", "independent_makespan_seconds",
        "independent_minus_witness_seconds", "independent_to_witness_ratio", "expanded",
        "planning_elapsed_seconds", "reason", "witness_source", "scope")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(entries)
    batch.atomic_text(destination, stream.getvalue())
    batch.atomic_json(path.parent / "precheck_comparison.json", {"updated_at": batch.utc_now(), "rows": entries})
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description="准备本轮规划、真实仿真抽样和门控测试清单；不执行搜索或仿真")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="新目录生成每族40任务，加入13项复核和200项独立规划")
    select = commands.add_parser("select", help="只从实际独立规划 solved 记录中抽样")
    select.add_argument("--selection", type=int, required=True, choices=(1, 2))
    select.add_argument("--seed", type=int, required=True)
    commands.add_parser("gates", help="加入五项明确指定的真实仿真 unittest")
    commands.add_parser("rebuild", help="仅重建规划复核的见证/独立搜索对照")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = prepare_plan(args.manifest)
        elif args.command == "select":
            result = prepare_selection(args.selection, args.seed, args.manifest)
        elif args.command == "gates":
            result = prepare_gates(args.manifest)
        else:
            result = {"comparison_rows": len(rebuild_precheck_comparison(args.manifest))}
        print(json.dumps({"manifest": str(args.manifest), "jobs": len(result.get("jobs", [])),
            "preparation": result.get("preparation"), **({"comparison_rows": result["comparison_rows"]} if "comparison_rows" in result else {})},
            ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"清单准备停止：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
