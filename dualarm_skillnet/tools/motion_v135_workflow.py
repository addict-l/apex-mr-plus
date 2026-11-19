"""Explicit prepare/run/rebuild for the original frozen motion-comparison cohort.

Importing this module never starts a solver, test, simulator, or batch. Every
formal attempt pins its inputs and complete current code before any worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import motion_v135_control as control

BASE = control.BASE
SOURCE = ROOT / "results/sim_v1_3/v134/manifests"
STAGES = ("planning200", "fixed20", "walls3", "gates")
SOURCE_STAGE = {"planning200": "planning200", "fixed20": "sim20", "walls3": "walls3", "gates": "gates"}
PHASE = {"planning200": "plan", "fixed20": "sim", "walls3": "sim", "gates": "gates"}
COUNTS = {"planning200": 200, "fixed20": 20, "walls3": 3, "gates": 5}
WORKER_FILES = tuple(sorted({
    *(p.relative_to(ROOT).as_posix() for p in (ROOT/"tools").glob("motion_v135*.py")),
    *(p.relative_to(ROOT).as_posix() for p in (ROOT/"tests").glob("*.py")),
    "tools/overnight_batch.py", "tools/overnight_regression.py", "tools/v134_batch.py",
    "tools/v134_originals.py", "tools/v134_guard.py", "tools/check_v13_archives.py"}))


def attempt_root(attempt):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt):
        raise ValueError("attempt只能使用安全的字母数字标识")
    path = (BASE / "experiments" / attempt).resolve()
    if not path.is_relative_to((BASE / "experiments").resolve()):
        raise ValueError("attempt目录越界")
    return path


def worker_version():
    return {name: old.sha256(ROOT/name) for name in WORKER_FILES}


def _stop(root, reason):
    row = {"at": old.utc_now(), "reason": reason, "global_stop": reason, "attempt": root.name}
    old.atomic_json(root/"global_stop.json", row)
    old.atomic_json(BASE/"global_stop.json", row)
    return row


def _stopped(root):
    for path in (BASE/"global_stop.json", root/"global_stop.json"):
        if path.exists():
            return old.read_json(path)
    return None


def _preflight_record(path, code, tools, expected_sha=None):
    """Bind an existing complete proof without repairing or rewriting it."""
    from tools.motion_v135_preflight import STEPS

    def require(ok, reason):
        if not ok:
            raise ValueError("preflight前置证明："+reason)

    require(path is not None, "需要 --preflight-proofs PATH")
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()) and path.is_file(), "路径不在本轮结果目录或文件不存在")
    digest = old.sha256(path)
    require(expected_sha is None or digest == expected_sha, "登记的证明文件SHA改变")
    data = old.read_json(path)
    require(data.get("formal_ready") is True, "formal_ready不为true")
    version = data.get("version", {})
    require(isinstance(version, dict) and version.get("core") == code, "core版本不是当前交付实现")
    declared_tools = version.get("tools")
    require(isinstance(declared_tools, dict) and bool(declared_tools), "缺少工具指纹")
    require(all(tools.get(name) == value for name, value in declared_tools.items()), "声明的工具SHA不一致或未登记")
    if data.get("version_sha256") is not None:
        version_digest = hashlib.sha256(json.dumps(version, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        require(data["version_sha256"] == version_digest, "version_sha256不一致")
    required = data.get("required_steps")
    require(isinstance(required, list) and bool(required) and all(isinstance(name, str) for name in required)
            and len(required) == len(set(required)) and set(STEPS) <= set(required),
            "required_steps不完整，不能跳过固定前置步骤")
    steps, evidence = data.get("steps", {}), {}
    require(isinstance(steps, dict), "steps格式无效")
    for name in required:
        proof = steps.get(name, {})
        require(isinstance(proof, dict) and proof.get("passed") is True
                and proof.get("completed") is True, "步骤未完整通过："+name)
        require(proof.get("version") == version, "步骤使用不同版本："+name)
        files = proof.get("evidence_files")
        require(isinstance(files, dict) and bool(files), "步骤缺少证据SHA："+name)
        for source, sha in files.items():
            source = Path(source).resolve()
            require(source.is_relative_to(ROOT.resolve()) and source.is_file()
                    and old.sha256(source) == sha, "步骤证据SHA改变/缺失："+str(source))
            evidence[str(source)] = sha
    return {"path": str(path), "sha256": digest, "version": deepcopy(version),
            "required_steps": list(required), "evidence_files": evidence}


def verify(root, *, check_code=True):
    selection_path = root/"selection.json"
    selection = old.read_json(selection_path)
    if check_code and (control.version() != selection["code_sha256"]
                       or worker_version() != selection["worker_code_sha256"]):
        raise ValueError("当前实现与prepare快照不一致；禁止混版本，需新attempt")
    prerequisite = selection.get("preflight_proofs", {})
    _preflight_record(prerequisite.get("path"), selection["code_sha256"],
                      selection["worker_code_sha256"], prerequisite.get("sha256"))
    for stage, row in selection["manifests"].items():
        path = root/"manifests"/(stage+".json")
        if old.sha256(path) != row["sha256"]:
            raise ValueError("已准备清单被修改："+stage)
        if old.sha256(row["baseline_manifest"]) != row["baseline_manifest_sha256"]:
            raise ValueError("正式基线清单被修改："+stage)
        manifest = old.load_manifest(path)
        if len(manifest["jobs"]) != COUNTS[stage]:
            raise ValueError("固定样本分母改变："+stage)
        for job in manifest["jobs"]:
            if job.get("task") and job["task_sha256"] != job["prepared_task_sha256"]:
                raise ValueError("任务输入SHA不匹配："+job["id"])
    return selection


def prepare(attempt, *, preflight_proofs=None):
    root = attempt_root(attempt)
    with old.execution_lock(BASE/"workflow.lock"):
        if (root/"selection.json").exists():
            selection = verify(root)
            if (preflight_proofs is not None
                    and str(Path(preflight_proofs).resolve()) != selection["preflight_proofs"]["path"]):
                raise ValueError("既有attempt不能更换preflight证明，保留现场并使用新attempt")
            return {"status": "already_prepared", "attempt": attempt, "selection": str(root/"selection.json"),
                    "code_version_sha256": old.fingerprint(selection["code_sha256"])}
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("attempt目录已有未封存文件；保留现场，使用新attempt")
        checked = control.audit()
        if not checked["passed"]:
            raise RuntimeError("历史数据/冻结规划核心保护失败")
        if _stopped(root):
            raise RuntimeError("存在global_stop；先核对现场，禁止自动清除")
        code = control.version()
        runners = worker_version()
        prerequisites = _preflight_record(preflight_proofs, code, runners)
        root.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "namespace": "motion_v135", "attempt": attempt,
                    "prepared_at": old.utc_now(), "code_sha256": code,
                    "worker_code_sha256": runners, "manifests": {},
                    "preflight_proofs": prerequisites,
                    "baseline": "v1.3.4 c_wall_v134/runtime_fix2",
                    "selection_policy": "exact original task identities and SHA, no resampling",
                    "budgets_policy": "copy original task and phase budgets without extension",
                    "preparation_does_not_start_work": True}
        for name, digest in {**code, **runners}.items():
            destination = root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/name).read_bytes())
            if old.sha256(destination) != digest:
                raise ValueError("快照期间文件变化："+name)
        for stage in STAGES:
            source_path = SOURCE/(SOURCE_STAGE[stage]+".json")
            source = old.read_json(source_path)
            if len(source["jobs"]) != COUNTS[stage]:
                raise ValueError("基线样本数量不符："+stage)
            jobs = []
            for index, baseline in enumerate(source["jobs"]):
                job = deepcopy(baseline)
                old_id = job["id"]
                for name in ("planning_code_sha256", "motion_code_sha256", "experiment_namespace",
                             "experiment_selection_sha256", "current_planning_job_id", "current_plan_path",
                             "current_plan_sha256", "definition_sha256"):
                    if name in job:
                        job["baseline_"+name] = job.pop(name)
                logical = "motion-v135-"+attempt+"-"+stage+"-"+str(index).zfill(3)
                job.update(id=logical, logical_job_id=logical, run_attempt=0,
                           source_job_id=baseline.get("source_job_id", old_id), baseline_job_id=old_id,
                           baseline_manifest_sha256=old.sha256(source_path),
                           output=str(root/stage/str(index).zfill(3)),
                           experiment_namespace="motion_v135", workflow_attempt=attempt,
                           workflow_root=str(root), code_version_sha256=old.fingerprint(code))
                job["worker_code_version_sha256"] = old.fingerprint(runners)
                if job.get("task"):
                    task_sha = old.sha256(job["task"])
                    expected = baseline.get("prepared_task_sha256", baseline.get("task_sha256"))
                    if expected is None or task_sha != expected:
                        raise ValueError("基线任务缺SHA或被改动："+old_id)
                    job["prepared_task_sha256"] = expected
                    job.setdefault("source_task_sha256", expected)
                jobs.append(job)
            manifest = {**{k: deepcopy(v) for k, v in source.items() if k != "jobs"},
                        "session_id": "motion_v135_"+attempt+"_"+stage,
                        "state_dir": str(root/"states"/stage),
                        "report_dir": str(root/"reports"/stage), "jobs": jobs}
            path = root/"manifests"/(stage+".json")
            old.atomic_json(path, manifest)
            old.load_manifest(path)  # Static schema/input validation only.
            metadata["manifests"][stage] = {"sha256": old.sha256(path),
                "baseline_manifest": str(source_path), "baseline_manifest_sha256": old.sha256(source_path),
                "requested": len(jobs), "phase_budgets_seconds": manifest["phase_budgets_seconds"],
                "task_budgets_seconds": sorted({j["timeout_seconds"] for j in jobs})}
        if code != control.version() or runners != worker_version():
            raise ValueError("prepare期间代码发生变化，保留现场；使用新attempt")
        old.atomic_json(root/"selection.json", metadata)
        control.record("M6_prepare_"+attempt, "prepared", evidence=str(root/"selection.json"))
        rebuild(attempt)
        return {"status": "prepared", "attempt": attempt, "selection": str(root/"selection.json"),
                "requested": dict(COUNTS), "code_version_sha256": old.fingerprint(code)}


def _logical_identity(row, canonical):
    explicit = row.get("logical_job_id")
    if explicit in canonical:
        return explicit
    value = row["id"]
    if value in canonical:
        return value
    # overnight_batch's interruption recovery predates logical identities and
    # does not copy our extra job fields into its interrupted result record.
    for logical in canonical:
        suffix = value[len(logical):] if value.startswith(logical) else ""
        if re.fullmatch(r"-resume[1-9][0-9]*", suffix):
            return logical
    return None


def _canonical_rows(root, stage):
    manifest = old.load_manifest(root/"manifests"/(stage+".json"))
    records = old.Journal(manifest["state_dir"]).load()
    canonical = {job["id"]: job for job in manifest["jobs"]}
    selected = {}
    for row in records.values():
        logical = _logical_identity(row, canonical)
        if logical not in canonical:
            continue
        # A recovered interrupted worker is evidence of an attempt, not a
        # completed logical task. Its replacement has a distinct durable ID.
        if row.get("status") != "interrupted":
            if logical in selected:
                raise ValueError("同一正式任务存在多个已完成结果："+logical)
            selected[logical] = row
    return manifest, records, selected


def rebuild(attempt):
    root = attempt_root(attempt)
    summaries = {}
    for stage in STAGES:
        manifest, records, selected = _canonical_rows(root, stage)
        counts = Counter(row.get("status", "missing") for row in selected.values())
        summaries[stage] = {"requested": len(manifest["jobs"]), "committed": len(selected),
            "pending": len(manifest["jobs"])-len(selected), "statuses": dict(counts),
            "passed": sum(row.get("passed") is True for row in selected.values()),
            "attempt_rows": len(records), "interrupted_attempts": sum(r.get("status") == "interrupted" for r in records.values()),
            "journal": str(Path(manifest["state_dir"])/"completed.jsonl")}
        old.atomic_json(root/"reports"/(stage+"_selected.json"), {
            "stage": stage, "requested": len(manifest["jobs"]),
            "selection_basis": "one committed non-interrupted result per original logical identity",
            "cases": [{"logical_job_id": job["id"], "source_job_id": job["source_job_id"],
                       "baseline_job_id": job["baseline_job_id"],
                       "task_sha256": job.get("task_sha256"),
                       "result": selected.get(job["id"])} for job in manifest["jobs"]]})
    result = {"at": old.utc_now(), "attempt": attempt, "stages": summaries,
              "execution_complete": all(row["pending"] == 0 for row in summaries.values()),
              "acceptance_not_assessed": True}
    old.atomic_json(root/"counts.json", result)
    lines = ["# 运动优化批次进度", "", "更新："+result["at"], "",
             "已提交不等于通过；所有失败/未知/预算/超时均保留。中断尝试单列，续跑结果不得覆盖它。", "",
             "|阶段|请求|已提交|通过|待处理|状态|", "|---|---:|---:|---:|---:|---|"]
    for stage, row in summaries.items():
        lines.append(f"|{stage}|{row['requested']}|{row['committed']}|{row['passed']}|{row['pending']}|{row['statuses']}|")
    old.atomic_text(root/"counts.md", "\n".join(lines)+"\n")
    return result


def _execution_manifest(root, stage):
    """Keep interrupted attempts, and retry only uncommitted logical jobs."""
    canonical_path = root/"manifests"/(stage+".json")
    canonical = old.load_manifest(canonical_path)
    raw = old.read_json(canonical_path)
    state = old._state(canonical)
    journal = old.Journal(canonical["state_dir"])
    old._recover_active(canonical, state, journal)
    old.save_state(canonical, state)
    records = journal.load()
    jobs = []
    for original in raw["jobs"]:
        logical = original["id"]
        attempts = [row for row in records.values() if _logical_identity(row, {j["id"]: j for j in raw["jobs"]}) == logical]
        completed = [row for row in attempts if row.get("status") != "interrupted"]
        if len(completed) > 1:
            raise ValueError("正式任务存在重复完成记录："+logical)
        if completed:
            selected = completed[0]
            if selected["id"] == logical:
                jobs.append(original)
            else:
                jobs.append(old.read_json(root/"recovery"/stage/(selected["id"]+".json"))["job"])
        elif attempts:
            number = len(attempts)
            if number >= 3:
                raise RuntimeError("同一任务已中断三次；保留现场，需独立恢复决定："+logical)
            replacement = deepcopy(original)
            replacement.update(id=logical+"-resume"+str(number), run_attempt=number,
                               recovery_of=attempts[-1]["id"],
                               output=str(root/"recovery_outputs"/stage/logical/str(number).zfill(3)))
            receipt = {"at": old.utc_now(), "logical_job_id": logical,
                       "reason": "previous worker did not commit a result; original evidence retained",
                       "previous_records": attempts, "job": replacement}
            path = root/"recovery"/stage/(replacement["id"]+".json")
            if path.exists():
                if old.read_json(path)["job"] != replacement:
                    raise ValueError("恢复任务定义发生改变")
            else:
                old.atomic_json(path, receipt)
            jobs.append(replacement)
        else:
            jobs.append(original)
    execution = {**{k: v for k, v in raw.items() if k != "jobs"}, "jobs": jobs}
    digest = old.fingerprint(execution)
    path = root/"execution_manifests"/(stage+"_"+digest[:20]+".json")
    if path.exists():
        if old.read_json(path) != execution:
            raise ValueError("执行清单摘要冲突")
    else:
        old.atomic_json(path, execution)
    return old.load_manifest(path)


def _startup_streak(root, extra=None):
    records = []
    for stage in STAGES:
        directory = root/"states"/stage
        if directory.exists():
            records += list(old.Journal(directory).load().values())
    if extra is not None:
        records.append(extra)
    streak = 0
    for row in sorted(records, key=lambda r: r.get("completed_at", "9999")):
        if row.get("phase") not in {"sim", "gates"} or row.get("status") in {"skipped", "interrupted"}:
            continue
        failed = not row.get("passed") and row.get("failure_phase") in {"simulator_start", "simulator_connect"}
        streak = streak+1 if failed else 0
    return streak


def run_stage(root, stage):
    verify(root)
    manifest = _execution_manifest(root, stage)
    execution_path = manifest["manifest_path"]

    def environment(archive):
        if _stopped(root):
            return {"ok": False, "global_stop": True, "reason": "运动优化global_stop已记录"}
        try:
            verify(root)
            protection = control.audit()
        except (OSError, ValueError, KeyError) as exc:
            _stop(root, "代码/输入/准备版本校验失败："+str(exc))
            return {"ok": False, "global_stop": True, "reason": str(exc)}
        if not protection["passed"]:
            _stop(root, "历史数据/冻结规划核心/归档校验变化")
            return {"ok": False, "global_stop": True, "reason": "历史数据或冻结规划核心校验失败", "protection": protection}
        return {**old.preflight(archive), "protection": protection}

    def factory(command, log_path, start_signal):
        command = list(command)
        command[3] = str(Path(__file__).resolve())
        path = Path(command[-1])
        spec = old.read_json(path)
        state = old.read_json(Path(manifest["state_dir"])/"state.json")
        seconds = min(spec["job"]["timeout_seconds"], old.remaining_seconds(state, PHASE[stage]))
        spec.update(execution_manifest=execution_path, hard_deadline_epoch=time.time()+seconds)
        old.atomic_json(path, spec)
        return old.OwnedProcess(command, log_path, start_signal)

    def adapter(job, result):
        from tools.v134_batch import adapt_result
        value = adapt_result(job, result)
        value.update(logical_job_id=job["logical_job_id"], source_job_id=job["source_job_id"],
                     baseline_job_id=job["baseline_job_id"], run_attempt=job["run_attempt"],
                     code_version_sha256=job["code_version_sha256"],
                     worker_code_version_sha256=job.get("worker_code_version_sha256"))
        incoming = {**value, "phase": job["phase"], "completed_at": old.utc_now()}
        if _startup_streak(root, incoming) >= 3:
            _stop(root, "连续三项CoppeliaSim启动/连接失败")
        return value

    result = old.run_phase(manifest, PHASE[stage], environment_check=environment,
                           process_factory=factory, result_adapter=adapter)
    if result.get("global_stop"):
        _stop(root, str(result["global_stop"]))
    if _stopped(root):
        result.update(status="global_stop", global_stop="运动优化整体停止已记录")
        old.save_state(manifest, result)
    return result


def run(attempt, stages=STAGES):
    root = attempt_root(attempt)
    requested = set(stages)
    if not requested or requested-set(STAGES):
        raise ValueError("未知或空阶段选择")
    ordered = [stage for stage in STAGES if stage in requested]
    path = root/"workflow_state.json"
    cursor = old.read_json(path) if path.exists() else {"attempt": attempt, "stages": {}}
    current = None
    with old.execution_lock(BASE/"workflow.lock"):
        try:
            protection = control.audit()
            if not protection["passed"]:
                _stop(root, "历史数据/冻结规划核心保护失败")
            verify(root)
            if _stopped(root):
                cursor["global_stop"] = "已有整体停止记录"
                raise RuntimeError(cursor["global_stop"])
            cursor.update(status="running", pid=os.getpid(), birth=old.process_birth(os.getpid()),
                          requested_stages=ordered, started_at=old.utc_now())
            old.atomic_json(path, cursor)
            for stage in ordered:
                current = stage
                if stage != "planning200":
                    _, _, plans = _canonical_rows(root, "planning200")
                    if len(plans) != COUNTS["planning200"]:
                        raise RuntimeError("planning200未全部提交，禁止启动native批次")
                cursor["current_stage"] = stage
                old.atomic_json(path, cursor)
                result = run_stage(root, stage)
                cursor["stages"][stage] = result
                cursor["updated_at"] = old.utc_now()
                old.atomic_json(path, cursor)
                summary = rebuild(attempt)
                control.record("M6_"+attempt+"_"+stage, result["status"],
                               evidence=str(root/"states"/stage/"completed.jsonl"),
                               counts=summary["stages"][stage])
                if result.get("global_stop"):
                    cursor["global_stop"] = result["global_stop"]
                    raise RuntimeError(str(result["global_stop"]))
                if result.get("status") != "phase_complete" or summary["stages"][stage]["pending"]:
                    raise RuntimeError("阶段仍有未提交正式任务，保留断点："+stage)
            cursor.update(status="finished", current_stage=None, reason=None)
        except BaseException as exc:
            cursor.update(status="global_stop" if cursor.get("global_stop") or _stopped(root) else "interrupted",
                          reason=f"{type(exc).__name__}: {exc}", current_stage=current)
        finally:
            cursor["updated_at"] = old.utc_now()
            old.atomic_json(path, cursor)
            if (root/"selection.json").exists():
                cursor["counts"] = rebuild(attempt)["stages"]
                old.atomic_json(path, cursor)
            control.record("M6_"+attempt+"_supervisor", cursor["status"], evidence=str(path))
    return cursor


def worker(path):
    spec = old.read_json(path)
    waited = time.monotonic()
    while not Path(spec["start_signal"]).exists():
        if time.monotonic()-waited > 30:
            raise RuntimeError("未收到自有进程握手")
        time.sleep(.03)
    job = spec["job"]
    root = attempt_root(job["workflow_attempt"])
    began = time.monotonic()
    deadline = began+max(.001, spec["hard_deadline_epoch"]-time.time()-.75)
    try:
        verify(root)
        execution = old.load_manifest(spec["execution_manifest"])
        expected = next(j for j in execution["jobs"] if j["id"] == job["id"])
        if job != expected:
            raise ValueError("worker输入与准备的执行清单不符")
        if job["phase"] == "plan":
            from tools.v134_batch import plan_job
            result = plan_job(job, deadline)
        else:
            from simbridge import runner
            original_owner = runner.Simulator
            class NamespaceSimulator(original_owner):
                def __init__(self, ignored_root, *args, **kwargs):
                    super().__init__(Path(job["output"])/"instance", *args, **kwargs)
            runner.Simulator = NamespaceSimulator
            runtime_job = {**job, "timeout_seconds": min(job["timeout_seconds"], max(.001, deadline-time.monotonic()))}
            try:
                result = old.sim_job(runtime_job) if job["phase"] == "sim" else old.gate_job(runtime_job)
            finally:
                runner.Simulator = original_owner
        pinned = old.read_json(root/"selection.json")
        unchanged = (control.version() == pinned["code_sha256"]
                     and worker_version() == pinned["worker_code_sha256"])
        result["code_unchanged_during_job"] = unchanged
        if not unchanged:
            result.update(uncredited_status=result.get("status"), status="invalid_version", passed=False,
                          reason="执行期间代码改变，结果不计为正式通过")
    except BaseException as exc:
        result = {"status": "worker_failed", "passed": False, "failure_phase": "worker",
                  "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    if "phase" in result:
        result["execution_phase"] = result.pop("phase")
    result.update(definition_sha256=job["definition_sha256"], worker_wall_seconds=time.monotonic()-began,
                  logical_job_id=job["logical_job_id"], run_attempt=job["run_attempt"],
                  source_job_id=job["source_job_id"], baseline_job_id=job["baseline_job_id"],
                  experiment_namespace="motion_v135", workflow_attempt=job["workflow_attempt"],
                  code_version_sha256=job["code_version_sha256"])
    result["worker_code_version_sha256"] = job.get("worker_code_version_sha256")
    result.setdefault("wall_seconds", result["worker_wall_seconds"])
    old.atomic_json(spec["result_path"], result)
    return 0 if result.get("passed") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="运动优化固定任务批次：准备、串行运行、断点重建")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command")
    for name in ("prepare", "run", "rebuild"):
        command = commands.add_parser(name)
        command.add_argument("--attempt", default="formal_a1")
        if name == "prepare":
            command.add_argument("--preflight-proofs", type=Path,
                                 help="当前版本完整formal_ready证明索引；仅验证并登记，不生成证明")
        if name == "run":
            command.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.command is None:
        parser.error("需要prepare、run或rebuild")
    functions = {"prepare": prepare, "run": run, "rebuild": rebuild}
    result = (run(args.attempt, args.stages) if args.command == "run" else
              prepare(args.attempt, preflight_proofs=args.preflight_proofs) if args.command == "prepare" else
              rebuild(args.attempt))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") in {None, "prepared", "already_prepared", "finished"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
