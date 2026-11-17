"""Resume authorized steps 6/7 after M4 finishes; importing starts no work.

The existing batch journals, not this cursor, determine completed experiments.
Only a true ``m4_outcome.finished`` marker enables execution. The marker need
not claim M4 passed. Run this synchronous supervisor in a hidden background
process; ``--rebuild`` reconstructs reports without running any experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as batch
from tools import overnight_prepare as prepare
from tools import overnight_summary as summary


BASE = ROOT / "results/sim_v1_3"


class GlobalStop(RuntimeError):
    pass


class Blocked(RuntimeError):
    pass


@dataclass
class Locations:
    base: Path = BASE

    def __post_init__(self):
        self.base = batch.confined(self.base)
        self.overnight = self.base / "overnight"
        self.manifest = self.overnight / "manifest.json"
        self.cursor = self.overnight / "final_batches_state.json"
        self.m4 = self.overnight / "m4_outcome.json"
        self.report = self.base / "overnight_report.md"
        self.out = ROOT / "out" if self.base == BASE else self.base / "out"
        self.regression = self.out / "overnight_final_regression.json"
        self.regression_log = self.out / "overnight_final_regression.log"
        self.archive_log = self.out / "overnight_final_archives.log"


def read(path, default=None):
    return batch.read_json(path) if Path(path).is_file() else default


def load_cursor(paths):
    return read(paths.cursor, {"schema_version": 1, "status": "ready", "stages": {},
        "active_regression": None, "created_at": batch.utc_now()})


def save_cursor(paths, cursor):
    cursor["updated_at"] = batch.utc_now()
    batch.atomic_json(paths.cursor, cursor)


def manifest_facts(paths):
    manifest = batch.load_manifest(paths.manifest)
    records = batch.Journal(manifest["state_dir"]).load()
    return manifest, records


def verify_planning_complete(manifest, records):
    jobs = [job for job in manifest["jobs"] if job["phase"] == "plan"]
    precheck = [job for job in jobs if job.get("cohort") == "precheck"]
    planning = [job for job in jobs if job.get("cohort") == "planning_200"]
    if len(jobs) != 213 or len(precheck) != 13 or len(planning) != 200:
        raise Blocked("第4步清单必须含13项复核及200项独立规划")
    if any(job["id"] not in records for job in jobs):
        raise Blocked("第4步213项仍有未提交结论，不能启动第6/7步")
    for job in jobs:
        if records[job["id"]].get("definition_sha256") != job["definition_sha256"]:
            raise Blocked("第4步完成记录与任务定义不符：" + job["id"])


def global_stop_reason(paths):
    own = read(paths.cursor, {})
    if own.get("global_stop"):
        return str(own["global_stop"])
    for path in sorted(paths.overnight.glob("*state/state.json")):
        state = read(path, {})
        if state.get("global_stop"):
            return str(path.relative_to(ROOT)) + "：" + str(state["global_stop"])
    outcome = read(paths.m4, {})
    if outcome.get("global_stop"):
        return "M4：" + str(outcome["global_stop"])
    return None


def guard(paths, archive_guard):
    stopped = global_stop_reason(paths)
    if stopped:
        raise GlobalStop(stopped)
    checked = batch.preflight(archive_guard)
    if not checked["ok"]:
        error = GlobalStop if checked.get("global_stop") else Blocked
        raise error(checked["reason"])
    return checked


def section(paths, key, content):
    """Replace our marked block while retaining all other report prose."""
    start, end = f"<!-- final-batches:{key}:start -->", f"<!-- final-batches:{key}:end -->"
    existing = paths.report.read_text(encoding="utf-8") if paths.report.exists() else ""
    rendered = start + "\n" + content.rstrip() + "\n" + end
    if start in existing:
        first = existing.index(start)
        last = existing.find(end, first)
        if last < 0:
            raise ValueError("报告标记不完整，保留原文：" + key)
        result = existing[:first] + rendered + existing[last+len(end):]
    else:
        result = existing.rstrip() + "\n\n" + rendered + "\n"
    batch.atomic_text(paths.report, result)


def round_facts(manifest, records, selection):
    jobs = [job for job in manifest["jobs"] if job["phase"] == "sim" and job.get("selection", 1) == selection]
    facts = summary.counts(jobs, records)
    facts["selection"] = selection
    facts["all_slots_processed"] = len(jobs) == 20 and facts["recorded"] == 20
    return facts


def rebuild(paths=None, *, cursor=None, rebuild_summary=True):
    paths = paths or Locations()
    cursor = load_cursor(paths) if cursor is None else cursor
    manifest, records = manifest_facts(paths)
    state_path = Path(manifest["state_dir"]) / "state.json"
    state = read(state_path, {})
    batch.rebuild_reports(manifest, state=state)
    prepare.rebuild_precheck_comparison(paths.manifest)
    for selection in (1, 2):
        facts = round_facts(manifest, records, selection)
        stage = cursor.get("stages", {}).get("sim_round_" + str(selection), {})
        remaining = batch.remaining_seconds(state, "sim") if state else None
        stage_status = stage.get("status")
        if facts["all_slots_processed"]:
            stage_status = "all_slots_recorded"
        elif facts["requested"] and remaining is not None and remaining <= 0:
            stage_status = "phase_budget_exhausted"
        elif not stage_status and facts["recorded"]:
            stage_status = "partial_records"
        lines = [f"## 第6步：抽样第{selection}轮", "",
            f"状态：{stage_status or '尚未开始'}；目标20个槽位，当前清单{facts['requested']}项。", "",
            "| 已记录槽位 | 实际任务尝试 | 显式跳过 | 验收通过 | 待处理 |", "|---:|---:|---:|---:|---:|",
            f"| {facts['recorded']} | {facts['executed']} | {facts['skipped']} | {facts['passed']} | {facts['pending']} |", "",
            "实际任务尝试包含失败/超时，不能等同于完成20项真实仿真。两轮共用首次仿真开始后的五小时墙钟，重启不重置。"]
        if remaining is not None and math.isfinite(remaining):
            lines.append(f"重建时剩余共享预算：{remaining:.3f}秒。")
        if stage.get("reason"):
            lines.append("原因：" + str(stage["reason"]))
        lines.append("证据：`sim_20.csv` / 两轮累计 `sim_40.csv`（存在时）；逐项完成JSONL、选择清单和完整任务目录。")
        section(paths, "step6-round" + str(selection), "\n".join(lines))
    try:
        regression = read(paths.regression, None)
    except (ValueError, UnicodeDecodeError):
        regression = {"status": "incomplete_artifact", "reason": "原回归JSON不完整，保留证据"}
    if regression is None:
        regression = cursor.get("stages", {}).get("regression", {})
    gate_jobs = [job for job in manifest["jobs"] if job["phase"] == "gates"]
    gate_counts = summary.counts(gate_jobs, records)
    lines = ["## 第7步：修改后额外回归与五项真实门控", "",
        f"额外完整回归：{regression.get('status', '尚未开始')}；测试数{regression.get('tests_run', '未知')}；"
        f"跳过{regression.get('skipped', '未知')}；证据 `{paths.regression.relative_to(ROOT)}`。",
        "完整回归与真实仿真严格串行；回归失败如实记录后继续独立的五项门控，不重复该回归。", "",
        f"五项门控：请求{gate_counts['requested']}，已记录{gate_counts['recorded']}，任务尝试{gate_counts['executed']}，"
        f"跳过{gate_counts['skipped']}，通过{gate_counts['passed']}，待处理{gate_counts['pending']}。", ""]
    for job in gate_jobs:
        row = records.get(job["id"], {})
        lines.append(f"- `{job['test_name']}`：{row.get('status', 'pending')}；{row.get('reason') or '无额外原因'}；证据 `{job['output']}`。")
    archive = cursor.get("stages", {}).get("archives", {})
    lines.extend(("", f"最终归档校验：{archive.get('status', '尚未开始')}；证据 `{paths.archive_log.relative_to(ROOT)}`。"))
    if cursor.get("reason"):
        lines.extend(("", "接续脚本停止原因：" + str(cursor["reason"])))
    section(paths, "step7", "\n".join(lines))
    if rebuild_summary:
        summary.rebuild(paths.base)
    return {"round1": round_facts(manifest, records, 1), "round2": round_facts(manifest, records, 2), "gates": gate_counts}


def _regression_record(paths):
    if not paths.regression.exists():
        return None
    try:
        return batch.read_json(paths.regression)
    except (ValueError, UnicodeDecodeError):
        backup = paths.regression.with_name(paths.regression.stem + ".incomplete_" + uuid.uuid4().hex + ".json")
        shutil.copy2(paths.regression, backup)
        record = {"label": "final_after_sim", "status": "interrupted", "tests_run": None,
            "reason": "上次回归JSON未写完整；保留原始字节，不自动重跑", "raw_evidence_backup": str(backup)}
        batch.atomic_json(paths.regression, record)
        return record


def run_regression(paths, cursor):
    """Run at most once, with native ownership and a persisted start handshake."""
    existing = _regression_record(paths)
    active = cursor.get("active_regression")
    if active and active.get("birth") is not None and batch.process_birth(active["pid"]) == active["birth"]:
        raise Blocked("已有额外回归worker仍在运行，拒绝重复启动")
    if existing is not None or active:
        record = existing or {"label": "final_after_sim", "tests_run": None}
        if record.get("status") in {None, "running"}:
            record.update(status="interrupted", reason="上次回归未提交最终结论；不自动重跑")
            batch.atomic_json(paths.regression, record)
        cursor["active_regression"] = None
        return record
    paths.out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    spec = paths.overnight / ("final_regression_" + token + ".json")
    start_signal = spec.with_suffix(".start")
    batch.atomic_json(spec, {"result": str(paths.regression), "start_signal": str(start_signal)})
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--regression-worker", str(spec)]
    # The full suite itself tests the shared batch lock with fake jobs. A
    # separate regression lock avoids blocking those tests. Its actual child
    # command is overnight_regression.py, recognized by batch's live preflight;
    # this supervisor never overlaps its own full suite with simulation.
    with batch.execution_lock(paths.overnight / "final_regression.lock"):
        owner = batch.OwnedProcess(command, paths.regression_log, start_signal)
        started = time.monotonic()
        timed_out = False
        environment_stop = None
        try:
            cursor["active_regression"] = {"pid": owner.pid, "birth": owner.birth,
                "started_at": batch.utc_now(), "log": str(paths.regression_log)}
            save_cursor(paths, cursor)
            owner.release()
            while owner.process.poll() is None:
                if time.monotonic()-started >= batch.HARD_TIMEOUT:
                    timed_out = True
                    owner.terminate()
                    break
                if shutil.disk_usage("C:\\" if os.name == "nt" else ROOT).free < batch.MIN_C_FREE:
                    owner.terminate()
                    environment_stop = "额外回归期间 C 盘剩余不足5GiB"
                    break
                time.sleep(.5)
            returncode = owner.process.poll()
        finally:
            owner.close()
        record = _regression_record(paths) or {"label": "final_after_sim", "tests_run": None}
        if timed_out or record.get("status") in {None, "running"}:
            record.update(status="timeout" if timed_out else "interrupted", reason="额外完整回归未提交成功结论；不自动重跑")
        if environment_stop:
            record.update(status="stopped", fatal_environment=True, reason=environment_stop)
        record.update(returncode=returncode, supervisor_seconds=time.monotonic()-started, log=str(paths.regression_log))
        batch.atomic_json(paths.regression, record)
        cursor["active_regression"] = None
        if environment_stop:
            cursor["stages"]["regression"] = {key: record.get(key) for key in ("status", "tests_run", "reason", "log")}
            save_cursor(paths, cursor)
            raise GlobalStop(environment_stop)
        return record


def regression_worker(specification):
    data = batch.read_json(batch.confined(specification))
    signal_path = batch.confined(data["start_signal"])
    started = time.monotonic()
    while not signal_path.exists():
        if time.monotonic()-started > 30:
            raise RuntimeError("额外回归没有收到所有权确认，不启动测试")
        time.sleep(.05)
    os.environ.pop("SKILLNET_LIVE_SIM", None)
    os.environ.pop("SKILLNET_LIVE_MODEL", None)
    environment = dict(os.environ)
    command = [sys.executable, "-X", "utf8", "tools/overnight_regression.py", "--worker",
               "--json", str(batch.confined(data["result"])), "--label", "final_after_sim"]
    # Descendants inherit the supervisor's native Job Object ownership.
    result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return result.returncode


def check_archives(paths):
    paths.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("SKILLNET_LIVE_SIM", None)
    environment.pop("SKILLNET_LIVE_MODEL", None)
    with paths.archive_log.open("ab") as stream:
        result = subprocess.run([sys.executable, "-X", "utf8", "tools/check_v13_archives.py"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            timeout=batch.HARD_TIMEOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode,
            "log": str(paths.archive_log), "finished_at": batch.utc_now()}


def orchestrate(base=BASE):
    paths, cursor = Locations(base), None
    with batch.execution_lock(paths.overnight / "final_batches.lock"):
        cursor = load_cursor(paths)
        archive_guard = batch.ArchiveGuard()
        try:
            if read(paths.m4, {}).get("finished") is not True:
                raise Blocked("M4尚未标记finished:true，不启动后续实验")
            manifest, records = manifest_facts(paths)
            verify_planning_complete(manifest, records)
            cursor.update(status="running", reason=None)
            save_cursor(paths, cursor)
            guard(paths, archive_guard)
            prepare.prepare_selection(1, 20261116, paths.manifest)
            manifest, records = manifest_facts(paths)
            first = round_facts(manifest, records, 1)
            if not first["all_slots_processed"]:
                state = batch.run_phase(manifest, "sim")
                cursor["stages"]["sim_round_1"] = {"status": state["status"]}
            else:
                cursor["stages"]["sim_round_1"] = {"status": "already_recorded"}
            save_cursor(paths, cursor)
            rebuild(paths, cursor=cursor)
            if cursor["stages"]["sim_round_1"]["status"] == "environment_blocked":
                raise Blocked("第6步第一轮环境阻塞；保留记录，等待同命令续跑")
            guard(paths, archive_guard)
            manifest, records = manifest_facts(paths)
            state = read(Path(manifest["state_dir"]) / "state.json", {})
            has_second = any(job["phase"] == "sim" and job.get("selection") == 2 for job in manifest["jobs"])
            eligible = round_facts(manifest, records, 1)["all_slots_processed"] and state and batch.remaining_seconds(state, "sim") > 0
            if has_second or eligible:
                if not has_second:
                    prepare.prepare_selection(2, 20261117, paths.manifest)
                manifest, records = manifest_facts(paths)
                if not round_facts(manifest, records, 2)["all_slots_processed"]:
                    phase = batch.run_phase(manifest, "sim")
                    cursor["stages"]["sim_round_2"] = {"status": phase["status"]}
                else:
                    cursor["stages"]["sim_round_2"] = {"status": "already_recorded"}
            else:
                cursor["stages"]["sim_round_2"] = {"status": "not_prepared",
                    "reason": "第一轮槽位未全部处理或共享五小时预算已耗尽；不声称已执行40项"}
            save_cursor(paths, cursor)
            rebuild(paths, cursor=cursor)
            if cursor["stages"]["sim_round_2"]["status"] == "environment_blocked":
                raise Blocked("第6步第二轮环境阻塞；保留记录，等待同命令续跑")
            guard(paths, archive_guard)
            regression = run_regression(paths, cursor)
            cursor["stages"]["regression"] = {key: regression.get(key) for key in (
                "status", "tests_run", "skipped", "total_seconds", "reason", "log")}
            save_cursor(paths, cursor)
            rebuild(paths, cursor=cursor)
            guard(paths, archive_guard)
            prepare.prepare_gates(paths.manifest)
            manifest, records = manifest_facts(paths)
            gate_state = batch.run_phase(manifest, "gates")
            cursor["stages"]["gates"] = {"status": gate_state["status"]}
            save_cursor(paths, cursor)
            rebuild(paths, cursor=cursor)
            if gate_state["status"] == "environment_blocked":
                raise Blocked("第7步门控环境阻塞；保留记录，等待同命令续跑")
            guard(paths, archive_guard)
            cursor["stages"]["archives"] = check_archives(paths)
            if cursor["stages"]["archives"]["status"] != "passed":
                raise GlobalStop("最终归档或受保护文件校验不一致")
            cursor["status"] = "finished"
        except GlobalStop as exc:
            cursor.update(status="global_stop", reason=str(exc), global_stop=str(exc))
        except Blocked as exc:
            cursor.update(status="blocked", reason=str(exc))
        except Exception as exc:
            cursor.update(status="interrupted", reason=f"{type(exc).__name__}: {exc}")
        finally:
            save_cursor(paths, cursor)
            if paths.manifest.exists():
                rebuild(paths, cursor=cursor)
        return cursor


def main(argv=None):
    parser = argparse.ArgumentParser(description="M4结束后按序继续抽样、额外回归、真实门控与报告")
    parser.add_argument("--rebuild", action="store_true", help="仅从事实日志重建报告")
    parser.add_argument("--regression-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.regression_worker:
        return regression_worker(args.regression_worker)
    if args.rebuild:
        rebuild()
        return 0
    cursor = orchestrate()
    print(json.dumps(cursor, ensure_ascii=False, indent=2), flush=True)
    return 0 if cursor["status"] == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
