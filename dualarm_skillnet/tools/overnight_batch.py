"""Resumable, serial unattended planning/simulation jobs, with durable evidence.

This entry point never launches work at import time. Run an authorized manifest
with ``--phase plan|sim|gates``; ``--rebuild`` only reconstructs reports. Every
task is a fresh Python subprocess. Completed IDs are never retried implicitly.
All paths are project-confined and the simulator port is always 23210.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
PORT = 23210
HARD_TIMEOUT = 1200.0
MIN_C_FREE = 5 * 1024**3
DEFAULT_SIM_BUDGET = 5 * 60 * 60
PHASES = ("plan", "sim", "gates")
LIVE_TESTS = tuple("tests.test_live_sim.LiveSimTests.test_" + name for name in (
    "s1_parallel_place", "s1_reach_handover", "s2_tower",
    "s3a_cantilever", "s3a_platform"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def confined(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("批处理的输入、日志、状态和输出必须位于项目目录内")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def atomic_text(path, text):
    """Replace fsynced data, allowing bounded Windows sharing-denial retries.

    Reuse the same complete temporary file and keep the previous destination
    untouched until replacement succeeds. Eight attempts wait at most .9 s in
    total; permanent permission errors and all other errors still propagate.
    """
    destination = confined(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                dir=destination.parent, prefix=destination.name + ".",
                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.02 * 2**attempt, .2))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _budget(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(field + " 必须是有限正数或 null")
    return float(value)


def load_manifest(path):
    path = confined(path)
    data = read_json(path)
    if data.get("schema_version", 1) != 1:
        raise ValueError("只支持 schema_version=1")
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError("session_id 只允许字母、数字、下划线、句点和连字符")
    if data.get("port", PORT) != PORT:
        raise ValueError("端口固定为 23210")
    data["manifest_path"] = str(path)
    data["port"] = PORT
    data["state_dir"] = str(confined(data.get("state_dir",
        f"results/sim_v1_3/overnight/{session_id}")))
    data["report_dir"] = str(confined(data.get("report_dir", "results/sim_v1_3")))
    data["session_budget_seconds"] = _budget(data.get("session_budget_seconds"), "session budget")
    budgets = {"plan": None, "sim": DEFAULT_SIM_BUDGET, "gates": None}
    budgets.update(data.get("phase_budgets_seconds", {}))
    if set(budgets) - set(PHASES):
        raise ValueError("存在未知的阶段预算")
    data["phase_budgets_seconds"] = {
        phase: _budget(budgets[phase], phase + " budget") for phase in PHASES}
    if data["phase_budgets_seconds"]["sim"] is None or data["phase_budgets_seconds"]["sim"] > DEFAULT_SIM_BUDGET:
        raise ValueError("sim 阶段总墙钟预算不得超过 5 小时")
    jobs, ids, outputs = [], set(), set()
    for raw in data.get("jobs", []):
        job = dict(raw)
        job_id, phase = job.get("id"), job.get("phase")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or job_id in ids:
            raise ValueError("每个 job.id 必须是唯一、安全的标识符")
        if phase not in PHASES:
            raise ValueError("job.phase 必须为 plan、sim 或 gates")
        ids.add(job_id)
        job["timeout_seconds"] = _budget(job.get("timeout_seconds", HARD_TIMEOUT), "task timeout")
        if job["timeout_seconds"] is None or job["timeout_seconds"] > HARD_TIMEOUT:
            raise ValueError("单任务硬超时不得超过 1200 秒")
        job["output"] = str(confined(job.get("output",
            Path(data["state_dir"]) / "jobs" / job_id)))
        if job["output"] in outputs:
            raise ValueError("不同任务必须有不同的 output 目录")
        outputs.add(job["output"])
        if "skip_reason" in job and (not isinstance(job["skip_reason"], str) or not job["skip_reason"].strip()):
            raise ValueError("skip_reason 必须是明确的非空原因")
        if phase == "gates":
            if job.get("test_name") not in LIVE_TESTS:
                raise ValueError("gates 只允许五项明确指定的真实仿真测试")
        elif not job.get("skip_reason") or job.get("task"):
            job["task"] = str(confined(job["task"]))
            if not Path(job["task"]).is_file():
                raise ValueError("任务文件不存在：" + job["task"])
            job["task_sha256"] = sha256(job["task"])
            if job.get("prepared_task_sha256") is not None and job["prepared_task_sha256"] != job["task_sha256"]:
                raise ValueError("已准备任务的 SHA256 改变，禁止运行未复核的内容：" + job["id"])
        job["family"] = str(job.get("family", "fixture"))
        job["cohort"] = str(job.get("cohort", "planning_200" if phase == "plan" else phase))
        job["selection"] = int(job.get("selection", 1))
        if job["selection"] not in {1, 2}:
            raise ValueError("selection 必须为 1 或 2")
        job["definition_sha256"] = fingerprint(job)
        jobs.append(job)
    if not jobs:
        raise ValueError("清单没有 jobs")
    data["jobs"] = jobs
    return data


class Journal:
    """Append-only completion records; a torn last line is never overwritten."""
    def __init__(self, directory):
        self.directory = confined(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.warnings = []

    def paths(self):
        first = self.directory / "completed.jsonl"
        return ([first] if first.exists() else []) + sorted(self.directory.glob("completed.recovery_*.jsonl"))

    def load(self):
        rows = {}
        self.warnings = []
        for path in self.paths():
            lines = path.read_bytes().splitlines(keepends=True)
            for number, line in enumerate(lines, 1):
                if number == len(lines) and not line.endswith(b"\n"):
                    self.warnings.append(f"保留未提交尾行：{path.name}:{number}")
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"完成日志损坏：{path}:{number}") from exc
                key = row["id"]
                if key in rows and rows[key] != row:
                    raise ValueError("完成日志包含冲突记录：" + key)
                rows[key] = row
        return rows

    def append(self, row):
        rows = self.load()
        if row["id"] in rows:
            if rows[row["id"]] != row:
                raise ValueError("禁止覆盖已完成任务：" + row["id"])
            return
        paths = self.paths()
        path = paths[-1] if paths else self.directory / "completed.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell():
                    stream.seek(-1, os.SEEK_END)
                    torn = stream.read(1) != b"\n"
                else:
                    torn = False
            if torn:
                path = self.directory / f"completed.recovery_{len(paths):04d}.jsonl"
        encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


@contextlib.contextmanager
def execution_lock(path):
    """One runner across every manifest and phase in the project."""
    destination = confined(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open("a+b")
    try:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有项目批处理运行，拒绝并发执行") from exc
        yield
    finally:
        stream.close()


def process_snapshot():
    """Native Windows parent-PID inventory; never enumerate for global kills."""
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class ENTRY(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent_pid", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ENTRY)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ENTRY)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows, item = [], ENTRY()
    item.dwSize = ctypes.sizeof(item)
    try:
        more = kernel.Process32FirstW(handle, ctypes.byref(item))
        while more:
            rows.append({"pid": int(item.pid), "parent_pid": int(item.parent_pid), "name": item.name})
            more = kernel.Process32NextW(handle, ctypes.byref(item))
    finally:
        kernel.CloseHandle(handle)
    return rows


def process_birth(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return "alive"
        except OSError:
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    values = [wintypes.FILETIME() for _ in range(4)]
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
            return None
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return None
        return str(values[0].dwHighDateTime << 32 | values[0].dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)


class OwnedProcess:
    """One owned worker and its descendants, fenced by a native Job Object.

    A worker waits for a file handshake until assignment succeeds. The job's
    KILL_ON_JOB_CLOSE also closes descendants if this supervisor crashes. No
    process-name kill, taskkill, external listener, or unrelated PID is used.
    """
    def __init__(self, command, log_path, start_signal):
        self.handle, self.process, self.stream = None, None, None
        self.known = {}
        self.start_signal = start_signal
        self.stream = confined(log_path).open("ab")
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=self.stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt")
            self.pid = self.process.pid
            self.birth = process_birth(self.pid)
            if os.name == "nt":
                self._assign_job()
            self.refresh()
        except BaseException:
            if self.process is not None:
                self.process.kill()  # This Popen handle is ours; no descendants can start before the handshake.
                self.process.wait(timeout=10)
            self.close()
            raise

    def release(self):
        # The supervisor first commits the PID/job definition to state.json.
        atomic_text(self.start_signal, "assigned\n")

    def _assign_job(self):
        import ctypes
        from ctypes import wintypes

        class BASIC(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t),
                ("maximum", ctypes.c_size_t), ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("basic", BASIC), ("io", IO), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.kernel.AssignProcessToJobObject(self.handle, int(self.process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def refresh(self):
        if os.name != "nt":
            self.known[self.pid] = {"pid": self.pid, "parent_pid": os.getpid(), "birth": self.birth}
            return list(self.known.values())
        import ctypes
        from ctypes import wintypes
        for row in process_snapshot():
            handle = self.kernel.OpenProcess(0x1000, False, row["pid"])
            if not handle:
                continue
            owned = wintypes.BOOL()
            try:
                if (self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(owned)) and owned.value):
                    self.known[row["pid"]] = {**row, "birth": process_birth(row["pid"]),
                                               "owned_by_job_object": True}
            finally:
                self.kernel.CloseHandle(handle)
        return list(self.known.values())

    def terminate(self):
        self.refresh()
        if os.name == "nt":
            if not self.kernel.TerminateJobObject(self.handle, 124):
                import ctypes
                raise ctypes.WinError(ctypes.get_last_error())
        elif self.process.poll() is None:
            os.killpg(self.pid, signal.SIGKILL)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise RuntimeError("自有任务进程在终止后仍未退出")

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            deadline = time.monotonic() + 10.
            while any(row.get("birth") is not None and process_birth(pid) == row["birth"]
                      for pid, row in self.known.items()):
                if time.monotonic() >= deadline:
                    raise RuntimeError("自有进程树清理未完成，拒绝继续下一项")
                time.sleep(.1)
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class ArchiveGuard:
    """Read-only archive check; unchanged size/mtime entries reuse an in-process hash."""
    def __init__(self, baseline=None):
        self.baseline = confined(baseline or "results/sim_v1_3/archive_before.json")
        self.cache = {}

    def check(self):
        before = read_json(self.baseline)
        changes = []
        for name, archive in before["archives"].items():
            directory = Path(archive["root"])
            expected = {item["path"]: item for item in archive["files"]}
            current = {path.relative_to(directory).as_posix(): path
                       for path in directory.rglob("*") if path.is_file()}
            for relative in sorted(expected.keys() | current.keys()):
                if relative not in expected or relative not in current:
                    changes.append(name + "/" + relative)
                    continue
                path, saved = current[relative], expected[relative]
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                digest = self.cache.get(key)
                if digest is None:
                    digest = sha256(path)
                    self.cache[key] = digest
                if stat.st_size != saved["size"] or digest != saved["sha256"]:
                    changes.append(name + "/" + relative)
        return {"passed": not changes, "changes": changes, "checked_at": utc_now()}


def full_test_processes():
    if os.name != "nt":
        return []
    # Return only identifying PIDs, never other applications' command-line data.
    script = """$ErrorActionPreference='Stop'
$rows = @(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'pytest.exe'\" | Where-Object { $_.ProcessId -ne %d -and ($_.CommandLine -match 'overnight_regression\\.py|unittest\\s+.*discover|unittest\\s+discover|-m\\s+unittest(?:\\s+-(?:v|q))*\\s*$|(?:-m\\s+pytest|pytest\\.exe)(?:\\s|$)') } | Select-Object ProcessId,ParentProcessId)
ConvertTo-Json -InputObject $rows -Compress
""" % os.getpid()
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
        timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("不能检查完整测试进程，拒绝启动下一任务")
    value = json.loads(result.stdout.strip() or "[]")
    return value if isinstance(value, list) else [value]


def preflight(archive_guard):
    free = shutil.disk_usage("C:\\" if os.name == "nt" else ROOT).free
    if free < MIN_C_FREE:
        return {"ok": False, "global_stop": True, "reason": "C 盘剩余不足 5 GiB", "c_free_bytes": free}
    archive = archive_guard.check()
    if not archive["passed"]:
        return {"ok": False, "global_stop": True, "reason": "归档 SHA256 与基线不符", "archive": archive}
    foreign = [row for row in process_snapshot() if row["name"].lower().startswith("coppeliasim")]
    if foreign:
        return {"ok": False, "global_stop": False, "reason": "检测到现有 CoppeliaSim，未改变该进程", "processes": foreign}
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, PORT), timeout=.2):
                return {"ok": False, "global_stop": False, "reason": "23210 端口已有监听者，未改变该进程"}
        except (OSError, TimeoutError):
            pass
    tests = full_test_processes()
    if tests:
        return {"ok": False, "global_stop": False, "reason": "完整默认测试仍在运行，拒绝并发", "processes": tests}
    return {"ok": True, "c_free_bytes": free, "archive": archive, "port": PORT}


def _csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, allow_nan=False)
                         if key in JSON_REPORT_COLUMNS else
                         (json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key])
                         for key in columns if key in row or key in JSON_REPORT_COLUMNS})
    return stream.getvalue()


JSON_REPORT_COLUMNS = ("phase_seconds", "step_durations", "feedback", "qualification_calls")


REPORT_COLUMNS = ("id", "family", "task", "status", "passed", "makespan", "expanded",
    "elapsed_seconds", "search_elapsed_seconds", "skills", "auxiliary", "auxiliary_skills",
    "failure_phase", "reason", "feedback_count", "compile_failed", "nominal_makespan_seconds",
    "actual_makespan_seconds", "actual_to_estimate_ratio", "wall_seconds", "selection", "executed", "output",
    *JSON_REPORT_COLUMNS, "tpg_seconds", "execution_wall_seconds", "executor_wall_seconds")


def rebuild_reports(manifest, journal=None, state=None):
    journal = journal or Journal(manifest["state_dir"])
    completed = journal.load()
    canonical_phases = {job["id"]: job["phase"] for job in manifest["jobs"]}
    # Runner phase describes its last internal stage (e.g. "complete").
    # Classify an ephemeral report view by the manifest's experiment phase;
    # leave committed records and the returned journal facts unchanged.
    rows = [{**row, "phase": canonical_phases.get(row["id"], row.get("phase"))}
            for row in completed.values()]
    report = Path(manifest["report_dir"])
    groups = {
        "planning_precheck.csv": [r for r in rows if r["phase"] == "plan" and r.get("cohort") == "precheck"],
        "planning_200.csv": [r for r in rows if r["phase"] == "plan" and r.get("cohort") != "precheck"],
        "sim_20.csv": [r for r in rows if r["phase"] == "sim" and r.get("selection", 1) == 1],
        "gates.csv": [r for r in rows if r["phase"] == "gates"],
    }
    if any(job["phase"] == "sim" and job.get("selection") == 2 for job in manifest["jobs"]):
        groups["sim_40.csv"] = [r for r in rows if r["phase"] == "sim"]
    for name, entries in groups.items():
        atomic_text(report / name, _csv_text(entries, REPORT_COLUMNS))
    lines = ["# 无人值守批处理进度", "", "更新：" + utc_now(), "",
        "完成日志是事实来源；未完成任务不计为成功。规划层 worker 未使用构造见证或 proposal hint。", "",
        f"清单请求 {len(manifest['jobs'])} 项；本清单已记录 "
        f"{sum(job['id'] in completed for job in manifest['jobs'])} 项；累计日志 {len(rows)} 项，其中明确跳过 "
        f"{sum(row.get('status') == 'skipped' for row in rows)} 项；清单待处理 "
        f"{sum(job['id'] not in completed for job in manifest['jobs'])} 项。", "",
        "| 阶段 | 任务族 | 已记录 | 通过 | 未通过 | 辅助步骤 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    by_family = defaultdict(list)
    for row in rows:
        by_family[(row["phase"], row.get("family", "fixture"))].append(row)
    for (phase, family), entries in sorted(by_family.items()):
        passed = sum(bool(row.get("passed")) for row in entries)
        lines.append(f"| {phase} | {family} | {len(entries)} | {passed} | {len(entries)-passed} | "
                     f"{sum(bool(row.get('auxiliary')) for row in entries)} |")
    lines.extend(("", "失败/未知任务：", ""))
    for row in rows:
        if not row.get("passed"):
            reason = str(row.get("reason", row["status"])).replace("\n", " ")
            lines.append(f"- {row['id']}：{row['status']} / {row.get('failure_phase') or 'planning'}：{reason}")
    ratios = [r["actual_to_estimate_ratio"] for r in rows
              if isinstance(r.get("actual_to_estimate_ratio"), (int, float))]
    if ratios:
        ordered = sorted(ratios)
        lines.extend(("", f"实际/估算耗时：n={len(ratios)}，min={min(ratios):.4f}，"
            f"median={ordered[len(ordered)//2]:.4f}，max={max(ratios):.4f}。"))
    if state:
        lines.extend(("", "运行状态：", "", "```json", json.dumps(state, ensure_ascii=False, indent=2), "```"))
    if journal.warnings:
        lines.extend(("", "日志恢复提示：", "", *["- " + item for item in journal.warnings]))
    atomic_text(report / "batch_summary.md", "\n".join(lines) + "\n")
    atomic_json(Path(manifest["state_dir"]) / "report_status.json", {
        "updated_at": utc_now(), "requested": len(manifest["jobs"]), "completed": len(rows),
        "manifest_completed": sum(job["id"] in completed for job in manifest["jobs"]),
        "skipped": sum(row.get("status") == "skipped" for row in rows),
        "pending": sum(job["id"] not in completed for job in manifest["jobs"]),
        "reports": {name: len(entries) for name, entries in groups.items()},
        "warnings": journal.warnings})
    return completed


def _state(manifest):
    path = Path(manifest["state_dir"]) / "state.json"
    if path.exists():
        state = read_json(path)
        if state["session_id"] != manifest["session_id"]:
            raise ValueError("state_dir 已属于另一个 session")
        # Extending/restarting a manifest must never reset or extend a clock.
        for phase, value in manifest["phase_budgets_seconds"].items():
            old = state["phase_budgets_seconds"].get(phase)
            if old is not None and (value is None or value > old):
                raise ValueError("续跑不得延长已保存的阶段预算")
        old = state.get("session_budget_seconds")
        new = manifest.get("session_budget_seconds")
        if old is not None and (new is None or new > old):
            raise ValueError("续跑不得延长已保存的 session 预算")
        state["phase_budgets_seconds"] = manifest["phase_budgets_seconds"]
        state["session_budget_seconds"] = new
        return state
    return {"schema_version": 1, "session_id": manifest["session_id"], "session_started_epoch": None,
        "session_budget_seconds": manifest["session_budget_seconds"],
        "phase_budgets_seconds": manifest["phase_budgets_seconds"], "phase_started_epoch": {},
        "consecutive_start_connect_failures": 0, "active_job": None, "global_stop": None,
        "status": "ready", "created_at": utc_now()}


def remaining_seconds(state, phase, now=None):
    now = time.time() if now is None else now
    remaining = []
    for start, budget in ((state.get("session_started_epoch"), state.get("session_budget_seconds")),
                         (state.get("phase_started_epoch", {}).get(phase), state["phase_budgets_seconds"].get(phase))):
        if start is not None and budget is not None:
            remaining.append(start + budget - now)
    return max(0., min(remaining)) if remaining else math.inf


def save_state(manifest, state):
    state["updated_at"] = utc_now()
    now = time.time()
    state["session_wall_seconds"] = max(0., now - state["session_started_epoch"]) if state.get("session_started_epoch") else 0.
    state["phase_wall_seconds"] = {name: max(0., now - start) for name, start in state["phase_started_epoch"].items()}
    atomic_json(Path(manifest["state_dir"]) / "state.json", state)


def _completion(job, result, **extra):
    return {**{key: job.get(key) for key in ("id", "phase", "family", "cohort", "selection", "task",
            "test_name", "output", "definition_sha256", "task_sha256")},
            **result, **extra, "completed_at": utc_now()}


def _record_result(manifest, state, journal, job, result, **extra):
    row = _completion(job, result, **extra)
    journal.append(row)  # Commit before the state/report references this completion.
    if job["phase"] in {"sim", "gates"} and row.get("status") != "skipped":
        failed_connect = (not row.get("passed") and row.get("failure_phase") in {"simulator_start", "simulator_connect"})
        state["consecutive_start_connect_failures"] = state.get("consecutive_start_connect_failures", 0) + 1 if failed_connect else 0
        if state["consecutive_start_connect_failures"] >= 3:
            state["global_stop"] = "连续三项 CoppeliaSim 启动/连接失败"
    state["active_job"] = None
    state["last_completed_id"] = job["id"]
    save_state(manifest, state)
    rebuild_reports(manifest, journal, state)
    return row


def _reconcile_failures(state, rows):
    # A crash between journal fsync and state replacement cannot lose a failure.
    failures = 0
    for row in rows.values():
        if row["phase"] in {"sim", "gates"} and row.get("status") != "skipped":
            failed_connect = (not row.get("passed") and row.get("failure_phase") in {"simulator_start", "simulator_connect"})
            failures = failures + 1 if failed_connect else 0
    state["consecutive_start_connect_failures"] = failures
    if failures >= 3:
        state["global_stop"] = "连续三项 CoppeliaSim 启动/连接失败"


def _recover_active(manifest, state, journal):
    active = state.get("active_job")
    if not active:
        return
    if active["id"] in journal.load():
        state["active_job"] = None
        return
    job = active["job"]
    result_path = Path(active["result_path"])
    if result_path.exists():
        result = read_json(result_path)
        if result.get("definition_sha256") != job["definition_sha256"]:
            raise ValueError("恢复结果与任务定义不符")
        _record_result(manifest, state, journal, job, result, recovered=True)
    elif active.get("birth") is not None and process_birth(active["pid"]) == active["birth"]:
        raise RuntimeError("上次自有 worker 仍在运行，拒绝重复启动或接管")
    else:
        observed = active.get("elapsed_seconds")
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
            observed = None
        _record_result(manifest, state, journal, job, {"status": "interrupted", "passed": False,
            "failure_phase": "supervisor", "reason": "上次任务未提交完成记录；保留证据，不自动重试",
            "wall_seconds": None, "wall_seconds_is_exact": False,
            "last_observed_seconds": observed,
            "recovery_elapsed_upper_bound_seconds": max(0., time.time()-active["started_epoch"])}, recovered=True)


def run_phase(manifest, phase, *, poll_seconds=2., environment_check=preflight, process_factory=OwnedProcess,
              result_adapter=None):
    directory = Path(manifest["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        journal = Journal(directory)
        state = _state(manifest)
        _recover_active(manifest, state, journal)
        completed = journal.load()
        _reconcile_failures(state, completed)
        for job in manifest["jobs"]:
            if job["id"] in completed and completed[job["id"]].get("definition_sha256") != job["definition_sha256"]:
                raise ValueError("已完成 ID 的定义或任务文件已改变：" + job["id"])
        if state.get("global_stop"):
            state["status"] = "global_stop"
            save_state(manifest, state)
            rebuild_reports(manifest, journal, state)
            return state
        guard = ArchiveGuard(manifest.get("archive_baseline"))
        jobs = [job for job in manifest["jobs"] if job["phase"] == phase and job["id"] not in completed]
        if not jobs:
            state["status"] = "phase_complete"
            save_state(manifest, state)
            rebuild_reports(manifest, journal, state)
            return state
        now = time.time()
        if state.get("session_started_epoch") is None:
            state["session_started_epoch"] = now
        state["phase_started_epoch"].setdefault(phase, now)
        state["status"], state["phase"] = "running", phase
        save_state(manifest, state)
        for job in jobs:
            if job.get("skip_reason"):
                _record_result(manifest, state, journal, job, {"status": "skipped", "passed": False,
                    "executed": False, "failure_phase": "selection", "reason": job["skip_reason"], "wall_seconds": 0.})
                continue
            if remaining_seconds(state, phase) <= 0:
                state["status"] = "phase_budget_exhausted"
                break
            check = environment_check(guard)
            state["last_preflight"] = check
            if not check["ok"]:
                state["status"] = "global_stop" if check.get("global_stop") else "environment_blocked"
                if check.get("global_stop"):
                    state["global_stop"] = check["reason"]
                break
            if remaining_seconds(state, phase) <= 0:
                state["status"] = "phase_budget_exhausted"
                break
            output = Path(job["output"])
            output.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            specification = output / ("worker_" + token + ".json")
            result_path = output / ("worker_" + token + ".result.json")
            start_signal = output / ("worker_" + token + ".start")
            atomic_json(specification, {"job": job, "result_path": str(result_path), "start_signal": str(start_signal)})
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(specification)]
            started = time.time()
            started_monotonic = time.monotonic()
            timeout = min(job["timeout_seconds"], remaining_seconds(state, phase))
            owner = process_factory(command, output / ("worker_" + token + ".log"), start_signal)
            timed_out, phase_timeout, supervision_error = False, False, None
            try:
                state["active_job"] = {"id": job["id"], "job": job, "pid": owner.pid, "birth": owner.birth,
                    "started_epoch": started, "hard_deadline_epoch": started+timeout, "result_path": str(result_path),
                    "owned_processes": owner.refresh()}
                save_state(manifest, state)
                owner.release()
                while owner.process.poll() is None:
                    if time.monotonic() >= started_monotonic + timeout or remaining_seconds(state, phase) <= 0:
                        phase_timeout = remaining_seconds(state, phase) <= 0
                        timed_out = True
                        owner.terminate()
                        break
                    free = shutil.disk_usage("C:\\" if os.name == "nt" else ROOT).free
                    if free < MIN_C_FREE:
                        state["global_stop"] = "C 盘剩余不足 5 GiB"
                        owner.terminate()
                        supervision_error = state["global_stop"]
                        break
                    state["active_job"]["owned_processes"] = owner.refresh()
                    state["active_job"]["elapsed_seconds"] = time.monotonic()-started_monotonic
                    save_state(manifest, state)
                    time.sleep(min(poll_seconds, max(.01, started_monotonic+timeout-time.monotonic())))
                owned = owner.refresh()
                returncode = owner.process.poll()
            finally:
                owner.close()  # Native job close only affects this worker's descendants.
            if timed_out:
                result = {"status": "timeout", "passed": False, "failure_phase": "phase_budget" if phase_timeout else "task_wall_timeout",
                    "reason": "阶段/会话墙钟预算已用完" if phase_timeout else "单任务达到硬墙钟超时", "wall_seconds": time.monotonic()-started_monotonic}
            elif supervision_error:
                result = {"status": "stopped", "passed": False, "failure_phase": "environment", "reason": supervision_error,
                          "wall_seconds": time.monotonic()-started_monotonic}
            elif result_path.exists():
                result = read_json(result_path)
                if result.get("definition_sha256") != job["definition_sha256"]:
                    raise ValueError("worker 结果与任务定义不符")
            else:
                result = {"status": "worker_failed", "passed": False, "failure_phase": "worker",
                    "reason": f"worker 退出码 {returncode}，没有提交结果；详见日志", "wall_seconds": time.monotonic()-started_monotonic}
            result.setdefault("executed", True)
            try:
                postflight = environment_check(guard)
            except Exception as exc:
                postflight = {"ok": False, "global_stop": True,
                              "reason": f"任务后环境复核失败：{type(exc).__name__}: {exc}"}
            state["last_postflight"] = postflight
            if postflight.get("global_stop"):
                state["global_stop"] = postflight["reason"]
            if result_adapter is not None:
                result = result_adapter(job, result)
            _record_result(manifest, state, journal, job, result, returncode=returncode, owned_processes=owned)
            print(json.dumps({"completed": job["id"], "status": result["status"], "at": utc_now()}, ensure_ascii=False), flush=True)
            if state.get("global_stop"):
                state["status"] = "global_stop"
                break
            if not postflight["ok"]:
                state["status"] = "environment_blocked"
                break
            if phase_timeout:
                state["status"] = "phase_budget_exhausted"
                break
        else:
            state["status"] = "phase_complete"
        save_state(manifest, state)
        rebuild_reports(manifest, journal, state)
        return state


class NoProposal:
    def decompose(self, problem, feedback=None):
        return None


def plan_job(job):
    from skillnet.domains.blocks import load_problem
    from skillnet.pipeline import run
    from skillnet.report import plan_to_dict

    problem = load_problem(job["task"], ROOT / "skills/blocks_v1.toml")
    began = time.monotonic()
    result = run(problem, decomposer=NoProposal())
    elapsed = time.monotonic()-began
    plan = plan_to_dict(result)
    plan["batch_independent_planning"] = {"proposal_enabled": False, "witness_loaded": False,
        "default_search_budget_unchanged": True}
    atomic_json(Path(job["output"]) / "plan.json", plan)
    skills = sorted({step.action.name for step in result.schedule.steps})
    auxiliary = {name for name in skills if name in {"handover", "support_on", "support_off", "detach", "co_pick", "co_insert"}}
    for step in result.schedule.steps:
        bindings = dict(zip(step.action.skill.params, step.action.args))
        location = problem.hooks.locations.get(bindings.get("loc"), {})
        slot = problem.hooks.slots.get(bindings.get("slot"), {})
        if location.get("kind") == "buffer" or slot.get("temporary"):
            auxiliary.add(step.action.name)
    reason = None if result.status == "solved" else "; ".join(result.feedback) or (
        "未知能力：" + ", ".join(plan["unknown"]) if plan["unknown"] else "搜索结论：" + result.status)
    return {"status": result.status, "passed": result.status == "solved", "task_name": problem.name,
        "makespan": result.schedule.makespan, "expanded": result.schedule.expanded,
        "generated": result.schedule.generated, "elapsed_seconds": elapsed,
        "search_elapsed_seconds": result.schedule.elapsed, "skills": skills,
        "auxiliary": bool(auxiliary), "auxiliary_skills": sorted(auxiliary),
        "rejected": result.schedule.rejected, "unknown": plan["unknown"], "reason": reason,
        "failure_phase": None if result.status == "solved" else "planning",
        "proposal_enabled": False, "witness_loaded": False}


def sim_job(job):
    from simbridge.runner import run_task
    summary = run_task(job["task"], port=PORT, headless=True, video=False,
                       out=Path(job["output"]) / "simulation", timeout_wall_s=job["timeout_seconds"])
    problem_dir = Path(job["output"]) / "simulation" / summary.get("task", Path(job["task"]).stem)
    initial_path = problem_dir / "initial_plan.json"
    initial = read_json(initial_path) if initial_path.exists() else {}
    estimate = initial.get("makespan")
    actual = summary.get("actual_makespan_seconds")
    ratio = actual / estimate if isinstance(actual, (int, float)) and isinstance(estimate, (int, float)) and estimate > 0 else None
    return {**summary, "evidence_directory": str(problem_dir), "makespan": estimate,
        "feedback_count": summary.get("motion_feedback_count", 0),
        "compile_failed": summary.get("failure_phase") in {"real_planning", "compile"},
        "actual_to_estimate_ratio": ratio,
        "elapsed_seconds": summary.get("wall_seconds")}


def gate_job(job):
    import unittest
    os.environ["SKILLNET_LIVE_SIM"] = "1"
    module = importlib.import_module("tests.test_live_sim")
    original = module.run_task
    summaries = []

    def recorded_run_task(task, **kwargs):
        # Only relocate evidence; the actual public runner and acceptance stay unchanged.
        kwargs.update(out=Path(job["output"]) / "simulation", port=PORT,
                      headless=True, video=False, timeout_wall_s=job["timeout_seconds"])
        summary = original(task, **kwargs)
        summaries.append(summary)
        return summary

    module.run_task = recorded_run_task
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName(job["test_name"])
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    finally:
        module.run_task = original
    summary = summaries[-1] if summaries else {}
    passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped and bool(summary.get("passed"))
    evidence = Path(job["output"]) / "simulation" / summary.get("task", "unknown_task")
    return {**summary, "status": "passed" if passed else "failed", "passed": passed,
        "evidence_directory": str(evidence),
        "test_name": job["test_name"], "tests_run": result.testsRun,
        "test_failures": len(result.failures), "test_errors": len(result.errors), "test_skipped": len(result.skipped),
        "failure_phase": None if passed else summary.get("failure_phase", "gated_test"),
        "reason": None if passed else summary.get("reason") or str(result.failures + result.errors + result.skipped)}


def worker(specification):
    data = read_json(confined(specification))
    job, result_path = data["job"], confined(data["result_path"])
    start_signal = confined(data["start_signal"])
    waiting = time.monotonic()
    while not start_signal.exists():
        if time.monotonic()-waiting > 30:
            raise RuntimeError("supervisor 未确认进程所有权，拒绝开始任务")
        time.sleep(.05)
    began = time.monotonic()
    try:
        if job["phase"] != "gates" and sha256(confined(job["task"])) != job["task_sha256"]:
            raise ValueError("任务文件在清单加载后发生改变")
        result = {"plan": plan_job, "sim": sim_job, "gates": gate_job}[job["phase"]](job)
    except BaseException as exc:
        result = {"status": "worker_failed", "passed": False, "failure_phase": "worker",
            "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    result["worker_wall_seconds"] = time.monotonic()-began
    result.setdefault("wall_seconds", result["worker_wall_seconds"])
    result["definition_sha256"] = job["definition_sha256"]
    atomic_json(result_path, result)
    return 0 if result.get("passed") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="可恢复、单线、逐任务持久化的规划/仿真批处理")
    parser.add_argument("--manifest", type=Path, help="项目内 JSON 清单")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--rebuild", action="store_true", help="仅重建 CSV/报告，不启动任务")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            return worker(args.worker)
        if not args.manifest or (not args.rebuild and not args.phase):
            parser.error("指定 --manifest 和 --phase，或使用 --rebuild")
        manifest = load_manifest(args.manifest)
        if args.rebuild:
            state_path = Path(manifest["state_dir"]) / "state.json"
            rebuild_reports(manifest, state=read_json(state_path) if state_path.exists() else None)
            return 0
        state = run_phase(manifest, args.phase)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0 if state["status"] == "phase_complete" else 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"批处理停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
