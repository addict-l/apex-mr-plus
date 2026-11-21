"""Resumable, serial, owned-process jobs for the motion optimization round."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools.overnight_regression import health

BASE = ROOT / "results/sim_v1_3/motion_v135"
_archive_guard = None
_EXCEPTION_PATH = "results/apex/geometry/rollout_verification.json"
_EXCEPTION_ORIGINAL = "b1860c844c677aca51185a371f9cd984ec14516d9e2ab97b984a694e950a56ad"
_EXCEPTION_ACCEPTED = "8a8a1a66e4d98ecef2caefbf7a01d359a5cb737102760138ab86f0cd2335c47c"


def _authorized_history_exception(name, original, actual):
    """Recognize only the user's one explicit exact-byte exception.

    The immutable original baseline stays intact. Neither a different artifact
    nor a later overwrite can inherit this authorization.
    """
    if (name, original, actual) != (_EXCEPTION_PATH, _EXCEPTION_ORIGINAL, _EXCEPTION_ACCEPTED):
        return None
    path = BASE/"protection_exceptions.json"
    if not path.is_file():
        return None
    try:
        data = old.read_json(path)
        allowed = {"path": _EXCEPTION_PATH, "original_sha256": _EXCEPTION_ORIGINAL,
                   "accepted_sha256": _EXCEPTION_ACCEPTED, "not_acceptance_evidence": True}
        if (data.get("authorization_id") != "user-motion-v135-resume-20261117"
                or data.get("baseline_sha256") != old.sha256(BASE/"baseline.json")
                or data.get("exceptions") != [allowed]):
            return None
    except (OSError, ValueError, TypeError):
        return None
    return {**allowed, "authorization_id": data["authorization_id"],
            "register_sha256": old.sha256(path)}


def version():
    return {p.relative_to(ROOT).as_posix(): old.sha256(p)
            for name in ("skillnet", "simbridge", "apexmr", "gen", "skills")
            for p in (ROOT/name).rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and p.suffix in (".py", ".lua", ".toml", ".json")}


def audit(*, full=False):
    global _archive_guard
    if _archive_guard is None:
        _archive_guard = old.ArchiveGuard()
    baseline = old.read_json(BASE/"baseline.json")
    changed, exceptions = [], []
    for row in baseline["source_files"]:
        name = row["path"]
        immutable = (name.startswith(("skillnet/", "gen/", "skills/", "tasks/"))
                     and name != "skillnet/pipeline.py") or name in {
                         "SPEC.md", "HANDOFF_v1.3_SIM.md", "FIX_v1.3.4.md"}
        if immutable and (not (ROOT/name).is_file() or old.sha256(ROOT/name) != row["sha256"]):
            changed.append(name)
    expected = {row["path"]: row for row in baseline["history_files"]}
    actual = {p.relative_to(ROOT).as_posix(): p for p in (ROOT/"results").rglob("*")
              if p.is_file() and not p.is_relative_to(BASE)}
    for name in expected.keys() | actual.keys():
        if name not in expected or name not in actual:
            changed.append(name)
            continue
        p, row = actual[name], expected[name]
        stat = p.stat()
        if full or name == _EXCEPTION_PATH or stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            actual_sha = old.sha256(p)
            if actual_sha != row["sha256"]:
                authorized = _authorized_history_exception(name, row["sha256"], actual_sha)
                if authorized is None:
                    changed.append(name)
                else:
                    exceptions.append(authorized)
    archives = (old.ArchiveGuard() if full else _archive_guard).check()
    result = {"at": old.utc_now(), "passed": not changed and archives["passed"],
              "full_history_hash": full, "changed": sorted(changed), "archives": archives,
              "authorized_exceptions": exceptions}
    if not result["passed"]:
        old.atomic_json(BASE/"global_stop.json", result)
    return result


def record(step, status, **detail):
    with old.execution_lock(BASE/"progress.lock"):
        data = old.read_json(BASE/"progress_state.json")
        row = {"step": step, "status": status, "at": old.utc_now(), **detail}
        data["steps"].append(row)
        data["updated_at"] = row["at"]
        old.atomic_json(BASE/"progress_state.json", data)


def worker(path):
    data = old.read_json(path)
    deadline = time.monotonic()+60
    while not Path(data["signal"]).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned-process assignment handshake not received")
        time.sleep(.02)
    return subprocess.run(data["command"], cwd=ROOT, stdin=subprocess.DEVNULL).returncode


def run_job(name, command, seconds, *, live=False):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Job name must be a simple unique identifier")
    output = BASE/"jobs"/name
    output.mkdir(parents=True, exist_ok=True)
    committed = output/"result.json"
    if committed.exists():
        return old.read_json(committed)
    with old.execution_lock(ROOT/"results/sim_v1_3/overnight_execution.lock"):
        if (BASE/"global_stop.json").exists():
            raise RuntimeError("Global stop exists; inspect before resuming")
        protection = audit()
        if not protection["passed"]:
            raise RuntimeError("Archive/frozen-source discrepancy: overall stop")
        machine = health()
        if not machine["passed"]:
            if machine["c_free_gib"] < 5:
                old.atomic_json(BASE/"global_stop.json", machine)
            raise RuntimeError("Environment is occupied or below the disk guard")
        if old.full_test_processes():
            raise RuntimeError("A default test suite is already running")
        token = uuid.uuid4().hex
        signal = output/(token+".start")
        spec = output/(token+".worker.json")
        old.atomic_json(spec, {"signal": str(signal), "command": command})
        revision = version()
        old.atomic_json(output/"code_version.json", revision)
        owner = old.OwnedProcess([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                  "--worker", str(spec)], output/"stdout.log", signal)
        began = time.monotonic()
        state = {"name": name, "status": "running", "pid": owner.pid, "birth": owner.birth,
                 "command": command, "seconds": seconds, "live": live,
                 "started_at": old.utc_now(), "code_version": str(output/"code_version.json")}
        timed_out = False
        try:
            old.atomic_json(output/"state.json", state)
            owner.release()
            while owner.process.poll() is None:
                elapsed = time.monotonic()-began
                if elapsed >= seconds:
                    timed_out = True
                    owner.terminate()
                    break
                if __import__("shutil").disk_usage("C:\\").free < old.MIN_C_FREE:
                    old.atomic_json(BASE/"global_stop.json", {"at": old.utc_now(), "reason": "C drive below 5 GiB"})
                    owner.terminate()
                    break
                state.update(elapsed_seconds=elapsed, owned_processes=owner.refresh())
                old.atomic_json(output/"state.json", state)
                time.sleep(1)
            state.update(status="timeout" if timed_out or owner.process.returncode == 124 else ("passed" if owner.process.returncode == 0 else "failed"),
                         returncode=owner.process.returncode, wall_seconds=time.monotonic()-began,
                         ended_at=old.utc_now())
        finally:
            owner.close()
        state["cleanup"] = health()
        if not state["cleanup"]["passed"]:
            old.atomic_json(BASE/"global_stop.json", {"at": old.utc_now(), "reason": "Owned process cleanup failed", "health": state["cleanup"]})
        state["code_unchanged_during_job"] = version() == revision
        old.atomic_json(committed, state)
        old.atomic_json(output/"state.json", state)
        record(name, state["status"], evidence=str(committed))
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.audit:
        result = audit(full=args.full_history)
        old.atomic_json(BASE/"latest_protection.json", result)
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not args.name or not command or args.seconds <= 0:
            parser.error("--name and a command after -- are required")
        result = run_job(args.name, command, args.seconds, live=args.live)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("passed", result.get("status") == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
