"""Detached resumable gate -> fixed-tape00 -> fixed-tape02 supervisor.

Run this supervisor directly, never inside control.run_job. Only each worker
enters its own owned Job Object and the shared simulation execution lock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as old
from tools import motion_v135_control as control
from tools.motion_v135_m5_chain import guard
from tools.motion_v135_preflight import verify_native

BASE = control.BASE
CASES = {"00": "02/simulation/gen_f1_s20261115_014", "02": "04/simulation/gen_f1_s20261115_024"}
TOOLS = ("motion_v135_abba_chain.py", "motion_v135_fast_idle.py", "motion_v135_tpg_profile.py",
         "motion_v135_control.py", "motion_v135_single.py", "motion_v135_preflight.py",
         "motion_v135_m5_chain.py", "motion_v135_detached.py", "overnight_batch.py", "overnight_regression.py")


def read(path):
    return old.read_json(path) if Path(path).is_file() else None


def snapshot():
    manifest = ROOT/"results/sim_v1_3/v134/manifests/sim20.json"
    inputs = [manifest, ROOT/"tasks/s1_parallel_place.toml"]
    jobs = old.read_json(manifest)["jobs"]
    for case, suffix in CASES.items():
        job = next(j for j in jobs if j["id"] == "v134-fixed20-"+case)
        if job["timeout_seconds"] != 1200 or old.sha256(job["task"]) != job["prepared_task_sha256"]:
            raise ValueError("Original task SHA or1200s wall changed")
        reference = BASE/"diagnostics/failed14_a1/cases"/suffix
        inputs += [Path(job["task"]), reference/"tasks.json", reference/"workcell.json"]
    return {"core": control.version(), "tools": {"tools/"+p: old.sha256(ROOT/"tools"/p) for p in TOOLS},
            "inputs": {str(p.resolve()): old.sha256(p) for p in inputs}}


def save(root, state, **changes):
    state.update(changes, updated_at=old.utc_now())
    old.atomic_json(root/"state.json", state)
    old.atomic_json(root/"report.json", state)
    lines = ["# FastIdle ABBA 串行监督", "", "状态："+state["status"], "", "更新："+state["updated_at"], "",
             "|步骤|结论|证据|原因|", "|---|---|---|---|"]
    for label, row in state.get("steps", {}).items():
        lines.append(f"|{label}|{row.get('status')}|{row.get('output')}|{row.get('reason', '')}|")
    lines += ["", "固定tape是停止场景诊断，不计作装配通过；原生门单独严格验收。"]
    old.atomic_text(root/"report.md", "\n".join(lines)+"\n")


def job_core(job):
    value = job.get("code_version")
    return value if isinstance(value, dict) else read(value) if value else None


def command_out(command):
    try:
        return Path(command[command.index("--out")+1]).resolve()
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("Job has no explicit evidence --out") from exc


def check_job(job, identity, output, *, gate=False):
    if job_core(job) != identity["core"] or job.get("code_unchanged_during_job") is not True:
        raise ValueError("Completed job core version differs or unchanged proof missing")
    if not job.get("cleanup", {}).get("passed"):
        raise RuntimeError("Job cleanup failed; do not start another simulator")
    command = job.get("command", [])
    if command_out(command) != output:
        raise ValueError("Job output identity differs")
    if job.get("seconds") not in (None, 1200., 1200):
        raise ValueError("Job changed original1200s wall")
    if gate and ("--gate" not in command or command[command.index("--gate")+1] != old.LIVE_TESTS[0]):
        raise ValueError("Existing gate command did not name original parallel-place test")


def obtain(root, state, label, job_name, output, command, identity, snapshotter, environment_check, job_runner, poll):
    for attempt in range(3):
        name = job_name if attempt == 0 else job_name+"_resume"+str(attempt)
        destination = output if attempt == 0 else root/(label+"_resume"+str(attempt))
        cmd = list(command)
        cmd[cmd.index("--out")+1] = str(destination)
        folder = BASE/"jobs"/name
        while True:
            if snapshotter() != identity:
                raise ValueError("Code/tool/input fingerprint changed; refusing mixed-version chain")
            if (BASE/"global_stop.json").exists():
                raise RuntimeError("GLOBAL_STOP exists")
            result, previous = read(folder/"result.json"), read(folder/"state.json")
            if result is not None:
                check_job(result, identity, destination, gate=label == "gate_parallel")
                return result, destination, name
            if previous and previous.get("birth") and old.process_birth(previous["pid"]) == previous["birth"]:
                if job_core(previous) != identity["core"] or command_out(previous.get("command", [])) != destination:
                    raise ValueError("Active job is not the same version/output")
                # Expected active sim owns23210: do not call idle health guard.
                if __import__('shutil').disk_usage("C:\\").free < old.MIN_C_FREE:
                    old.atomic_json(BASE/"global_stop.json", {"at": old.utc_now(), "reason": "C drive below5GiB"})
                    raise RuntimeError("GLOBAL_STOP disk")
                save(root, state, status="waiting_existing_job", active_label=label,
                     active_job=name, active_pid=previous["pid"], command=previous.get("command"))
                time.sleep(poll)
                continue
            if previous or destination.exists():
                receipt = root/"interruptions"/(name+".json")
                old.atomic_json(receipt, {"at": old.utc_now(), "job_state": previous,
                                         "output": str(destination), "next_attempt": attempt+1,
                                         "reason": "No complete controlled result or live PID/birth; preserve attempt"})
                break
            environment_check(root)
            save(root, state, status="running", active_label=label, active_job=name,
                 active_output=str(destination), command=cmd)
            result = job_runner(name, cmd, 1200., live=True)
            if snapshotter() != identity:
                raise ValueError("Code/tool/input changed during worker")
            check_job(result, identity, destination, gate=label == "gate_parallel")
            return result, destination, name
    raise RuntimeError("Three interrupted attempts retained; no further automatic rerun")


def profile_proof(case, output, job):
    profile = read(output/"profile.json") or {}
    passed = (job.get("status") == "passed" and profile.get("status") == "completed"
              and profile.get("cleanup_passed") is True and profile.get("code_unchanged") is True
              and profile.get("case") == case and profile.get("budget_seconds") == 1200
              and profile.get("abba", {}).get("passed") is True
              and [p.get("phase") for p in profile.get("abba", {}).get("phases", [])] == ["A1", "B1", "B2", "A2"]
              and all(p.get("passed") is True for p in profile.get("abba", {}).get("phases", [])))
    if passed:
        from tools.motion_v135_fast_idle import code_version
        passed = profile.get("code_sha256") == code_version()
    failure_status = "timeout" if profile.get("status") == "timeout" or job.get("status") == "timeout" else "failed"
    return {"passed": bool(passed), "status": "passed" if passed else failure_status,
            "reason": profile.get("reason") or (None if passed else "Missing/incomplete matched-query proof"),
            "profile": str(output/"profile.json"), "profile_status": profile.get("status", "missing"),
            "diagnostic_only": True, "formal_assembly_pass": False,
            "ratio_A_over_B": profile.get("abba", {}).get("matched_query_api_ratio_A_over_B")}


def run(run_id, gate_job=None, gate_out=None, *, snapshotter=None, environment_check=None, job_runner=None, poll_seconds=2.):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Use a safe run-id")
    snapshotter, environment_check, job_runner = snapshotter or snapshot, environment_check or guard, job_runner or control.run_job
    root = BASE/"m5"/run_id
    root.mkdir(parents=True, exist_ok=True)
    gate_job = gate_job or "abba_"+run_id+"_parallel_gate"
    gate_out = Path(gate_out).resolve() if gate_out else root/"parallel_gate"
    identity = snapshotter()
    path = root/"state.json"
    state = read(path) or {"run_id": run_id, "steps": {}, "created_at": old.utc_now(),
                          "identity": identity, "gate_job": gate_job, "gate_out": str(gate_out)}
    if state["identity"] != identity or state["gate_job"] != gate_job or state["gate_out"] != str(gate_out):
        raise ValueError("Refusing changed version/input/gate for existing run-id")
    with old.execution_lock(root/"chain.lock"):
        try:
            save(root, state, status="running")
            steps = [("gate_parallel", gate_job, gate_out,
                [sys.executable, "-B", "-X", "utf8", str(ROOT/"tools/motion_v135_single.py"),
                 "--gate", old.LIVE_TESTS[0], "--out", str(gate_out), "--seconds", "1200"])]
            for case, suffix in CASES.items():
                output = root/("F1_014" if case == "00" else "F1_024")
                steps.append(("abba_"+case, "abba_"+run_id+"_"+case, output,
                    [sys.executable, "-B", "-X", "utf8", str(ROOT/"tools/motion_v135_fast_idle.py"),
                     "--case", case, "--reference-dir", str(BASE/"diagnostics/failed14_a1/cases"/suffix),
                     "--out", str(output), "--seconds", "1200", "--batches", "256", "--warmup", "8"]))
            for label, name, output, command in steps:
                if snapshotter() != identity:
                    raise ValueError("Changed chain fingerprint before step")
                previous = state["steps"].get(label)
                if previous is not None:
                    job, destination = previous["job"], Path(previous["output"])
                    check_job(job, identity, destination, gate=label == "gate_parallel")
                else:
                    job, destination, name = obtain(root, state, label, name, output, command, identity,
                                                   snapshotter, environment_check, job_runner, poll_seconds)
                proof = (verify_native("gate_parallel", destination, job, identity) if label == "gate_parallel"
                         else profile_proof(label[-2:], destination, job))
                row = {**proof, "status": proof.get("status", "passed" if proof["passed"] else "failed"), "output": str(destination),
                       "job_name": name, "job": job, "completed_at": old.utc_now()}
                state["steps"][label] = row
                save(root, state, status="running", active_label=None, active_job=None)
                control.record("M5_ABBA_"+run_id+"_"+label, row["status"], evidence=str(root/"report.json"))
                environment_check(root)
                if label == "gate_parallel" and not proof["passed"]:
                    save(root, state, status="gate_failed", requires_rollback=True,
                         reason=proof.get("reason"), finished_at=old.utc_now())
                    return state
            status = "finished" if all(row["passed"] for row in state["steps"].values()) else "finished_with_failures"
            save(root, state, status=status, finished_at=old.utc_now())
        except BaseException as exc:
            save(root, state, status="global_stop" if (BASE/"global_stop.json").exists() else "interrupted",
                 reason=f"{type(exc).__name__}: {exc}")
        finally:
            control.record("M5_ABBA_chain_"+run_id, state["status"], evidence=str(path))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gate-job")
    parser.add_argument("--gate-out", type=Path)
    args = parser.parse_args(argv)
    if args.gate_out and not args.gate_out.resolve().is_relative_to(BASE):
        parser.error("gate-out must stay in motion_v135")
    result = run(args.run_id, args.gate_job, args.gate_out)
    print(json.dumps({key: result.get(key) for key in ("run_id", "status", "reason")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
