"""Resumable M6 prerequisites, never a launcher for the formal cohort.

Run directly in a detached process. This supervisor calls dispatch/run_job;
wrapping it in control.run_job would acquire the same serial lock twice.
Every native gate has a 1200-second hard budget. Retained-context probes have
180 seconds and explicitly remain diagnostics, not native end-to-end passes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import motion_v135_control as control
from tools import motion_v135_regression as regression

GATES = {
    "gate_parallel": (old.LIVE_TESTS[0], "tasks/s1_parallel_place.toml"),
    "gate_handover": (old.LIVE_TESTS[1], "tasks/s1_reach_handover.toml"),
    "gate_tower": (old.LIVE_TESTS[2], "tasks/sim/s2_tower_sim.toml"),
    "gate_cantilever": (old.LIVE_TESTS[3], "tasks/sim/s3a_cantilever_sim.toml"),
    "gate_platform": (old.LIVE_TESTS[4], "tasks/sim/s3a_platform_sim.toml")}
PROBES = {"probe_table": "06/simulation/gen_f2_s20261115_014",
          "probe_branch": "07/simulation/gen_f2_s20261115_012"}
STEPS = ("tpg_profiles", "default", "original11", *GATES, *PROBES)
TOOLS = ("motion_v135_preflight.py", "motion_v135_regression.py", "motion_v135_control.py",
         "motion_v135_single.py", "motion_v135_replay.py", "motion_v135_tpg_profile.py")


def snapshot():
    return {"core": control.version(), "tools": {"tools/"+name: old.sha256(ROOT/"tools"/name) for name in TOOLS}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _files(paths):
    return {str(Path(path).resolve()): old.sha256(path) for path in paths}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _check_job(job, version):
    _require(job.get("status") == "passed", "controlled job did not pass")
    _require(job.get("code_unchanged_during_job") is True, "job code version changed or missing")
    _require(job.get("cleanup", {}).get("passed") is True, "job cleanup proof missing/failed")
    path = Path(job.get("code_version", ""))
    _require(path.is_file() and old.read_json(path) == version["core"], "native/offline proof has old core version")


def _connection(worker):
    phase = worker.get("startup_failure_phase") or worker.get("failure_phase") or worker.get("phase")
    startup = phase in {"simulator_start", "simulator_connect"}
    connected = bool(worker.get("passed")) or (not startup and (
        "workcell" in worker.get("phase_seconds", {}) or phase in {
            "real_planning", "compile", "execute", "validation", "complete", "home_path", "compile_action"}))
    return {"startup_failure": startup, "connected": connected, "failure_phase": phase}


def verify_native(step, output, job, version):
    output = Path(output).resolve()
    worker_path = output/"worker_result.json"
    worker = old.read_json(worker_path) if worker_path.is_file() else {}
    proof = {"passed": False, "scope": "native_dynamics_and_independent_validation",
             "worker_result": str(worker_path), **_connection(worker)}
    try:
        _check_job(job, version)
        _require(worker_path.is_file(), "missing worker_result.json")
        test, task = GATES[step]
        expected_sha = old.sha256(ROOT/task)
        _require(worker.get("passed") is True and worker.get("status") == "passed", "native worker failed")
        _require(worker.get("test_name") == test and worker.get("tests_run") == 1
                 and all(worker.get(key) == 0 for key in ("test_skipped", "test_failures", "test_errors")),
                 "native gate did not actually run exactly one unskipped original test")
        _require(worker.get("task_sha256") == expected_sha, "native worker task SHA mismatch")
        evidence = Path(worker["evidence_directory"]).resolve()
        _require(evidence.is_relative_to(output) and evidence.name == Path(task).stem, "native evidence directory mismatch")
        names = ("summary.json", "validation.json", "execution.json", "execution_validation.json",
                 "reference_receipt.json", "tasks.json", "tpg.json", "geometry/rollout_verification.json")
        documents = {}
        for name in names:
            path = evidence/name
            _require(path.is_file(), "missing native evidence: "+name)
            documents[name] = old.read_json(path)
        summary, validation, execution = (documents[name] for name in ("summary.json", "validation.json", "execution.json"))
        rollout = documents["geometry/rollout_verification.json"]
        _require(summary.get("passed") is True and summary.get("success") is True
                 and summary.get("status") == "passed" and summary.get("simulator_closed") is True
                 and summary.get("task_sha256") == expected_sha
                 and summary.get("real_planning_status") == "solved"
                 and not summary.get("fatal_environment") and not summary.get("remote_interface_failure"), "native summary not fully passed")
        _require(validation.get("success") is True and validation.get("violations") == []
                 and validation.get("expected_nodes", 0) > 0
                 and validation.get("actual_node_intervals") == validation.get("expected_nodes"), "independent native validation failed/incomplete")
        _require(execution.get("success") is True and execution.get("done") is True
                 and execution.get("independent_validation_passed") is True
                 and execution.get("total", 0) > 0 and execution.get("completed") == execution.get("total")
                 and execution.get("tracking_failures") == [] and execution.get("collision_failures") == []
                 and not execution.get("timeout") and not execution.get("remote_interface_failure")
                 and bool(execution.get("task_schedule")), "native execution missing/incomplete/failed")
        _require(rollout.get("valid") is True and rollout.get("coverage_complete") is True
                 and rollout.get("no_sampled_collision") is True and rollout.get("stationary_waiting_poses_included") is True
                 and rollout.get("planned_samples", 0) > 0
                 and rollout.get("tested_samples") == rollout.get("planned_samples"), "nominal sampled rollout missing/incomplete")
        proof.update(passed=True, task=task, task_sha256=expected_sha, connected=True,
            evidence_directory=str(evidence), evidence_files=_files([worker_path, Path(job["code_version"]), *[evidence/name for name in names]]),
            sampling_scope="nominal sampled rollout plus native monitor/independent validation; no continuous collision guarantee")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        proof["reason"] = str(exc)
    return proof


def verify_profiles(paths, version, *, require_complete=False):
    proof = {"passed": False, "scope": "retained_reference_TPG_diagnostics", "profiles": [], "evidence_files": {}}
    try:
        _require(len(paths) == 2, "需要00/02两份M5同输入profile.json")
        found = set()
        for path in paths:
            path = old.confined(path)
            data = old.read_json(path)
            case = data.get("case")
            _require(case in {"00", "02"} and case not in found, "M5 profile case重复/不正确")
            found.add(case)
            original = control.BASE/"m1"/("tpg"+case+"_a1")/"profile.json"
            reference = old.read_json(original)
            _require(all(data.get(key) == reference.get(key) for key in
                         ("task_sha256", "source_tasks_sha256", "source_workcell_sha256")), "M5 profile changed original input")
            hashes = {key.replace("\\", "/"): value for key, value in data.get("code_sha256", {}).items()}
            required = {"apexmr/geometry.py", "apexmr/tpg.py", "apexmr/program.py", "apexmr/motion.py", "apexmr/motion_backend.lua"}
            _require(required <= hashes.keys() and all(version["core"].get(key) == value for key, value in hashes.items()), "M5 profile old/missing relevant code SHA")
            _require(data.get("diagnostic_only") is True and data.get("formal_run") is False, "M5 profile diagnostic scope missing")
            completed = data.get("status") == "completed" and data.get("cleanup_passed") is True
            if data.get("until") == "rollout":
                completed = completed and data.get("rollout", {}).get("valid") is True
            controlled = None
            if require_complete:
                _require(completed and data.get("until") == "rollout"
                         and data.get("graph_counts", {}).get("nodes", 0) > 0,
                         "正式M5需要完成完整图与rollout，不能采用ABBA/超时前缀")
                _require(data.get("joint_sample_step_rad") == reference.get("joint_sample_step_rad") == .07
                         and data.get("rollout_step_s") == reference.get("rollout_step_s") == .05,
                         "完整M5改变或缺少原几何采样精度")
                _require(data.get("project_code_sha256") == version["core"]
                         and data.get("profile_tool_sha256") == version["tools"].get("tools/motion_v135_tpg_profile.py"),
                         "完整M5的项目/工具指纹缺失或不属于当前版本")
                rollout = data.get("rollout", {})
                _require(rollout.get("coverage_complete") is True
                         and rollout.get("no_sampled_collision") is True
                         and rollout.get("stationary_waiting_poses_included") is True
                         and rollout.get("planned_samples", 0) > 0
                         and rollout.get("tested_samples") == rollout.get("planned_samples"),
                         "完整M5的名义rollout覆盖不完整")
                idle = data.get("fast_idle_scope", {})
                before, after = idle.get("before"), idle.get("after")
                physical = {"simulation_state", "simulation_time_s", "simulation_timestep_s",
                            "dynamics_step_s", "dynamics_engine", "gravity", "idle_fps", "fast_idle_tag_hex"}
                _require(data.get("fast_idle_enabled") is True and idle.get("enabled") is True
                         and idle.get("restored") is True and isinstance(before, dict)
                         and physical <= before.keys() and before == after
                         and before["simulation_state"] == 0, "停止场景优化未确认精确物理/app恢复")
                controlled = old.confined(data.get("controlled_job_result", ""))
                _require(controlled.is_relative_to(control.BASE/"jobs") and controlled.is_file(),
                         "完整M5缺少当前受控作业证明")
                job = old.read_json(controlled)
                _check_job(job, version)
                command = job.get("command", [])
                _require(job.get("live") is True and job.get("seconds") == 1200
                         and 0 < data.get("budget_seconds", 0) <= 1200
                         and any(Path(str(arg)).name == "motion_v135_tpg_profile.py" for arg in command)
                         and "--fast-idle" in command
                         and command[command.index("--case")+1] == case
                         and Path(command[command.index("--out")+1]).resolve() == path.parent.resolve()
                         and command[command.index("--until")+1] == "rollout",
                         "完整M5受控命令、案例、输出或原预算不匹配")
            row = {"case": case, "path": str(path), "status": data.get("status"),
                "diagnostic_scope_passed": completed, "scope": data.get("until"),
                "elapsed_seconds": data.get("elapsed_seconds"), "stage_seconds": data.get("stage_seconds", {}),
                "remaining": list(data.get("stages_not_executed", [])), "native_dynamics_proven": False}
            row["complete_current_controlled_geometry"] = bool(require_complete and completed)
            proof["profiles"].append(row)
            proof["evidence_files"].update(_files([path, original]))
            if controlled is not None:
                proof["evidence_files"].update(_files([controlled, Path(job["code_version"])]))
        proof["passed"] = all(row["diagnostic_scope_passed"] for row in proof["profiles"])
        if not proof["passed"]:
            proof["reason"] = "M5诊断仍有timeout/失败；保留阶段范围与剩余项"
    except (ValueError, KeyError, TypeError, OSError, IndexError, AttributeError) as exc:
        proof["reason"] = str(exc)
    return proof


def _regression_proof(step, result, version):
    proof = {"passed": False, "scope": "offline_"+step, "evidence_files": {}}
    try:
        _require(result.get("kind") == step and result.get("passed") is True and len(result.get("runs", [])) == 1,
                 "regression dispatch not passed/exactly one run")
        row = result["runs"][0]
        _require(row.get("passed") is True, "regression worker proof failed")
        _check_job(row["job"], version)
        path = old.confined(row["worker_json"])
        worker, request = old.read_json(path), old.read_json(path.parent/"request.json")
        _require(request.get("tool_sha256") == version["tools"]["tools/motion_v135_regression.py"], "regression wrapper tool SHA differs")
        _require(worker.get("status") == "passed", "regression worker status failed")
        if step == "default":
            _require(worker.get("tests_discovered", 0) > 0 and worker.get("tests_run") == worker.get("tests_discovered")
                     and worker.get("failure_count") == worker.get("error_count") == 0, "default evidence incomplete")
        else:
            _require(worker.get("passed") is True and len(worker.get("rows", [])) == 11
                     and all(item.get("passed") is True and item.get("rule_replay", {}).get("passed") is True
                             for item in worker["rows"]), "original11 replay/certificate evidence incomplete")
        proof.update(passed=True, regression=result,
            evidence_files=_files([path, path.parent/"request.json", Path(row["job"]["code_version"])]))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        proof["reason"] = str(exc)
    return proof


def _stem(attempt, step):
    return "m6_preflight_"+attempt+"_"+step


def perform(step, output, attempt, version, tpg_profiles):
    if step == "tpg_profiles":
        return verify_profiles(tpg_profiles, version, require_complete=True)
    stem = _stem(attempt, step)
    if step in {"default", "original11"}:
        result = regression.dispatch(stem, step)
        return _regression_proof(step, result, version)
    if step in GATES:
        command = [sys.executable, "-X", "utf8", "-u", "tools/motion_v135_single.py", "--gate", GATES[step][0],
                   "--out", str(output), "--seconds", "1200"]
        job = control.run_job(stem, command, 1200., live=True)
        return {**verify_native(step, output, job, version), "controlled_job": job}
    source = ROOT/"results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/sim20"/PROBES[step]
    command = [sys.executable, "-X", "utf8", "-u", "tools/motion_v135_replay.py", "--source", str(source),
        "--record-index", "1", "--mode", "action", "--use-backend", "--diagnostic-seconds", "180", "--out", str(output)]
    job = control.run_job(stem, command, 180., live=True)
    proof = {"passed": False, "scope": "retained_context_action_diagnostic_not_native_end_to_end",
             "diagnostic_seconds": 180., "controlled_job": job, "evidence_files": {}}
    candidates = list(output.glob("action_*/result.json"))
    worker = old.read_json(candidates[0]) if len(candidates) == 1 else {}
    proof.update(_connection(worker))
    try:
        _check_job(job, version)
        _require(len(candidates) == 1, "缺失或存在多个不明确的动作诊断结果")
        path = candidates[0]
        supervisor = old.read_json(path.parent/"supervisor.json")
        _require(worker.get("status") == "reference_action_passed" and worker.get("passed") is True
                 and worker.get("simulator_closed") is True, "保留上下文动作诊断未通过")
        _require(worker.get("source") == str(source) and worker.get("record_index") == 1
                 and worker.get("task_sha256") == old.read_json(source/"summary.json")["task_sha256"], "诊断来源身份或SHA不匹配")
        _require(supervisor.get("status") == "completed" and supervisor.get("cleanup", {}).get("passed") is True
                 and not supervisor.get("cleanup_errors"), "诊断拥有者清理失败/证据不完整")
        proof.update(passed=True, connected=True, source=str(source), task_sha256=worker["task_sha256"],
            evidence_files=_files([path, path.parent/"supervisor.json", path.parent/"restored_context.json",
                path.parent/"compile_evidence.json", path.parent/"queries.jsonl", Path(job["code_version"])]))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        proof["reason"] = str(exc)
    return proof


def _dependencies(step):
    if step in {"default", "tpg_profiles"}:
        return ()
    if step == "original11":
        return ("default",)
    return ("tpg_profiles", "default", "original11")+( () if step == "gate_parallel" else ("gate_parallel",))


def _verify_saved(proof, version):
    _require(proof.get("version") == version, "已有证明来自不同版本")
    for path, expected in proof.get("evidence_files", {}).items():
        _require(Path(path).is_file() and old.sha256(path) == expected, "已有证明证据SHA变化："+path)


def _reuse(path, step, version):
    data = old.read_json(old.confined(path))
    if "version" in data and "passed" in data:
        _verify_saved(data, version)
        return {**data, "reused_from": str(path)}
    if step in {"default", "original11"}:
        proof = _regression_proof(step, data, version)
        _require(proof["passed"], proof.get("reason", "regression reuse failed"))
        return {**proof, "version": deepcopy(version), "reused_from": str(path)}
    raise ValueError("native/诊断外部证明必须包含当时工具和core版本快照；不能将旧M3/M4门冒充最终码证明")


def _stop(detail):
    old.atomic_json(control.BASE/"global_stop.json", detail)


def _run(attempt, *, base=None, steps=None, tpg_profiles=None, reuse=None,
        snapshotter=snapshot, performer=None, stop_writer=_stop):
    _require(bool(attempt) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in attempt), "attempt必须是安全名称")
    base = regression._output(base or control.BASE)
    selected = set(STEPS if steps is None else steps)
    _require(selected <= set(STEPS), "未知preflight步骤")
    folder = base/"m6"/("preflight_"+attempt)
    state_folder = folder/"state"
    state_folder.mkdir(parents=True, exist_ok=True)
    state_path, proofs_path = state_folder/"state.json", state_folder/"proofs.json"
    version = snapshotter()
    paths = [str(old.confined(path)) for path in (tpg_profiles or [])]
    if state_path.exists():
        state = old.read_json(state_path)
        _require(state["version"] == version, "已有preflight版本不同，保留现场并使用新attempt")
        if paths:
            _require(paths == state["tpg_profiles"], "已有TPG输入不能重置")
        else:
            paths = state["tpg_profiles"]
        for proof in state["steps"].values():
            _verify_saved(proof, version)
    else:
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "--attempt", attempt]
        if paths:
            command += ["--tpg-profile", *paths]
        state = {"attempt": attempt, "created_at": old.utc_now(), "version": version,
            "version_sha256": _digest(version), "tpg_profiles": paths, "steps": {}, "history": [],
            "current_step": None, "supervisor_pid": os.getpid(), "worker_pid": None,
            "startup_failures": 0, "global_stop": False, "requires_rollback": False,
            "resume_command": command, "resume_command_text": subprocess.list2cmdline(command)}
        old.atomic_json(folder/"code_version.json", version)
    state.update(supervisor_pid=os.getpid(), status="running", formal_ready=False, stop_reason=None)

    def save():
        state["updated_at"] = old.utc_now()
        old.atomic_json(state_path, state)
        old.atomic_json(proofs_path, {"version": version, "version_sha256": state["version_sha256"],
            "formal_ready": state.get("formal_ready", False), "steps": state["steps"],
            "required_steps": list(STEPS)})

    def commit(step, proof):
        proof = {**proof, "version": deepcopy(version), "completed_at": old.utc_now()}
        proof.setdefault("status", "passed" if proof.get("passed") else "failed")
        proof.setdefault("completed", True)
        state["steps"][step] = proof
        state["history"].append({"step": step, **deepcopy(proof)})
        save()
        print(json.dumps({"step": step, "status": proof["status"], "passed": proof.get("passed"),
            "reason": proof.get("reason"), "proofs": str(proofs_path)}, ensure_ascii=False), flush=True)

    for step, path in (reuse or {}).items():
        _require(step in STEPS, "未知reuse step")
        proof = _reuse(path, step, version)
        commit(step, proof)
    save()
    if performer is None:
        # A completed-proof resume must not bypass the environmental/history
        # guards merely because it has no new run_job to start.
        if (control.BASE/"global_stop.json").exists():
            state["global_stop"] = old.read_json(control.BASE/"global_stop.json")
        else:
            protection = control.audit()
            machine = control.health()
            state["entry_health"] = machine
            if not protection["passed"]:
                state["global_stop"] = protection
            elif not machine["passed"]:
                state.update(status="blocked", stop_reason="environment_guard", formal_ready=False)
                if machine.get("c_free_gib", 0) < 5:
                    state["global_stop"] = machine
                    stop_writer(machine)
                save()
                return state
    if state["global_stop"] or state["requires_rollback"]:
        state["status"] = "global_stop" if state["global_stop"] else "normal_gate_failed_requires_rollback"
        save()
        return state
    do_step = perform if performer is None else performer
    for step in STEPS:
        if step not in selected:
            continue
        previous = state["steps"].get(step)
        if previous and previous.get("completed"):
            if step in {"default", "original11", "gate_parallel"} and not previous.get("passed"):
                state["stop_reason"] = step+"_failed"
                break
            continue
        missing = [name for name in _dependencies(step) if not state["steps"].get(name, {}).get("passed")]
        if missing:
            commit(step, {"passed": False, "status": "blocked", "completed": False,
                "reason": "missing_current_version_prerequisites", "missing": missing, "evidence_files": {}})
            continue
        stem = _stem(attempt, step)
        job_name = stem+"_"+step if step in {"default", "original11"} else stem
        state.update(current_step=step, current_job=job_name,
            worker_pid=None, log=str(control.BASE/"jobs"/job_name/"stdout.log"),
            active_job_state=str(control.BASE/"jobs"/job_name/"state.json"))
        save()
        # Reading the owned controller's progress is lightweight and survives
        # interrupted supervisors; no second lock or simulator is introduced.
        done = threading.Event()

        def heartbeat():
            while not done.wait(1.):
                path = Path(state["active_job_state"])
                try:
                    if path.is_file():
                        state["worker_pid"] = old.read_json(path).get("pid")
                        save()
                except (OSError, ValueError):
                    pass

        watcher = threading.Thread(target=heartbeat, daemon=True) if performer is None and step != "tpg_profiles" else None
        if watcher:
            watcher.start()
        try:
            proof = do_step(step, folder/step, attempt, version, paths)
        except Exception as exc:
            proof = {"passed": False, "status": "blocked", "completed": False,
                     "reason": str(exc), "traceback": traceback.format_exc(), "evidence_files": {}}
            if performer is None and (control.BASE/"global_stop.json").exists():
                state["global_stop"] = old.read_json(control.BASE/"global_stop.json")
        finally:
            done.set()
            if watcher:
                watcher.join(timeout=3.)
        if snapshotter() != version:
            proof.update(passed=False, reason="code_or_tool_version_changed_during_preflight")
            state["stop_reason"] = "version_changed"
        commit(step, proof)
        if proof.get("startup_failure"):
            state["startup_failures"] += 1
        elif proof.get("connected"):
            state["startup_failures"] = 0
        if state["startup_failures"] >= 3:
            state["global_stop"] = {"at": old.utc_now(), "reason": "three consecutive simulator start/connect failures", "step": step}
            stop_writer(state["global_stop"])
        if not proof.get("passed") and step == "gate_parallel":
            state.update(requires_rollback=True, stop_reason="normal_gate_failed")
        elif not proof.get("passed") and step in {"default", "original11"}:
            state["stop_reason"] = step+"_failed"
        state["current_step"] = None
        save()
        if state["global_stop"] or state["requires_rollback"] or state["stop_reason"]:
            break
    state["formal_ready"] = not state["global_stop"] and not state["requires_rollback"] and all(
        state["steps"].get(step, {}).get("passed") is True for step in STEPS)
    state["status"] = ("ready_for_formal_review" if state["formal_ready"] else
        "global_stop" if state["global_stop"] else "requires_rollback" if state["requires_rollback"] else "not_ready")
    state["current_step"] = None
    save()
    return state


def run(attempt, **kwargs):
    _require(bool(attempt) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in attempt), "attempt必须是安全名称")
    base = regression._output(kwargs.get("base") or control.BASE)
    folder = base/"m6"/("preflight_"+attempt)
    folder.mkdir(parents=True, exist_ok=True)
    # This is a distinct supervisor lock, never the run_job/dispatch execution
    # lock. Two detached resumptions cannot overwrite one another's state.
    with old.execution_lock(folder/"supervisor.lock"):
        return _run(attempt, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="串行收集当前版本M6先决，不自动启动正式批次")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--tpg-profile", nargs=2, metavar=("CASE00_PROFILE", "CASE02_PROFILE"))
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--reuse", action="append", default=[], metavar="STEP=PROOF_JSON")
    args = parser.parse_args(argv)
    reuse = dict(item.split("=", 1) for item in args.reuse)
    result = run(args.attempt, steps=args.steps, tpg_profiles=args.tpg_profile, reuse=reuse)
    print(json.dumps({key: result.get(key) for key in ("status", "formal_ready", "stop_reason", "resume_command_text")}, ensure_ascii=False), flush=True)
    return 0 if result["formal_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
