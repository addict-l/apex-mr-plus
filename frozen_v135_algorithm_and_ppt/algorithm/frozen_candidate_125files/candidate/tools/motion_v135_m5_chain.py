"""Resumable serial short-profile comparison, explicitly launched after a gate.

Import and rebuild never start experiments. A 120-second diagnostic limit is
not a replacement for the unchanged 1200-second formal task budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import motion_v135_control as control
from tools.overnight_regression import health

BASE = control.BASE
PROFILE = ROOT/"tools/motion_v135_tpg_profile.py"
STEPS = (("00", "optimized"), ("00", "legacy"), ("02", "optimized"), ("02", "legacy"))


def profile_command(case, variant, directory, complete_optimized=False):
    full = complete_optimized and variant == "optimized"
    command = [sys.executable, "-X", "utf8", str(PROFILE), "--case", case,
               "--seconds", "1180" if full else "120", "--out", str(directory)]
    if complete_optimized:
        command += ["--until", "rollout"]
    if full:
        command.append("--fast-idle")
    if variant == "legacy":
        command.append("--legacy-sample-reset")
    return command, 1200. if full else 220.


def read(path):
    return old.read_json(path) if Path(path).is_file() else None


def root_for(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run-id必须为简单安全标识")
    return BASE/"m5"/run_id


def finite_ratio(numerator, denominator):
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) and denominator > 0:
        value = numerator/denominator
        return value if math.isfinite(value) else None
    return None


def saved_version(result):
    value = result.get("code_version")
    return value if isinstance(value, dict) else read(value) if value else None


def guard(root):
    stop = read(BASE/"global_stop.json")
    if stop:
        raise RuntimeError("GLOBAL_STOP: 已存在整体停止记录；本工具不清除它")
    checked = control.audit()
    if not checked["passed"]:
        raise RuntimeError("GLOBAL_STOP: 历史数据/冻结规划核心/归档校验失败")
    machine = health()
    if machine["c_free_gib"] < 5:
        old.atomic_json(BASE/"global_stop.json", {"at": old.utc_now(), "reason": "C盘不足5GiB", "health": machine})
        raise RuntimeError("GLOBAL_STOP: C盘不足5GiB")
    if not machine["passed"]:
        raise RuntimeError("当前有实例或23210占用；保留现场，禁止并发")
    return checked, machine


def checkpoint(root, state, **changes):
    state.update(changes, updated_at=old.utc_now())
    old.atomic_json(root/"state.json", state)


def wait_gate(root, state, gate_job, wait_seconds):
    gate_dir = BASE/"jobs"/gate_job
    began = time.monotonic()
    while True:
        if (BASE/"global_stop.json").exists():
            raise RuntimeError("GLOBAL_STOP: 等待普通取放门期间已整体停止")
        result = read(gate_dir/"result.json")
        if result is not None:
            if result.get("status") != "passed" or not result.get("cleanup", {}).get("passed"):
                raise RuntimeError("普通取放回归门未通过或清理未通过；不启动profile")
            if result.get("code_unchanged_during_job") is not True or saved_version(result) != control.version():
                raise RuntimeError("普通取放门与当前代码版本不一致")
            return result
        active = read(gate_dir/"state.json")
        if active and active.get("status") == "running" and active.get("birth"):
            if old.process_birth(active["pid"]) != active["birth"]:
                raise RuntimeError("普通取放门进程已结束但没有提交结果；保留断点")
        if time.monotonic()-began >= wait_seconds:
            raise TimeoutError("等待普通取放门结果超过诊断等待上限")
        checkpoint(root, state, status="waiting_gate", waiting_gate=str(gate_dir/"result.json"),
                   gate_wait_seconds=time.monotonic()-began)
        time.sleep(2.)


def normalized_replay(profile_path, profile):
    replayed = Path(profile_path).parent/"tasks_replayed.json"
    source = Path(profile["source_reference"])/"tasks.json"
    if not replayed.is_file() or not source.is_file():
        return {"passed": False, "reason": "missing replayed or source tasks"}
    tasks = old.read_json(replayed)
    mapping = profile.get("contact_handle_remapping", {})
    reverse = {row["new_handle"]: int(previous) for previous, row in mapping.items()}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "contact_exclusions":
                    for pair in item:
                        for index, endpoint in enumerate(pair):
                            if isinstance(endpoint, int) and not isinstance(endpoint, bool):
                                if endpoint not in reverse:
                                    raise ValueError("重放接触handle缺少原始别名映射")
                                pair[index] = reverse[endpoint]
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(tasks)
    return {"passed": tasks == old.read_json(source), "replayed_tasks": str(replayed),
            "source_tasks": str(source), "basis": "Only numeric contact endpoints remapped by original object alias; every other field compared exactly"}


def row_from(label, case, variant, directory, result, *, recovered=False):
    profile_path = directory/"profile.json"
    profile = read(profile_path)
    progress = read(directory/"progress.json")
    row = {"label": label, "case": case, "variant": variant, "output": str(directory),
           "control_result": result, "control_status": result.get("status"),
           "profile_path": str(profile_path) if profile else None,
           "profile_sha256": old.sha256(profile_path) if profile else None,
           "profile_status": profile.get("status") if profile else "missing",
           "recovered_from_profile_commit": recovered,
           "cleanup": result.get("cleanup"),
           "diagnostic_limit_seconds": (profile or {}).get("budget_seconds", 120.),
           "hard_limit_seconds": result.get("seconds", 220.)}
    data = profile or progress or {}
    row["last_stage"] = data.get("stage")
    row["startup_failed"] = (data.get("stage") in {"simulator_start", "simulator_connect"}
        and (data.get("status") in {"failed", "timeout"} or result.get("status") in {"failed", "timeout", "stopped"}))
    if profile:
        row["identity"] = {key: profile.get(key) for key in ("task", "task_sha256", "source_tasks_sha256", "source_workcell_sha256",
            "joint_sample_step_rad", "rollout_step_s", "skip_ordered_pairs", "until", "code_sha256")}
        row["alias_remapping"] = {key: value["alias"] for key, value in profile.get("contact_handle_remapping", {}).items()}
        row["reference_replay_audit"] = normalized_replay(profile_path, profile)
        geometry, rpc = profile.get("geometry", {}), profile.get("rpc", {})
        server = profile.get("lua_sample_batch_statistics", {})
        row.update(graph_built=bool(profile.get("graph_counts")),
                   graph_completed_within_soft_limit=profile.get("status") == "completed" and bool(profile.get("graph_counts")),
                   graph_counts=profile.get("graph_counts"), structural_graph_counts=profile.get("structural_graph_counts"),
                   stage_seconds=profile.get("stage_seconds"), geometry=geometry, rpc=rpc,
                   completed_mesh_states_per_oracle_second=finite_ratio(geometry.get("mesh_state_queries"), geometry.get("wall_s")),
                   rpc_requested_samples_per_tpg_second=finite_ratio(rpc.get("samples"), profile.get("stage_seconds", {}).get("tpg_build")),
                   server_completed_samples_per_batch_second=finite_ratio(server.get("samples_checked"), server.get("batch_seconds")),
                   codec=profile.get("codec"), codec_restored=profile.get("codec_restored"),
                   lua_sample_batch_statistics=server,
                   server_statistics_snapshot_age_s=profile.get("lua_statistics_snapshot_age_s"),
                   lua_restore_fraction=finite_ratio(server.get("restore_seconds"), server.get("batch_seconds")),
                   lua_context_reuse_fraction=finite_ratio(server.get("world_reuse_samples"), server.get("samples_checked")),
                   native_cleanup_passed=profile.get("cleanup_passed"))
    return row


def common_prefix(first, second):
    paths = [Path(row["output"])/"geometry/geometry_pair_checks.jsonl" for row in (first, second)]
    def load(path):
        rows = []
        if path.is_file():
            for line in path.read_bytes().splitlines():
                try:
                    rows.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    break
        return rows
    a, b = map(load, paths)
    count = min(len(a), len(b))
    mismatches, details = [], []
    for index in range(count):
        left, right = a[index], b[index]
        if (left.get("a"), left.get("b"), left.get("conflict")) != (right.get("a"), right.get("b"), right.get("conflict")):
            mismatches.append(index)
        fields = ("problem", "qA", "qB", "payloads", "placed", "grid_shape", "pair_states_total", "pair_states_considered",
                  "new_mesh_queries", "cache_hits", "checked_all_pair_states")
        if any(left.get("evidence", {}).get(key) != right.get("evidence", {}).get(key) for key in fields):
            details.append(index)
    return {"journal_paths": list(map(str, paths)), "lengths": [len(a), len(b)], "common_prefix_records": count,
            "pair_or_conflict_mismatches": len(mismatches), "first_pair_or_conflict_mismatches": mismatches[:10],
            "detail_mismatches": len(details), "first_detail_mismatches": details[:10],
            "scope": "Observed common prefix only. Different 120s prefix lengths do not establish full-graph equivalence; DLL/scalar equivalence evidence is separate."}


def rebuild(run_id):
    root = root_for(run_id)
    state = read(root/"state.json") or {"steps": {}, "status": "not_started"}
    comparisons = {}
    for case in ("00", "02"):
        fast, legacy = (state.get("steps", {}).get(case+"_"+variant) for variant in ("optimized", "legacy"))
        if not fast or not legacy:
            comparisons[case] = {"status": "pending"}
            continue
        identity = bool(fast.get("identity")) and fast.get("identity") == legacy.get("identity")
        aliases = fast.get("alias_remapping") == legacy.get("alias_remapping")
        replay = all(row.get("reference_replay_audit", {}).get("passed") is True for row in (fast, legacy))
        comparisons[case] = {"status": "collected" if identity and aliases and replay else "invalid_or_missing_inputs",
            "same_source_sha_precision_and_code": identity, "same_contact_alias_mapping": aliases,
            "only_alias_handles_changed_in_replay": replay,
            "optimized": fast, "legacy": legacy,
            "observed_completed_mesh_throughput_ratio": finite_ratio(fast.get("completed_mesh_states_per_oracle_second"), legacy.get("completed_mesh_states_per_oracle_second")),
            "throughput_scope": "Observed prefixes may have different work/mode distributions; this ratio is not a controlled whole-task wall speedup",
            "collision_result_scope": common_prefix(fast, legacy)}
    report = {"at": old.utc_now(), "run_id": run_id, "chain_status": state.get("status"),
              "complete_optimized": state.get("complete_optimized", False),
              "completed_diagnostics": len(state.get("steps", {})), "requested_diagnostics": 4,
              "comparisons": comparisons, "formal_acceptance_claimed": False,
              "rollout_and_dynamics_not_executed": not any("rollout" in row.get("stage_seconds", {})
                  for row in state.get("steps", {}).values()),
              "dynamics_not_executed": True,
              "reference_randomness": "The exact frozen joint reference tasks are replayed; no new IK/RRT sampling seed is supplied or changed",
              "equivalence_unit_evidence": "out/motion_v135_m5_focused_a1.log (same-sample scalar/batch full-mesh API tests)"}
    old.atomic_json(root/"comparison.json", report)
    lines = ["# M5 同输入 TPG 诊断对照", "", "更新（UTC）："+report["at"], "",
             "各变体预算/阶段范围见JSON：完整优化模式在原1200秒硬界内尝试图与rollout；legacy保留120秒诊断。均未运行动力学，不计为正式场景通过。", "",
             "|案例/变体|profile结论|图在预算内完成|mesh状态查询|RPC次数|mesh状态/geometry秒|Lua恢复比例|清理|",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for label, row in state.get("steps", {}).items():
        lines.append(f"|{label}|{row['profile_status']}|{row.get('graph_completed_within_soft_limit')}|{row.get('geometry',{}).get('mesh_state_queries')}|{row.get('rpc',{}).get('rpc_calls')}|{row.get('completed_mesh_states_per_oracle_second')}|{row.get('lua_restore_fraction')}|{row.get('cleanup',{}).get('passed')}|")
    for case, row in comparisons.items():
        lines += ["", "## 案例 "+case, "", "状态："+row["status"]]
        if row["status"] != "pending":
            lines += ["", "任务/原轨迹/原workcell SHA、采样精度、代码一致："+str(row["same_source_sha_precision_and_code"])+
                      "；别名接触映射一致："+str(row["same_contact_alias_mapping"])+
                      "；重放任务仅接触handle重映射："+str(row["only_alias_handles_changed_in_replay"])+"。",
                      "", "共同前缀检查：`"+json.dumps(row["collision_result_scope"], ensure_ascii=False)+"`。",
                      "", "短时前缀长度不同，吞吐比不构成完整图或整任务等价/加速证明。DLL逐样本完整mesh对照另行保存。"]
    lines += ["", "完整Lua分阶段、codec、RPC和快照年龄见 comparison.json，各尝试原始目录保留。"]
    old.atomic_text(root/"comparison.md", "\n".join(lines)+"\n")
    return report


def run(run_id="profiles_a1", gate_job="m5_parallel_gate_a1", wait_seconds=1800., *, complete_optimized=False):
    root = root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root/"state.json"
    state = read(path) or {"schema_version": 1, "run_id": run_id, "steps": {}, "startup_failure_streak": 0,
                           "gate_job": gate_job, "created_at": old.utc_now()}
    if state["gate_job"] != gate_job:
        raise ValueError("既有chain不得改换普通门控来源")
    if path.exists() and state.get("complete_optimized", False) != bool(complete_optimized):
        raise ValueError("既有chain不得修改阶段/预算模式；使用新ID")
    state["complete_optimized"] = bool(complete_optimized)
    with old.execution_lock(root/"chain.lock"):
        try:
            gate = wait_gate(root, state, gate_job, wait_seconds)
            guard(root)
            current = control.version()
            profile_sha = old.sha256(PROFILE)
            if state.get("code_sha256") is not None and state["code_sha256"] != current:
                raise ValueError("诊断chain不可混代码版本；使用新的run-id")
            if state.get("profile_tool_sha256") is not None and state["profile_tool_sha256"] != profile_sha:
                raise ValueError("诊断工具版本改变；保留当前chain，使用新的run-id")
            checkpoint(root, state, status="running", code_sha256=current,
                       profile_tool_sha256=profile_sha, gate_result=str(BASE/"jobs"/gate_job/"result.json"),
                       gate_code_version_sha256=old.fingerprint(saved_version(gate)))
            for case, variant in STEPS:
                label = case+"_"+variant
                if label in state["steps"]:
                    continue
                guard(root)
                if control.version() != current or old.sha256(PROFILE) != profile_sha:
                    raise ValueError("执行期间代码或profile工具改变")
                trial = 0
                while True:
                    suffix = "" if trial == 0 else "_resume"+str(trial)
                    job_name = "m5_profile_"+run_id+"_"+label+suffix
                    directory = root/label if trial == 0 else root/label/("resume_"+str(trial))
                    job_dir = BASE/"jobs"/job_name
                    result = read(job_dir/"result.json")
                    previous = read(job_dir/"state.json")
                    if result is not None:
                        if saved_version(result) != current:
                            raise ValueError("既有诊断job不属于当前代码")
                        break
                    if previous and previous.get("birth") and old.process_birth(previous["pid"]) == previous["birth"]:
                        checkpoint(root, state, status="waiting_existing_job", active_label=label,
                                   active_job=job_name, active_pid=previous["pid"])
                        time.sleep(2.)
                        continue
                    native = read(directory/"profile.json")
                    if native and native.get("cleanup_passed") and previous and saved_version(previous) == current:
                        guard(root)
                        result = {**previous, "status": "interrupted_after_profile_commit", "cleanup": health(),
                                  "code_unchanged_during_job": None, "recovered_native_profile": True,
                                  "recovered_code_start_and_current_match": True,
                                  "recovery_scope": "Native profile committed; the interrupted supervisor's final version assertion is unavailable"}
                        old.atomic_json(root/label/("recovery_receipt"+suffix+".json"), {
                            "at": old.utc_now(), "reason": "Native profile committed before supervisor result; no rerun",
                            "result": result, "profile": str(directory/"profile.json")})
                        break
                    if (directory/"progress.json").exists() or previous:
                        old.atomic_json(root/label/("interruption_receipt"+suffix+".json"), {
                            "at": old.utc_now(), "job_state": previous, "output": str(directory),
                            "reason": "Uncommitted diagnostic attempt retained; next distinct output"})
                        trial += 1
                        if trial >= 3:
                            raise RuntimeError("同一诊断连续三次中断；保留现场，不重复无证据尝试")
                        continue
                    break
                if result is None:
                    command, hard_limit = profile_command(case, variant, directory, complete_optimized)
                    command += ["--controlled-job-result", str(BASE/"jobs"/job_name/"result.json")]
                    checkpoint(root, state, status="running", active_label=label, active_job=job_name, command=command)
                    result = control.run_job(job_name, command, hard_limit, live=True)
                if saved_version(result) != current or not result.get("cleanup", {}).get("passed"):
                    raise RuntimeError("诊断代码版本或实际清理校验失败")
                row = row_from(label, case, variant, directory, result, recovered=result.get("recovered_native_profile", False))
                state["steps"][label] = row
                state["startup_failure_streak"] = state.get("startup_failure_streak", 0)+1 if row["startup_failed"] else 0
                checkpoint(root, state, status="running", active_label=None, active_job=None)
                rebuild(run_id)
                control.record("M5_profile_"+run_id+"_"+label, row["profile_status"],
                               evidence=row["profile_path"] or str(directory), control_status=result["status"])
                if state["startup_failure_streak"] >= 3:
                    old.atomic_json(BASE/"global_stop.json", {"at": old.utc_now(), "reason": "连续三次仿真器启动/连接失败"})
                    raise RuntimeError("GLOBAL_STOP: 连续三次仿真器启动/连接失败")
            checkpoint(root, state, status="finished", finished_at=old.utc_now())
        except BaseException as exc:
            checkpoint(root, state, status="global_stop" if (BASE/"global_stop.json").exists() else "interrupted",
                       reason=f"{type(exc).__name__}: {exc}")
        finally:
            rebuild(run_id)
            control.record("M5_chain_"+run_id, state["status"], evidence=str(path))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description="普通取放门后串行四项120秒TPG诊断；可断点续跑")
    parser.add_argument("command", choices=("run", "rebuild"))
    parser.add_argument("--run-id", default="profiles_a1")
    parser.add_argument("--gate-job", default="m5_parallel_gate_a1")
    parser.add_argument("--wait-seconds", type=float, default=1800.)
    parser.add_argument("--complete-optimized", action="store_true", help="优化变体在原1200秒内完整图/rollout，legacy仍为短诊断")
    args = parser.parse_args(argv)
    result = (rebuild(args.run_id) if args.command == "rebuild" else
              run(args.run_id, args.gate_job, args.wait_seconds, complete_optimized=args.complete_optimized))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status", result.get("chain_status")) in {"finished", "not_started"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
