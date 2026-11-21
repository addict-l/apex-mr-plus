"""Owned, bounded replay of retained motion contexts; never runs symbolic search.

Examples (run from the project root)::

    python -X utf8 tools/motion_v135_replay.py --source CASE_DIR \
        --mode action --record-index 0 --diagnostic-seconds 180
    python -X utf8 tools/motion_v135_replay.py --source CASE_DIR \
        --mode home --home-arm B --diagnostic-seconds 120

CASE_DIR must contain the old summary.json, workcell.json and
motion_backend.json. Historical artifacts are only read. A fresh workcell is
built on exclusive port 23210. This probes a retained reference context, not
the feasibility of an omitted prefix, a full schedule, TPG, or dynamics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overnight_batch as batch

DEFAULT_OUT = ROOT / "results/sim_v1_3/motion_v135/m1"


def _read(path):
    return batch.read_json(path)


def _write(path, value):
    batch.atomic_json(path, value)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_q(left, right):
    return (set(left) == set(right) and all(len(left[a]) == len(right[a])
        and max((abs(x-y) for x, y in zip(left[a], right[a])), default=0.) < 1e-8
        for a in left))


def _source(spec):
    source = batch.confined(spec["source"])
    summary = _read(source / "summary.json")
    backend = _read(source / "motion_backend.json")
    workcell = _read(source / "workcell.json")
    task = batch.confined(spec.get("task") or summary["task_file"])
    expected = summary.get("task_sha256")
    actual = _digest(task)
    if not expected or expected != actual:
        raise ValueError("旧任务 SHA 缺失或已变化，拒绝猜测回放输入")
    return source, summary, backend, workcell, task


def _contact_map(old_cfg, cfg):
    """Map recorded scene-local bearing handles to the rebuilt scene."""
    mapping = {}
    for name in ("baseplate", "table"):
        if isinstance(old_cfg.get(name), int) and isinstance(cfg.get(name), int):
            mapping[old_cfg[name]] = cfg[name]
    for field in ("pads", "blocks"):
        for name, old in old_cfg.get(field, {}).items():
            if isinstance(old, int) and name in cfg.get(field, {}):
                mapping[old] = cfg[field][name]
    for side, arm in old_cfg.get("arms", {}).items():
        for field in ("support_pad", "support_sensor", "palm"):
            old, new = arm.get(field), cfg.get("arms", {}).get(side, {}).get(field)
            if isinstance(old, int) and isinstance(new, int):
                mapping[old] = new
        for old, new in zip(arm.get("fingers", []), cfg.get("arms", {}).get(side, {}).get("fingers", [])):
            mapping[old] = new
    return mapping


def _prefix(records, selected):
    """Only retain an actually recorded contiguous successful joint chain."""
    first = selected
    target = records[selected]["start_q"]
    while first > 0:
        before = records[first-1]
        if (before.get("status") != "VALID" or not before.get("tasks")
                or not _same_q(before.get("end_q", {}), target)):
            break
        target = before["start_q"]
        first -= 1
    tasks = [deepcopy(t) for r in records[first:selected] for t in r.get("tasks", [])]
    return tasks, {"first_record": first, "selected_record": selected,
        "retained_records": selected-first, "joint_chain_verified": True,
        "full_historical_prefix_claimed": False,
        "scope": "仅使用现存连续成功记录；未保存的更早前缀、动态状态或任务依赖不补造"}


def _home_context(backend, old_cfg, arm):
    certs = list(backend.get("path_certificates", {}).items())
    direct = [(key, row) for key, row in certs
              if row.get("context", {}).get("metadata", {}).get("final_home")
              and row["context"].get("arm") == arm]
    if direct:
        key, row = direct[-1]
        return deepcopy(row["context"]), {"source_certificate": key,
            "derivation": "记录的该回位请求上下文"}
    if arm == "B":
        previous = [(key, row) for key, row in certs
            if row.get("context", {}).get("metadata", {}).get("final_home")
            and row["context"].get("arm") == "A" and row.get("path")]
        if previous:
            key, row = previous[-1]
            context = deepcopy(row["context"])
            context.update(arm="B", target_q=deepcopy(old_cfg["arms"]["B"]["home_q"]))
            context["start_q"]["A"] = deepcopy(row["path"][-1])
            context["held"] = deepcopy(context.get("payloads", {}).get("B"))
            return context, {"source_certificate": key,
                "derivation": "A→B既有回位顺序；A用已验证home_A路径终点，B保留该上下文起点",
                "full_historical_prefix_claimed": False}
    raise ValueError("没有足够的已保存回位上下文；不由符号状态猜测实际关节角")


def _worker(spec):
    from skillnet.capability.oracle import Status
    # The parent releases this only after its native Job Object owns us.
    signal = Path(spec["start_signal"])
    while not signal.exists():
        if time.monotonic() >= spec["deadline_monotonic"]:
            return 124
        time.sleep(.02)
    from apexmr.motion import MotionPlanner
    from apexmr.sim_connection import Simulator
    from simbridge.backend import CoppeliaBackend
    from simbridge.compiler import CompileFailure
    from simbridge.frames import Frames
    from simbridge.primitives import MotionRequest
    from simbridge.workcell import build
    from skillnet.core.engine import Engine
    from skillnet.core.logic import parse_fact
    from skillnet.domains.blocks import load_problem

    output = batch.confined(spec["output"])
    result = {"status": "running", "mode": spec["mode"], "passed": False,
        "scope": "保留几何上下文的参考运动探针；不证明缺失前缀、完整时间表或动力学执行",
        "deadline_monotonic": spec["deadline_monotonic"], "phase_seconds": {}}
    owner = compiler = None
    phase, phase_start = "source", time.monotonic()

    def enter(name):
        nonlocal phase, phase_start
        now = time.monotonic()
        result["phase_seconds"][phase] = now-phase_start
        phase, phase_start = name, now
        result.update(phase=phase, elapsed_seconds=now-spec["started_monotonic"])
        _write(output / "progress.json", result)
        if now >= spec["deadline_monotonic"]:
            raise TimeoutError("诊断达到总墙钟截止时间")

    def append(name, row):
        with (output / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()

    try:
        source, summary, old_backend, workcell, task = _source(spec)
        result.update(task_file=str(task), task_sha256=_digest(task), source=str(source),
            source_sha256={name: _digest(source/name)
                for name in ("summary.json", "motion_backend.json", "workcell.json")},
            code_sha256={name: _digest(ROOT/name) for name in (
                "tools/motion_v135_replay.py", "simbridge/compiler.py", "simbridge/backend.py",
                "simbridge/primitives.py", "simbridge/frames.py", "simbridge/workcell.py",
                "apexmr/motion.py", "apexmr/motion_backend.lua")})
        enter("simulator_start")
        # Simulator.root is used for log placement only. The workcell builder,
        # geometry handles, task and backend code still use the original paths.
        owner = Simulator(output/"instance", port=23210, timeout=30.)
        owner.start(headless=True)
        result["simulator_pid"] = owner.process.pid
        enter("simulator_connect")
        client, sim = owner.connect()
        enter("workcell")
        cfg = build(sim, client, task, owner.install, output)
        old_cfg = workcell.get("cfg", {})
        if not old_cfg.get("arms"):
            raise ValueError("旧workcell缺少cfg.arms，不猜测历史home定义")
        # Replaying retained q uses the original fixed home definition; the
        # diagnostic builder's newly solved home is not substituted silently.
        for arm in ("A", "B"):
            cfg["arms"][arm]["home_q"] = deepcopy(old_cfg["arms"][arm]["home_q"])
            cfg["free_q"][arm] = deepcopy(old_cfg["arms"][arm]["home_q"])
        if "ik_seeds" in old_cfg:
            cfg["ik_seeds"] = deepcopy(old_cfg["ik_seeds"])
        cfg["profile"] = "skillnet_v1_3"
        _write(output/"replay_cfg.json", cfg)
        with MotionPlanner(sim, client, cfg) as planner:
            backend = CoppeliaBackend(planner, cfg, Frames(task, arm_config=cfg["arms"]))
            compiler = backend.compiler
            compiler.deadline = spec["deadline_monotonic"]
            original_call, original_emit = compiler._call, compiler._emit

            def call(kind, request, function, *args, **kwargs):
                row = {"query": kind, "request": deepcopy(vars(request)),
                    "call_args": deepcopy(args), "call_kwargs": deepcopy(kwargs),
                    "start_q": deepcopy(compiler.q), "held": deepcopy(compiler.held),
                    "placed": deepcopy(compiler.placed), "last_arm": deepcopy(compiler._last_arm),
                    "emitted_task_ids": [t["id"] for t in compiler.tasks],
                    "elapsed_seconds": time.monotonic()-spec["started_monotonic"]}
                _write(output/"pending_query.json", row)
                began = time.monotonic()
                try:
                    answer = original_call(kind, request, function, *args, **kwargs)
                    row["answer"] = deepcopy(answer)
                    return answer
                except BaseException as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    row["seconds"] = time.monotonic()-began
                    append("queries.jsonl", row)

            def emit(*args, **kwargs):
                task_row = original_emit(*args, **kwargs)
                append("emitted_tasks.jsonl", task_row)
                return task_row

            compiler._call, compiler._emit = call, emit
            if spec["mode"] == "action":
                records = old_backend.get("validations", [])
                selected = spec.get("record_index")
                if selected is None:
                    selected = next((i for i in range(len(records)-1, -1, -1)
                        if records[i].get("status") != "VALID"), None)
                if selected is None or not 0 <= selected < len(records):
                    raise ValueError("找不到所选旧动作验证记录")
                record = records[selected]
                for field in ("start_q", "held", "placed", "state", "action"):
                    if field not in record:
                        raise ValueError("旧动作记录缺少实际起点字段："+field)
                result.update(record_index=selected, original_status=record["status"],
                    original_detail=record.get("detail"), source_action=record["action"])
                tasks, prefix_receipt = _prefix(records, selected)
                compiler.q = deepcopy(record["start_q"])
                compiler.held, compiler.placed = deepcopy(record["held"]), deepcopy(record["placed"])
                compiler.tasks = tasks
                compiler._last_arm = {t["arm"]: t["id"] for t in tasks}
                # The actual placed/held snapshot is authoritative. This mark
                # prevents _sync_state from replacing it with a symbolic pose.
                compiler._touched = set(compiler.frames.blocks)
                state = frozenset(parse_fact(text) for text in record["state"])
                compiler._symbolic_context = tuple(sorted(map(str, state)))
                action = Engine(load_problem(task)).parse_action(record["action"])
                backend.seed_overrides[action.key] = int(record.get("ik_seed_limit", 8))
                result["prefix_restore"] = prefix_receipt
                _write(output/"prefix_tasks.json", tasks)
                _write(output/"restored_context.json", {"source_record": record,
                    "prefix_restore": prefix_receipt, "touch_policy": "保留源几何快照，不补造缺失前缀",
                    "restored_q": compiler.q, "restored_held": compiler.held,
                    "restored_placed": compiler.placed, "seed_limit": record.get("ik_seed_limit", 8)})
                enter("compile_action")
                if spec.get("use_backend"):
                    hint_log = Path(spec.get("qualification_log") or task.parent/"qualify_log.json")
                    if hint_log.is_file():
                        hint_log = batch.confined(hint_log)
                        hint_data = _read(hint_log)
                        if hint_data.get("task") != load_problem(task).name:
                            raise ValueError("资格提示与回放任务身份不一致")
                        imported = backend.import_qualification_hints(hint_data, source=str(hint_log))
                        _write(output/"qualification_hint_import.json", {**imported,
                            "source_sha256": _digest(hint_log), "scope": "point IK suggestions; revalidation required"})
                    backend._explicit_reference = True
                    validation = backend.validate(action, state)
                    if validation.status == Status.UNKNOWN and backend.validation_context.get("retryable"):
                        validation = backend.validate_escalated(action, state)
                    _write(output/"motion_backend.json", backend.audit())
                    result.update(status="reference_action_passed" if validation.status == Status.VALID else "reproduced_failure",
                        passed=validation.status == Status.VALID, motion_status=validation.status.value,
                        reason=validation.detail, validation_context=deepcopy(backend.validation_context),
                        new_task_count=len(compiler.tasks)-len(tasks))
                else:
                    new_tasks = compiler.compile_action(action, state, step_id="replay")
                    result.update(status="reference_action_passed", passed=True, new_task_count=len(new_tasks))
            else:
                arm = spec["home_arm"]
                context, derivation = _home_context(old_backend, old_cfg, arm)
                compiler.q = deepcopy(context["start_q"])
                compiler.held = deepcopy(context.get("payloads", {}))
                compiler.placed = deepcopy(context["placed"])
                compiler._symbolic_context = tuple(context.get("state", ()))
                mapping = _contact_map(old_cfg, cfg)
                contacts = [[mapping[value] if isinstance(value, int) and value in mapping else value
                             for value in pair] for pair in context.get("contacts", [])]
                if any(isinstance(value, int) and value not in mapping.values()
                       for pair in contacts for value in pair):
                    raise ValueError("回位上下文包含无法映射的旧场景接触句柄")
                request = MotionRequest("home_"+arm, arm, "transit",
                    target_q=deepcopy(old_cfg["arms"][arm]["home_q"]),
                    held=deepcopy(compiler.held.get(arm)), payloads=deepcopy(compiler.held),
                    placed=deepcopy(compiler.placed), contacts=contacts,
                    metadata={"source_action": "home", "final_home": True})
                _write(output/"restored_context.json", {"context": context,
                    "derivation": derivation, "request": vars(request),
                    "contact_handle_mapping": mapping, "actual_start_q": compiler.q})
                enter("home_goal_collision")
                result["home_goal_validation"] = compiler._call("goal_collision", request,
                    planner.validate_joint, arm, request.target_q, payload=request.held,
                    placed=request.placed, options=compiler._options(request))
                _write(output/"home_goal_validation.json", result["home_goal_validation"])
                enter("home_path")
                compiler._emit(request, {}, "home_replay", [])
                result.update(status="reference_home_passed", passed=True)
    except CompileFailure as exc:
        result.update(status="reproduced_failure", motion_status=exc.status.value,
            failed_segment=exc.segment, reason=exc.detail, failure_evidence=deepcopy(exc.evidence))
    except BaseException as exc:
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
            reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["phase_seconds"][phase] = time.monotonic()-phase_start
        if compiler is not None:
            _write(output/"compile_evidence.json", compiler.evidence)
            _write(output/"tasks.json", compiler.tasks)
        if owner is not None:
            try:
                owner.close()
            except Exception as exc:
                result["close_error"] = f"{type(exc).__name__}: {exc}"
            result["simulator_closed"] = owner.process is not None and owner.process.poll() is not None
            result["simulator_log"] = str(owner.log_path)
        result["elapsed_seconds"] = time.monotonic()-spec["started_monotonic"]
        _write(output/"result.json", result)
    return 0 if result["status"] in {"reference_action_passed", "reference_home_passed", "reproduced_failure"} else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="回放旧几何上下文，不运行符号搜索或动力学")
    parser.add_argument("--source")
    parser.add_argument("--task")
    parser.add_argument("--mode", choices=("action", "home"), default="action")
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--home-arm", choices=("A", "B"), default="B")
    parser.add_argument("--use-backend", action="store_true",
                        help="动作回放包含当前后端有限候选与历史点IK提示重验")
    parser.add_argument("--qualification-log")
    parser.add_argument("--diagnostic-seconds", type=float, default=180.)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        return _worker(_read(batch.confined(args.worker_spec)))
    if not args.source:
        parser.error("需要 --source CASE_DIR")
    if not math.isfinite(args.diagnostic_seconds) or not 0 < args.diagnostic_seconds <= 1200:
        parser.error("诊断总预算必须在 (0,1200] 秒")
    if args.record_index is not None and args.record_index < 0:
        parser.error("record-index必须非负")
    started = time.monotonic()
    spec = {"source": str(batch.confined(args.source)), "task": args.task,
        "mode": args.mode, "record_index": args.record_index, "home_arm": args.home_arm,
        "use_backend": args.use_backend, "qualification_log": args.qualification_log,
        "started_monotonic": started, "deadline_monotonic": started+args.diagnostic_seconds,
        "diagnostic_seconds": args.diagnostic_seconds}
    _source(spec)  # Validate historical source/task identity before starting a process.
    output = batch.confined(args.out)/f"{args.mode}_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    spec.update(output=str(output), start_signal=str(output/"worker.assigned"))
    _write(output/"spec.json", spec)
    preflight = batch.preflight(batch.ArchiveGuard())
    _write(output/"preflight.json", preflight)
    if not preflight["ok"]:
        _write(output/"supervisor.json", {"status": "blocked", "preflight": preflight})
        print(json.dumps({"status": "blocked", "output": str(output), "reason": preflight["reason"]}, ensure_ascii=False))
        return 2
    if time.monotonic() >= spec["deadline_monotonic"]:
        _write(output/"supervisor.json", {"status": "timeout", "reason": "preflight_exhausted_budget"})
        print(json.dumps({"status": "timeout", "output": str(output)}, ensure_ascii=False))
        return 124
    process = None
    status = "running"
    try:
        process = batch.OwnedProcess([sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
            "--worker-spec", str(output/"spec.json")], output/"worker.log", output/"worker.assigned")
        _write(output/"supervisor.json", {"status": status, "pid": process.pid,
            "birth": process.birth, "deadline_monotonic": spec["deadline_monotonic"]})
        process.release()
        while process.process.poll() is None:
            if time.monotonic() >= spec["deadline_monotonic"]:
                process.terminate()
                status = "timeout"
                break
            time.sleep(.2)
        if status == "running":
            status = "completed" if process.process.returncode == 0 else "worker_failed"
    finally:
        known, cleanup_errors = [], []
        if process is not None:
            try:
                known = process.refresh()
            except Exception as exc:
                cleanup_errors.append("refresh: "+str(exc))
            finally:
                try:
                    process.close()
                except Exception as exc:
                    cleanup_errors.append("close: "+str(exc))
        from tools.overnight_regression import health
        cleanup = health()
        _write(output/"supervisor.json", {"status": status, "elapsed_seconds": time.monotonic()-started,
            "owned_processes": known, "cleanup": cleanup, "cleanup_errors": cleanup_errors,
            "result_exists": (output/"result.json").is_file()})
    print(json.dumps({"status": status, "output": str(output), "cleanup": cleanup["passed"]}, ensure_ascii=False))
    return 124 if status == "timeout" else (0 if status == "completed" and cleanup["passed"] and not cleanup_errors else 2)


if __name__ == "__main__":
    raise SystemExit(main())
