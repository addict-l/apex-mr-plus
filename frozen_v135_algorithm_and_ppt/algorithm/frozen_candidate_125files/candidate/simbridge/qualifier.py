"""Bounded real IK reach-cache generation, with durable partial snapshots.

Importing this module starts no simulator. qualify_task is the single-worker
API; the CLI supervisor additionally supplies a native hard 60-second limit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import tomllib
import uuid

from skillnet.capability.oracle import Status
from skillnet.core.logic import Fact
from tools.overnight_batch import atomic_json, atomic_text
from .taskio import task_data_to_toml


ROOT = Path(__file__).resolve().parents[1]
VERSION_ROOT = ROOT / "results/sim_v1_3/v134"
DEFAULT_OUT = VERSION_ROOT / "qualified"
PORT = 23210
MAX_SECONDS = 60.


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _facts(problem):
    return [Fact("reach", (arm, place)) for arm in problem.hooks.arms
            for place in (*problem.hooks.locations, *problem.hooks.slots)]


def qualified_task_data(source, progress):
    """Project only committed IK outcomes; never retain untested prior claims."""
    result = deepcopy(source)
    table = {"default": "UNKNOWN", **{status.lower(): {arm: [] for arm in source["arms"]}
                                     for status in ("VALID", "INVALID", "UNKNOWN")}}
    for cell in progress["updates"]:
        table[cell["status"].lower()][cell["arm"]].append(cell["place"])
    result.setdefault("capabilities", {})["reach"] = table
    if progress["checked"]:
        result["capability_source"] = "ik_qualified"
    else:
        result.pop("capability_source", None)
    return result


def _slot_depth(problem, place):
    rows = problem.hooks.slots
    if not isinstance(rows, dict):
        rows = {row["id"]: row for row in getattr(problem, "data", {}).get("slot", [])}

    def depth(name, visiting):
        if name not in rows:
            return 0
        if name in visiting:
            raise ValueError("槽位below成环")
        return 1+max((depth(lower, visiting | {name}) for lower in rows[name].get("below", [])), default=0)
    return depth(place, set())


def _escalate(problem, backend, fact, original, progress, deadline, persist):
    """Apply the bounded pose-set policy, never treat interruption as proof."""
    row = {"fact": str(fact), "arm": fact.args[0], "place": fact.args[1],
           "slot_depth": _slot_depth(problem, fact.args[1]), "initial": deepcopy(original),
           "attempts": [], "status": "UNKNOWN", "accepted_pose_mode": None}
    progress["escalation"].append(row)
    modes = [None]+([30, 45, 60] if row["slot_depth"] >= 4 else [])
    row["required_pose_modes"] = ["catalog" if mode is None else f"side_{mode}" for mode in modes]
    if not callable(getattr(backend, "qualify_candidate", None)):
        row["detail"] = "后端没有显式升级接口，证据不完整，保留UNKNOWN"
        persist()
        return Status.UNKNOWN
    for mode in modes:
        if time.monotonic() >= deadline or backend.fatal_error:
            break
        began, first = time.monotonic(), len(backend.qualification_records)
        answer = backend.qualify_candidate(fact, seed_limit=32, pitch_degrees=mode)
        status = Status.UNKNOWN if answer is None else Status(answer)
        record = deepcopy(backend.qualification_records[-1]) if len(backend.qualification_records) > first else {}
        late = time.monotonic() >= deadline
        row["attempts"].append({"pose_mode": "catalog" if mode is None else f"side_{mode}",
            "seed_limit": 32, "seconds": time.monotonic()-began, "status": status.value,
            "deadline_exhausted": late, "backend_record": record})
        if late or backend.fatal_error:
            row["detail"] = "预算/接口中断，升级证据不完整，保留UNKNOWN"
            persist()
            break
        if status == Status.VALID:
            row.update(status="VALID", accepted_pose_mode=row["attempts"][-1]["pose_mode"],
                       detail="同一候选姿态的所有对应类型抓取/接近位均通过；侧姿资格不证明目录顶抓执行")
            persist()
            return Status.VALID
        persist()
    if (len(row["attempts"]) == len(modes) and not backend.fatal_error
            and time.monotonic() < deadline
            and all(not attempt["deadline_exhausted"] and attempt["backend_record"].get("candidate_exhausted")
                    for attempt in row["attempts"])):
        row.update(status="INVALID", bounded_pose_set_rejection=True,
            detail=("顶抓 32 种子 + 侧抓 3 姿态 × 32 种子无解" if len(modes) == 4 else "顶抓 32 种子无解")
                   +"；按FIX v1.3.4限定候选集拒绝，不宣称连续空间不可达证明")
        persist()
        return Status.INVALID
    row.setdefault("detail", "候选集证据不完整或类型结论不一致，保留UNKNOWN")
    persist()
    return Status.UNKNOWN


def qualify_workspace(problem, backend, *, deadline, checkpoint=None, inherited=None):
    """Check every arm/place, independent of the oracle's current reach table.

    This also powers runner prequalification in its already-created workcell.
    The compiler's original deadline is restored for the subsequent pipeline.
    """
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("资格验证需要有限的绝对 monotonic deadline")
    facts, started = _facts(problem), time.monotonic()
    prior_mode = inherited is not None
    inherited = inherited or {}
    covered = {index for index, fact in enumerate(facts) if inherited.get(fact) in (Status.VALID, Status.INVALID)}
    progress = {"status": "running", "total": len(facts), "checked": len(covered), "records": [],
        "inherited_count": len(covered), "queried_count": 0, "escalation": [],
        "updates": [{"arm": fact.args[0], "place": fact.args[1], "status": "UNKNOWN"} for fact in facts],
        "source_before": [{"fact": str(fact), "status": problem.oracle.check(fact).status.value} for fact in facts],
        "counts": {"VALID": 0, "INVALID": 0, "UNKNOWN": len(facts)}, "elapsed_seconds": 0.,
        "scope": "实际类型、叠放高度的抓取与接近位IK；不证明完整轨迹、支撑力学或执行成功"}
    for index, fact in enumerate(facts):
        status = inherited[fact] if index in covered else Status.UNKNOWN
        progress["updates"][index]["status"] = status.value
        problem.oracle.update(fact, status, "已核验旧IK缓存确定结论" if index in covered else "真实IK预验证尚未检查")
    query_indices = [index for index in range(len(facts)) if index not in covered]
    if prior_mode:
        goals = {condition.fact.args[0] for condition in getattr(problem, "goal", ())
                 if not condition.negated and condition.fact.predicate == "filled"}
        def priority(index):
            place = facts[index].args[1]
            depth = _slot_depth(problem, place)
            return (0, -depth, index) if place in goals and depth >= 4 else (1, 0, index)
        query_indices.sort(key=priority)
    progress["query_priority"] = ("仅显式prior：正目标高层UNKNOWN按深度降序、原索引稳定；其余UNKNOWN保持原顺序，预算不变"
                                  if prior_mode else "全组合原顺序；不继承输入能力声明")
    progress["planned_query_order"] = [str(facts[index]) for index in query_indices]
    progress["query_order"] = []

    def persist():
        progress["elapsed_seconds"] = time.monotonic()-started
        progress["remaining_unchecked"] = [str(fact) for index, fact in enumerate(facts) if index not in covered]
        progress["counts"] = {status: sum(cell["status"] == status for cell in progress["updates"])
                              for status in ("VALID", "INVALID", "UNKNOWN")}
        if checkpoint is not None:
            checkpoint(deepcopy(progress))

    previous_deadline = backend.compiler.deadline
    backend.compiler.deadline = deadline if previous_deadline is None else min(deadline, previous_deadline)
    try:
        persist()
        for index in query_indices:
            fact = facts[index]
            if time.monotonic() >= deadline or backend.fatal_error:
                break
            progress["query_order"].append(str(fact))
            began, first = time.monotonic(), len(backend.qualification_records)
            try:
                answer = backend.qualify(fact)
                status = Status.UNKNOWN if answer is None else Status(answer)
                record = deepcopy(backend.qualification_records[-1]) if len(backend.qualification_records) > first else {}
                if status == Status.UNKNOWN and time.monotonic() < deadline and not backend.fatal_error:
                    status = _escalate(problem, backend, fact, record, progress, deadline, persist)
            except Exception as exc:
                from .errors import is_transport_failure
                status, record = Status.UNKNOWN, {"detail": f"{type(exc).__name__}: {exc}"}
                progress["reason"] = record["detail"]
                progress["backend_exception"] = True
                if is_transport_failure(exc):
                    backend.fatal_error = record["detail"]
            if time.monotonic() >= deadline:
                status = Status.UNKNOWN
                record["deadline_exhausted"] = True
            progress["records"].append({"fact": str(fact), "arm": fact.args[0], "place": fact.args[1],
                "status": status.value, "seconds": time.monotonic()-began, "backend_record": record})
            progress["updates"][index]["status"] = status.value
            progress["checked"] += 1
            progress["queried_count"] += 1
            covered.add(index)
            problem.oracle.update(fact, status, "真实IK抓取/接近位预验证")
            persist()
            if progress.get("backend_exception"):
                break
        progress["status"] = ("failed" if backend.fatal_error or progress.get("backend_exception") else
                              "completed" if progress["checked"] == len(facts) else "partial")
        progress["fatal_environment"] = bool(backend.fatal_error)
        if backend.fatal_error:
            progress["reason"] = backend.fatal_error
        elif progress["checked"] < len(facts):
            progress.setdefault("reason", "任务资格预算耗尽，未检查组合保持UNKNOWN")
        progress["all_reach_resolved"] = all(cell["status"] != "UNKNOWN" for cell in progress["updates"])
        persist()
        return progress
    finally:
        backend.compiler.deadline = previous_deadline


def _environment(port):
    from .runner import check_environment
    from tools.overnight_batch import full_test_processes, process_snapshot
    checked = check_environment(port)
    if any("coppelia" in row["name"].lower() for row in process_snapshot()):
        raise RuntimeError("已有CoppeliaSim进程；不接管或改变该实例")
    if full_test_processes():
        raise RuntimeError("完整默认测试仍在运行，不启动真实资格验证")
    return checked


def _simulator(port, timeout):
    from apexmr.sim_connection import Simulator
    return Simulator(ROOT, port=port, timeout=timeout)


def _build(sim, client, task, install, output):
    from .workcell import build
    return build(sim, client, task, install, output)


def _planner(sim, client, cfg):
    from apexmr.motion import MotionPlanner
    return MotionPlanner(sim, client, cfg)


def _backend(planner, cfg, data):
    from .backend import CoppeliaBackend
    from .frames import Frames
    return CoppeliaBackend(planner, cfg, Frames(data))


def _result(log):
    return {key: deepcopy(log.get(key)) for key in ("task", "source_task", "source_sha256", "qualified_task",
        "log_file", "status", "total", "checked", "counts", "remaining_unchecked", "elapsed_seconds",
        "reason", "failure_phase", "fatal_environment", "simulator_pid", "simulator_closed", "finished", "recovered",
        "inherited_count", "queried_count", "prior_qualification", "escalation")}


def _layout(data):
    data = deepcopy(data)
    data.pop("capability_source", None)
    data.setdefault("capabilities", {}).pop("reach", None)
    return data


def _prior_cache(path, source_data, problem):
    """Accept an actual immutable IK projection with its original proof log."""
    from skillnet.capability.oracle import CapabilityOracle
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("旧资格缓存必须位于项目内")
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("capability_source") != "ik_qualified" or _layout(data) != _layout(source_data):
        raise ValueError("旧资格缓存来源标记或完整任务布局不匹配")
    log_path = path.parent/"qualify_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    origin = Path(log.get("source_task", "")).resolve()
    facts = _facts(problem)
    if (Path(log.get("qualified_task", "")).resolve() != path
            or not origin.is_relative_to(ROOT) or not origin.is_file()
            or log.get("source_sha256") != _digest(origin)
            or _layout(tomllib.loads(origin.read_text(encoding="utf-8-sig"))) != _layout(data)
            or log.get("checked") != len(facts) or log.get("total") != len(facts)
            or log.get("status") != "completed" or log.get("simulator_closed") is not True
            or not isinstance(log.get("simulator_pid"), int) or log["simulator_pid"] <= 0):
        raise ValueError("旧资格缓存的完整覆盖、来源SHA或真实会话证据不匹配")
    oracle = CapabilityOracle(data.get("capabilities", {}))
    expected = {(fact.args[0], fact.args[1]): oracle.check(fact).status.value for fact in facts}
    cells = log.get("updates", [])
    actual = {(cell["arm"], cell["place"]): cell["status"] for cell in cells}
    if len(cells) != len(expected) or actual != expected:
        raise ValueError("旧资格缓存的能力单元与原始证据不匹配")
    inherited = {fact: oracle.check(fact).status for fact in facts
                 if oracle.check(fact).status != Status.UNKNOWN}
    metadata = {"path": str(path), "sha256": _digest(path), "log_file": str(log_path),
                "log_sha256": _digest(log_path), "source_task": str(origin),
                "source_sha256": log["source_sha256"], "inherited_count": len(inherited)}
    return inherited, metadata


def qualify_task(task, *, out=DEFAULT_OUT, timeout_wall_s=MAX_SECONDS, deadline=None, port=PORT,
                 prior_qualification=None):
    """Build once, save partial results after each cell, and never rewrite input.

    out is a v134 output root. A repeated source/output pair adopts its saved
    result without launching another experiment, including a partial result.
    """
    from skillnet.domains.blocks import load_problem
    started = time.monotonic()
    if (isinstance(timeout_wall_s, bool) or not isinstance(timeout_wall_s, (int, float))
            or not math.isfinite(timeout_wall_s) or not 0 < timeout_wall_s <= MAX_SECONDS):
        raise ValueError("每任务资格预算必须大于0且不超过60秒")
    if port != PORT:
        raise ValueError("端口固定为23210")
    if deadline is not None and (not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("deadline必须为有限monotonic时间")
    deadline = min(started+timeout_wall_s, deadline) if deadline is not None else started+timeout_wall_s
    # The supervisor's 60-second wall includes final partial snapshots and
    # native shutdown. Do not spend that cleanup window on another IK query.
    cleanup_reserve = min(15., max(0., deadline-started)*.25)
    query_deadline = deadline-cleanup_reserve
    source, output_root = Path(task).resolve(), Path(out).resolve()
    if not source.is_relative_to(ROOT) or not output_root.is_relative_to(VERSION_ROOT):
        raise ValueError("输入必须在项目内，资格结果必须写入v134新目录")
    data = tomllib.loads(source.read_text(encoding="utf-8-sig"))
    problem = load_problem(source, ROOT / "skills/blocks_v1.toml")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", problem.name) or problem.name in {".", ".."}:
        raise ValueError("任务名不能用于安全的输出目录")
    output = output_root / problem.name
    qualified = output / (problem.name+".qualified.toml")
    log_path = output / "qualify_log.json"
    if qualified.resolve() == source:
        raise ValueError("资格副本不能覆盖输入任务")
    source_hash = _digest(source)
    inherited, prior = ({}, None) if prior_qualification is None else _prior_cache(prior_qualification, data, problem)
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        if (log.get("source_sha256") != source_hash or log.get("source_task") != str(source)
                or log.get("prior_qualification") != prior):
            raise FileExistsError("保留已有不同来源的资格结果："+str(output))
        if log.get("status") == "running":
            # Proofs are committed before the task projection. Rebuild a lagging
            # projection without granting another 60-second experimental run.
            atomic_text(qualified, task_data_to_toml(qualified_task_data(data, log)))
            log.update(status="partial", recovered=True, finished=_now(),
                       reason="上次资格任务中断；保留已提交证据，不自动重跑")
            atomic_json(log_path, log)
        return _result(log)
    output.mkdir(parents=True, exist_ok=True)
    cells = [{"arm": fact.args[0], "place": fact.args[1], "status": inherited.get(fact, Status.UNKNOWN).value}
             for fact in _facts(problem)]
    log = {"task": problem.name, "source_task": str(source), "source_sha256": source_hash,
        "qualified_task": str(qualified), "log_file": str(log_path), "status": "running", "phase": "load",
        "total": len(cells), "checked": len(inherited), "updates": cells, "records": [], "elapsed_seconds": 0.,
        "prior_qualification": prior, "inherited_count": len(inherited), "queried_count": 0, "escalation": [],
        "timeout_wall_s": timeout_wall_s, "port": port, "started": _now(), "fatal_environment": False,
        "outer_deadline_monotonic": deadline, "query_deadline_monotonic": query_deadline,
        "cleanup_reserve_seconds": cleanup_reserve,
        "remaining_unchecked": [str(fact) for fact in _facts(problem) if fact not in inherited],
        "counts": {status: sum(cell["status"] == status for cell in cells) for status in ("VALID", "INVALID", "UNKNOWN")}}

    def checkpoint(progress=None):
        if progress is not None:
            log.update(deepcopy(progress))
        log["elapsed_seconds"] = time.monotonic()-started
        atomic_json(log_path, log)  # Commit proofs first; the task may safely lag.
        text = task_data_to_toml(qualified_task_data(data, log))
        if qualified.exists() and log["checked"] == 0 and qualified.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError("保留已有不同资格副本："+str(qualified))
        atomic_text(qualified, text)

    def enter(phase):
        log["phase"] = phase
        checkpoint()
        if time.monotonic() >= query_deadline:
            raise TimeoutError("资格查询预算耗尽；在原总预算内预留日志与原生关闭时间")

    owner = None
    initial = task_data_to_toml(qualified_task_data(data, log))
    if qualified.exists() and qualified.read_text(encoding="utf-8-sig") != initial:
        raise FileExistsError("保留已有不同资格副本："+str(qualified))
    checkpoint()  # A useful all-UNKNOWN snapshot exists before simulator imports/start.
    try:
        enter("environment")
        log["environment"] = _environment(port)
        enter("simulator_start")
        owner = _simulator(port, max(.001, min(30., query_deadline-time.monotonic())))
        owner.start(headless=True)
        log["simulator_pid"] = owner.process.pid
        enter("simulator_connect")
        client, sim = owner.connect()
        enter("workcell")
        cfg = _build(sim, client, source, owner.install, output)
        enter("qualification")
        with _planner(sim, client, cfg) as planner:
            backend = _backend(planner, cfg, data)
            qualify_workspace(problem, backend, deadline=query_deadline, checkpoint=checkpoint,
                              inherited=inherited if prior is not None else None)
    except TimeoutError as exc:
        log.update(status="partial", reason=str(exc))
    except Exception as exc:
        log.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                   fatal_environment=log["phase"] == "environment")
    finally:
        if owner is not None:
            try:
                owner.close()
                log["simulator_closed"] = owner.process is not None and owner.process.poll() is not None
                if not log["simulator_closed"]:
                    log.update(status="failed", fatal_environment=True, reason="自有模拟器未关闭")
            except Exception as exc:
                log.update(status="failed", fatal_environment=True, reason=f"关闭自有模拟器失败：{exc}")
        log["finished"] = _now()
        log["failure_phase"] = None if log["status"] == "completed" else log["phase"]
        checkpoint()
    return _result(log)


def qualify_tasks(tasks, *, out=DEFAULT_OUT, prior_qualification=None):
    """Serial CLI guard: own each worker tree and hard-stop it after 60 s."""
    from tools.overnight_batch import OwnedProcess, execution_lock
    out = Path(out).resolve()
    if not out.is_relative_to(VERSION_ROOT):
        raise ValueError("资格结果必须位于v134新目录")
    priors = [None]*len(tasks) if prior_qualification is None else prior_qualification
    if isinstance(priors, (str, Path)):
        priors = [priors]
    if len(priors) != len(tasks):
        raise ValueError("旧资格缓存数量必须与任务逐项对应")
    out.mkdir(parents=True, exist_ok=True)
    results, startup_failures = [], 0
    with execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        for task, prior in zip(tasks, priors):
            token = uuid.uuid4().hex
            spec = out / ("qualify_worker_"+token+".json")
            start_signal = spec.with_suffix(".start")
            worker_result = spec.with_suffix(".result.json")
            hard_deadline = time.monotonic()+MAX_SECONDS
            atomic_json(spec, {"task": str(Path(task).resolve()), "out": str(out), "start_signal": str(start_signal),
                "deadline": hard_deadline-1., "result": str(worker_result),
                **({} if prior is None else {"prior_qualification": str(Path(prior).resolve())})})
            owner = OwnedProcess([sys.executable, "-X", "utf8", "-m", "simbridge.qualifier", "--worker", str(spec)],
                                 spec.with_suffix(".log"), start_signal)
            try:
                owner.release()
                while owner.process.poll() is None:
                    if time.monotonic() >= hard_deadline:
                        owner.terminate()
                        break
                    time.sleep(.1)
            finally:
                owner.close()
            if worker_result.exists():
                result = json.loads(worker_result.read_text(encoding="utf-8"))
            else:
                # No queries are possible under this expired deadline. Adopt
                # committed partial proofs, or create an all-UNKNOWN snapshot.
                result = qualify_task(task, out=out, timeout_wall_s=.001, deadline=time.monotonic()-1.,
                                      **({} if prior is None else {"prior_qualification": prior}))
            results.append(result)
            startup_failures = startup_failures+1 if result.get("failure_phase") in {"simulator_start", "simulator_connect"} else 0
            atomic_json(out / "qualify_batch.json", {"requested": len(tasks), "recorded": len(results), "results": results})
            if result.get("fatal_environment") or startup_failures >= 3:
                break
    return {"status": "completed" if len(results) == len(tasks) else "stopped", "results": results}


def _worker(spec):
    data = json.loads(Path(spec).read_text(encoding="utf-8"))
    while not Path(data["start_signal"]).exists():
        if time.monotonic() >= data["deadline"]:
            return 2
        time.sleep(.02)
    remaining = max(.001, data["deadline"]-time.monotonic())
    result = qualify_task(data["task"], out=data["out"], timeout_wall_s=min(MAX_SECONDS, remaining), deadline=data["deadline"],
                          **({} if data.get("prior_qualification") is None else {"prior_qualification": data["prior_qualification"]}))
    atomic_json(data["result"], result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(sys.argv[2]))
    raise SystemExit("请使用 python -m simbridge qualify --tasks ... --out DIR")
