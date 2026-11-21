"""Project-confined single-task and sequential batch simulator orchestration."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import tomllib
import traceback

from apexmr.motion import MotionPlanner
from apexmr.sim_connection import Simulator
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import plan_to_dict
from tools.overnight_batch import atomic_json as atomic_progress_json
from .compiler import CompileFailure
from .backend import CoppeliaBackend
from .errors import is_transport_failure
from .frames import Frames, read_task
from .workcell import build


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/sim_v1_3/v133"
PORT = 23210
VIDEO_LIMIT = 2 * 1024**3


class _PlanningIncomplete(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class MotionEvidenceJournal:
    """Commit motion boundaries before a hard task timeout can kill a worker."""

    def __init__(self, output, started, phase):
        self.output, self.started, self.phase = Path(output), started, phase
        self.events, self.wall_seconds = 0, 0.

    def __call__(self, event):
        began = time.monotonic()
        row = {"at": datetime.now(timezone.utc).isoformat(), "sequence": self.events+1,
               "elapsed_seconds": began-self.started, "pid": os.getpid(), "event": event}
        # Encode before opening the append-only file: invalid data must not
        # leave a torn JSON line or overwrite any previous committed event.
        encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/"motion_live.jsonl").open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.events += 1
        atomic_progress_json(self.output/"motion_progress.json", {
            **{key: value for key, value in row.items() if key != "event"},
            "phase": self.phase()+"/"+event.get("query", "motion")+":"+event.get("phase", "event"),
            "event": {key: value for key, value in event.items() if key != "record"},
            "journal": str(self.output/"motion_live.jsonl")})
        self.wall_seconds += time.monotonic()-began


def write_json(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


def output_directory(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("仿真输出必须位于项目根目录内")
    return result


def check_environment(port=PORT):
    if port != PORT:
        raise ValueError("本轮仿真端口固定为 23210")
    free = shutil.disk_usage("C:\\").free
    if free < 5 * 1024**3:
        raise RuntimeError("C 盘剩余不足 5 GB，按交接单停止")
    return {"c_free_bytes": free, "port": port}


def video_allowed(requested, root=DEFAULT_OUT):
    total = sum(path.stat().st_size for path in Path(root).rglob("*.mp4"))
    return bool(requested and total < VIDEO_LIMIT)


def run_task(task, *, port=PORT, headless=True, video=False,
             out=DEFAULT_OUT, timeout_wall_s=1200.):
    """Run one task; preserve failure evidence and always close our instance.

    The timeout covers planning, construction, compilation and execution.
    Failures are returned with a phase, reason and non-success status, allowing
    an authorized batch to continue except on environmental stop conditions.
    """
    if not math.isfinite(timeout_wall_s) or timeout_wall_s <= 0:
        raise ValueError("墙钟超时必须为有限正数")
    destination = output_directory(out)
    task_path = Path(task).resolve()
    started = time.monotonic()
    deadline = started + timeout_wall_s
    phase = "load"
    phase_start = started
    phase_times = {}
    summary = {"task_file": str(task_path), "status": "failed", "passed": False,
               "profile": "skillnet_v1_3", "port": port,
               "timeout_wall_s": timeout_wall_s, "fatal_environment": False}
    output = destination / task_path.stem
    owner = None
    compiler = None
    backend = None
    motion_journal = None

    def enter(name):
        nonlocal phase, phase_start
        now = time.monotonic()
        phase_times[phase] = phase_times.get(phase, 0.) + now - phase_start
        phase, phase_start = name, now
        # Every stage is durable before a blocking call, including motion and
        # TPG. A killed worker must retain its last completed timings.
        atomic_progress_json(output / "run_progress.json", {
            "phase": name, "pid": summary.get("simulator_pid"),
            "elapsed_seconds": now-started, "phase_seconds": dict(phase_times),
            "task": summary.get("task", task_path.stem),
            "task_file": str(task_path), "task_sha256": summary.get("task_sha256")})
        if now >= deadline:
            raise TimeoutError("任务达到墙钟超时")

    try:
        problem = load_problem(task_path, ROOT / "skills/blocks_v1.toml")
        task_data = read_task(task_path)
        prequalify = ("reach" not in problem.oracle.defaults
                      or task_data.get("capability_source") == "distance_prior")
        output = destination / problem.name
        output.mkdir(parents=True, exist_ok=True)
        summary.update(task=problem.name,
                       task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest())
        enter("environment")
        try:
            summary["environment"] = check_environment(port)
        except (RuntimeError, OSError):
            summary["fatal_environment"] = True
            raise
        enter("planning")
        if prequalify:
            summary["planning_status"] = "awaiting_ik_prequalification"
        else:
            result = run(problem, deadline=deadline)
            summary["planning_status"] = result.status
            summary["feedback"] = list(result.feedback)
            write_json(output / "plan.json", plan_to_dict(result))
            write_json(output / "initial_plan.json", plan_to_dict(result))
        # UNKNOWN may be resolved by the real backend's capability queries.
        # The scene must exist before qualify can check those poses.
        if not prequalify and result.status not in {"solved", "unknown"}:
            raise _PlanningIncomplete(result.status, f"符号规划未求解：{result.status}")
        enter("simulator_start")
        owner = Simulator(output / "instance", port=port, timeout=30)
        try:
            owner.start(headless=headless)
            summary["simulator_pid"] = owner.process.pid
            enter("simulator_connect")
            client, sim = owner.connect()
        except Exception:
            summary["fatal_environment"] = True
            raise
        enter("workcell")
        cfg = build(sim, client, task_path, owner.install, output)
        cfg["profile"] = "skillnet_v1_3"
        with MotionPlanner(sim, client, cfg) as planner:
            backend = CoppeliaBackend(planner, cfg, Frames(task_path))
            motion_journal = MotionEvidenceJournal(output, started, lambda: phase)
            backend.progress_callback = motion_journal
            compiler = backend.compiler
            compiler.deadline = deadline
            qualification_log = task_path.parent / "qualify_log.json"
            if qualification_log.is_file():
                hints = json.loads(qualification_log.read_text(encoding="utf-8-sig"))
                if hints.get("task") in (None, problem.name):
                    receipt = backend.import_qualification_hints(hints, source={
                        "path": str(qualification_log),
                        "sha256": hashlib.sha256(qualification_log.read_bytes()).hexdigest()})
                    write_json(output / "qualification_hint_import.json", receipt)
            if prequalify:
                from .qualifier import qualified_task_data, qualify_workspace
                from .taskio import task_data_to_toml
                from tools.overnight_batch import atomic_json, atomic_text

                enter("prequalification")
                def save_prequalification(progress):
                    atomic_json(output / "prequalification.json", progress)
                    atomic_text(output / (problem.name+".qualified.toml"),
                                task_data_to_toml(qualified_task_data(task_data, progress)))
                prequalification = qualify_workspace(problem, backend,
                    deadline=min(deadline, time.monotonic()+60.), checkpoint=save_prequalification)
                summary["prequalification"] = {key: prequalification[key] for key in
                                               ("status", "total", "checked", "counts")}
                if backend.fatal_error:
                    raise ConnectionError("IK预验证远程接口失联："+backend.fatal_error)
                enter("planning")
                result = run(problem, deadline=deadline)
                summary["planning_status"] = result.status
                summary["feedback"] = list(result.feedback)
                write_json(output / "plan.json", plan_to_dict(result))
                write_json(output / "initial_plan.json", plan_to_dict(result))
                if result.status not in {"solved", "unknown"}:
                    raise _PlanningIncomplete(result.status, f"IK预验证后符号规划未求解：{result.status}")
            enter("real_planning")
            result = run(problem, backend=backend, deadline=deadline, post_first_solution_limit=2.0)
            summary["real_planning_status"] = result.status
            summary["symbolic_makespan_seconds"] = result.schedule.makespan
            summary["symbolic_lower_bound"] = result.schedule.lower_bound
            summary["symbolic_suboptimality_bound"] = result.schedule.suboptimality_bound
            summary["feedback"] = list(result.feedback)
            write_json(output / "plan.json", plan_to_dict(result))
            write_json(output / "motion_backend.json", backend.audit())
            if backend.fatal_error:
                raise ConnectionError(f"真实运动后端远程接口失联，停止后续任务：{backend.fatal_error}")
            if result.status != "solved":
                raise _PlanningIncomplete(result.status, f"真实运动后端重规划未求解：{result.status}")
            enter("compile")
            receipt = backend.export_reference(result.schedule, result.graph)
            tasks = receipt["tasks"]
            write_json(output / "reference_receipt.json", receipt)
            summary["reference_reused"] = True
            compile_seconds = time.monotonic() - phase_start
            summary["backend_nominal_seconds"] = backend.evaluate(result.schedule, result.graph)
            write_json(output / "tasks.json", tasks)
            write_json(output / "compile_evidence.json", compiler.evidence)
            enter("execute")
            from .executor import execute_plan
            measured = execute_plan(
                sim, client, cfg, tasks, problem, output,
                planner=planner, plan_steps=result.schedule.steps,
                planning_seconds=(phase_times.get("planning", 0.)
                                  + phase_times.get("real_planning", 0.)),
                compile_seconds=compile_seconds,
                video=video_allowed(video, DEFAULT_OUT),
                stopped_geometry_context=partial(owner.fast_idle_stopped, deadline=deadline),
                timeout_wall_s=max(.001, deadline-time.monotonic()))
            summary.update(measured)
            if measured.get("remote_interface_failure"):
                summary["fatal_environment"] = True
            summary["passed"] = bool(measured.get("passed"))
            summary["status"] = "passed" if summary["passed"] else measured.get("status", "failed")
            if not summary["passed"]:
                summary.setdefault("reason", "执行或验收未通过，详见 validation.json")
                summary["failure_phase"] = measured.get("failure_phase", "validation")
    except Exception as exc:
        summary.setdefault("original_failure_phase", phase)
        summary.setdefault("original_failure_reason", f"{type(exc).__name__}: {exc}")
        if phase in {"simulator_start", "simulator_connect"}:
            summary.setdefault("startup_failure_phase", phase)
        if is_transport_failure(exc):
            summary["fatal_environment"] = True
        summary.update(status="timeout" if isinstance(exc, TimeoutError) else
                       exc.status if isinstance(exc, _PlanningIncomplete) else "failed",
                       passed=False, failure_phase=phase,
                       reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if isinstance(exc, CompileFailure):
            summary["motion_status"] = exc.status.value
            summary["failed_segment"] = exc.segment
            summary["failure_evidence"] = exc.evidence
            if isinstance(exc.evidence, dict) and exc.evidence.get("timeout"):
                summary["status"] = "timeout"
    finally:
        phase_times[phase] = phase_times.get(phase, 0.) + time.monotonic()-phase_start
        if compiler is not None:
            write_json(output / "compile_evidence.json", compiler.evidence)
        if backend is not None:
            backend_audit = backend.audit()
            summary["motion_backend_cache"] = backend_audit["cache"]
            summary["motion_feedback_count"] = sum(
                record["status"] in {"INVALID", "UNKNOWN"} for record in backend.validation_records)
            summary["motion_feedback_rounds"] = len(backend_audit.get("feedback_rounds", []))
            summary["qualification_calls"] = len(backend.qualification_records)
            write_json(output / "motion_backend.json", backend_audit)
        if owner is not None:
            close_error = None
            try:
                owner.close()
            except Exception as exc:
                close_error = f"{type(exc).__name__}: {exc}"
            summary["simulator_log"] = str(owner.log_path)
            summary["simulator_closed"] = owner.process is not None and owner.process.poll() is not None
            if close_error or (not summary["simulator_closed"] and owner.process is not None):
                if summary.get("failure_phase"):
                    summary.setdefault("original_failure_phase", summary["failure_phase"])
                    summary.setdefault("original_failure_reason", summary.get("reason"))
                summary["close_reason"] = close_error or "项目实例未能关闭，停止后续任务"
                summary.update(passed=False, status="failed", fatal_environment=True,
                               failure_phase="simulator_close", reason=summary["close_reason"])
        summary["phase_seconds"] = phase_times
        summary["initial_planning_seconds"] = phase_times.get("planning", 0.)
        summary["real_planning_seconds"] = phase_times.get("real_planning", 0.)
        summary["planning_seconds"] = (summary["initial_planning_seconds"]
                                       + summary["real_planning_seconds"])
        summary["wall_seconds"] = time.monotonic()-started
        if motion_journal is not None:
            summary["motion_journal"] = {"events": motion_journal.events,
                "persistence_wall_seconds": motion_journal.wall_seconds,
                "path": str(output/"motion_live.jsonl")}
        write_json(output / "summary.json", summary)
    return summary


def batch(manifest):
    """Execute manifest entries serially; stop on fatal or three same failures."""
    path = Path(manifest).resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    entries = data.get("run", data.get("task", []))
    if not entries:
        raise ValueError("清单缺少 [[run]] 任务")
    settings = data.get("settings", {})
    results = []
    failures = Counter()
    for entry in entries:
        row = {**settings, **entry}
        task = Path(row.pop("task", row.pop("path", "")))
        if not task.is_absolute():
            task = ROOT / task
        result = run_task(task, **{key: value for key, value in row.items()
                                  if key in {"out", "port", "headless", "video", "timeout_wall_s"}})
        results.append(result)
        if result.get("fatal_environment"):
            break
        if not result["passed"]:
            key = (result.get("failure_phase"), result.get("reason"))
            failures[key] += 1
            if failures[key] >= 3:
                break
    summary = {"passed": len(results) == len(entries) and all(row["passed"] for row in results),
               "requested": len(entries), "completed": len(results), "results": results}
    write_json(DEFAULT_OUT / "batch_summary.json", summary)
    return summary
