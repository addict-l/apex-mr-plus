"""Build a collision-sampled TPG, execute actual dynamics, and audit evidence."""
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

from apexmr.geometry import SampledConflictOracle, verify_rollout
from apexmr.motion import MotionPlanner
from apexmr.program import parameterize
from apexmr.sim_connection import FastIdleRestorationError
from apexmr.runtime import execute, install_controller
from apexmr.tpg import build_tpg
from apexmr.tpg import build_tpg as _structural_tpg
from apexmr.validation import validate_execution
from tools.overnight_batch import atomic_json
from .frames import Frames
from .errors import is_transport_failure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "skillnet_v1_3"


class _BudgetedPlanner:
    """Check a shared wall deadline around each geometry backend call."""

    def __init__(self, planner, check, statistics=None, checkpoint=None, after_samples=None):
        self._planner, self._check = planner, check
        self._statistics = statistics
        self._checkpoint = checkpoint
        self._after_samples = after_samples

    def __getattr__(self, name):
        value = getattr(self._planner, name)
        if not callable(value):
            return value
        def bounded(*args, **kwargs):
            self._check()
            stats = self._statistics
            if stats is not None:
                stats["attempted_calls"] += 1
                item = stats["methods"].setdefault(name, {"attempted": 0, "completed": 0, "wall_seconds": 0.})
                item["attempted"] += 1
                if name == "sample_state_pairs":
                    stats["sample_batches"] += 1
                    stats["sample_states"] += len(args[0])
            if self._checkpoint is not None:
                self._checkpoint()
            started = time.perf_counter()
            try:
                answer = value(*args, **kwargs)
                if stats is not None:
                    stats["completed_calls"] += 1
                    item["completed"] += 1
            finally:
                if stats is not None:
                    spent = time.perf_counter()-started
                    stats["wall_seconds"] += spent
                    item["wall_seconds"] += spent
                if self._checkpoint is not None:
                    self._checkpoint()
            self._check()
            if name == "sample_state_pairs" and self._after_samples is not None:
                self._after_samples()
            return answer
        return bounded


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def build_requirements(problem, cfg, tasks, frames=None):
    """Derive goals from the task and final declared install/detach transitions."""
    frames = frames or Frames(problem)
    installed = {block: slot for slot, block in cfg.get("initial_installed", {}).items()}
    for task in sorted(tasks, key=lambda row: row.get("reference_order", 0)):
        metadata, block = task.get("metadata", {}), task.get("part")
        if metadata.get("action") == "install":
            installed[block] = metadata["slot"]
        elif metadata.get("action") == "detach":
            installed.pop(block, None)
    goals, required_installed, filled, forbidden_installed, forbidden_poses = {}, {}, [], [], []
    for condition in problem.goal:
        fact = condition.fact
        if fact.predicate == "at":
            block, location = fact.args
            pose = frames.block_pose(block, location)
            if condition.negated:
                forbidden_poses.append([block, pose])
            else:
                goals[block] = pose
        elif fact.predicate == "installed":
            block, slot = fact.args
            if condition.negated:
                forbidden_installed.append([block, slot])
            else:
                required_installed[block] = slot
                goals[block] = frames.slot_pose(slot)
        elif fact.predicate == "filled" and not condition.negated:
            slot = fact.args[0]
            candidates = [block for block, target in installed.items() if target == slot]
            if len(candidates) != 1:
                raise ValueError(f"编译后的安装状态未为目标槽位 {slot} 提供唯一积木")
            block = candidates[0]
            filled.append(slot)
            required_installed[block] = slot
            goals[block] = frames.slot_pose(slot)
    return {"profile": PROFILE, "goal_poses": goals,
            "goal_position_tolerance_m": .005, "goal_orientation_tolerance_rad": .02,
            "home_tolerance_rad": .01, "support_force_min_N": cfg.get("runtime", {}).get("support_force_min_N", 1.),
            "required_installed": required_installed, "required_filled_slots": filled,
            "forbidden_installed": forbidden_installed, "forbidden_poses": forbidden_poses,
            "temporary_slots": {slot: {"pose": frames.slot_pose(slot), "size": list(frames.catalog[row["type"]].size)}
                                for slot, row in frames.slots.items() if row.get("temporary")},
            "object_sizes": {block: list(frames.catalog[row["type"]].size) for block, row in frames.blocks.items()},
            "home_q": {arm: list(cfg["arms"][arm]["home_q"]) for arm in ("A", "B")},
            "required_dynamic_objects": list(frames.blocks)}


def _step_times(graph, execution, plan_steps):
    measured = {}
    for event in execution.get("events", []):
        if event.get("kind") not in {"node_started", "node_completed"}:
            continue
        measured.setdefault(event.get("node"), {})[
            "start" if event["kind"] == "node_started" else "end"] = event["time"]
    groups = {}
    for node in graph["nodes"]:
        step = node.get("metadata", {}).get("source_step")
        if isinstance(step, int):
            groups.setdefault(step, []).append(node)
    estimates = {index: float(step.end-step.start) for index, step in
                 enumerate(sorted(plan_steps or [], key=lambda item: item.start), 1)}
    rows = []
    for number, nodes in sorted(groups.items()):
        times = [measured.get(node["id"], {}) for node in nodes]
        actual = (max(row["end"] for row in times)-min(row["start"] for row in times)
                  if all("start" in row and "end" in row for row in times) else None)
        estimate = estimates.get(number)
        rows.append({"step": number, "action": nodes[0].get("metadata", {}).get("source_action"),
                     "estimated_seconds": estimate, "actual_seconds": actual,
                     "estimated_actual_ratio": estimate/actual if estimate is not None and actual else None})
    return rows


def execute_plan(sim, client, cfg, tasks, problem, outdir, *, video=False,
                 timeout_wall_s=1200., planner=None, plan_steps=None,
                 planning_seconds=0., compile_seconds=0., stopped_geometry_context=None):
    """Return a persisted summary, including a phase/reason on every failure.

    An optional existing stopped-scene planner is reused. This function never
    owns the simulator process. The runner must convert failed summaries to a
    nonzero CLI exit code and retain control of its own process lifecycle.
    An optional zero-argument owned context covers only stopped geometry. Its
    original task deadline is bound by the runner; verified restoration must
    precede controller installation and dynamics execution.
    """
    output = Path(outdir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("仿真输出必须位于项目根目录内")
    if not math.isfinite(timeout_wall_s) or timeout_wall_s <= 0:
        raise ValueError("仿真墙钟超时必须为有限正数")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"task": problem.name, "profile": PROFILE, "success": False,
               "status": "failed", "phase": "requirements", "reason": None,
               "planning_seconds": planning_seconds, "compile_seconds": compile_seconds,
               "motion_task_count": len(tasks), "video": bool(video), "step_durations": [],
               "remote_interface_failure": False}
    started = time.monotonic()
    deadline = started+timeout_wall_s
    stage_started = started
    timed_phase = summary["phase"]
    checkpoint_at = 0.
    server_statistics_at = -float("inf")
    stage_seconds = {}
    oracle = None
    queries = {"attempted_calls": 0, "completed_calls": 0, "wall_seconds": 0.,
               "sample_batches": 0, "sample_states": 0, "methods": {},
               "scope": "planner API calls including client/server and codec; not isolated network time"}
    summary.update(geometry_scope="sampled swept conflicts at 0.07 rad and nominal rollout at 0.05 s; no continuous collision guarantee",
                   stage_seconds=stage_seconds, query_statistics=queries)
    geometry_context = {"enabled": stopped_geometry_context is not None,
                        "scope": "stopped TPG/parameterization/nominal rollout only",
                        "enter_seconds": 0., "restore_seconds": 0., "receipt": None}
    summary["stopped_geometry_context"] = geometry_context

    def checkpoint(force=False):
        nonlocal checkpoint_at
        now = time.monotonic()
        if not force and now-checkpoint_at < 2.:
            return
        checkpoint_at = now
        partial = dict(stage_seconds)
        partial[timed_phase] = partial.get(timed_phase, 0.)+now-stage_started
        atomic_json(output/"executor_progress.json", {
            "phase": summary["phase"], "elapsed_seconds": now-started,
            "remaining_seconds": max(0., deadline-now), "stage_seconds": partial,
            "tpg_nodes": summary.get("tpg_nodes"),
            "candidate_conflict_pairs": summary.get("candidate_conflict_pairs"),
            "query_statistics": queries,
             "stopped_geometry_context": geometry_context,
            "server_geometry_statistics": summary.get("server_geometry_statistics"),
            "geometry_statistics": {} if oracle is None else dict(getattr(oracle, "stats", {}))})

    def enter(name):
        nonlocal stage_started, timed_phase
        now = time.monotonic()
        stage_seconds[timed_phase] = stage_seconds.get(timed_phase, 0.)+now-stage_started
        summary["phase"], timed_phase, stage_started = name, name, now
        checkpoint(force=True)
        check_deadline()

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"执行器的 {timeout_wall_s:g} 秒统一墙钟预算已耗尽（阶段：{summary['phase']}）")

    try:
        check_deadline()
        _save(output/"tasks.json", tasks)
        requirements = build_requirements(problem, cfg, tasks)
        cfg["profile"], cfg["validation_requirements"] = PROFILE, requirements
        cfg["goal_poses"] = deepcopy(requirements["goal_poses"])
        cfg["slot_below"] = {slot: list(row.get("below", [])) for slot, row in problem.hooks.slots.items()}
        cfg["slot_types"] = {slot: row["type"] for slot, row in problem.hooks.slots.items()}
        _save(output/"requirements.json", requirements)
        check_deadline()
        with ExitStack() as resources:
            if planner is None:
                planner = resources.enter_context(MotionPlanner(sim, client, cfg))
            check_deadline()
            def capture_server_statistics(force=False):
                nonlocal server_statistics_at
                now = time.monotonic()
                if (not force and now-server_statistics_at < 5.) or deadline-now < .1:
                    return
                method = getattr(planner, "sample_batch_statistics", None)
                if callable(method):
                    server_statistics_at = now
                    statistics = budgeted_planner.sample_batch_statistics()
                    if isinstance(statistics, dict):
                        summary["server_geometry_statistics"] = statistics
                        summary.setdefault("server_geometry_statistics_before_tpg", deepcopy(statistics))
                    checkpoint(force=True)
            budgeted_planner = _BudgetedPlanner(planner, check_deadline, queries, checkpoint,
                                               capture_server_statistics)
            # This independent lexical stack closes on success, early return,
            # or exception. It never encloses install_controller or execute.
            with ExitStack() as geometry_resources:
                scope_entered = False
                try:
                    if stopped_geometry_context is not None:
                        enter("geometry_context_enter")
                        context_started = time.monotonic()
                        try:
                            geometry_context["receipt"] = geometry_resources.enter_context(stopped_geometry_context())
                            scope_entered = True
                        finally:
                            geometry_context["enter_seconds"] += time.monotonic()-context_started
                    capture_server_statistics(force=True)
                    tpg_started = time.monotonic()
                    enter("tpg_structure")
                    structural = _structural_tpg(tasks)
                    summary.update(tpg_nodes=len(structural["nodes"]),
                                   candidate_conflict_pairs=structural["counts"]["cross_arm_pairs"],
                                   structural_ordered_pairs_skipped=structural["counts"]["already_ordered_pairs_skipped"])
                    enter("tpg_geometry")
                    oracle = SampledConflictOracle(budgeted_planner, cfg, output/"geometry")

                    def timed_oracle(first, second):
                        check_deadline()
                        answer = oracle(first, second)
                        check_deadline()
                        return answer

                    graph = build_tpg(tasks, timed_oracle)
                    check_deadline()
                    capture_server_statistics(force=True)
                    summary["geometry_statistics"] = deepcopy(getattr(oracle, "stats", {}))
                    summary["tpg_counts"] = deepcopy(graph["counts"])
                    enter("parameterize")
                    parameterize(graph, vmax=.60, amax=1.20, jmax=8.)
                    check_deadline()
                    graph["validation_requirements"] = requirements
                    _save(output/"tpg.json", graph)
                    summary.update(tpg_seconds=time.monotonic()-tpg_started,
                                   tpg_nodes=len(graph["nodes"]), tpg_edges=len(graph["edges"]),
                                   nominal_makespan_seconds=graph["nominal_makespan"])
                    enter("rollout")
                    rollout = verify_rollout(graph, budgeted_planner, cfg, step_s=.05, outdir=output/"geometry")
                    check_deadline()
                    capture_server_statistics(force=True)
                    if not rollout["valid"]:
                        summary["reason"] = "名义轨迹的运动、等待位姿或状态转移未通过几何验收"
                        summary["rollout_failure"] = rollout.get("first_violation") or rollout.get("world_state_gaps")
                        return summary
                    graph["certificate"]["waiting_endpoints_verified"] = True
                    graph["certificate"]["nominal_rollout_checked"] = True
                    _save(output/"tpg.json", graph)
                finally:
                    if stopped_geometry_context is not None:
                        restore_started = time.monotonic()
                        try:
                            try:
                                # Preserve the original failure stage on an
                                # exception/invalid-rollout early return.
                                if sys.exc_info()[0] is None and summary["reason"] is None:
                                    enter("geometry_context_restore")
                            finally:
                                try:
                                    geometry_resources.close()
                                except FastIdleRestorationError:
                                    raise
                                except BaseException as exc:
                                    raise FastIdleRestorationError(str(exc), geometry_context["receipt"]) from exc
                            receipt = geometry_context["receipt"]
                            if scope_entered and (not isinstance(receipt, dict) or receipt.get("restored") is not True):
                                raise FastIdleRestorationError("Stopped geometry restoration was not confirmed", receipt)
                        finally:
                            geometry_context["restore_seconds"] += time.monotonic()-restore_started
            check_deadline()
            enter("install")
            check_deadline()
            install_controller(sim, cfg, graph, str(output/"planned.ttt"))
            check_deadline()
            enter("execution")
            remaining = deadline-time.monotonic()
            check_deadline()
            measured = execute(sim, client, cfg, output, timeout_wall_s=max(.001, remaining), video=video)
            _save(output/"execution.json", measured)
            if measured.get("remote_interface_failure"):
                summary.update(remote_interface_failure=True, status="failed", phase="execution",
                               reason=measured.get("failure") or "实际执行期间远程接口连接中断",
                               remote_interface_errors=deepcopy(measured.get("remote_interface_errors", [])),
                               actual_makespan_seconds=measured.get("makespan"),
                               execution_wall_seconds=measured.get("wall_time_s"))
                _save(output/"validation.json", {"success": False, "violations": [
                    {"code": "remote_interface_failure", "detail": summary["reason"]}],
                    "scope": "远程接口中断，无法完成独立执行验收"})
                return summary
            check_deadline()
            enter("acceptance")
            audit = validate_execution(graph, measured, requirements)
            _save(output/"validation.json", audit)
            check_deadline()
            summary.update(actual_makespan_seconds=measured.get("makespan"),
                           execution_wall_seconds=measured.get("wall_time_s"),
                           step_durations=_step_times(graph, measured, plan_steps),
                           success=bool(measured.get("success") and audit["success"]))
            if measured.get("timeout"):
                summary.update(status="timeout", phase="execution", reason=measured.get("failure"))
            elif summary["success"]:
                enter("complete")
                summary.update(status="passed")
            else:
                summary.update(phase="acceptance", reason=measured.get("failure") or audit["violations"])
            return summary
    except FastIdleRestorationError as exc:
        previous = summary["phase"]
        geometry_context["receipt"] = exc.receipt
        summary.update(success=False, status="failed", phase="geometry_context_restore",
                       reason=f"{type(exc).__name__}: {exc}",
                       geometry_context_failure=True, geometry_context_previous_phase=previous)
        if is_transport_failure(exc):
            summary.update(remote_interface_failure=True, remote_interface_errors=[{
                "stage": "geometry_context_restore", "type": type(exc).__name__, "message": str(exc)}])
        return summary
    except TimeoutError as exc:
        summary.update(success=False, status="timeout", reason=f"TimeoutError: {exc}")
        return summary
    except Exception as exc:
        summary["success"] = False
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        if is_transport_failure(exc):
            cause = exc.__cause__ or exc.__context__
            summary.update(remote_interface_failure=True, remote_interface_errors=[{
                "stage": summary["phase"], "type": type(exc).__name__,
                "module": type(exc).__module__, "message": str(exc),
                "cause": None if cause is None else f"{type(cause).__name__}: {cause}"}])
        return summary
    finally:
        stage_seconds[timed_phase] = stage_seconds.get(timed_phase, 0.)+time.monotonic()-stage_started
        if oracle is not None:
            summary["geometry_statistics"] = deepcopy(getattr(oracle, "stats", {}))
        else:
            summary.setdefault("geometry_statistics", {})
        summary["passed"] = bool(summary["success"])
        summary["failure_phase"] = None if summary["passed"] else summary["phase"]
        summary["executor_wall_seconds"] = time.monotonic()-started
        _save(output/"summary.json", summary)
