"""Build a collision-sampled TPG, execute actual dynamics, and audit evidence."""
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from apexmr.geometry import SampledConflictOracle, verify_rollout
from apexmr.motion import MotionPlanner
from apexmr.program import parameterize
from apexmr.runtime import execute, install_controller
from apexmr.tpg import build_tpg
from apexmr.validation import validate_execution
from .frames import Frames
from .errors import is_transport_failure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "skillnet_v1_3"


class _BudgetedPlanner:
    """Check a shared wall deadline around each geometry backend call."""

    def __init__(self, planner, check):
        self._planner, self._check = planner, check

    def __getattr__(self, name):
        value = getattr(self._planner, name)
        if not callable(value):
            return value
        def bounded(*args, **kwargs):
            self._check()
            answer = value(*args, **kwargs)
            self._check()
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
                 planning_seconds=0., compile_seconds=0.):
    """Return a persisted summary, including a phase/reason on every failure.

    An optional existing stopped-scene planner is reused. This function never
    owns the simulator process. The runner must convert failed summaries to a
    nonzero CLI exit code and retain control of its own process lifecycle.
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
            budgeted_planner = _BudgetedPlanner(planner, check_deadline)
            summary["phase"] = "tpg"
            tpg_started = time.monotonic()
            oracle = SampledConflictOracle(budgeted_planner, cfg, output/"geometry")

            def timed_oracle(first, second):
                check_deadline()
                answer = oracle(first, second)
                check_deadline()
                return answer

            graph = build_tpg(tasks, timed_oracle)
            check_deadline()
            parameterize(graph, vmax=.60, amax=1.20, jmax=8.)
            check_deadline()
            graph["validation_requirements"] = requirements
            _save(output/"tpg.json", graph)
            summary.update(tpg_seconds=time.monotonic()-tpg_started,
                           tpg_nodes=len(graph["nodes"]), tpg_edges=len(graph["edges"]),
                           nominal_makespan_seconds=graph["nominal_makespan"])
            summary["phase"] = "rollout"
            rollout = verify_rollout(graph, budgeted_planner, cfg, step_s=.05, outdir=output/"geometry")
            check_deadline()
            if not rollout["valid"]:
                summary["reason"] = "名义轨迹的运动、等待位姿或状态转移未通过几何验收"
                summary["rollout_failure"] = rollout.get("first_violation") or rollout.get("world_state_gaps")
                return summary
            graph["certificate"]["waiting_endpoints_verified"] = True
            graph["certificate"]["nominal_rollout_checked"] = True
            _save(output/"tpg.json", graph)
            summary["phase"] = "install"
            check_deadline()
            install_controller(sim, cfg, graph, str(output/"planned.ttt"))
            check_deadline()
            summary["phase"] = "execution"
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
            summary["phase"] = "acceptance"
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
                summary.update(status="passed", phase="complete")
            else:
                summary.update(phase="acceptance", reason=measured.get("failure") or audit["violations"])
            return summary
    except TimeoutError as exc:
        summary.update(status="timeout", reason=f"TimeoutError: {exc}")
        return summary
    except Exception as exc:
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        if is_transport_failure(exc):
            cause = exc.__cause__ or exc.__context__
            summary.update(remote_interface_failure=True, remote_interface_errors=[{
                "stage": summary["phase"], "type": type(exc).__name__,
                "module": type(exc).__module__, "message": str(exc),
                "cause": None if cause is None else f"{type(cause).__name__}: {cause}"}])
        return summary
    finally:
        summary["passed"] = bool(summary["success"])
        summary["failure_phase"] = None if summary["passed"] else summary["phase"]
        summary["executor_wall_seconds"] = time.monotonic()-started
        _save(output/"summary.json", summary)
