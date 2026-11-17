"""Validated generation and symbolic witness replay, never planner search."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from statistics import pstdev

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.cost.model import DistanceCost
from skillnet.domains.blocks import problem_from_data
from .families import FAMILIES, sample_family, validate_parameters
from .geometry import (POSSIBLE_RADIUS, VALID_RADIUS, check_generation_layout,
                       distance_prior_copy)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/sim_v1_3/v133/generated_samples"
DEFAULT_DIVERSITY_OUT = ROOT / "results/sim_v1_3/v133/diversity_samples"


class GenerationError(ValueError):
    def __init__(self, family, rejections):
        self.family, self.rejections = family, rejections
        super().__init__(f"{family} 在 {len(rejections)} 次候选内未生成合法任务：{rejections[-1]}")


def _output_directory(out):
    directory = Path(out).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("生成文件必须位于项目根目录内")
    return directory


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "["+", ".join(_toml_value(item) for item in value)+"]"
    if isinstance(value, dict):
        return "{"+", ".join(_toml_value(str(key))+" = "+_toml_value(item)
                              for key, item in value.items())+"}"
    raise TypeError(f"不支持的任务字段类型：{type(value).__name__}")


def task_to_toml(task):
    """Serialize the supported task schema without proposal/witness hints."""
    from simbridge.taskio import task_data_to_toml
    return task_data_to_toml(task)


def replay_witness(task, steps, *, allow_unknown=False):
    """Replay rules without search; UNKNOWN is conditional only if requested."""
    problem = task if hasattr(task, "hooks") else problem_from_data(task)
    engine, state = Engine(problem), problem.initial
    violation = engine.hooks.invariant(state)
    if violation:
        raise ValueError("初始状态不稳定："+str(violation))
    cost, estimate, trace, unknown = DistanceCost(problem.hooks), 0., [], set()
    for number, text in enumerate(steps, 1):
        action = engine.parse_action(text)
        failures = engine.failed(action, state)
        if failures:
            raise ValueError(f"见证第 {number} 步前提不满足：{text}：{failures}")
        status, evidence = engine.oracle.evaluate(action.capability)
        if status == Status.INVALID or (status == Status.UNKNOWN and not allow_unknown):
            raise ValueError(f"见证第 {number} 步能力不满足：{text}：{evidence}")
        unknown.update(str(item.fact) for item in evidence if item.status == Status.UNKNOWN)
        duration = cost.estimate(action, state)
        # Engine.execute checks both start and finish invariants. No search or
        # timer is involved, and the trace is stored outside the task input.
        state = engine.execute(action, state)
        estimate += duration
        trace.append({"step": number, "action": text, "estimate_seconds": duration,
                      "capability_status": status.value,
                      "state_after": [str(fact) for fact in sorted(state)]})
    if not engine.satisfied(engine.goal, state):
        raise ValueError("构造见证没有达到全部任务目标")
    return {"task": problem.name, "rules_replayed": True, "goal_satisfied": True,
            "initial_invariant_valid": True, "steps": list(steps), "trace": trace,
            "serial_witness_estimate_seconds": estimate,
            "all_capabilities_valid": not unknown, "conditional": bool(unknown),
            "qualification_needed": bool(unknown), "unknown_facts": sorted(unknown),
            "all_witness_capabilities_noninvalid": True,
            "search_performed": False, "wall_budget_experiment_performed": False,
            "independent_planner_solved": None, "optimal_makespan_seconds": None,
            "scope": "构造见证的前提、起止不变量和目标重放；UNKNOWN 能力仅条件成立，需后续资格确认；不是独立规划搜索、IK或动力学验收"}


def _write_new_or_identical(path, text):
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"保留已有不同内容，未覆盖：{path}")
    path.write_text(text, encoding="utf-8")


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"


def generate(family, seed, index=0, out=None, *, max_attempts=64, parameters=None, distance_prior=False):
    """Generate one reproducible task, retrying invalid geometry/rule samples.

    Optional parameters constrain documented family ranges for inspection.
    ``out=None`` returns data only. Any files are confined to this project.
    The family/index stream is stable across process hashes and request order.
    """
    family = str(family).upper()
    parameters = validate_parameters(family, parameters)
    for key, value in (("seed", seed), ("index", index), ("max_attempts", max_attempts)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} 必须为整数")
    if index < 0 or max_attempts < 1:
        raise ValueError("index 必须非负，max_attempts 必须大于零")
    destination = None if out is None else _output_directory(out)
    stream = hashlib.sha256(f"{family}:{seed}:{index}".encode("ascii")).digest()
    rng = random.Random(int.from_bytes(stream, "big"))
    seed_name = str(seed) if seed >= 0 else "n"+str(-seed)
    name = f"gen_{family.lower()}_s{seed_name}_{index:03d}"
    rejections = []
    for attempt in range(1, max_attempts+1):
        try:
            task, witness, sampled = sample_family(family, rng, name, parameters=parameters, index=index)
            if distance_prior:
                task = distance_prior_copy(task)
            problem = problem_from_data(task)
            if problem.hooks.invariant(problem.initial):
                raise ValueError("初始状态不稳定")
            layout = check_generation_layout(task, family, sampled)
            if not layout["valid"]:
                rejections.append({"attempt": attempt, "stage": "geometry",
                                   "violations": deepcopy(layout["violations"])})
                continue
            replay = replay_witness(problem, witness, allow_unknown=True)
        except (ValueError, AssertionError) as exc:
            rejections.append({"attempt": attempt, "stage": "load_or_witness",
                               "reason": str(exc)})
            continue
        report = {"family": family, "seed": seed, "index": index, "attempts": attempt,
                  "parameters": sampled, "rejections": rejections,
                  "task": task, "layout": layout, "witness": replay, "path": None,
                  "search_performed": False, "ik_performed": False, "dynamics_performed": False}
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            path = destination/(name+".toml")
            _write_new_or_identical(path, task_to_toml(task))
            _write_new_or_identical(destination/(name+".geometry.json"), _json(layout))
            _write_new_or_identical(destination/(name+".witness.json"), _json(replay))
            metadata = {key: value for key, value in report.items()
                        if key not in {"task", "layout", "witness", "path"}}
            metadata["task_file"] = str(path.relative_to(ROOT))
            _write_new_or_identical(destination/(name+".generation.json"), _json(metadata))
            report["path"] = str(path)
        return report
    raise GenerationError(family, rejections)


def generate_samples(seed=20261115, count=2, out=DEFAULT_OUT, *, distance_prior=False):
    """Generate samples only; all planning is constructive rule replay."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 40:
        raise ValueError("每族生成数必须在 1–40 之间；离线冒烟默认仅 2 个")
    destination = _output_directory(out)
    rows = []
    for family in FAMILIES:
        for index in range(count):
            result = generate(family, seed, index, destination, distance_prior=distance_prior)
            rows.append({"family": family, "seed": seed, "index": index,
                         "path": result["path"], "attempts": result["attempts"],
                         "parameters": result["parameters"],
                         "geometry_valid": result["layout"]["valid"],
                         "rules_replayed": result["witness"]["rules_replayed"],
                         "goal_satisfied": result["witness"]["goal_satisfied"],
                         "witness_steps": len(result["witness"]["steps"]),
                         "conditional": result["witness"]["conditional"],
                         "all_capabilities_valid": result["witness"]["all_capabilities_valid"],
                         "unknown_facts": result["witness"]["unknown_facts"],
                         "serial_witness_estimate_seconds": result["witness"]["serial_witness_estimate_seconds"],
                         "search_performed": False})
    summary = {"seed": seed, "per_family": count, "tasks_generated": len(rows), "samples": rows,
               "geometry_and_rule_replay_passed": all(row["geometry_valid"] and row["goal_satisfied"] for row in rows),
               "conditional_witnesses": sum(row["conditional"] for row in rows),
               "independent_search_performed": False, "wall_budget_experiments_performed": False,
               "real_ik_performed": False, "dynamics_performed": False,
               "note": "仅生成、纯几何检查和规则重放；UNKNOWN 保留为待确认，不算能力通过；未运行规划搜索或仿真任务"}
    _write_new_or_identical(destination/"smoke_summary.json", _json(summary))
    return summary


def generate_diversity(seed=20261115, per_family=10, out=DEFAULT_DIVERSITY_OUT, report_path=None):
    """Persist an untimed generation/property check, never a planning batch."""
    if per_family < 2:
        raise ValueError("离散程度检查每族至少需要 2 个任务")
    destination = _output_directory(out)
    summary = generate_samples(seed, per_family, destination)
    families, rows, failures = {}, [], []
    for family in FAMILIES:
        samples = [row for row in summary["samples"] if row["family"] == family]
        coordinates, spacings, com_margins = [], [], []
        minimum_reference, ranges = [], []
        for sample in samples:
            stem = Path(sample["path"]).with_suffix("")
            layout = json.loads(Path(str(stem)+".geometry.json").read_text(encoding="utf-8"))
            checks = layout["metrics"]["generator_checks"]
            coordinates.append(sample["parameters"]["primary_target_xy"])
            if checks["minimum_target_spacing_m"] is not None:
                spacings.append(checks["minimum_target_spacing_m"])
            com_margins.extend(item["minimum_margin_m"] for item in checks["assembled_com"].values())
            spacing = layout["metrics"]["minimum_reference_spacing_m"]
            if spacing is not None:
                minimum_reference.append(spacing)
            ranges.extend(item["horizontal_distance_m"] for item in layout["metrics"]["stock_base_ranges"].values())
            rows.append({key: sample[key] for key in ("family", "index", "path", "attempts", "geometry_valid",
                "rules_replayed", "conditional", "all_capabilities_valid", "unknown_facts")}
                        | {"primary_target_xy": coordinates[-1], "goal_satisfied": sample["goal_satisfied"]})
        sx, sy = (pstdev(point[axis] for point in coordinates) for axis in (0, 1))
        vector = (sx*sx+sy*sy)**.5
        checks = {"x_std_ge_0_05": sx >= .05, "y_std_ge_0_05": sy >= .05,
                  "vector_std_ge_0_05": vector >= .05, "all_witness_capabilities_noninvalid": True}
        families[family] = {"count": len(samples), "x_std_m": sx, "y_std_m": sy,
            "vector_std_m": vector, "minimum_target_spacing_m": min(spacings, default=None),
            "minimum_reference_spacing_m": min(minimum_reference, default=None),
            "stock_base_min_m": min(ranges, default=None), "stock_base_max_m": max(ranges, default=None),
            "minimum_assembled_com_margin_m": min(com_margins, default=None),
            "all_geometry_valid": all(row["geometry_valid"] for row in samples),
            "all_rules_replayed": all(row["rules_replayed"] and row["goal_satisfied"] for row in samples),
            "conditional_witnesses": sum(row["conditional"] for row in samples),
            "unknown_reach_facts": sum(len(row["unknown_facts"]) for row in samples), "checks": checks}
        for axis, value in (("x", sx), ("y", sy)):
            if value < .05:
                failures.append(f"{family} 的主目标/结构中心 {axis} 标准差 {value:.9f} m < 0.05 m")
    reason = ("F4 当前实际采样的共同区域 x 标准差未达 0.05 m；"
              "保留实测采样统计及 UNKNOWN，不把距离先验当作 IK 证据。")
    if not families["F4"]["checks"]["x_std_ge_0_05"]:
        failures.append(reason)
    exact = not failures
    report = {"seed": seed, "per_family": per_family, "tasks_generated": len(rows),
        "geometry_and_rule_replay_passed": summary["geometry_and_rule_replay_passed"],
        "capability_policy": {"distance": "XY reference to base (±0.70,0.13)",
            "valid_radius_m": VALID_RADIUS, "possible_radius_m": POSSIBLE_RADIUS,
            "reach_table_written": False, "distance_prior_requires_explicit_switch": True,
            "unknown_policy": "保留 UNKNOWN；条件见证仅做规则重放，不表示能力全部通过"},
        "exact_peraxis_contract_passed": exact, "exact_contract_passed": exact,
        "exact_contract_failure_reasons": failures,
        "y_and_vector_contract_passed": all(row["checks"]["y_std_ge_0_05"]
            and row["checks"]["vector_std_ge_0_05"] for row in families.values()),
        "families": families, "samples": rows,
        "independent_search_performed": False, "wall_budget_experiments_performed": False,
        "real_ik_performed": False, "dynamics_performed": False,
        "note": "50 个任务的无计时生成属性检查；不是 200 任务规划求解验收；每轴标准差合同未通过时如实保留失败"}
    target = destination.parent/"generator_diversity.json" if report_path is None else Path(report_path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("离散程度报告必须位于项目根目录内")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_new_or_identical(target, _json(report))
    return report
