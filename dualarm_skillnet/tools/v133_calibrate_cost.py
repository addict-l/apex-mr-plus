"""Fit recorded real timings using only the standard library; run no planner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sim_v1_3/v133"
REFERENCE_SPEED = .25
SOURCE_ROOTS = ("results/sim_v1_3/m3_attempt_5", "results/sim_v1_3/overnight/m4",
                "results/sim_v1_3/overnight/sim_selection_1", "results/sim_v1_3/overnight/sim_selection_2")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def real_passed(data):
    actual = data.get("actual_makespan_seconds")
    return (data.get("profile") == "skillnet_v1_3" and data.get("status") == "passed"
            and data.get("passed") is True and data.get("success") is True
            and isinstance(data.get("simulator_pid"), int) and data["simulator_pid"] > 0
            and data.get("simulator_closed") is True and isinstance(actual, (int, float))
            and math.isfinite(actual) and actual > 0)


def collect_samples():
    library = tomllib.loads((ROOT / "skills/blocks_v1.toml").read_text(encoding="utf-8"))
    skills = {row["name"]: row for row in library["skill"]}
    samples, sources = [], []
    paths = sorted({path for folder in SOURCE_ROOTS for path in (ROOT/folder).rglob("summary.json")})
    for path in paths:
        summary = read(path)
        if not real_passed(summary):
            continue
        task_path = Path(summary.get("task_file", ""))
        if not task_path.is_file():
            name = summary["task"]
            task_path = next(p for p in (ROOT/"tasks"/(name+".toml"), ROOT/"tasks/sim"/(name+".toml")) if p.is_file())
        task = tomllib.loads(task_path.read_text(encoding="utf-8-sig"))
        plan_path = path.with_name("plan.json")
        plan = read(plan_path)
        positions = {row["id"]: row["pos"] for key in ("location", "slot") for row in task.get(key, [])}
        positions.update({"home_"+row["id"]: row["home"] for row in task["arm"]})
        speeds = {row["id"]: row["speed"] for row in task["arm"]}
        begin = len(samples)
        for observed in summary.get("step_durations", []):
            step = plan["schedule"][observed["step"]-1]
            if step["action"] != observed["action"]:
                raise ValueError("真实步骤与计划动作不匹配："+str(path))
            name, arguments = observed["action"].split("(", 1)
            binding = dict(zip(skills[name]["params"], (a.strip() for a in arguments[:-1].split(","))))
            distances, travel = {}, []
            for arm_var, place_var in skills[name].get("move_to", {}).items():
                arm, destination = binding[arm_var[1:]], binding[place_var[1:]]
                fact = next(f for f in step["state_before"] if f.startswith("arm_at("+arm+","))
                origin = fact.split(",", 1)[1].strip()[:-1]
                distances[arm] = math.dist(positions[origin], positions[destination])
                if speeds[arm] <= 0:
                    raise ValueError("来源任务的速度必须大于0")
                travel.append(distances[arm]/speeds[arm])
            original = skills[name]["op_time"]+max(travel, default=0.)
            if abs(original-observed["estimated_seconds"]) > 1e-6:
                raise ValueError("独立重算旧估计与保存值不符："+str(path))
            actual = observed["actual_seconds"]
            if not math.isfinite(actual) or actual <= 0:
                raise ValueError("真实动作时长必须为有限正数")
            samples.append({"source_summary": relative(path), "source_plan": relative(plan_path),
                "task_file": relative(task_path), "task": plan["task"], "step": observed["step"],
                "action": observed["action"], "skill": name, "arms": step["arms"],
                "distance_m": max(distances.values(), default=0.), "arm_distances_m": distances,
                "speeds_m_s": {arm: speeds[arm] for arm in distances}, "op_time": skills[name]["op_time"],
                "old_estimated_seconds": observed["estimated_seconds"], "actual_seconds": actual})
        sources.append({"source_summary": relative(path), "source_plan": relative(plan_path),
            "task_file": relative(task_path), "task": plan["task"], "sample_count": len(samples)-begin,
            "actual_makespan_seconds": summary["actual_makespan_seconds"]})
    if len({(row["source_summary"], row["step"]) for row in samples}) != len(samples):
        raise ValueError("重复真实步骤不能计为多个样本")
    if len(samples) < 30:
        raise ValueError("真实标定步骤少于30条")
    return skills, samples, sources


def constrained_fit(samples, constraints):
    """Exact 2D convex least squares over half-planes A*a+B*b >= C."""
    x = [row["distance_m"] for row in samples]
    y = [row["actual_seconds"] for row in samples]
    candidates = []
    variance = sum((value-mean(x))**2 for value in x)
    if variance > 1e-15:
        b = sum((value-mean(x))*(actual-mean(y)) for value, actual in zip(x, y))/variance
        candidates.append((mean(y)-b*mean(x), b))
    for A, B, C in constraints:
        point = (C/A, 0.) if abs(A) >= abs(B) else (0., C/B)
        direction = (-B, A)
        features = [direction[0]+direction[1]*value for value in x]
        denominator = sum(value*value for value in features)
        if denominator:
            t = sum(feature*(actual-point[0]-point[1]*value)
                    for feature, value, actual in zip(features, x, y))/denominator
            candidates.append((point[0]+t*direction[0], point[1]+t*direction[1]))
    for i, (A, B, C) in enumerate(constraints):
        for D, E, F in constraints[i+1:]:
            determinant = A*E-B*D
            if abs(determinant) > 1e-15:
                candidates.append(((C*E-B*F)/determinant, (A*F-C*D)/determinant))
    feasible = [(a, b) for a, b in candidates
                if all(A*a+B*b >= C-1e-9 for A, B, C in constraints)]
    if not feasible:
        raise ValueError("标定约束无可行解")
    return min(feasible, key=lambda pair: sum((actual-pair[0]-pair[1]*value)**2 for value, actual in zip(x, y)))


def route_constraints(pick, place, margin=.25):
    source, home, buffer, goal = (.4, -.2, 0.), (-.3, .3, .2), (0., -.3, 0.), (-.15, .2, 0.)
    far, near = (0., 1.2, .2), (0., .1, .2)
    duration = lambda fit, d: fit[0]+fit[1]*d
    buffer_cost = (duration(place, math.dist(source, buffer))+duration(pick, math.dist(home, buffer))
                   +duration(place, math.dist(buffer, goal)))
    far_d, near_d = (max(math.dist(source, point), math.dist(home, point)) for point in (far, near))
    far_floor = buffer_cost-duration(place, math.dist(far, goal))+margin
    near_ceiling = buffer_cost-duration(place, math.dist(near, goal))-margin
    return [(1., far_d, far_floor), (-1., -near_d, -near_ceiling)], {
        "source": "旧near/far任务的两条可行路线，扣除共同B取料；不调用规划器或被测estimate",
        "minimum_ordering_margin_seconds": margin, "buffer_after_common_pick_seconds": buffer_cost,
        "far_handover_distance_m": far_d, "near_handover_distance_m": near_d,
        "far_handover_min_seconds": far_floor, "near_handover_max_seconds": near_ceiling}


def build_calibration():
    raw, samples, sources = collect_samples()
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["skill"]].append(row)
    fits, metadata = {}, {}
    for name in ("pick", "place"):
        fits[name] = constrained_fit(grouped[name], [(1., 0., raw[name]["op_time"]), (0., 1., 4.)])
        metadata[name] = {"status": "fitted", "fit_constraints": ["a>=原op_time", "b>=4秒/米"]}
    ordering, ordering_evidence = route_constraints(fits["pick"], fits["place"])
    fits["handover"] = constrained_fit(grouped["handover"], [(1., 0., raw["handover"]["op_time"]), (0., 1., 4.), *ordering])
    metadata["handover"] = {"status": "fitted_sparse_order_constrained",
        "fit_constraints": ["a>=原op_time", "b>=4秒/米", "near交接优于buffer，far交接劣于buffer至少0.25秒"],
        "limitation": "仅3条/2种距离；不宣称独立识别了远距离斜率"}
    # Three inserts from one tower confound arm parking with distance. Keep all
    # observations and shrink the slope to the well-sampled pick/place prior.
    prior_b = (fits["pick"][1]+fits["place"][1])/2
    fits["insert"] = (max(raw["insert"]["op_time"], mean(row["actual_seconds"]-prior_b*row["distance_m"]
                                                   for row in grouped["insert"])), prior_b)
    metadata["insert"] = {"status": "sparse_slope_prior",
        "fit_constraints": ["保留全部3条实测，不剔除31.1秒样本", "b固定为pick/place斜率均值>=4，仅拟合非负额外截距"],
        "limitation": "仅1个塔任务，含异臂/停车开销；无约束OLS负斜率未采用"}
    table = {}
    for name, definition in raw.items():
        data = grouped[name]
        a, b = fits.get(name, (definition["op_time"], 4.))
        fixed, movement = max(0., a-definition["op_time"]), max(1., b*REFERENCE_SPEED)
        predictions = [a+b*row["distance_m"] for row in data]
        table[name] = {"reference_speed": REFERENCE_SPEED, "fixed_delta": fixed,
            "movement_multiplier": movement, "a_skill": definition["op_time"]+fixed,
            "b_skill": movement/REFERENCE_SPEED, "samples": len(data),
            "training_tasks": sorted({row["task"] for row in data}),
            **metadata.get(name, {"status": "uncalibrated", "fit_constraints": ["无真实成功样本，保持原模型"]}),
            "distance_range_m": [min((r["distance_m"] for r in data), default=None), max((r["distance_m"] for r in data), default=None)],
            "rmse_seconds": (mean((p-r["actual_seconds"])**2 for p, r in zip(predictions, data)))**.5 if data else None}
    per_source = []
    for source in sources:
        arm_end, structural_end = defaultdict(float), 0.
        data = [row for row in samples if row["source_summary"] == source["source_summary"]]
        for row in data:
            parameters = table[row["skill"]]
            duration = (row["op_time"]+parameters["fixed_delta"]+parameters["movement_multiplier"]
                        *max((distance/row["speeds_m_s"][arm] for arm, distance in row["arm_distances_m"].items()), default=0.))
            start = max((arm_end[arm] for arm in row["arms"]), default=0.)
            if raw[row["skill"]].get("structural"):
                start = max(start, structural_end)
                structural_end = start+duration
            for arm in row["arms"]:
                arm_end[arm] = start+duration
        estimate = max(arm_end.values(), default=0.)
        per_source.append({**source, "replayed_estimate_seconds": estimate,
            "estimate_to_actual": estimate/source["actual_makespan_seconds"]})
    # Step timings omit cross-arm waits introduced later by the collision TPG.
    # Impose the requested task-level interval jointly, with one common scale
    # rather than separate per-task corrections. This remains a linear cost
    # for every skill and preserves all existing route-order inequalities.
    ratio_floor = max(.7/row["estimate_to_actual"] for row in per_source)
    ratio_ceiling = min(1.3/row["estimate_to_actual"] for row in per_source)
    coverage_scale = max(1., ratio_floor)+1e-12
    if coverage_scale > ratio_ceiling:
        raise ValueError("单一共享标定比例不能满足训练任务的0.7–1.3区间，不能伪造通过")
    if any(row["skill"] not in fits for row in samples):
        raise ValueError("训练重计时包含未标定技能，需要重新核算共同覆盖约束")
    for name, (a, b) in fits.items():
        parameters = table[name]
        parameters["unscaled_a_skill"], parameters["unscaled_b_skill"] = a, b
        parameters["coverage_scale"] = coverage_scale
        parameters["a_skill"], parameters["b_skill"] = a*coverage_scale, b*coverage_scale
        parameters["fixed_delta"] = parameters["a_skill"]-raw[name]["op_time"]
        parameters["movement_multiplier"] = parameters["b_skill"]*REFERENCE_SPEED
        parameters["fit_constraints"].append("共享非缩小比例满足14个训练来源的资源重计时/实际处于0.7–1.3；不作独立泛化声明")
        parameters["rmse_seconds"] = mean((parameters["a_skill"]+parameters["b_skill"]*row["distance_m"]
                                            -row["actual_seconds"])**2 for row in grouped[name])**.5
    for source in per_source:
        source["unscaled_estimate_to_actual"] = source["estimate_to_actual"]
        source["replayed_estimate_seconds"] *= coverage_scale
        source["estimate_to_actual"] *= coverage_scale
    ordering_evidence["coverage_scale"] = coverage_scale
    ordering_evidence["values_before_shared_scale"] = True
    for key in ("minimum_ordering_margin_seconds", "buffer_after_common_pick_seconds",
                "far_handover_min_seconds", "near_handover_max_seconds"):
        ordering_evidence["final_"+key] = ordering_evidence[key]*coverage_scale
    return {"schema_version": 1, "version": "v1.3.3-cost", "reference_speed": REFERENCE_SPEED,
        "total_samples": len(samples), "samples": samples, "sources": sources,
        "training_tasks": sorted({row["task"] for row in samples}), "skills": table,
        "ordering_constraints": ordering_evidence,
        "shared_coverage_constraint": {"scale": coverage_scale, "feasible_interval": [max(1., ratio_floor), ratio_ceiling],
            "target_interval": [.7, 1.3], "limitation": "补偿训练任务的TPG跨臂等待；新任务仍需独立真实验证，不保证泛化"},
        "lower_bound_contract": "new=action.op_time+fixed_delta+movement_multiplier*max(distance/current_speed)，fixed_delta>=0且multiplier>=1，逐动作不低于旧模型",
        "training_schedule_replay": per_source,
        "training_makespan_ratio_median": median(row["estimate_to_actual"] for row in per_source),
        "training_scope": "固定旧可行路线的资源顺序重计时，不运行搜索，不当作独立最优或held-out验收",
        "holdout_policy": "新sim20优先选择不在training_tasks中的任务，单独报告训练任务重叠"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅用旧真实成功步骤标定cost；不运行规划或仿真")
    parser.add_argument("--out", type=Path, default=BASE/"calibration.json")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if not output.is_relative_to(BASE.resolve()):
        parser.error("标定产物只能写入新的v133目录")
    report = build_calibration()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": relative(output), "total_samples": report["total_samples"],
        "skills": {k: {key: v[key] for key in ("a_skill", "b_skill", "samples", "status")} for k, v in report["skills"].items()},
        "training_ratio_median": report["training_makespan_ratio_median"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
