"""Actual HiGHS MILP allocation for scene3 support and transport modes.

``solve_allocation(problem, banned_modes=(), banned_combinations=())`` accepts
geometrically qualified finite modes. Support modes choose a fixture position,
removal order and placing/removing arm. Task modes choose a task's arm/grasp.
Cross-arm supply/installation requires a qualified buffer handoff. The MILP
chooses these jointly with a peak-workload objective; it does not enumerate
complete solutions and call that ILP.

The result includes mandatory event dependencies for the motion compiler.
Buffer capacity is enforced by serializing deposit/pick intervals through
those dependencies. This allocator does NOT claim to solve continuous-time
scheduling, certify paths, or predict actual execution makespan.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


ARMS = ("A", "B")
TOL = 1e-6


def _number(value, label, minimum=0.0):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _items(value, label):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return list(value)


def _records(value, label):
    records = _items(value, label)
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError(f"{label} must contain mappings")
    return [dict(record) for record in records]


def _boolean(record, key, label):
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


def solve_allocation(problem, banned_modes=(), banned_combinations=()):
    """Solve joint support/task mode allocation and return a JSON-safe audit.

    Required keys:
      support_modes: id, candidate, removal_order, placer_arm, remover_arm,
        feasible, cost, rated_load_N, contact_overlap_m, blocked_after_gate.
      tasks: id, kind (supply/install/gate), part and optional allowed_arms.
      task_modes: id, task, arm, feasible, cost, optional grasp and
        compatible_support_modes (IDs).
      handoffs: part, supply_task, install_task, from_arm, to_arm, feasible,
        cost, optional id/deposit_event/pick_event.

    Scene defaults: A initially holds the beam; K1/K2 supply is A-exclusive;
    assembly order K1 then K2; one shared buffer. Input geometry flags must
    come from the caller's IK/collision qualification. A false or absent
    handoff qualification cannot silently become a feasible transport mode.
    Retrying after a motion failure can ban a single mode or the joint set of
    mode IDs returned in selected_mode_ids, without excluding unrelated modes.
    """
    started = time.perf_counter()
    if not isinstance(problem, Mapping):
        raise ValueError("problem must be a mapping")
    arms = _items(problem.get("arms", list(ARMS)), "arms")
    if sorted(arms) != list(ARMS):
        raise ValueError("scene3 requires exactly arms A and B")
    initial_holder = problem.get("initial_holder", "A")
    if initial_holder not in arms:
        raise ValueError("initial_holder must be A or B")
    if problem.get("buffer_capacity", 1) != 1:
        raise ValueError("scene3 allocator currently implements buffer_capacity=1")
    requirement = _number(problem.get("required_support_load_N", 25.86305357142857),
                          "required_support_load_N")
    overlap = _number(problem.get("minimum_contact_overlap_m", .010),
                      "minimum_contact_overlap_m")
    balance = _number(problem.get("load_balance_weight", .15), "load_balance_weight")
    unit = _name(problem.get("cost_unit", "proxy_cost"), "cost_unit")
    time_limit = _number(problem.get("time_limit_s", 30.), "time_limit_s")
    gap_target = _number(problem.get("mip_rel_gap", 0.), "mip_rel_gap")
    if time_limit <= 0:
        raise ValueError("time_limit_s must be positive")
    supports = _records(problem.get("support_modes"), "support_modes")
    tasks = _records(problem.get("tasks"), "tasks")
    task_modes = _records(problem.get("task_modes"), "task_modes")
    handoffs = _records(problem.get("handoffs", []), "handoffs")
    task_by_id = {}
    for task in tasks:
        name = _name(task.get("id"), "task.id")
        if name in task_by_id:
            raise ValueError("duplicate task id")
        if task.get("kind") not in {"supply", "install", "gate"}:
            raise ValueError(f"{name}: unknown task kind")
        if task["kind"] != "gate":
            _name(task.get("part"), name + ".part")
        allowed = _items(task.get("allowed_arms", arms), name + ".allowed_arms")
        if any(a not in arms for a in allowed):
            raise ValueError("allowed_arms contains an unknown arm")
        task["allowed_arms"] = allowed
        task_by_id[name] = task
    supplies = {t["part"]: t["id"] for t in tasks if t["kind"] == "supply"}
    installs = {t["part"]: t["id"] for t in tasks if t["kind"] == "install"}
    if (len(supplies) != sum(t["kind"] == "supply" for t in tasks) or
            len(installs) != sum(t["kind"] == "install" for t in tasks)):
        raise ValueError("a part must have at most one supply and one install task")
    if set(supplies) != set(installs):
        raise ValueError("every supplied part needs one installation task and vice versa")
    assembly_order = _items(problem.get("assembly_order", list(supplies)), "assembly_order")
    if len(assembly_order) != len(set(assembly_order)) or set(assembly_order) != set(supplies):
        raise ValueError("assembly_order must list each supplied part once")
    exclusive = problem.get("supply_access", {part: ["A"] for part in supplies})
    if not isinstance(exclusive, Mapping):
        raise ValueError("supply_access must be a mapping")
    # The default scene's two source parts remain A-exclusive even when a caller
    # accidentally marks a B endpoint as IK-reachable.
    access = {}
    for part in supplies:
        access[part] = _items(exclusive.get(part, ["A"]), f"supply_access.{part}")
        if any(a not in arms for a in access[part]):
            raise ValueError("supply_access contains an unknown arm")
        if part in {"K1", "K2"}:
            access[part] = [a for a in access[part] if a == "A"]
    initial_work = problem.get("initial_arm_workloads", {})
    if not isinstance(initial_work, Mapping):
        raise ValueError("initial_arm_workloads must be a mapping")
    initial_work = {a: _number(initial_work.get(a, 0.), f"initial_arm_workloads.{a}") for a in arms}
    variables, rejected, indexes = [], [], {}

    def add_variable(record, group, workload, reasons):
        name = _name(record.get("id"), group + ".id")
        if name == "__peak_workload":
            raise ValueError("__peak_workload is a reserved variable name")
        if name in indexes:
            raise ValueError(f"duplicate mode id: {name}")
        cost = _number(record.get("cost"), name + ".cost")
        if "arm_workload" in record:
            supplied = record["arm_workload"]
            if not isinstance(supplied, Mapping) or any(a not in arms for a in supplied):
                raise ValueError("arm_workload must map A/B to nonnegative costs")
            workload = {a: _number(supplied.get(a, 0.), name + ".arm_workload") for a in arms}
        indexes[name] = len(variables)
        variables.append({"id": name, "group": group, "record": record,
                          "cost": cost, "arm_workload": workload,
                          "upper_bound": 0.0 if reasons else 1.0})
        rejected.extend({"mode_id": name, "reason": reason} for reason in reasons)

    for mode in supports:
        name = _name(mode.get("id"), "support.id")
        _name(mode.get("candidate"), name + ".candidate")
        if mode.get("removal_order") not in {"before_gate", "after_gate"}:
            raise ValueError("removal_order must be before_gate or after_gate")
        if mode.get("placer_arm") not in arms or mode.get("remover_arm") not in arms:
            raise ValueError("support placer/remover must be A or B")
        feasible = _boolean(mode, "feasible", name)
        blocked = _boolean(mode, "blocked_after_gate", name)
        load = _number(mode.get("rated_load_N"), name + ".rated_load_N")
        contact = _number(mode.get("contact_overlap_m"), name + ".contact_overlap_m")
        cost = _number(mode.get("cost"), name + ".cost")
        reasons = []
        if not feasible: reasons.append("support_mode_failed_geometry_qualification")
        if mode["placer_arm"] == initial_holder: reasons.append("initial_holder_cannot_place_support")
        if load + 1e-9 < requirement: reasons.append("insufficient_support_load_rating")
        if contact + 1e-9 < overlap: reasons.append("insufficient_contact_overlap")
        if blocked and mode["removal_order"] == "after_gate": reasons.append("closed_gate_traps_support")
        workload = {a: 0.0 for a in arms}
        workload[mode["placer_arm"]] += cost / 2
        workload[mode["remover_arm"]] += cost / 2
        add_variable(mode, "support", workload, reasons)
    support_ids = {m["id"] for m in supports}
    for mode in task_modes:
        name = _name(mode.get("id"), "task_mode.id")
        if mode.get("task") not in task_by_id or mode.get("arm") not in arms:
            raise ValueError(f"{name}: unknown task or arm")
        feasible = _boolean(mode, "feasible", name)
        task = task_by_id[mode["task"]]
        reasons = []
        if not feasible: reasons.append("task_mode_failed_geometry_qualification")
        if mode["arm"] not in task["allowed_arms"]: reasons.append("task_arm_not_allowed")
        if task["kind"] == "supply" and mode["arm"] not in access[task["part"]]:
            reasons.append("A_exclusive_supply_access")
        if "compatible_support_modes" in mode:
            compatible = _items(mode["compatible_support_modes"], name + ".compatible_support_modes")
            if any(m not in support_ids for m in compatible):
                raise ValueError("compatible_support_modes references an unknown support mode")
        cost = _number(mode.get("cost"), name + ".cost")
        add_variable(mode, "task", {a: cost if a == mode["arm"] else 0. for a in arms}, reasons)
    handoff_pairs = {}
    for handoff in handoffs:
        part = handoff.get("part")
        if part not in supplies:
            raise ValueError("handoff references an unknown part")
        if handoff.get("supply_task") != supplies[part] or handoff.get("install_task") != installs[part]:
            raise ValueError("handoff supply/install task does not match its part")
        source, target = handoff.get("from_arm"), handoff.get("to_arm")
        if source not in arms or target not in arms or source == target:
            raise ValueError("handoff needs two different known arms")
        key = (part, source, target)
        if key in handoff_pairs:
            raise ValueError("duplicate handoff arm pair for a part")
        handoff.setdefault("id", f"handoff_{part}_{source}_{target}")
        handoff.setdefault("deposit_event", f"deposit_{part}")
        handoff.setdefault("pick_event", f"pick_{part}")
        _name(handoff["deposit_event"], "deposit_event")
        _name(handoff["pick_event"], "pick_event")
        feasible = _boolean(handoff, "feasible", handoff["id"])
        cost = _number(handoff.get("cost"), handoff["id"] + ".cost")
        add_variable(handoff, "handoff", {a: cost/2 for a in arms},
                     [] if feasible else ["buffer_handoff_failed_geometry_qualification"])
        handoff_pairs[key] = handoff["id"]
    bans = set(_items(banned_modes, "banned_modes")) | set(_items(problem.get("banned_modes", []), "banned_modes"))
    if not bans <= set(indexes):
        raise ValueError("banned_modes references an unknown mode")
    for name in bans:
        variables[indexes[name]]["upper_bound"] = 0.0
        rejected.append({"mode_id": name, "reason": "motion_planning_retry_ban"})
    combinations = (_items(banned_combinations, "banned_combinations") +
                    _items(problem.get("banned_combinations", []), "banned_combinations"))
    combinations = [_items(c, "banned combination") for c in combinations]
    for combination in combinations:
        if not combination or len(set(combination)) != len(combination) or not set(combination) <= set(indexes):
            raise ValueError("a banned combination must contain distinct known mode IDs")

    n = len(variables)
    matrix, lower, upper, descriptions = [], [], [], []

    def constraint(label, coefficients, lo=-np.inf, hi=np.inf):
        row = np.zeros(n + 1)
        for name, value in coefficients.items():
            row[n if name == "__peak_workload" else indexes[name]] += value
        matrix.append(row); lower.append(lo); upper.append(hi)
        descriptions.append({"constraint": label, "coefficients": coefficients,
                             "lower": None if not math.isfinite(lo) else lo,
                             "upper": None if not math.isfinite(hi) else hi})

    constraint("exactly_one_support_mode", {m: 1. for m in support_ids}, 1., 1.)
    task_arm_modes = {}
    for task in tasks:
        matching = [m for m in task_modes if m["task"] == task["id"]]
        constraint("exactly_one_task_mode:" + task["id"], {m["id"]: 1. for m in matching}, 1., 1.)
        for arm in arms:
            task_arm_modes[(task["id"], arm)] = [m["id"] for m in matching if m["arm"] == arm]
    for mode in task_modes:
        if "compatible_support_modes" in mode:
            coefficients = {mode["id"]: 1.}
            coefficients.update({s: -1. for s in mode["compatible_support_modes"]})
            constraint("task_support_compatibility:" + mode["id"], coefficients, hi=0.)
    for part in assembly_order:
        for source, target in [("A", "B"), ("B", "A")]:
            supply_vars = task_arm_modes[(supplies[part], source)]
            install_vars = task_arm_modes[(installs[part], target)]
            pair = handoff_pairs.get((part, source, target))
            coefficients = {v: 1. for v in supply_vars + install_vars}
            if pair is None:
                constraint(f"unqualified_cross_arm_transport:{part}:{source}:{target}", coefficients, hi=1.)
            else:
                # h == supplier(source) AND installer(target).
                coefficients[pair] = -1.
                constraint("handoff_required:" + pair, coefficients, hi=1.)
                constraint("handoff_supplier:" + pair,
                           {pair: 1., **{v: -1. for v in supply_vars}}, hi=0.)
                constraint("handoff_installer:" + pair,
                           {pair: 1., **{v: -1. for v in install_vars}}, hi=0.)
    for arm in arms:
        coefficients = {v["id"]: v["arm_workload"][arm] for v in variables}
        coefficients["__peak_workload"] = -1.
        constraint("peak_arm_workload:" + arm, coefficients, hi=-initial_work[arm])
    for index, combination in enumerate(combinations):
        constraint(f"no_good_motion_retry:{index}", {m: 1. for m in combination}, hi=float(len(combination)-1))
    objective = np.array([v["cost"] for v in variables] + [balance])
    bound_upper = np.array([v["upper_bound"] for v in variables] + [np.inf])
    array = np.asarray(matrix)
    lo, hi = np.asarray(lower), np.asarray(upper)
    base = {"solver": "scipy.optimize.milp / HiGHS", "scipy_version": scipy.__version__,
            "model": "joint binary support/task/handoff modes with peak workload auxiliary",
            "cost_unit": unit, "objective_definition": "sum(selected mode costs) + load_balance_weight * peak_arm_workload",
            "objective_is_makespan": False, "load_balance_weight": balance,
            "required_support_load_N": requirement, "minimum_contact_overlap_m": overlap,
            "assembly_order": assembly_order, "buffer_capacity": 1,
            "buffer_capacity_enforcement": "mandatory pick(previous)->deposit(next) dependencies emitted for the motion compiler",
            "geometry_scope": "Caller-provided candidate qualification; endpoints do not certify connecting paths.",
            "constraint_count": len(matrix), "constraints": descriptions,
            "binary_variable_count": n, "continuous_variable_count": 1,
            "rejected_modes": rejected, "retry_banned_modes": sorted(bans),
            "retry_banned_combinations": combinations}
    result = milp(objective, integrality=np.array([1]*n+[0]),
                  bounds=Bounds(np.zeros(n+1), bound_upper),
                  constraints=LinearConstraint(csc_matrix(array), lo, hi),
                  options={"time_limit": time_limit, "mip_rel_gap": gap_target, "presolve": True})

    def finite(value):
        return None if value is None or not math.isfinite(float(value)) else float(value)

    audit = {**base, "solver_called": True, "solver_status": int(result.status),
             "solver_message": str(result.message), "mip_gap": finite(getattr(result, "mip_gap", None)),
             "mip_dual_bound": finite(getattr(result, "mip_dual_bound", None)),
             "mip_node_count": finite(getattr(result, "mip_node_count", None)),
             "elapsed_s": time.perf_counter() - started}
    failed = {"solved": False, "optimal": False, "objective": None,
              "choices": [], "selected_support": None, "handoffs": [], "selected_mode_ids": []}
    if result.x is None:
        return {**audit, **failed, "status": {1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "solver_error")}
    vector = np.asarray(result.x)
    residual = array @ vector
    valid = (np.all(np.isfinite(vector)) and np.all(vector >= -TOL) and
             np.all(vector <= bound_upper + TOL) and
             np.all(np.abs(vector[:n] - np.round(vector[:n])) <= TOL) and
             np.all(residual >= lo - TOL) and np.all(residual <= hi + TOL))
    chosen = [v for v, x in zip(variables, vector[:n]) if x > .5]
    chosen_supports = [v["record"] for v in chosen if v["group"] == "support"]
    chosen_tasks = [v["record"] for v in chosen if v["group"] == "task"]
    chosen_handoffs = [v["record"] for v in chosen if v["group"] == "handoff"]
    valid = valid and len(chosen_supports) == 1 and all(
        sum(m["task"] == t["id"] for m in chosen_tasks) == 1 for t in tasks)
    if not valid or result.status not in (0, 1):
        return {**audit, **failed, "status": "invalid_incumbent"}
    selected_support = chosen_supports[0]
    by_task = {m["task"]: m for m in chosen_tasks}
    chosen_handoffs.sort(key=lambda h: assembly_order.index(h["part"]))
    buffer_edges = []
    for handoff in chosen_handoffs:
        buffer_edges.append({"from": handoff["deposit_event"], "to": handoff["pick_event"],
                             "reason": "buffer_material_handoff"})
    for previous, following in zip(chosen_handoffs, chosen_handoffs[1:]):
        buffer_edges.append({"from": previous["pick_event"], "to": following["deposit_event"],
                             "reason": "shared_buffer_capacity_one"})
    lifecycle_edges = [
        {"from": "place_support", "to": "verify_support_load"},
        {"from": "verify_support_load", "to": "release_A"},
        {"from": "verify_permanent_structure", "to": "close_gate"},
        {"from": "verify_permanent_structure", "to": "unload_support"},
        {"from": "unload_support", "to": "extract_support"},
        {"from": "extract_support", "to": "return_support"}]
    lifecycle_edges += [{"from": "release_A", "to": supplies[p]} for p in assembly_order]
    lifecycle_edges += [{"from": supplies[p], "to": installs[p]} for p in assembly_order]
    lifecycle_edges += [{"from": installs[p], "to": installs[q]}
                       for p, q in zip(assembly_order, assembly_order[1:])]
    if assembly_order:
        lifecycle_edges.append({"from": installs[assembly_order[-1]], "to": "verify_permanent_structure"})
    lifecycle_edges.append({"from": "return_support", "to": "close_gate"}
                           if selected_support["removal_order"] == "before_gate" else
                           {"from": "close_gate", "to": "unload_support"})
    workloads = {a: initial_work[a] + sum(v["arm_workload"][a] for v in chosen) for a in arms}
    true_objective = sum(v["cost"] for v in chosen) + balance * max(workloads.values())
    return {**audit, "solved": True, "optimal": result.status == 0,
            "status": "optimal" if result.status == 0 else "feasible_limit",
            "objective": float(true_objective), "solver_objective": finite(result.fun),
            "selected_support": selected_support,
            "choices": [by_task[t["id"]] for t in tasks],
            "task_assignments": {task: mode["arm"] for task, mode in by_task.items()},
            "handoffs": chosen_handoffs,
            "required_buffer_edges": buffer_edges, "required_lifecycle_edges": lifecycle_edges,
            "selected_mode_ids": [v["id"] for v in chosen],
            "arm_workloads": workloads, "peak_arm_workload": max(workloads.values()),
            "peak_auxiliary_value": float(vector[n]),
            "incumbent_validation": {"integrality": True, "bounds": True,
                                     "all_linear_constraints": True, "exact_task_coverage": True},
            "variables": [{"id": v["id"], "group": v["group"], "value": float(x),
                           "upper_bound": v["upper_bound"], "cost": v["cost"]}
                          for v, x in zip(variables, vector[:n])]}
