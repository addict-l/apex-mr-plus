"""MILP allocation of assembly parts, grasp modes and supporting arms.

The caller supplies geometrically checked candidates. This module does not
replace an IK/path oracle or select an assembly sequence. In particular, a
feasible endpoint candidate is not a certificate for its connecting path.

For every part, choose exactly one mode (installer, grasp, supporting arm or
none). A mode cannot assign the same arm to installing and supporting. The
quasi-static input model requires support when

    (mass_kg * gravity_m_s2 + insertion_force_N) * lever_arm_m
        > connection_moment_limit_Nm.

This is a conservative screening model for the specified loading condition,
not a general rigid-body stability or dynamic contact solver. Its assumptions
and all rejected candidates are included in the returned audit record.

Objective = sum(chosen main cost + support cost)
            + load_balance_weight * maximum arm workload.

Every cost and workload uses the caller's common ``cost_unit`` (default:
``proxy_cost``). The balance weight is dimensionless. Each arm's workload
includes its installer cost and its support cost, as appropriate. The result
is an allocation cost, not the final asynchronously executed makespan.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


_SOLVER = "scipy.optimize.milp / HiGHS"


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sequence(value: Any, label: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return list(value)


def _vector(value: Any, dimension: int, label: str) -> list[float]:
    items = _sequence(value, label)
    if len(items) != dimension:
        raise ValueError(f"{label} must have {dimension} elements")
    result = []
    for item in items:
        if isinstance(item, (bool, str, bytes)):
            raise ValueError(f"{label} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain finite numbers")
        result.append(number)
    return result


def support_requirement(part: Mapping[str, Any], gravity: float = 9.81) -> dict:
    """Return the input load test; never silently honor a manual support flag.

    ``lever_arm`` must bound the load's moment arm at the relevant connection.
    ``connection_moment_limit`` is the assembly model's allowable moment.
    Equal required/allowable moment is accepted up to round-off only.
    """
    part_id = _name(part.get("id"), "part id")
    mass = _number(part.get("mass"), f"{part_id}.mass")
    insertion_force = _number(part.get("insertion_force", 0.0),
                              f"{part_id}.insertion_force")
    lever = _number(part.get("lever_arm", 0.0), f"{part_id}.lever_arm")
    limit = _number(part.get("connection_moment_limit", 0.0),
                    f"{part_id}.connection_moment_limit")
    gravity = _number(gravity, "gravity_m_s2")
    required = (mass * gravity + insertion_force) * lever
    if not math.isfinite(required):
        raise ValueError(f"{part_id}: computed load moment is not finite")
    tolerance = 1e-12 * max(1.0, required, limit)
    return {
        "part": part_id,
        "mass_kg": mass,
        "gravity_m_s2": gravity,
        "insertion_force_N": insertion_force,
        "lever_arm_m": lever,
        "required_moment_Nm": required,
        "connection_moment_limit_Nm": limit,
        "support_required": required > limit + tolerance,
    }


def solve_allocation(problem: Mapping[str, Any]) -> dict:
    """Solve allocation over finite IK-qualified assembly/support modes.

    Required problem fields are ``arms``, ``parts`` and ``candidates``. A
    feasible candidate contains part/arm/grasp/cost and optional
    ``support_candidates``. Each support candidate has arm/feasible/cost,
    ``tcp_position`` (world metres) and ``quaternion`` (xyzw unit quaternion).

    Returns a JSON-compatible audit record. Invalid input raises ValueError;
    an empty feasible set or solver infeasibility returns ``solved=False``.
    A time-limited incumbent is accepted only after independently checking
    integrality, part coverage and the linear constraints.
    """
    started = time.perf_counter()
    if not isinstance(problem, Mapping):
        raise ValueError("problem must be a mapping")
    arms = [_name(a, "arm") for a in _sequence(problem.get("arms"), "arms")]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be distinct and nonempty")
    parts = _sequence(problem.get("parts"), "parts")
    if any(not isinstance(p, Mapping) for p in parts):
        raise ValueError("parts must contain mappings")
    part_ids = [_name(p.get("id"), "part id") for p in parts]
    if len(set(part_ids)) != len(part_ids):
        raise ValueError("duplicate part id")
    by_part = dict(zip(part_ids, parts))
    gravity = _number(problem.get("gravity_m_s2", 9.81), "gravity_m_s2")
    weight = _number(problem.get("load_balance_weight", 0.15), "load_balance_weight")
    unit = _name(problem.get("cost_unit", "proxy_cost"), "cost_unit")
    time_limit = _number(problem.get("time_limit_s", 30.0), "time_limit_s")
    if time_limit <= 0:
        raise ValueError("time_limit_s must be positive")
    relative_gap = _number(problem.get("mip_rel_gap", 0.0), "mip_rel_gap")
    criteria = {p["id"]: support_requirement(p, gravity) for p in parts}
    grasps = {}
    for p in parts:
        values = _sequence(p.get("grasps", []), f"{p['id']}.grasps")
        if any(not isinstance(g, Mapping) for g in values):
            raise ValueError(f"{p['id']}.grasps must contain mappings")
        names = [_name(g.get("id"), f"{p['id']}.grasp.id") for g in values]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate grasp id for {p['id']}")
        grasps[p["id"]] = set(names)

    rejected = []
    modes = []
    candidates = _sequence(problem.get("candidates"), "candidates")

    def reject(index: int, reason: str, support_index: int | None = None):
        entry = {"candidate_index": index, "reason": reason}
        if support_index is not None:
            entry["support_candidate_index"] = support_index
        rejected.append(entry)

    def add_mode(index: int, candidate: Mapping, support: Mapping | None,
                 support_index: int | None):
        main_cost = _number(candidate.get("cost"), f"candidate {index}.cost")
        support_cost = 0.0 if support is None else _number(
            support.get("cost", 0.0), f"candidate {index} support {support_index}.cost")
        support_arm = None if support is None else support["arm"]
        workload = {arm: 0.0 for arm in arms}
        workload[candidate["arm"]] += main_cost
        if support_arm is not None:
            workload[support_arm] += support_cost
        support_position, support_quaternion = None, None
        if support is not None:
            support_position = _vector(support.get("tcp_position"), 3,
                                       f"candidate {index} support tcp_position")
            support_quaternion = _vector(support.get("quaternion"), 4,
                                         f"candidate {index} support quaternion")
            if not math.isclose(sum(q*q for q in support_quaternion), 1.0,
                                rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"candidate {index} support quaternion must be normalized xyzw")
        modes.append({
            "mode_id": f"m{len(modes):04d}",
            "candidate_index": index,
            "support_candidate_index": support_index,
            "part": candidate["part"],
            "arm": candidate["arm"],
            "grasp": candidate["grasp"],
            "support_arm": support_arm,
            "support_required": criteria[candidate["part"]]["support_required"],
            "support_tcp_position": support_position,
            "support_quaternion": support_quaternion,
            "main_cost": main_cost,
            "support_cost": support_cost,
            "mode_cost": main_cost + support_cost,
            "arm_workload": workload,
        })

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {index} must be a mapping")
        p = candidate.get("part")
        arm = candidate.get("arm")
        grasp = candidate.get("grasp")
        if p not in by_part:
            raise ValueError(f"candidate {index} references unknown part {p!r}")
        if arm not in arms:
            raise ValueError(f"candidate {index} references unknown arm {arm!r}")
        _name(grasp, f"candidate {index}.grasp")
        if grasps[p] and grasp not in grasps[p]:
            raise ValueError(f"candidate {index} references unknown grasp {grasp!r}")
        feasible = candidate.get("feasible")
        if not isinstance(feasible, bool):
            raise ValueError(f"candidate {index}.feasible must be a boolean")
        if not feasible:
            reject(index, "installer/grasp candidate is geometrically infeasible")
            continue
        _number(candidate.get("cost"), f"candidate {index}.cost")
        if not criteria[p]["support_required"]:
            add_mode(index, candidate, None, None)
        else:
            reject(index, "unsupported mode violates the specified load-moment limit")
        supports = _sequence(candidate.get("support_candidates", []),
                             f"candidate {index}.support_candidates")
        for support_index, support in enumerate(supports):
            if not isinstance(support, Mapping):
                raise ValueError(f"candidate {index} support {support_index} must be a mapping")
            support_arm = support.get("arm")
            if support_arm not in arms:
                raise ValueError(f"candidate {index} support references unknown arm {support_arm!r}")
            if not isinstance(support.get("feasible"), bool):
                raise ValueError(f"candidate {index} support feasible must be a boolean")
            if support_arm == arm:
                reject(index, "installer and support arm must be different", support_index)
                continue
            if not support["feasible"]:
                reject(index, "support candidate is geometrically infeasible", support_index)
                continue
            add_mode(index, candidate, support, support_index)

    base = {
        "solver": _SOLVER,
        "scipy_version": scipy.__version__,
        "model": "binary assembly modes with a continuous peak-workload auxiliary",
        "cost_unit": unit,
        "load_balance_weight": weight,
        "objective_definition": "sum(mode_cost * x) + load_balance_weight * peak_arm_workload",
        "objective_is_makespan": False,
        "support_criterion": {
            "formula": "(mass_kg * gravity_m_s2 + insertion_force_N) * lever_arm_m > connection_moment_limit_Nm",
            "scope": "quasi-static load screening at the supplied connection/load point",
            "assumption": "each feasible support candidate supplies the needed counter-support; caller must certify its geometry and capacity",
            "per_part": list(criteria.values()),
        },
        "constraints_summary": {
            "exactly_one_mode_per_part": len(parts),
            "arm_peak_workload_inequalities": len(arms),
            "installer_support_same_arm_forbidden": True,
            "unsupported_over_limit_mode_forbidden": True,
            "binary_mode_count": len(modes),
            "continuous_auxiliary_count": 1,
            "assembly_sequence": "provided externally; not optimized by this model",
        },
        "rejected_candidates": rejected,
    }
    if not parts:
        return {**base, "status": "optimal", "solved": True, "optimal": True,
                "objective": 0.0, "solver_objective": 0.0, "choices": [],
                "arm_workloads": {a: 0.0 for a in arms}, "peak_arm_workload": 0.0,
                "mip_gap": 0.0, "solver_called": False, "variables": [],
                "elapsed_s": time.perf_counter() - started}
    uncovered = [p for p in part_ids if not any(m["part"] == p for m in modes)]
    if uncovered:
        return {**base, "status": "infeasible", "solved": False, "optimal": False,
                "objective": None, "choices": [], "mip_gap": None,
                "solver_called": False, "reason": "some parts have no feasible assembly/support mode",
                "uncovered_parts": uncovered, "variables": [],
                "elapsed_s": time.perf_counter() - started}

    n, peak_index = len(modes), len(modes)
    objective = np.array([m["mode_cost"] for m in modes] + [weight], dtype=float)
    integrality = np.array([1] * n + [0], dtype=np.uint8)
    matrix = np.zeros((len(parts) + len(arms), n + 1), dtype=float)
    lower = np.full(len(parts) + len(arms), -np.inf)
    upper = np.zeros(len(parts) + len(arms))
    for row, p in enumerate(part_ids):
        matrix[row, :n] = [float(mode["part"] == p) for mode in modes]
        lower[row] = upper[row] = 1.0
    for offset, arm in enumerate(arms):
        row = len(parts) + offset
        matrix[row, :n] = [mode["arm_workload"][arm] for mode in modes]
        matrix[row, peak_index] = -1.0
    bounds = Bounds(np.zeros(n + 1), np.array([1.0] * n + [np.inf]))
    result = milp(
        objective, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(csc_matrix(matrix), lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": relative_gap, "presolve": True},
    )
    gap_value = getattr(result, "mip_gap", None)
    gap = None if gap_value is None or not math.isfinite(float(gap_value)) else float(gap_value)
    result_info = {**base, "solver_called": True, "solver_status": int(result.status),
                   "solver_message": str(result.message), "mip_gap": gap,
                   "elapsed_s": time.perf_counter() - started}
    if result.x is None:
        status = {1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "solver_error")
        return {**result_info, "status": status, "solved": False, "optimal": False,
                "objective": None, "choices": [], "variables": []}

    vector = np.asarray(result.x)
    residual = matrix @ vector
    valid = (np.all(np.isfinite(vector))
             and np.all(vector >= -1e-6)
             and np.all(vector[:n] <= 1 + 1e-6)
             and np.all(np.abs(vector[:n] - np.round(vector[:n])) <= 1e-6)
             and np.all(residual >= lower - 1e-6)
             and np.all(residual <= upper + 1e-6))
    selected = [mode for mode, value in zip(modes, vector[:n]) if value > .5]
    valid = valid and all(sum(m["part"] == p for m in selected) == 1 for p in part_ids)
    if not valid or result.status not in (0, 1):
        return {**result_info, "status": "invalid_incumbent", "solved": False,
                "optimal": False, "objective": None, "choices": [], "variables": []}
    selected_by_part = {m["part"]: m for m in selected}
    workloads = {arm: sum(mode["arm_workload"][arm] for mode in selected) for arm in arms}
    peak = max(workloads.values(), default=0.0)
    actual_objective = sum(m["mode_cost"] for m in selected) + weight * peak
    return {
        **result_info,
        "status": "optimal" if result.status == 0 else "feasible_limit",
        "solved": True,
        "optimal": result.status == 0,
        "objective": float(actual_objective),
        "solver_objective": float(result.fun),
        "choices": [dict(selected_by_part[p]) for p in part_ids],
        "arm_workloads": workloads,
        "peak_arm_workload": peak,
        "peak_auxiliary_value": float(vector[peak_index]),
        "variables": [{"mode_id": mode["mode_id"], "part": mode["part"],
                       "arm": mode["arm"], "grasp": mode["grasp"],
                       "support_arm": mode["support_arm"], "value": float(value),
                       "mode_cost": mode["mode_cost"]}
                      for mode, value in zip(modes, vector[:n])],
    }
