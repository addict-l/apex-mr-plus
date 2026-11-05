"""Dependency-scoped evidence reuse and a planar quasi-static LP oracle.

The LP models ONE connected structure and upward point contacts. It checks
Fx/Fz/M balance (M = x*Fz - z*Fx), unilateral normal forces, Coulomb friction and bounded
connection reactions. It does not certify grasp friction, 3-D stability,
continuous motion, joint torque limits or transient contact establishment.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Callable

import numpy as np
from scipy.optimize import linprog

from .model import Action, Problem, State


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EvidenceCache:
    def __init__(self):
        self.entries: dict[tuple, dict] = {}
        self.counts = Counter()

    def query(self, kind: str, key, dependencies: dict, compute: Callable[[], dict]) -> dict:
        identity = (kind, _json(key))
        previous = self.entries.get(identity)
        signature = _json(dependencies)
        if previous is not None and previous["signature"] == signature:
            self.counts[f"{kind}_hits"] += 1
            return deepcopy(previous["answer"])
        self.counts[f"{kind}_misses"] += 1
        answer = compute()
        # Search/solver budget failures must be retried, not permanently cached.
        if answer["status"] != "UNKNOWN":
            self.entries[identity] = {"signature": signature, "dependencies": set(dependencies),
                                      "answer": deepcopy(answer)}
        return deepcopy(answer)

    def invalidate(self, changed_facts: list[str]) -> list[dict]:
        invalid = []
        for key, value in list(self.entries.items()):
            hits = [fact for fact in changed_facts if any(
                dep == fact or dep.startswith(fact + "/") or fact.startswith(dep + "/")
                for dep in value["dependencies"])]
            if hits:
                invalid.append({"kind": key[0], "key": key[1], "changed_facts": hits})
                del self.entries[key]
        return invalid


class DeclaredCapabilities:
    """Finite input facts. This class deliberately performs no IK/RRT query."""
    def __init__(self, problem: Problem, cache: EvidenceCache):
        self.problem, self.cache = problem, cache
        self.unknown_queries = set()

    def check(self, arm: str, operation: str, target: str) -> dict:
        fact = f"capabilities/{arm}/{operation}/{target}"
        value = self.problem.status(arm, operation, target)
        answer = self.cache.query("capability", [arm, operation, target], {fact: value}, lambda: {
            "status": value, "fact": fact, "source": "declared finite-model capability",
            "motion_verified": False,
        })
        if value == "UNKNOWN":
            self.unknown_queries.add(fact)
        return answer


class PlanarWrenchOracle:
    def __init__(self, problem: Problem, cache: EvidenceCache):
        self.problem, self.cache = problem, cache
        self.unknown_queries = 0

    def check(self, state: State, action: Action | None = None) -> dict:
        physics = self.problem.physics
        # Location of a free/held object is irrelevant to THIS structure LP;
        # installed loads and the complete structure model are dependencies.
        installed = sorted(t for t in state.completed if t in self.problem.tasks
                           and self.problem.tasks[t]["kind"] == "install")
        loading = action.target if action is not None and action.kind == "install" else None
        key = {"supports": state.supports, "installed": installed, "loading": loading}
        deps = {"physics": physics, "parts": self.problem.data.get("parts", []),
                "tasks": self.problem.data["tasks"]}
        answer = self.cache.query("wrench", key, deps, lambda: self._solve(state, loading))
        if answer["status"] == "UNKNOWN":
            self.unknown_queries += 1
        return answer

    def _solve(self, state: State, loading: str | None) -> dict:
        physics = self.problem.physics
        load = np.zeros(3)

        def add_load(fx, fz, x, z, moment=0.):
            load[:] += [fx, fz, x*fz-z*fx+moment]

        for external in physics.get("loads", []):
            add_load(external.get("fx_N", 0.), external.get("fz_N", 0.),
                     external.get("x_m", 0.), external.get("z_m", 0.), external.get("moment_Nm", 0.))
        for tid, task in self.problem.tasks.items():
            if task["kind"] == "install" and (tid in state.completed or tid == loading):
                weight = self.problem.parts[task["part"]]["mass"] * physics["gravity_m_s2"]
                insert = task.get("insertion_force_N", 0.) if tid == loading else 0.
                add_load(0., -weight-insert, task.get("load_x_m", 0.), task.get("load_z_m", 0.))
        connection = physics["connection"]
        moment_limit = connection["moment_limit_Nm"]
        if connection.get("reinforced_by") in state.completed:
            moment_limit = connection["reinforced_moment_limit_Nm"]
        limits = [connection["force_x_limit_N"], connection["force_z_limit_N"], moment_limit]
        contacts = [(arm, cid, physics["contacts"][cid]) for arm, cid in state.supports]
        dimensions = 3 + 2*len(contacts)
        equality = np.zeros((3, dimensions))
        equality[:, :3] = np.eye(3)
        bounds = [(-limit, limit) for limit in limits]
        inequalities = []
        for i, (_, _, contact) in enumerate(contacts):
            ix, iz = 3+2*i, 4+2*i
            equality[:, ix] = [1., 0., -contact.get("z_m", 0.)]
            equality[:, iz] = [0., 1., contact.get("x_m", 0.)]
            bounds.extend([(None, None), (0., contact["normal_limit_N"])])
            for sign in (-1., 1.):
                row = np.zeros(dimensions)
                row[ix], row[iz] = sign, -contact["friction"]
                inequalities.append(row)
        result = linprog(np.zeros(dimensions), A_eq=equality, b_eq=-load,
                         A_ub=np.asarray(inequalities) if inequalities else None,
                         b_ub=np.zeros(len(inequalities)) if inequalities else None,
                         bounds=bounds, method="highs", options={"time_limit": 2.})
        base = {"model": "planar_single_structure_quasistatic", "solver": "scipy.optimize.linprog/HiGHS",
                "external_wrench_Fx_Fz_M": load.tolist(), "solver_status": int(result.status)}
        if result.status == 2:
            return {**base, "status": "INVALID", "reason": "LP infeasible in the declared planar model"}
        if result.status != 0 or result.x is None or not np.all(np.isfinite(result.x)):
            return {**base, "status": "UNKNOWN", "reason": str(result.message)}
        residual = float(np.max(np.abs(equality @ result.x + load)))
        inequality_error = max((float(row @ result.x) for row in inequalities), default=0.)
        bound_error = max([0.] + [max((low-v if low is not None else 0.),
                                           (v-high if high is not None else 0.))
                                   for v, (low, high) in zip(result.x, bounds)])
        if max(residual, inequality_error, bound_error) > 1e-7:
            return {**base, "status": "UNKNOWN", "reason": "LP witness failed independent residual checks"}
        return {**base, "status": "VALID", "balance_residual": residual,
                "connection_reaction_Fx_Fz_M": result.x[:3].tolist(),
                "support_reactions": [{"arm": arm, "contact": cid,
                                       "fx_N": float(result.x[3+2*i]), "fz_N": float(result.x[4+2*i])}
                                      for i, (arm, cid, _) in enumerate(contacts)]}
