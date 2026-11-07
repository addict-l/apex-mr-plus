"""Transactional CoppeliaSim IK and genuine OMPL RRT-Connect queries.

This module plans joint paths in the stopped geometry model. It does not run
the physics controller. Attached object transforms are relative to the TCP.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable


def _vector(value: Iterable[float], length: int, label: str) -> list[float]:
    result = [float(x) for x in value]
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label} needs {length} finite values")
    return result


def _arm(value: str) -> str:
    if value not in ("A", "B"):
        raise ValueError("Arm must be A or B")
    return value


def _payload(value: Any) -> Any:
    if not value:
        return {}
    if isinstance(value, str):
        return {"name": value}
    result = deepcopy(value)
    if "part" in result and "name" not in result:
        result["name"] = result["part"]
    if "name" not in result:
        raise ValueError("A payload needs part/name")
    if "offset" in result:
        result["offset"] = _vector(result["offset"], 3, "payload offset")
    if "quaternion" in result:
        result["quaternion"] = _vector(result["quaternion"], 4, "payload quaternion")
    return result


class MotionPlanner:
    """Own a temporary planning script in a stopped APEX MR Plus scene.

    All calls restore the caller's arm/part/target poses, including failures.
    Contact exclusions are exact shape pairs. Block/table and block/block
    exemptions also require surface contact within contact_tolerance.
    """

    def __init__(self, sim: Any, client: Any, cfg: dict[str, Any]):
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("MotionPlanner must be initialized with simulation stopped")
        self.sim, self.client, self.cfg = sim, client, cfg
        self.script = None
        source = Path(__file__).with_name("motion_backend.lua").read_text(encoding="utf-8")
        self.script = sim.createScript(sim.scripttype_customization, source, 0, "lua")
        try:
            sim.setObjectAlias(self.script, "NormalMotionPlanningBackend")
            sim.setObjectParent(self.script, sim.getObject("/APEX_MR_Plus"), True)
            sim.initScript(self.script)
            self.info = self._call("mpSetup", cfg)
            if not self.info.get("ok"):
                raise RuntimeError(self.info.get("error", "Motion backend setup failed"))
        except BaseException:
            self.close()
            raise

    def _call(self, function: str, *args):
        if self.script is None:
            raise RuntimeError("MotionPlanner is closed")
        import zmq

        socket = self.client.socket
        before = socket.getsockopt(zmq.RCVTIMEO)
        try:
            socket.setsockopt(zmq.RCVTIMEO, 180_000)
            return self.sim.callScriptFunction(function, self.script, *args)
        finally:
            socket.setsockopt(zmq.RCVTIMEO, before)

    def solve_ik(self, arm, position, quaternion=None, payload=None, placed=None, options=None):
        arm = _arm(arm)
        quaternion = quaternion if quaternion is not None else self.cfg["arms"][arm]["quaternion"]
        result = self._call(
            "mpSolveIK", arm, _vector(position, 3, "position"),
            _vector(quaternion, 4, "quaternion"), _payload(payload), placed or {}, options or {},
        )
        solutions = result.get("solutions", [])
        result.setdefault("stats", {})["collision_geometry_policy"] = self.info["collision_geometry_policy"]
        if solutions:
            result["q"] = solutions[0]
        return result

    def plan(self, arm, q_start, q_goal, payload=None, placed=None, other_q=None,
             contact_exclusions=None, solve_time=2.0, options=None):
        """Return an exact RRTConnect joint solution, then densely recheck it.

        Densification interpolates a found OMPL path for validation/execution;
        it is not a replacement for the RRT-Connect planning call.
        """
        arm = _arm(arm)
        if not math.isfinite(float(solve_time)) or solve_time <= 0:
            raise ValueError("solve_time must be finite and positive")
        opts = deepcopy(options or {})
        opts["solve_time"] = float(solve_time)
        opts.setdefault("max_joint_step", 0.035)
        opts.setdefault("resolution", 0.001)
        opts.setdefault("simplify_time", 0.035)
        if other_q is not None:
            opts["other_q"] = _vector(other_q, 6, "other_q")
        if contact_exclusions is not None:
            opts["contact_exclusions"] = deepcopy(contact_exclusions)
        result = self._call("mpPlanJoint", arm, _vector(q_start, 6, "q_start"),
                            _vector(q_goal, 6, "q_goal"), "RRTConnect", _payload(payload),
                            placed or {}, opts)
        result["algorithm"] = "RRTConnect"
        result.setdefault("stats", {})["collision_geometry_policy"] = self.info["collision_geometry_policy"]
        return result

    # Compatible explicit name for pipeline callers familiar with the baseline.
    def plan_joint(self, arm, q_start, q_goal, algorithm="RRTConnect", payload=None,
                   placed=None, options=None):
        if algorithm != "RRTConnect":
            raise ValueError("The normal APEX MR flow currently uses RRTConnect")
        opts = deepcopy(options or {})
        solve_time = opts.pop("solve_time", 2.0)
        return self.plan(arm, q_start, q_goal, payload, placed, solve_time=solve_time, options=opts)

    def validate_joint(self, arm, q, payload=None, placed=None, options=None):
        return self._call("mpValidateJoint", _arm(arm), _vector(q, 6, "q"),
                          _payload(payload), placed or {}, options or {})

    def validate_edge(self, arm, q_start, q_goal, payload=None, placed=None, other_q=None,
                      contact_exclusions=None, step=.005, options=None):
        """Check a joint-space edge in the same geometry as the RRT backend."""
        if not math.isfinite(float(step)) or step <= 0:
            raise ValueError("step must be finite and positive")
        opts = deepcopy(options or {})
        opts["edge_check_step"] = float(step)
        if other_q is not None:
            opts["other_q"] = _vector(other_q, 6, "other_q")
        if contact_exclusions is not None:
            opts["contact_exclusions"] = deepcopy(contact_exclusions)
        return self._call("mpValidateJointEdge", _arm(arm), _vector(q_start, 6, "q_start"),
                          _vector(q_goal, 6, "q_goal"), _payload(payload), placed or {}, opts)

    def validate_pair(self, qA, qB, payloads=None, placed=None, contact_exclusions=None, options=None):
        opts = deepcopy(options or {})
        if contact_exclusions is not None:
            opts["contact_exclusions"] = deepcopy(contact_exclusions)
        bindings = {a: _payload(p) for a, p in (payloads or {}).items()}
        return self._call("mpValidatePair", _vector(qA, 6, "qA"), _vector(qB, 6, "qB"),
                          bindings, placed or {}, opts)

    def sample_state_pairs(self, samples, options=None):
        """Batch real-shape pair tests used to build/verify motion TPG edges.

        Samples accept qA/qB or A/B. Payloads use A/B names and each sample can
        supply placed and options (contact exclusions etc.) independently.
        """
        normalized = []
        for sample in samples:
            s = deepcopy(sample)
            if "qA" in s:
                s["A"] = s.pop("qA")
            if "qB" in s:
                s["B"] = s.pop("qB")
            for arm in ("A", "B"):
                if arm in s:
                    s[arm] = _vector(s[arm], 6, arm)
            s["payloads"] = {a: _payload(p) for a, p in s.get("payloads", {}).items()}
            normalized.append(s)
        return self._call("mpValidateSamples", normalized, options or {})

    validate_samples = sample_state_pairs

    def fk(self, arm, q):
        return self._call("mpFK", _arm(arm), _vector(q, 6, "q"))

    def invalidate(self):
        return self._call("mpInvalidate")

    def close(self):
        if self.script is not None:
            try:
                self._call("mpCleanup")
            finally:
                try:
                    self.sim.removeObjects([self.script])
                finally:
                    self.script = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
