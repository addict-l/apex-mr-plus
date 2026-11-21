"""Transactional collision-checked joint paths with OMPL RRT-Connect fallback.

This module plans joint paths in the stopped geometry model. It does not run
the physics controller. Attached object transforms are relative to the TCP.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import count
import math
import time
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


_PROBE_IDS = count()
_PRECHECK_DRAIN_SECONDS = .15


class PrecheckTimeout(TimeoutError):
    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt


class MotionPlanner:
    """Own a temporary planning script in a stopped APEX MR Scene 3 scene.

    All calls restore the caller's arm/part/target poses, including failures.
    Contact exclusions are exact shape pairs. Block/table and block/block
    exemptions also require surface contact within contact_tolerance.
    """

    def __init__(self, sim: Any, client: Any, cfg: dict[str, Any]):
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("MotionPlanner must be initialized with simulation stopped")
        self.sim, self.client, self.cfg = sim, client, cfg
        self.script = None
        self._precheck_deadline = None
        self._probe_transport_fault = None
        source = Path(__file__).with_name("motion_backend.lua").read_text(encoding="utf-8")
        self.script = sim.createScript(sim.scripttype_customization, source, 0, "lua")
        try:
            sim.setObjectAlias(self.script, "Scene3MotionPlanningBackend")
            sim.setObjectParent(self.script, cfg.get("root") or sim.getObject(cfg.get("root_path", "/APEX_MR_Scene3")), True)
            sim.initScript(self.script)
            self.info = self._call("mpSetup", cfg)
            if not self.info.get("ok"):
                raise RuntimeError(self.info.get("error", "Motion backend setup failed"))
        except BaseException:
            self.close()
            raise

    def _call(self, function: str, *args):
        if getattr(self, "_probe_transport_fault", None):
            raise ConnectionError(self._probe_transport_fault)
        if self.script is None:
            raise RuntimeError("MotionPlanner is closed")
        import zmq

        if self._precheck_deadline is not None:
            return self._call_bounded_precheck(function, *args)
        socket = self.client.socket
        before = socket.getsockopt(zmq.RCVTIMEO)
        try:
            socket.setsockopt(zmq.RCVTIMEO, 180_000)
            return self.sim.callScriptFunction(function, self.script, *args)
        finally:
            socket.setsockopt(zmq.RCVTIMEO, before)

    @contextmanager
    def precheck_deadline(self, deadline):
        """Bound optional probe waits without changing ordinary planning calls."""
        before = self._precheck_deadline
        self._precheck_deadline = deadline
        try:
            yield
        finally:
            self._precheck_deadline = before

    def _call_bounded_precheck(self, function, *args):
        import zmq

        client, hard_deadline = self.client, self._precheck_deadline
        work_deadline = hard_deadline - _PRECHECK_DRAIN_SECONDS
        if time.monotonic() >= work_deadline:
            raise PrecheckTimeout("Precheck work deadline reached before sending")
        request_id, options = None, None
        if function == "mpSolveIK":
            request_id = next(_PROBE_IDS)
            options = dict(args[-1])
            options["probe_request_id"] = request_id
            args = (*args[:-1], options)
        socket = client.socket
        send, recv = client._send, client._recv
        prior = {name: vars(client)[name] for name in ("_send", "_recv") if name in vars(client)}
        before = {option: socket.getsockopt(option) for option in (zmq.SNDTIMEO, zmq.RCVTIMEO)}
        sent_at = None

        def send_until(request):
            nonlocal sent_at
            now = time.monotonic()
            original = sent_at is None or isinstance(request, dict) and request.get("func") == "sim.callScriptFunction"
            deadline = work_deadline if original else hard_deadline
            ms = int(max(0., deadline - now) * 1000)
            if not ms:
                if sent_at is None:
                    raise PrecheckTimeout("Precheck work deadline reached before sending")
                raise ConnectionError("Precheck RPC could not complete its protocol handshake")
            if original and options is not None:
                options["probe_work_budget_s"] = deadline - now
            socket.setsockopt(zmq.SNDTIMEO, ms)
            send(request)
            if sent_at is None:
                sent_at = now

        def recv_until(*data, **kwargs):
            remaining = int(max(0., hard_deadline - time.monotonic()) * 1000)
            socket.setsockopt(zmq.RCVTIMEO, remaining)
            return recv(*data, **kwargs)

        client._send, client._recv = send_until, recv_until
        try:
            try:
                result = self.sim.callScriptFunction(function, self.script, *args)
                received_at = time.monotonic()
                if received_at >= hard_deadline or sent_at is None or not isinstance(result, dict):
                    raise ConnectionError("Precheck did not receive a final reply within its total deadline")
                if str(result.get("error", "")).startswith("Transaction restore failed"):
                    raise ConnectionError(result["error"])
                native = result.get("probe_receipt")
                if request_id is not None and not self._confirmed_probe(native, request_id, result):
                    raise ConnectionError("Precheck response did not confirm recovery of its IK transaction")
            except PrecheckTimeout:
                raise
            except Exception as exc:
                self._probe_transport_fault = f"Unsafe precheck transport: {exc}"
                raise ConnectionError(self._probe_transport_fault) from exc
            if result.get("probe_timeout") is True or received_at >= work_deadline:
                receipt = {"request_id": request_id, "request_sent_s": sent_at, "final_reply_s": received_at,
                           "client_work_deadline_s": work_deadline, "client_hard_deadline_s": hard_deadline,
                           "native": native}
                raise PrecheckTimeout("Precheck work expired and final transaction reply was received", receipt)
            return result
        finally:
            for name in ("_send", "_recv"):
                if name in prior:
                    setattr(client, name, prior[name])
                else:
                    delattr(client, name)
            for option, value in before.items():
                socket.setsockopt(option, value)

    @staticmethod
    def _confirmed_probe(receipt, request_id, result):
        if not isinstance(receipt, dict) or receipt.get("request_id") != request_id or receipt.get("restored") is not True:
            return False
        times = [receipt.get(name) for name in ("server_entered_s", "work_ended_s", "restored_s", "work_deadline_s")]
        if not all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) for value in times):
            return False
        entered, ended, restored, deadline = times
        timed_out = result.get("probe_timeout") is True
        return (entered <= ended <= restored and entered <= deadline
                and receipt.get("work_expired") is timed_out and (timed_out or ended < deadline)
                and (not timed_out or ended >= deadline and result.get("ok") is False
                     and isinstance(receipt.get("boundary"), str)))

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
        """Compare checked direct/reference candidates with genuine RRT paths.

        All sources use the same simulator collision context and dense final
        certificate. Results distinguish a validated direct segment from an
        actual OMPL call; interpolation alone is never collision evidence.
        """
        arm = _arm(arm)
        if not math.isfinite(float(solve_time)) or solve_time <= 0:
            raise ValueError("solve_time must be finite and positive")
        opts = deepcopy(options or {})
        opts["solve_time"] = float(solve_time)
        opts.setdefault("max_joint_step", 0.035)
        opts.setdefault("resolution", 0.001)
        opts.setdefault("direct_first", True)
        opts.setdefault("max_rrt_candidates", 3)
        opts.setdefault("shortcut_max_trials", 96)
        opts.setdefault("shortcut_budget_s", 0.6)
        opts.setdefault("quality_stretch_threshold", 1.15)
        if other_q is not None:
            opts["other_q"] = _vector(other_q, 6, "other_q")
        if contact_exclusions is not None:
            opts["contact_exclusions"] = deepcopy(contact_exclusions)
        result = self._call("mpPlanJoint", arm, _vector(q_start, 6, "q_start"),
                            _vector(q_goal, 6, "q_goal"), "RRTConnect", _payload(payload),
                            placed or {}, opts)
        result.setdefault("algorithm", result.get("stats", {}).get("algorithm", "RRTConnect"))
        result.setdefault("stats", {})["collision_geometry_policy"] = self.info["collision_geometry_policy"]
        return result

    # Compatible explicit name for pipeline callers familiar with the baseline.
    def plan_joint(self, arm, q_start, q_goal, algorithm="RRTConnect", payload=None,
                   placed=None, options=None):
        if algorithm != "RRTConnect":
            raise ValueError("The Scene 3 APEX MR flow uses RRTConnect")
        opts = deepcopy(options or {})
        solve_time = opts.pop("solve_time", 2.0)
        return self.plan(arm, q_start, q_goal, payload, placed, solve_time=solve_time, options=opts)

    def validate_joint(self, arm, q, payload=None, placed=None, options=None):
        return self._call("mpValidateJoint", _arm(arm), _vector(q, 6, "q"),
                          _payload(payload), placed or {}, options or {})

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

    def sample_batch_statistics(self, *, reset=False):
        """Read exact full-geometry batch counters; callers budget this RPC."""
        return self._call("mpSampleBatchStatistics", {"reset": bool(reset)})

    def fk(self, arm, q):
        return self._call("mpFK", _arm(arm), _vector(q, 6, "q"))

    def invalidate(self):
        return self._call("mpInvalidate")

    def close(self):
        if getattr(self, "_probe_transport_fault", None):
            # The owned process supervisor handles teardown on a broken REQ.
            self.script = None
            return
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
