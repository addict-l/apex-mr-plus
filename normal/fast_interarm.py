"""Fast sampled Type-2 collision oracle with an intentionally narrow scope.

This sampler checks only the two arms' physical collision proxies, tools, and
carried objects against the opposite arm/other payload. It does NOT check
environment obstacles, free parts, self-collision, or support stability. Use
the full MotionPlanner for RRT validation and simultaneous rollout validation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _q(value: Iterable[float]) -> list[float]:
    q = [float(x) for x in value]
    if len(q) != 6 or not all(math.isfinite(x) for x in q):
        raise ValueError("Joint state must contain six finite values")
    return q


def _payload(value: Any):
    if not value:
        return {}
    if isinstance(value, str):
        return {"name": value}
    result = deepcopy(value)
    result.setdefault("name", result.get("part"))
    if not result.get("name"):
        raise ValueError("A payload requires part/name")
    for field, length in (("offset", 3), ("quaternion", 4)):
        if field in result:
            numbers = [float(x) for x in result[field]]
            if len(numbers) != length or not all(math.isfinite(x) for x in numbers):
                raise ValueError(f"Payload {field} must have {length} finite values")
            result[field] = numbers
    return result


class FastInterarmSampler:
    """Temporary stopped-scene script; batch queries are transactional.

    Collections are built once from the same native collision geometry policy
    used by normal/motion_backend.lua. Each batch saves/restores both joint
    states and the union of touched payload world poses exactly once. Caller
    ``placed`` snapshots are ignored because static/free parts are outside this
    sampler's Type-2 scope; this is explicit in ``info`` and each result.
    """

    def __init__(self, sim, client, cfg):
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("FastInterarmSampler requires a stopped scene")
        self.sim, self.client, self.cfg = sim, client, cfg
        self._bounds_cache = {}
        self.script = None
        source = Path(__file__).with_suffix(".lua").read_text(encoding="utf-8")
        self.script = sim.createScript(sim.scripttype_customization, source, 0, "lua")
        try:
            sim.setObjectAlias(self.script, "NormalFastInterarmSampler")
            sim.setObjectParent(self.script, sim.getObject("/APEX_MR_Plus"), True)
            sim.initScript(self.script)
            self.info = self._call("fastSetup", cfg)
            if not self.info.get("ok"):
                raise RuntimeError(self.info.get("error", "Interarm sampler setup failed"))
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        if self.script is None:
            raise RuntimeError("FastInterarmSampler is closed")
        import zmq
        socket = self.client.socket
        previous = socket.getsockopt(zmq.RCVTIMEO)
        try:
            socket.setsockopt(zmq.RCVTIMEO, 120_000)
            result = self.sim.callScriptFunction(name, self.script, *args)
            if isinstance(result, dict) and result.get("ok") is False and "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            socket.setsockopt(zmq.RCVTIMEO, previous)

    def sample_state_pairs(self, samples, options=None):
        normalized = []
        for sample in samples:
            result = deepcopy(sample)
            if "qA" in result:
                result["A"] = result.pop("qA")
            if "qB" in result:
                result["B"] = result.pop("qB")
            for arm in ("A", "B"):
                if arm in result:
                    result[arm] = _q(result[arm])
            payloads = {}
            for arm, payload in result.get("payloads", {}).items():
                if arm not in ("A", "B"):
                    raise ValueError("Payload owner must be A or B")
                if payload:
                    payloads[arm] = _payload(payload)
            result["payloads"] = payloads
            # Preserve the field to make the omission auditable, even though
            # the Lua Type-2 implementation intentionally does not apply it.
            result.setdefault("placed", {})
            normalized.append(result)
        response = self._call("fastValidateSamples", normalized, options or {})
        if not isinstance(response, list) or len(response) != len(normalized):
            raise RuntimeError("Interarm sampler returned a malformed batch")
        return response

    validate_samples = sample_state_pairs

    def bounds_for_node(self, node):
        """Return a conservative nominal swept AABB for this entire node.

        The endpoint-union box is enlarged by
        ``2 m * max_adjacent_sum_abs_dq / 2``. This uses the stated model
        assumption that every proxy/tool/payload point lies within 2 m of each
        influencing revolute joint at every configuration on a segment. It is
        conservative for the installed UR5, its fixed tool, and the current
        <=0.416 m block models. The Lua side refuses the bound if even an
        observed bounding-box corner violates that radius assumption.

        Only disjoint boxes can skip the expensive relative-phase collision
        grid. Overlapping boxes mean 'unknown', never an actual collision.
        The bound covers the unchanged linear joint polyline (including a
        monotone quintic traversal), not compliance or tracking deviations.
        """
        arm = node.get("arm")
        if arm not in ("A", "B"):
            raise ValueError("Node arm must be A or B")
        source = node.get("q_samples") or node.get("q_path")
        if not source:
            if node.get("q_start") is None:
                raise ValueError("Node bounds need q_samples or q_start/q_end")
            source = [node["q_start"], node.get("q_end", node["q_start"])]
        path = [_q(q) for q in source]
        payload = _payload(node.get("payload"))
        cacheable = not payload or "offset" in payload
        # Bare payload names obtain their transform from the stopped baseline
        # scene. Mark that baseline explicitly in the cache identity. Normal
        # compiled tasks always supply offset/quaternion transforms.
        key = json.dumps({"arm": arm, "path": path, "payload": payload,
                          "model_bound_m": 2.0,
                          "baseline_payload_pose": self.cfg.get("initial", {}).get(payload.get("name")) if payload and "offset" not in payload else None},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
        if not cacheable or key not in self._bounds_cache:
            answer = self._call("fastNodeBounds", arm, path, payload, 2.0)
            if not isinstance(answer, dict) or not answer.get("ok"):
                raise RuntimeError("Interarm sampler returned an invalid node bound")
            for name in ("min", "max"):
                if len(answer.get(name, [])) != 3 or not all(math.isfinite(float(v)) for v in answer[name]):
                    raise RuntimeError("Node AABB has invalid coordinates")
            answer["node_geometry_fingerprint"] = hashlib.sha256(key.encode()).hexdigest()
            answer["cache_hit"] = False
            answer["cacheable"] = cacheable
            if cacheable:
                self._bounds_cache[key] = deepcopy(answer)
            return answer
        answer = deepcopy(self._bounds_cache[key])
        answer["cache_hit"] = True
        return answer

    def statistics(self):
        result = self._call("fastStatistics")
        result["cached_node_bounds"] = len(self._bounds_cache)
        return result

    def close(self):
        if self.script is not None:
            handle = self.script
            try:
                self._call("fastCleanup")
            finally:
                try:
                    self.sim.removeObjects([handle])
                finally:
                    self.script = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["FastInterarmSampler"]
