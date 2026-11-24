"""Sampled physical-mesh conflict checks for the Scene 3 multi-modal TPG.

The motion backend supplies transactional mesh collision checks. This module
only combines exact joint/object modes, preserves path corners while sampling,
and batches the resulting pairs. Sampling is not a continuous collision proof.
No simulator connection, scene mutation, or dynamics execution is performed
outside the supplied MotionPlanner's transactional query interface.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


ARMS = ("A", "B")
EPS = 1e-9
BATCH_SIZE = 128


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _q(value):
    result = [float(x) for x in value]
    if len(result) != 6 or not all(math.isfinite(x) for x in result):
        raise ValueError("Geometry queries require six finite joint positions")
    return result


def _resample(path, step):
    if not path:
        raise ValueError("A TPG node has no reference q_samples")
    points = [_q(q) for q in path]
    result = [points[0]]
    for start, end in zip(points, points[1:]):
        count = max(1, math.ceil(max(abs(a-b) for a,b in zip(start,end))/step))
        for index in range(1, count+1):
            sample = [a+(b-a)*index/count for a,b in zip(start,end)]
            if sample != result[-1]:
                result.append(sample)
    return result


def _node_path(node, step):
    points = node.get("q_samples") or node.get("q_path")
    if not points:
        if node.get("q_start") is None:
            raise ValueError(f"Missing joint geometry for node {node.get('id')}")
        points = [node["q_start"], node.get("q_end", node["q_start"])]
    return _resample(points, step)


def _payload_name(payload):
    if not payload:
        return None
    if isinstance(payload, str):
        return payload
    name = payload.get("name", payload.get("part"))
    if not name:
        raise ValueError("A nonempty payload mode must have a part/name")
    return name


def _parts(cfg):
    values = cfg.get("parts", [])
    return dict(values) if isinstance(values, Mapping) else {part["id"]: part for part in values}


def _pose_equal(a, b):
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return _canonical(a) == _canonical(b)
    left, right = list(a), list(b)
    if len(left) == 3:
        left += [0.,0.,0.,1.]
    if len(right) == 3:
        right += [0.,0.,0.,1.]
    if len(left) != len(right):
        return False
    if max(abs(float(x)-float(y)) for x,y in zip(left[:3],right[:3])) > 1e-8:
        return False
    if len(left) == 7:
        return min(max(abs(float(x)-float(y)) for x,y in zip(left[3:],right[3:])),
                   max(abs(float(x)+float(y)) for x,y in zip(left[3:],right[3:]))) <= 1e-8
    return max(abs(float(x)-float(y)) for x,y in zip(left,right)) <= 1e-8


def _pairs(*groups):
    result, seen = [], set()
    for group in groups:
        for pair in group or []:
            if len(pair) != 2:
                raise ValueError("Contact exclusions must contain exact pairs")
            token = tuple(sorted((_canonical(pair[0]), _canonical(pair[1]))))
            if token not in seen:
                seen.add(token); result.append(deepcopy(list(pair)))
    return result


def _resting_contacts(placed, payloads, cfg):
    """Enumerate exact intended assembly contacts, never workspace exemptions.

    The backend additionally enforces its surface-contact tolerance for
    block/block and block/table pairs. Listing a pair therefore cannot hide a
    deeply penetrating block trajectory.
    """
    parts = _parts(cfg)
    names = set(placed) | {_payload_name(p) for p in payloads.values() if p}
    contacts = []
    table = cfg.get("table")
    for name in sorted(names):
        part = parts.get(name)
        if not part:
            continue
        dependencies = part.get("dependencies", [])
        contacts.extend([[name, dependency] for dependency in dependencies if dependency in placed])
        if not dependencies:
            if table is not None:
                contacts.append([name, table])
            # Do not exempt every static obstacle merely because this is a
            # bottom-layer part. The compiler may supply the exact intended
            # source/goal contact surface in metadata, or explicitly register
            # a validated horizontal support surface here.
            surfaces = cfg.get("support_contact_surfaces", {}).get(name, [])
            contacts.extend([[name, surface] for surface in surfaces])
    return contacts


def _merge_modes(node_a, node_b, cfg):
    placed = {}
    payloads = {}
    contacts = []
    tolerances = []
    holders = {}
    articulations = {}
    for node in (node_a, node_b):
        arm = node["arm"]
        if arm not in ARMS:
            raise ValueError("Conflict nodes must belong to A or B")
        payload = deepcopy(node.get("payload"))
        if payload:
            name = _payload_name(payload)
            if name in holders:
                return None, {"reason": "inconsistent_mode", "part": name,
                              "detail": "two independent node modes both claim the same payload",
                              "arms": [holders[name], arm]}
            holders[name] = arm
            payloads[arm] = payload
    for node in (node_a, node_b):
        metadata = node.get("metadata", {})
        for name, value in metadata.get("options", {}).get("articulations", {}).items():
            if name in articulations and abs(articulations[name]-value) > EPS:
                return None, {"reason": "inconsistent_articulation_mode", "joint": name}
            articulations[name] = value
        for part, pose in metadata.get("placed", {}).items():
            if part in holders:
                continue
            if part in placed and not _pose_equal(placed[part], pose):
                return None, {"reason": "inconsistent_mode", "part": part,
                              "detail": "same unheld part has different declared world poses",
                              "pose_a": placed[part], "pose_b": pose}
            placed[part] = deepcopy(pose)
        contacts = _pairs(contacts, metadata.get("contact_exclusions", []),
                          metadata.get("options", {}).get("contact_exclusions", []))
        for candidate in (metadata.get("contact_tolerance"), metadata.get("options", {}).get("contact_tolerance")):
            if candidate is not None:
                tolerances.append(_positive(candidate, "contact_tolerance"))
    for arm, payload in payloads.items():
        name = _payload_name(payload)
        # Active TCP-object transforms take precedence over an earlier/later
        # nominal placement of that same object in the other arm's snapshot.
        placed.pop(name, None)
    contacts = _pairs(contacts, _resting_contacts(placed, payloads, cfg))
    context = {"payloads": payloads, "placed": placed,
               "options": {"contact_exclusions": contacts,
                           "articulations": articulations,
                           "contact_tolerance": min(tolerances) if tolerances else .003}}
    return context, None


def _answers(planner, samples):
    raw = planner.sample_state_pairs(samples)
    if isinstance(raw, Mapping):
        raw = raw.get("results", raw.get("samples"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(samples):
        raise RuntimeError("MotionPlanner returned a malformed batch validation result")
    result = []
    for answer in raw:
        if not isinstance(answer, Mapping) or not isinstance(answer.get("ok"), bool):
            raise RuntimeError("MotionPlanner validation result needs a boolean ok")
        result.append(deepcopy(dict(answer)))
    return result


class SampledConflictOracle:
    """All-relative-phase mesh checks, cached by exact complete world state."""

    def __init__(self, planner, cfg, outdir, step=.07):
        self.planner = planner
        self.cfg = cfg
        self.step = _positive(step, "step")
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        journal=self.outdir/'geometry_pair_checks.jsonl'
        if journal.exists():
            history=self.outdir/'diagnostics'/'geometry_history'
            history.mkdir(parents=True,exist_ok=True)
            journal.replace(history/f'geometry_pair_checks.{time.time_ns()}.jsonl')
        self.cache = {}
        self.stats = {"oracle_calls": 0, "conflicting_calls": 0, "mode_conflicts": 0,
                      "mesh_state_queries": 0, "cache_hits": 0, "states_considered": 0,
                      "batch_calls": 0, "wall_s": 0., "joint_sample_step_rad": self.step,
                      "batch_size": BATCH_SIZE, "continuous_collision_guarantee": False}

    def _record(self, a, b, evidence, started):
        self.stats["wall_s"] += time.perf_counter()-started
        self.stats["conflicting_calls"] += int(evidence["conflict"])
        audit = {"a": a.get("id"), "b": b.get("id"), **evidence}
        with (self.outdir/"geometry_pair_checks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False)+"\n")
        self.save_stats()
        return evidence

    def save_stats(self):
        (self.outdir/"geometry_statistics.json").write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def __call__(self, node_a, node_b):
        started = time.perf_counter()
        self.stats["oracle_calls"] += 1
        if node_a["arm"] == node_b["arm"]:
            raise ValueError("The cross-arm conflict oracle needs one node from each arm")
        context, mismatch = _merge_modes(node_a, node_b, self.cfg)
        base = {"joint_sample_step_rad": self.step, "all_relative_phases": True,
                "continuous_collision_guarantee": False, "backend": type(self.planner).__name__+".sample_state_pairs",
                "backend_scope": deepcopy(getattr(self.planner,'info',{})),
                "waiting_endpoint_verification": False}
        if mismatch:
            self.stats["mode_conflicts"] += 1
            return self._record(node_a, node_b, {"conflict": True, "evidence": {**base, "problem": mismatch}}, started)
        # A conservative whole-fragment broad phase is available for the
        # UR5 physical-proxy sampler. Overlapping boxes still use all phases.
        if hasattr(self.planner,'bounds_for_node'):
            ba=self.planner.bounds_for_node(node_a);bb=self.planner.bounds_for_node(node_b)
            separated=any(ba['max'][axis]<bb['min'][axis] or bb['max'][axis]<ba['min'][axis] for axis in range(3))
            if separated:
                self.stats['broadphase_disjoint_calls']=self.stats.get('broadphase_disjoint_calls',0)+1
                return self._record(node_a,node_b,{'conflict':False,'evidence':{**base,
                    'method':'conservative_swept_physical_proxy_AABB','bounds_A':ba,'bounds_B':bb}},started)
        paths = {node["arm"]: _node_path(node, self.step) for node in (node_a, node_b)}
        total = len(paths["A"])*len(paths["B"])
        considered, queried, cached = 0, 0, 0
        batch, batch_keys = [], []
        first_hit = None

        def flush():
            nonlocal queried, first_hit
            if not batch:
                return
            answers = _answers(self.planner, batch)
            self.stats["batch_calls"] += 1
            self.stats["mesh_state_queries"] += len(batch)
            queried += len(batch)
            for key, sample, answer in zip(batch_keys, batch, answers):
                self.cache[key] = answer
                if first_hit is None and not answer["ok"]:
                    first_hit = {"problem": answer.get("problem"), "qA": sample["A"], "qB": sample["B"],
                                 "payloads": context["payloads"], "placed": context["placed"]}
            batch.clear(); batch_keys.clear()

        for qa, qb in itertools.product(paths["A"], paths["B"]):
            sample = {"A": qa, "B": qb, **context}
            key = _canonical(sample)
            considered += 1
            if key in self.cache:
                cached += 1
                answer = self.cache[key]
                if not answer["ok"]:
                    first_hit = {"problem": answer.get("problem"), "qA": qa, "qB": qb,
                                 "payloads": context["payloads"], "placed": context["placed"]}
                    break
            else:
                batch.append(sample); batch_keys.append(key)
                if len(batch) >= BATCH_SIZE:
                    flush()
                    if first_hit is not None:
                        break
        if first_hit is None:
            flush()
        self.stats["states_considered"] += considered
        self.stats["cache_hits"] += cached
        evidence = {**base, "grid_shape": [len(paths["A"]), len(paths["B"])],
                    "pair_states_total": total, "pair_states_considered": considered,
                    "new_mesh_queries": queried, "cache_hits": cached,
                    "checked_all_pair_states": first_hit is None and considered == total,
                    "mode_fingerprint": hashlib.sha256(_canonical(context).encode()).hexdigest()}
        if first_hit is not None:
            evidence.update(first_hit)
        return self._record(node_a, node_b, {"conflict": first_hit is not None, "evidence": evidence}, started)


class _BaselinePlanner:
    """Instrument the original 2D route without advertising a bounds method."""

    def __init__(self, owner, info):
        self.owner = owner
        self.info = info

    def sample_state_pairs(self, samples):
        return self.owner._sample("fallback", samples, self.owner._fallback_projection)


class BroadphaseConflictOracle(SampledConflictOracle):
    """Reuse only positive projections of full-scene mesh queries.

    One instance is scoped to a fixed stopped scene and the immutable nodes of
    one TPG build. In a strictly separated pair the original mesh predicate
    decomposes into local A and B checks and a fixed-scene context. In particular,
    unheld geometry remains present in the original sample_state_pairs calls.
    """

    def __init__(self, planner, cfg, outdir, *, bounds_sampler, step=.07):
        self._remote = planner
        self._bounds_sampler = bounds_sampler
        self._fallback_projection = None
        self._node_bounds = {}
        blocks = cfg.get("blocks", {})
        handle_counts = Counter(_canonical(handle) for handle in blocks.values())
        self._canonical_block_names = {
            name for name, handle in blocks.items() if handle_counts[_canonical(handle)] == 1
        } if blocks else None
        # Logically keyed by arm, own q, own binding, opposing payload name,
        # placed and options. Group the q values by their common key prefix.
        self.single_ok_cache = {}
        self._full_context_ok = set()
        full_info = getattr(planner, "info", {})
        bounds_info = getattr(bounds_sampler, "info", {})
        shared_metadata = ("collision_geometry_policy", "collision_shape_counts")
        self._bounds_compatible = all(
            full_info[key] == bounds_info[key] for key in shared_metadata
            if key in full_info and key in bounds_info
        )
        super().__init__(_BaselinePlanner(self, full_info), cfg, outdir, step)
        self.stats.update({
            "aabb_seconds": 0.,
            "cross_query_seconds": 0.,
            "fallback_query_seconds": 0.,
            "cross_mesh_queries": 0,
            "fallback_mesh_queries": 0,
            "bounds_calls": 0,
            "bounds_cache_hits": 0,
            "bounds_sampler_cache_hits": 0,
            "broadphase_eligible_calls": 0,
            "broadphase_disjoint_calls": 0,
            "single_cache_lookups": 0,
            "single_cache_hits": 0,
            "cross_calls": 0,
            "zero_rpc_cross_calls": 0,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "bounds_rejected_nodes": 0,
            "bounds_rejections": [],
        })

    @staticmethod
    def _ordinary(node):
        if node.get("kind") not in (None, "transit") or node.get("node_type") not in (None, "pose"):
            return False
        if node.get("geometry_available") is False or node.get("mode"):
            return False
        metadata = node.get("metadata", {})
        unsupported = ("shared_payload", "closed_chain", "cohold", "coheld", "grasp_transforms")
        return not any(
            source.get(key) for source in (node, metadata, metadata.get("options", {}))
            for key in unsupported
        ) and not metadata.get("mode")

    def _eligibility(self, nodes, context):
        if not all(self._ordinary(node) for node in nodes):
            return "nonordinary_or_shared"
        if context["options"]["articulations"]:
            # Per-node bounds use their own options. Refuse their use where
            # the merged scene moves any articulated part of the workcell.
            return "merged_articulations"
        for payload in context["payloads"].values():
            # With a bare name each backend infers the binding from its current
            # snapshot at a different layer. Require the same explicit rigid
            # transform in both the bounds and native mesh query.
            if not isinstance(payload, Mapping) or not payload.get("offset") or not payload.get("quaternion"):
                return "implicit_payload_transform"
            if (self._canonical_block_names is not None
                    and _payload_name(payload) not in self._canonical_block_names):
                return "unregistered_or_aliased_payload"
        return None

    def _projection(self, context):
        payloads = context["payloads"]
        world = _canonical((context["placed"], context["options"]))
        pair = (world, _canonical(payloads))
        a = ("A", _canonical(payloads.get("A")), _payload_name(payloads.get("B")), world)
        b = ("B", _canonical(payloads.get("B")), _payload_name(payloads.get("A")), world)
        return pair, self.single_ok_cache.setdefault(a, set()), self.single_ok_cache.setdefault(b, set())

    def _sample(self, route, samples, projection):
        began = time.perf_counter()
        raw = self._remote.sample_state_pairs(samples)
        self.stats[f"{route}_query_seconds"] += time.perf_counter()-began
        self.stats[f"{route}_mesh_queries"] += len(samples)
        if projection is not None:
            answers = raw.get("results", raw.get("samples")) if isinstance(raw, Mapping) else raw
            if (isinstance(answers, Sequence) and not isinstance(answers, (str, bytes))
                    and len(answers) == len(samples)):
                full_key, good_a, good_b = projection
                for sample, answer in zip(samples, answers):
                    if isinstance(answer, Mapping) and answer.get("ok") is True:
                        self._full_context_ok.add(full_key)
                        good_a.add(tuple(sample["A"]))
                        good_b.add(tuple(sample["B"]))
        return raw

    def _bounds(self, node):
        # TPG hands the oracle deep-copied nodes. Use stable fragment identity
        # and inspect the exact geometry so a reused ID cannot reuse a stale box.
        identity = node.get("id")
        source = node.get("q_samples") or node.get("q_path")
        if not source:
            source = [node.get("q_start"), node.get("q_end", node.get("q_start"))]
        geometry = (node["arm"], source, node.get("payload"))
        saved = self._node_bounds.get(identity)
        if saved is not None and saved[0] == geometry:
            self.stats["bounds_cache_hits"] += 1
            return saved[1]
        began = time.perf_counter()
        self.stats["bounds_calls"] += 1
        box, path = None, None
        try:
            # Cover the exact discrete grid points checked by the reference
            # oracle directly. Keep the native sweep inflation and radius test;
            # equivalence on this finite grid needs no between-point assumption.
            path = _node_path(node, self.step)
            result = self._bounds_sampler.bounds_for_node(dict(node, q_samples=path))
            if not isinstance(result, Mapping):
                raise ValueError("Malformed node AABB")
            if result.get("ok") is False:
                raise ValueError(result.get("error") or "Node AABB refused")
            if not result.get("covers_between_linear_joint_samples", True):
                raise ValueError("Node AABB does not cover the sampled path")
            if node.get("payload") and not result.get("payload_included", True):
                raise ValueError("Node AABB omitted the carried geometry")
            low, high = tuple(map(float, result["min"])), tuple(map(float, result["max"]))
            if len(low) != 3 or len(high) != 3 or not all(
                math.isfinite(a) and math.isfinite(b) and a <= b for a, b in zip(low, high)
            ):
                raise ValueError("Node AABB has invalid coordinates")
            observed = result.get("max_observed_corner_joint_distance_m")
            radius = result.get("model_bound_m")
            if observed is not None and radius is not None and observed > radius+1e-8:
                raise ValueError("Observed geometry exceeds the revolute-joint radius bound")
            self.stats["bounds_sampler_cache_hits"] += int(bool(result.get("cache_hit")))
            box = low, high
        except Exception as error:
            from simbridge.errors import is_transport_failure

            if (isinstance(error, TimeoutError) or is_transport_failure(error)
                    or "restore failed" in str(error).lower()):
                raise
            self.stats["bounds_rejected_nodes"] += 1
            if len(self.stats["bounds_rejections"]) < 8:
                self.stats["bounds_rejections"].append({"node": node.get("id"), "reason": str(error)})
        finally:
            self.stats["aabb_seconds"] += time.perf_counter()-began
        # Memoize refusals for identical fragments as well as successful boxes.
        self._node_bounds[identity] = deepcopy(geometry), (box, path)
        return box, path

    def _fallback(self, a, b, reason, projection=None):
        self.stats["fallback_calls"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0)+1
        self._fallback_projection = projection
        try:
            return super().__call__(a, b)
        finally:
            self._fallback_projection = None

    def __call__(self, node_a, node_b):
        began = time.perf_counter()
        nodes = node_a, node_b
        if node_a["arm"] == node_b["arm"]:
            return super().__call__(*nodes)
        context, mismatch = _merge_modes(*nodes, self.cfg)
        if mismatch:
            return super().__call__(*nodes)
        reason = self._eligibility(nodes, context)
        if reason:
            return self._fallback(*nodes, reason)
        if not self._bounds_compatible:
            return self._fallback(*nodes, "collision_geometry_policy_mismatch")
        self.stats["broadphase_eligible_calls"] += 1
        projection = self._projection(context)
        a, b = (next(n for n in nodes if n["arm"] == arm) for arm in ARMS)
        (ba, path_a), (bb, path_b) = self._bounds(a), self._bounds(b)
        if ba is None or bb is None:
            return self._fallback(*nodes, "bounds_refused", projection)
        axis = next((i for i in range(3) if ba[1][i] < bb[0][i] or bb[1][i] < ba[0][i]), None)
        if axis is None:
            return self._fallback(*nodes, "bounds_overlap_or_touch", projection)
        self.stats["broadphase_disjoint_calls"] += 1
        self.stats["cross_calls"] += 1
        self.stats["oracle_calls"] += 1
        return self._cross(nodes, (path_a, path_b), context, projection, (ba, bb), axis, began)

    def _cross(self, nodes, paths, context, projection, bounds, axis, began):
        full_key, good_a, good_b = projection
        path_a, path_b = (list(map(tuple, path)) for path in paths)
        anchor = path_a[0], path_b[0]
        remaining_a = [q for q in path_a if q not in good_a]
        remaining_b = [q for q in path_b if q not in good_b]
        hits = len(path_a)+len(path_b)-len(remaining_a)-len(remaining_b)
        self.stats["single_cache_lookups"] += len(path_a)+len(path_b)
        self.stats["single_cache_hits"] += hits

        pairs = dict.fromkeys((q, anchor[1]) for q in remaining_a)
        pairs.update(dict.fromkeys((anchor[0], q) for q in remaining_b))
        if not pairs and full_key not in self._full_context_ok:
            pairs[anchor] = None

        queried, considered, exact_hits = 0, 0, 0
        problem = None
        pending = []
        for qa, qb in pairs:
            sample = {"A": list(qa), "B": list(qb), **context}
            considered += 1
            key = _canonical(sample)
            if key in self.cache:
                exact_hits += 1
                if not self.cache[key]["ok"]:
                    problem = self._collision(sample, self.cache[key])
                    break
            else:
                pending.append((sample, key))
                if len(pending) == BATCH_SIZE:
                    problem = self._cross_batch(pending, projection)
                    queried += len(pending)
                    pending.clear()
                    if problem is not None:
                        break
        if pending and problem is None:
            problem = self._cross_batch(pending, projection)
            queried += len(pending)

        self.stats["mesh_state_queries"] += queried
        self.stats["cache_hits"] += exact_hits
        self.stats["states_considered"] += considered
        self.stats["zero_rpc_cross_calls"] += int(queried == 0)
        evidence = {
            "method": "swept_AABB_separated_cross_full_scene",
            "separation_axis": axis,
            "bounds_A": {"min": bounds[0][0], "max": bounds[0][1]},
            "bounds_B": {"min": bounds[1][0], "max": bounds[1][1]},
            "joint_sample_step_rad": self.step,
            "all_relative_phases": True,
            "continuous_collision_guarantee": False,
            "waiting_endpoint_verification": False,
            "backend": type(self._remote).__name__+".sample_state_pairs",
            "grid_shape": [len(path_a), len(path_b)],
            "pair_states_total": len(path_a)*len(path_b),
            "pair_states_considered": considered,
            "new_mesh_queries": queried,
            "cache_hits": exact_hits,
            "single_ok_cache_hits": hits,
            "checked_all_pair_states": considered == len(path_a)*len(path_b) and problem is None,
            "mode_fingerprint": hashlib.sha256(_canonical(context).encode()).hexdigest(),
        }
        if problem is not None:
            evidence.update(problem)
        return self._record(*nodes, {"conflict": problem is not None, "evidence": evidence}, began)

    def _cross_batch(self, entries, projection):
        samples = [sample for sample, _ in entries]

        class Batch:
            def sample_state_pairs(_, batch):
                return self._sample("cross", batch, projection)

        answers = _answers(Batch(), samples)
        self.stats["batch_calls"] += 1
        problem = None
        for (sample, key), answer in zip(entries, answers):
            self.cache[key] = answer
            if not answer["ok"] and problem is None:
                problem = self._collision(sample, answer)
        return problem

    @staticmethod
    def _collision(sample, answer):
        return {"problem": deepcopy(answer.get("problem")), "qA": sample["A"], "qB": sample["B"],
                "payloads": sample["payloads"], "placed": sample["placed"]}

def _configuration_at(node, stamp, beginning, end):
    """Use runtime segment timing when supplied; otherwise reference arclength."""
    elapsed = max(0., stamp-beginning)
    segments = node.get("motion_segments")
    if segments:
        for segment in segments:
            duration = float(segment["duration"])
            if elapsed <= duration+EPS:
                if segment.get('geometry_profile'):
                    from .curve import eval_profile
                    return eval_profile(segment['geometry_profile'], elapsed+float(segment.get('profile_offset', 0.)))
                profile_duration = float(segment.get("profile_duration", duration))
                profile_offset = float(segment.get("profile_offset", 0.))
                u = min(1., max(0., (elapsed+profile_offset)/profile_duration)) if profile_duration > EPS else 1.
                v = u*u*u*(10+u*(-15+6*u))
                return [a+(b-a)*v for a,b in zip(segment["q0"],segment["q1"])]
            elapsed -= duration
        return _q(node.get("q_end", segments[-1]["q1"]))
    points = node.get("q_samples") or [node["q_start"],node.get("q_end",node["q_start"])]
    points = [_q(q) for q in points]
    if len(points)==1 or end-beginning <= EPS:
        return points[-1]
    lengths = [math.sqrt(sum((a-b)**2 for a,b in zip(q0,q1))) for q0,q1 in zip(points,points[1:])]
    total = sum(lengths)
    if total <= EPS:
        return points[-1]
    distance = min(1., max(0., (stamp-beginning)/(end-beginning)))*total
    for index,length in enumerate(lengths):
        if distance <= length+EPS:
            fraction = min(1., distance/length) if length>EPS else 1.
            return [a+(b-a)*fraction for a,b in zip(points[index],points[index+1])]
        distance -= length
    return points[-1]


def verify_rollout(tpg, planner, cfg, step_s=.05, *, outdir=None, stop_on_first_failure=True):
    """Validate a nominal simultaneous geometric rollout, including all waits.

    Source poses come from cfg.parts, completed place skills use their declared
    goal poses, and pick/release effects update the held-object mode. This is
    an explicit nominal world model, not measured dynamics. Unsupported events
    mark the world-state certificate incomplete instead of silently passing.
    """
    step_s = _positive(step_s, "step_s")
    output = Path(outdir or cfg.get("apex_output_dir") or Path(__file__).resolve().parents[1]/"results"/"apex"/"geometry")
    output.mkdir(parents=True, exist_ok=True)
    nodes = list(tpg["nodes"])
    timestamps = tpg.get("nominal_timestamps", {})
    timing = {n["id"]: timestamps.get(n["id"], n) for n in nodes}
    end = max((float(timing[n["id"]]["end"]) for n in nodes),default=0.)
    part_map = _parts(cfg)
    placed = {pid: deepcopy(p["source"]) for pid,p in part_map.items() if "source" in p}
    missing_sources = [pid for pid in part_map if pid not in placed]
    held = deepcopy(cfg.get("initial_payloads", {}))
    support = deepcopy(cfg.get("initial_support_contacts", {}))
    articulations = deepcopy(cfg.get("initial_articulations", {}))
    for payload in held.values():
        placed.pop(_payload_name(payload), None)
    histories = {arm: sorted([n for n in nodes if n["arm"]==arm],key=lambda n:float(timing[n["id"]]["start"])) for arm in ARMS}
    resting = {}
    for arm in ARMS:
        initial = cfg.get("arms",{}).get(arm,{}).get("home_q")
        if initial is None and histories[arm]:
            initial = histories[arm][0].get("q_start")
        if initial is None:
            raise ValueError(f"No initial joint state for arm {arm}")
        resting[arm] = _q(initial)
    finish_events = sorted(nodes,key=lambda n:(float(timing[n["id"]]["end"]),n.get("reference_index",0)))
    event_index = 0
    stamps = {0.,end}
    count = int(math.ceil(end/step_s))
    stamps.update(min(end,i*step_s) for i in range(count+1))
    for row in timing.values():
        stamps.update([float(row["start"]),float(row["end"])])
    stamps = sorted(stamps)
    unsupported = []
    transfers = {}
    first_violation = None
    tested, batches = 0, 0
    batch = []
    summaries = []
    started = time.perf_counter()

    def finish_event(node):
        arm = node["arm"]
        metadata = node.get("metadata",{})
        action = metadata.get("action")
        kind = node["kind"]
        if cfg.get("profile") == "skillnet_v1_3" and metadata.get("handover"):
            part = node.get("part")
            giver, taker = metadata.get("giver"), metadata.get("taker")
            if action == "grasp":
                binding = metadata.get("transfer_payload") or metadata.get("payload_after")
                if (not binding or _payload_name(held.get(giver)) != part
                        or taker in held or part in transfers):
                    unsupported.append({"node":node["id"],"reason":"invalid handover receiver mode"})
                else:
                    transfers[part] = {"giver":giver,"taker":taker,"payload":deepcopy(binding),
                                       "contacts":deepcopy(metadata.get("contact_exclusions", []))}
            elif action == "release":
                pending = transfers.pop(part, None)
                if (not pending or pending["giver"] != giver or pending["taker"] != taker
                        or _payload_name(held.get(giver)) != part):
                    unsupported.append({"node":node["id"],"reason":"handover release without matching completed receiver grasp"})
                else:
                    held.pop(giver, None)
                    held[taker] = pending["payload"]
                    placed.pop(part, None)
        elif kind=="pick" or action=="grasp":
            payload = metadata.get("payload_after")
            if not payload:
                unsupported.append({"node":node["id"],"reason":"pick has no payload_after transform"})
            else:
                name = _payload_name(payload)
                if arm in held or any(_payload_name(p)==name for p in held.values()):
                    unsupported.append({"node":node["id"],"reason":"pick violates single-holder/single-payload state"})
                held[arm] = deepcopy(payload)
                placed.pop(_payload_name(payload),None)
        elif kind=="place" or action=="release":
            part = node.get("part",metadata.get("part"))
            goal = metadata.get("goal",part_map.get(part,{}).get("goal"))
            if arm not in held or _payload_name(held[arm]) != part:
                unsupported.append({"node":node["id"],"reason":"place lacks matching declared held part"})
            if part is None or goal is None:
                unsupported.append({"node":node["id"],"reason":"place has no declared part/goal"})
            else:
                placed[part] = deepcopy(goal)
            held.pop(arm,None)
        elif kind=="support_start":
            support[arm] = deepcopy(metadata.get("contact_exclusions",[]))
        elif kind=="support_end":
            support.pop(arm,None)
        elif action not in (None,"wait","verify_support_load","verify_permanent_structure",
                            "verify_support","verify_permanent","load_test","lock_structure") and not metadata.get("effects"):
            unsupported.append({"node":node["id"],"reason":"unsupported action/world transition","action":action})
        # Custom contact/actuator skills must explicitly declare their geometric
        # state effects. These are nominal predictions, never measured events.
        effects=metadata.get("effects", {})
        for key in effects:
            if key not in {"placed","payloads","articulations","support_contacts"}:
                unsupported.append({"node":node["id"],"reason":"unknown geometric effect", "effect":key})
        for name, value in effects.get("placed", {}).items():
            if value is None: placed.pop(name, None)
            else: placed[name]=deepcopy(value)
        for side, value in effects.get("payloads", {}).items():
            if value is None: held.pop(side, None)
            else:
                held[side]=deepcopy(value)
                placed.pop(_payload_name(value), None)
        articulations.update(effects.get("articulations", {}))
        for side, value in effects.get("support_contacts", {}).items():
            if not value: support.pop(side, None)
            else: support[side]=deepcopy(value)

    def flush():
        nonlocal tested,batches,first_violation
        if not batch:
            return
        results = _answers(planner,[item["sample"] for item in batch])
        batches += 1; tested += len(batch)
        for item,answer in zip(batch,results):
            if not answer["ok"] and first_violation is None:
                first_violation = {"time_s":item["time_s"],"problem":answer.get("problem"),
                                   "active_nodes":item["active_nodes"],**item["sample"]}
        summaries.append({"start_s":batch[0]["time_s"],"end_s":batch[-1]["time_s"],
                          "samples":len(batch),"invalid_samples":sum(not r["ok"] for r in results)})
        batch.clear()

    for stamp in stamps:
        while event_index<len(finish_events) and float(timing[finish_events[event_index]["id"]]["end"]) <= stamp+EPS:
            node=finish_events[event_index]
            resting[node["arm"]]=_q(node["q_end"])
            finish_event(node)
            event_index+=1
        sample={"A":deepcopy(resting["A"]),"B":deepcopy(resting["B"]),
                "payloads":deepcopy(held),"placed":deepcopy(placed),"options":{}}
        active_ids={}
        contacts=_pairs(*support.values(), *[transfer["contacts"] for transfer in transfers.values()])
        sample_articulations=deepcopy(articulations)
        for arm in ARMS:
            active=[n for n in histories[arm] if float(timing[n["id"]]["start"])-EPS <= stamp < float(timing[n["id"]]["end"])-EPS]
            if len(active)>1:
                unsupported.append({"time_s":stamp,"reason":"two nodes occupy same arm","arm":arm})
            if active:
                node=active[0];row=timing[node["id"]]
                sample[arm]=_configuration_at(node,stamp,float(row["start"]),float(row["end"]))
                active_ids[arm]=node["id"]
                if node.get("payload"):
                    expected = _payload_name(node["payload"])
                    if arm not in held or _payload_name(held[arm]) != expected:
                        unsupported.append({"node":node["id"],"time_s":stamp,
                                            "reason":"active carrying mode has no matching completed grasp"})
                    sample["payloads"][arm]=deepcopy(node["payload"])
                contacts=_pairs(contacts,node.get("metadata",{}).get("contact_exclusions",[]))
                metadata=node.get("metadata",{})
                sample_articulations.update(metadata.get("options",{}).get("articulations",{}))
                for name, motion in metadata.get("articulation_motion",{}).items():
                    fraction=min(1.,max(0.,(stamp-float(row["start"]))/max(EPS,float(row["end"])-float(row["start"]))))
                    sample_articulations[name]=float(motion["start"])+(float(motion["end"])-float(motion["start"]))*fraction
        for payload in sample["payloads"].values():
            sample["placed"].pop(_payload_name(payload),None)
        sample["options"]={"contact_exclusions":_pairs(contacts,_resting_contacts(sample["placed"],sample["payloads"],cfg)),
                           "articulations":sample_articulations,
                           "contact_tolerance":.003}
        batch.append({"sample":sample,"time_s":stamp,"active_nodes":active_ids})
        if len(batch)>=BATCH_SIZE:
            flush()
            if first_violation is not None and stop_on_first_failure:
                break
    if first_violation is None or not stop_on_first_failure:
        flush()
    complete = tested==len(stamps)
    if transfers:
        unsupported.append({"reason":"unfinished handover transfer", "parts":sorted(transfers)})
    state_complete = not unsupported and not missing_sources
    report={"kind":"nominal_geometric_rollout_verification","step_s":step_s,
            "nominal_makespan_s":end,"planned_samples":len(stamps),"tested_samples":tested,
            "batch_calls":batches,"coverage_complete":complete,"batch_statistics":summaries,
            "first_violation":first_violation,"no_sampled_collision":first_violation is None and complete,
            "nominal_world_state_reconstructed":state_complete,"world_state_gaps":unsupported,
            "missing_source_poses":missing_sources,"stationary_waiting_poses_included":True,
            "valid":first_violation is None and complete and state_complete,
            "continuous_collision_guarantee":False,"dynamics_or_force_verified":False,
            "world_state_basis":"cfg source poses + declared pick payload transforms + completed place goal poses",
            "trajectory_timing_basis":"motion_segments quintic timing when supplied; otherwise reference-polyline arclength",
            "wall_s":time.perf_counter()-started}
    (output/"rollout_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


__all__=["SampledConflictOracle","BroadphaseConflictOracle","verify_rollout"]
