"""Sampled real-mesh conflict checks for the normal multi-modal TPG.

The motion backend supplies transactional mesh collision checks. This module
only combines exact joint/object modes, preserves path corners while sampling,
and batches the resulting pairs. Sampling is not a continuous collision proof.
No simulator connection, scene mutation, or dynamics execution is performed
outside the supplied MotionPlanner's transactional query interface.
"""
from __future__ import annotations

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
                profile_duration = float(segment.get('profile_duration', duration))
                profile_offset = float(segment.get('profile_offset', 0.))
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
    output = Path(outdir or cfg.get("normal_output_dir") or Path(__file__).resolve().parents[1]/"results"/"normal"/"geometry")
    output.mkdir(parents=True, exist_ok=True)
    nodes = list(tpg["nodes"])
    timestamps = tpg.get("nominal_timestamps", {})
    timing = {n["id"]: timestamps.get(n["id"], n) for n in nodes}
    end = max((float(timing[n["id"]]["end"]) for n in nodes),default=0.)
    part_map = _parts(cfg)
    placed = {pid: deepcopy(p["source"]) for pid,p in part_map.items() if "source" in p}
    missing_sources = [pid for pid in part_map if pid not in placed]
    held, support = {}, {}
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
        if kind=="pick" or action=="grasp":
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
        elif action not in (None,"wait"):
            unsupported.append({"node":node["id"],"reason":"unsupported action/world transition","action":action})

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
        contacts=_pairs(*support.values())
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
        for payload in sample["payloads"].values():
            sample["placed"].pop(_payload_name(payload),None)
        sample["options"]={"contact_exclusions":_pairs(contacts,_resting_contacts(sample["placed"],sample["payloads"],cfg)),
                           "contact_tolerance":.003}
        batch.append({"sample":sample,"time_s":stamp,"active_nodes":active_ids})
        if len(batch)>=BATCH_SIZE:
            flush()
            if first_violation is not None and stop_on_first_failure:
                break
    if first_violation is None or not stop_on_first_failure:
        flush()
    complete = tested==len(stamps)
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


__all__=["SampledConflictOracle","verify_rollout"]
