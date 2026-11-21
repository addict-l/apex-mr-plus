"""Compile symbolic skills into transactional, geometry-checked APEX tasks.

Compilation advances a sequential geometric reference while retaining the
symbolic graph's concurrency. A later overlapping symbolic start snapshot
cannot undo an object transition already emitted by this reference compiler.
"""
from contextlib import nullcontext
from copy import deepcopy
from heapq import heappop, heappush
import hashlib
import json
import math
from itertools import count
import time

from scipy.spatial.transform import Rotation, Slerp

from skillnet.capability.oracle import Status


from .errors import is_transport_failure


_PRECHECK_EVENTS = count()


class CompileFailure(RuntimeError):
    """A classified motion failure, with the failed request and evidence."""

    def __init__(self, status, detail, segment=None, evidence=None):
        self.status = Status(status)
        self.detail = str(detail)
        self.segment = segment
        self.evidence = evidence
        super().__init__(self.detail)


def _unique(values):
    return list(dict.fromkeys(values))


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _proven(evidence):
    return isinstance(evidence, dict) and bool(
        evidence.get("infeasible") is True or evidence.get("proven_infeasible") is True)


def _failure_status(answer, *, ik=False):
    """Sampling exhaustion is not a proof of geometric impossibility."""
    if answer.get("status") in ("INVALID", Status.INVALID):
        return Status.INVALID
    if answer.get("infeasible") is True or answer.get("proven_infeasible") is True:
        return Status.INVALID
    if not ik:
        error = str(answer.get("error", "")).lower()
        if (answer.get("collision") is True or error.startswith(("invalid start", "invalid goal"))
                or "outside joint bounds" in error or "dense path validation failed" in error):
            return Status.INVALID
    return Status.UNKNOWN


class MotionCompiler:
    """Compile with bounded IK seeds, RRT time, and Cartesian line sampling.

    ``compile(schedule, graph=None)`` resets to the workcell's initial state,
    accepts a SearchResult or SearchStep sequence, and returns APEX tasks.
    ``compile_action(action, state)`` is incremental and transactional. Failed
    calls restore q, payloads, placed objects and tasks, retaining diagnostics.
    ``held`` maps arm names to TCP-relative payload specifications.
    """

    def __init__(self, planner, cfg, frames, *, ik_seeds=8, rrt_timeout=10.,
                 linear_step=.01, joint_step=.035, force_rrt=True,
                 linear_joint_delta=.25, linear_position_tolerance=.002,
                 linear_orientation_tolerance=.02, local_box_margin=.6,
                 local_box_time=4., path_cache=None, action_candidate_limit=4,
                 qualification_hints=None):
        if isinstance(ik_seeds, bool) or not isinstance(ik_seeds, int) or ik_seeds < 1:
            raise ValueError("IK 种子数必须为正整数")
        if (isinstance(action_candidate_limit, bool) or not isinstance(action_candidate_limit, int)
                or action_candidate_limit < 1):
            raise ValueError("完整动作候选数必须为正整数")
        if not all(math.isfinite(x) and x > 0 for x in
                   (rrt_timeout, linear_step, joint_step, linear_joint_delta,
                    linear_position_tolerance, linear_orientation_tolerance)):
            raise ValueError("运动预算与采样间隔必须为有限正数")
        self.planner, self.cfg, self.frames = planner, cfg, frames
        self.ik_seeds, self.rrt_timeout = ik_seeds, float(rrt_timeout)
        self.linear_step, self.joint_step = float(linear_step), float(joint_step)
        self.force_rrt = bool(force_rrt)
        self.linear_joint_delta = float(linear_joint_delta)
        self.linear_position_tolerance = float(linear_position_tolerance)
        self.linear_orientation_tolerance = float(linear_orientation_tolerance)
        # Two-stage free motion search (FIX_v1.3.2.md section 3): a bounded
        # joint box around both endpoints first, then the full joint range
        # with the rest of the same RRT budget. Both stages keep every
        # certificate; a box failure alone never decides UNKNOWN.
        if local_box_margin is not None and (not math.isfinite(local_box_margin) or local_box_margin <= 0):
            raise ValueError("局部关节盒半径必须为正数或 None")
        if not math.isfinite(local_box_time) or local_box_time <= 0 or local_box_time >= self.rrt_timeout:
            raise ValueError("局部搜索时间必须为正数且小于 RRT 总预算")
        self.local_box_margin = None if local_box_margin is None else float(local_box_margin)
        self.local_box_time = float(local_box_time)
        self.path_cache = {} if path_cache is None else path_cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.action_candidate_limit = action_candidate_limit
        self.qualification_hints = deepcopy(qualification_hints or [])
        self._candidate_session = None
        self._endpoint_overrides = {}
        self._selected_endpoints = []
        self._action_bindings = {}
        self._last_transit_candidates = []
        self._last_cache_key = None
        self.evidence = []
        self.deadline = None
        self.reset()

    def reset(self):
        self.q = {a: list(self.cfg["arms"][a]["home_q"]) for a in ("A", "B")}
        self.held = deepcopy(self.cfg.get("initial_payloads", {}))
        self.placed = deepcopy(self.cfg.get("initial", {}))
        for name in self.frames.blocks:
            self.placed.setdefault(name, self.frames.block_pose(name))
        for payload in self.held.values():
            self.placed.pop(payload["name"], None)
        self.tasks, self._last_arm, self._touched = [], {}, set()
        self._symbolic_context = ()
        self.evidence = []

    def _snapshot(self):
        return deepcopy((self.q, self.held, self.placed, self.tasks, self._last_arm,
                         self._touched, self._symbolic_context))

    def _restore(self, snapshot):
        (self.q, self.held, self.placed, self.tasks,
         self._last_arm, self._touched, self._symbolic_context) = snapshot

    def _payload(self, block, arm):
        return self.frames.payload(block, arm)

    def _sync_state(self, state):
        """Read untouched objects from state_before, never undo emitted effects."""
        for fact in state:
            if fact.predicate in {"at", "installed"}:
                block, place = fact.args
                if block not in self._touched:
                    self.placed[block] = self.frames.block_pose(block, place)
            elif fact.predicate == "held":
                arm, block = fact.args
                if block not in self._touched:
                    # A symbolic held fact names the part but carries no
                    # measured TCP binding. Keep an explicitly known binding.
                    if self.held.get(arm, {}).get("name") != block:
                        self.held[arm] = self._payload(block, arm)
                    self.placed.pop(block, None)

    def _options(self, request):
        other = "B" if request.arm == "A" else "A"
        options = deepcopy(request.metadata.get("options", {}))
        options.update(other_q=list(self.q[other]),
                       other_payload=deepcopy(request.payloads.get(other, {})),
                       contact_exclusions=deepcopy(request.contacts), contact_tolerance=.003)
        return options

    def _call(self, kind, request, function, *args, **kwargs):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise CompileFailure(Status.UNKNOWN, "运动编译达到任务墙钟超时", request.id,
                                 {"timeout": True})
        try:
            answer = function(*args, **kwargs)
        except CompileFailure:
            raise
        except Exception as exc:
            self.evidence.append({"request": request.id, "arm": request.arm,
                                  "query": kind, "error": f"{type(exc).__name__}: {exc}"})
            raise CompileFailure(Status.UNKNOWN,
                f"运动段 {request.id} 的 {kind} 调用失败：{type(exc).__name__}: {exc}",
                request.id) from exc
        self.evidence.append({"request": request.id, "arm": request.arm,
                              "query": kind, "answer": deepcopy(answer)})
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise CompileFailure(Status.UNKNOWN, "运动编译达到任务墙钟超时", request.id,
                                 {"timeout": True, "last_answer": answer})
        return answer

    def _candidate_policy(self, request=None):
        return {"version": "complete-action-endpoints-v135-5",
            "adjacent_linear_precheck_seconds": 2.,
            "follower": self._follower_context(request),
            "endpoint_order": "wrist_family,elbow_family,joint_distance,source",
            "family_singular_tolerance": 1e-6,
            "endpoint_cluster_max_delta_rad": 1e-4,
            "independent_hint_fallback": "last_bounded_slot",
            "grasp_seed_transfer": "same_catalog_type_pose_binding-v1; fresh_IK_only",
            "support_exit_strategies": ["default_lower", "horizontal_only-v1"],
            "action_candidate_limit": self.action_candidate_limit,
            "endpoint_rank": self._endpoint_overrides.get(request.id) if request else None,
            "ik_seeds": deepcopy(self.cfg.get("ik_seeds", [])),
            "qualification_hints": deepcopy(self.qualification_hints)}

    @staticmethod
    def _joint_family(configuration, joint):
        value = math.sin(configuration[joint])
        return 0 if abs(value) <= 1e-6 else (1 if value > 0 else -1)

    def _rank_transit_candidates(self, rows, seed):
        """Continuity guides trials; every endpoint still requires a path."""
        start = [self._joint_family(seed, joint) for joint in (4, 2)]
        sources = {"qualification_hint": 0, "qualification_grasp_seed": 1, "bounded_ik": 2}
        for row in rows:
            families = [self._joint_family(row["q"], joint) for joint in (4, 2)]
            row["joint_distance"] = math.dist(row["q"], seed)
            row["families"] = {"wrist": families[0], "elbow": families[1]}
            row["family_mismatches"] = [bool(a and b and a != b) for a, b in zip(start, families)]
        ranked = sorted(rows, key=lambda row: (*row["family_mismatches"], row["joint_distance"],
                                               sources.get(row["source"], 2)))
        # Keep one independent historical strategy available if the new order
        # would otherwise spend every bounded slot on fresh configurations.
        limit = self.action_candidate_limit
        if limit > 1 and not any(row["source"] == "qualification_hint" for row in ranked[:limit]):
            fallback = next((i for i, row in enumerate(ranked[limit:], limit)
                             if row["source"] == "qualification_hint"), None)
            if fallback is not None:
                row = ranked.pop(fallback)
                row["retained_hint_fallback"] = True
                ranked.insert(limit-1, row)
        return ranked

    def _cluster_transit_candidates(self, rows, request):
        """Suppress numerical repeats, without averaging or proving infeasibility."""
        clusters = []
        for row in rows:
            families = [self._joint_family(row["q"], joint) for joint in (4, 2)]
            match = next((anchor for anchor in clusters
                if max(abs(a-b) for a, b in zip(anchor["q"], row["q"])) <= 1e-4
                and all(not a or not b or a == b for a, b in zip(families,
                    [self._joint_family(anchor["q"], joint) for joint in (4, 2)]))), None)
            if match is None:
                match = deepcopy(row)
                match["provenance"] = []
                clusters.append(match)
            match["provenance"].append(deepcopy(row))
        self.evidence.append({"request": request.id, "arm": request.arm,
            "query": "IK_candidate_clusters", "answer": {
                "scope": "finite endpoint sampling only; not an infeasibility proof",
                "max_delta_rad": 1e-4, "input_count": len(rows),
                "suppressed_count": len(rows)-len(clusters), "clusters": deepcopy(clusters)}})
        return clusters

    def _request_context(self, request):
        return {"scene": [id(self.planner), self.cfg.get("root"), self.cfg.get("task_name")],
            "geometry_version": self.cfg.get("geometry_version"),
            "start_q": deepcopy(self.q), "held": deepcopy(self.held),
            "placed": deepcopy(self.placed), "state": self._symbolic_context,
            "request": deepcopy(vars(request)),
            "follower": self._follower_context(request),
            "limits": [self.ik_seeds, self.rrt_timeout, self.linear_step, self.joint_step,
                self.force_rrt, self.linear_joint_delta, self.linear_position_tolerance,
                self.linear_orientation_tolerance, self.local_box_margin, self.local_box_time]}

    def _hint_candidates(self, request, pose):
        """A qualification q is a suggestion, rechecked in the current scene."""
        result = []
        place = self._action_bindings.get("loc", self._action_bindings.get("slot"))
        block = self.frames.blocks.get(request.part, {})
        strategy = request.metadata.get("grasp_strategy")
        for hint in self.qualification_hints:
            if (not isinstance(hint, dict) or hint.get("arm") != request.arm
                    or hint.get("part") != request.part or hint.get("block_type") != block.get("type")
                    or hint.get("place") != place or hint.get("pose_mode") != "catalog"
                    or hint.get("grasp_strategy") != strategy):
                continue
            desired, q = hint.get("pose"), hint.get("q")
            if (not isinstance(desired, (list, tuple)) or len(desired) != 7
                    or not isinstance(q, (list, tuple)) or len(q) != 6
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in [*desired, *q])):
                continue
            try:
                rotation_error = (Rotation.from_quat(desired[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude()
            except ValueError:
                continue
            if (math.dist(desired[:3], pose[:3]) > self.linear_position_tolerance
                    or rotation_error > self.linear_orientation_tolerance):
                continue
            receipt = {"hint_id": _digest(hint), "q": list(q), "pose": list(pose),
                "context_id": _digest(self._request_context(request)),
                "source": "qualification_hint", "scope": "current target configuration only"}
            try:
                self._validate(request, q)
                measured = self._call("hint_FK", request, self.planner.fk, request.arm, q)
                actual = measured.get("pose")
                if (not measured.get("ok") or not actual or len(actual) != 7
                        or math.dist(actual[:3], pose[:3]) > self.linear_position_tolerance
                        or (Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude()
                            > self.linear_orientation_tolerance):
                    receipt.update(ok=False, reason="qualification_hint_FK_mismatch")
                else:
                    receipt["ok"] = True
                    result.append({"q": list(q), "source": "qualification_hint", "hint_id": receipt["hint_id"]})
            except CompileFailure as exc:
                if exc.status != Status.INVALID or _proven(exc.evidence):
                    raise
                receipt.update(ok=False, reason=exc.detail, evidence=deepcopy(exc.evidence))
            self.evidence.append({"query": "qualification_hint", "request": request.id,
                                  "arm": request.arm, "answer": receipt})
        return result

    def _hint_seeds(self, request):
        """Grasp-stage solutions seed the approach; they are never endpoints."""
        place = self._action_bindings.get("loc", self._action_bindings.get("slot"))
        block = self.frames.blocks.get(request.part, {})
        rows = []
        for hint in self.qualification_hints:
            if (not isinstance(hint, dict) or hint.get("stage") != "grasp"
                    or hint.get("arm") != request.arm
                    or hint.get("block_type") != block.get("type") or hint.get("place") != place
                    or hint.get("pose_mode") != "catalog"
                    or hint.get("grasp_strategy") != request.metadata.get("grasp_strategy")):
                continue
            source_part = hint.get("part")
            if source_part != request.part and not self._representative_seed_compatible(hint, request):
                continue
            q = hint.get("q")
            if (isinstance(q, (list, tuple)) and len(q) == 6
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) and abs(v) <= 2*math.pi for v in q)):
                provenance = {"source": "qualification_grasp_seed", "hint_id": _digest(hint),
                    "source_part": source_part, "requested_part": request.part,
                    "scope": "untrusted grasp seed; fresh approach IK and complete action required"}
                existing = next((row for row in rows
                    if max(abs(a-b) for a, b in zip(row[0], q)) <= 1e-4), None)
                if existing is None:
                    rows.append((list(q), provenance))
        return rows

    def _representative_seed_compatible(self, hint, request):
        """Transfer a seed only when the physical grasp is the same.

        A representative object's point is never an endpoint certificate and
        supplies neither contacts nor world state to the actual request.
        """
        source = hint.get("part")
        if (source not in self.frames.blocks or request.part not in self.frames.blocks
                or self.frames.blocks[source]["type"] != self.frames.blocks[request.part]["type"]):
            return False
        place = hint["place"]
        try:
            old = self.frames.grasp_pose(source, place, request.arm)
            current = self.frames.grasp_pose(request.part, place, request.arm)
            recorded = hint.get("pose")
            def same_pose(left, right):
                return (isinstance(left, (tuple, list)) and len(left) == 7
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in left)
                    and math.dist(left[:3], right[:3]) <= self.linear_position_tolerance
                    and (Rotation.from_quat(left[3:]).inv()*Rotation.from_quat(right[3:])).magnitude()
                        <= self.linear_orientation_tolerance)
            if not same_pose(recorded, current) or not same_pose(old, current):
                return False
            old_binding = self.frames.payload(source, request.arm, pose=old,
                body_pose=self.frames.block_pose(source, place))
            binding = self.frames.payload(request.part, request.arm, pose=current,
                body_pose=self.frames.block_pose(request.part, place))
            bindings = [old_binding] + ([request.held] if request.held else [])
            return all(same_pose([*value["offset"], *value["quaternion"]],
                                [*binding["offset"], *binding["quaternion"]]) for value in bindings)
        except (KeyError, TypeError, ValueError):
            return False

    def _ik(self, request, pose, seed, *, max_joint_delta=None):
        collect_transit = request.kind == "transit" and max_joint_delta is None
        hint_seeds = self._hint_seeds(request)[:max(0, self.ik_seeds-1)] if collect_transit else []
        candidates = [*([row[0] for row in hint_seeds]), list(seed), *deepcopy(self.cfg.get("ik_seeds", [])),
                      [-2.5, -1.2, -2., -1.57, -1.57, 0.],
                      [0., -1.15, 1.7, -2.1, -1.57, 0.],
                      [-2.35, -.25, 1.57, .25, -1.57, -.78],
                      [-1.37, 1.24, -1.57, 1.90, -1.57, .20],
                      [math.pi, -1.15, 1.7, -2.1, -1.57, 0.],
                      [-1., -1.2, -2., -1.57, 1.57, 0.],
                      [2., 1.2, 2., -1.57, 1.57, 0.]]
        seeds = []
        for candidate in candidates:
            if candidate not in seeds:
                seeds.append(candidate)
        # Repeat a deterministic seed only when the caller explicitly requests
        # more attempts than the available branch seeds. The count stays finite.
        while len(seeds) < self.ik_seeds:
            seeds.append(list(seed))
        failures, solutions = [], []
        if collect_transit:
            solutions.extend(self._hint_candidates(request, pose))
        for attempt, candidate in enumerate(seeds[:self.ik_seeds], 1):
            source = next((row[1] for row in hint_seeds if row[0] == candidate), {"source": "bounded_ik"})
            if collect_transit:
                self.evidence.append({"query": "IK_seed", "request": request.id, "arm": request.arm,
                    "answer": {"seed": list(candidate), "request_kind": request.kind,
                        "seed_index": attempt, "seed_limit": self.ik_seeds, **deepcopy(source)}})
            options = self._options(request)
            options.update(seed=candidate, multiple=False, max_solutions=1, ik_time=.1)
            answer = self._call("IK", request, self.planner.solve_ik,
                request.arm, pose[:3], pose[3:], payload=request.held,
                placed=request.placed, options=options)
            if answer.get("ok"):
                q = answer.get("q") or next(iter(answer.get("solutions", [])), None)
                if q is None or len(q) != 6 or not all(math.isfinite(v) for v in q):
                    raise CompileFailure(Status.UNKNOWN, f"运动段 {request.id} 的 IK 未返回有效关节解", request.id, answer)
                # Fallback IK seeds can return an equivalent cyclic UR5 joint
                # winding near the trial seed. Express it near the actual arm
                # configuration before measuring branch continuity or planning.
                period = 2*math.pi
                q = [min((value+turn*period for turn in range(-2, 3)
                          if -period-1e-9 <= value+turn*period <= period+1e-9),
                         key=lambda equivalent: abs(equivalent-current), default=value)
                     for value, current in zip(q, seed)]
                delta = max(abs(a-b) for a,b in zip(q, seed))
                if max_joint_delta is not None and delta > max_joint_delta:
                    failures.append({"ok": False, "problem": {
                        "reason": "IK_branch_discontinuity", "max_joint_delta_rad": delta,
                        "limit_rad": max_joint_delta}, "answer": deepcopy(answer)})
                    continue
                if not collect_transit:
                    return list(q)
                solutions.append({"q": list(q), "seed_index": attempt, **deepcopy(source)})
                continue
            failures.append(answer)
            if not solutions and _failure_status(answer, ik=True) == Status.INVALID:
                raise CompileFailure(Status.INVALID,
                    f"运动段 {request.id} 明确不可达：{answer.get('problem') or answer.get('error')}", request.id, answer)
        if solutions:
            solutions = self._rank_transit_candidates(self._cluster_transit_candidates(solutions, request), seed)
            start_family = self._joint_family(seed, 2)
            self._last_transit_candidates = deepcopy(solutions)
            selected = solutions[0]["q"]
            # Family agreement is a candidate-ranking heuristic only. The
            # selected endpoint still needs the ordinary RRT and dense collision
            # certificate; it is not evidence of a connected free-space path.
            self.evidence.append({"request": request.id, "arm": request.arm,
                "query": "IK_selection", "answer": {
                    "ok": True, "q": list(selected), "start_q": list(seed),
                    "valid_candidates": [list(row["q"]) for row in solutions], "seed_budget": self.ik_seeds,
                    "start_elbow_family": start_family,
                    "selected_elbow_family": self._joint_family(selected, 2),
                    "candidate_details": deepcopy(solutions),
                    "selection_rule": "wrist family, elbow family, nearest actual q, source tie; RRT and complete action required"}})
            return list(selected)
        raise CompileFailure(Status.UNKNOWN,
            f"运动段 {request.id} 的 IK 已耗尽 {self.ik_seeds} 个种子："
            f"{failures[-1].get('problem') or failures[-1].get('error') or '未找到解'}",
            request.id, failures)

    def _validate(self, request, q):
        answer = self._call("collision", request, self.planner.validate_joint,
            request.arm, q, payload=request.held, placed=request.placed,
            options=self._options(request))
        if not answer.get("ok"):
            status = Status.UNKNOWN if answer.get("error") and not answer.get("problem") else Status.INVALID
            raise CompileFailure(status,
                f"运动段 {request.id} 的轨迹点碰撞或无效：{answer.get('problem') or answer.get('error')}",
                request.id, answer)

    def _line(self, request, start):
        answer = self._call("FK", request, self.planner.fk, request.arm, start)
        if not answer.get("ok") or len(answer.get("pose", [])) != 7:
            raise CompileFailure(Status.UNKNOWN, f"运动段 {request.id} 无法获取起点位姿", request.id, answer)
        beginning, ending = answer["pose"], request.target_pose
        rotations = Rotation.from_quat([beginning[3:], ending[3:]])
        angle = (rotations[0].inv()*rotations[1]).magnitude()
        count = max(1, math.ceil(math.dist(beginning[:3], ending[:3])/self.linear_step),
                    math.ceil(angle/.05))
        interpolate = Slerp([0., 1.], rotations)
        path = [list(start)]
        self._validate(request, path[0])
        for index in range(1, count+1):
            fraction = index/count
            position = [a+(b-a)*fraction for a, b in zip(beginning[:3], ending[:3])]
            pose = position+interpolate([fraction]).as_quat()[0].tolist()
            goal = self._ik(request, pose, path[-1], max_joint_delta=self.linear_joint_delta)
            before = path[-1]
            samples = max(1, math.ceil(max(abs(a-b) for a, b in zip(before, goal))/self.joint_step))
            for sample in range(1, samples+1):
                q = [a+(b-a)*sample/samples for a, b in zip(before, goal)]
                self._validate(request, q)
                # Collision-free interpolation can still leave the requested
                # Cartesian line (notably after a branch switch). Certify the
                # actual FK at every point emitted for execution.
                phase = (index-1+sample/samples)/count
                expected_position = [a+(b-a)*phase for a,b in zip(beginning[:3], ending[:3])]
                expected_rotation = interpolate([phase])[0]
                actual = self._call("line_FK", request, self.planner.fk, request.arm, q)
                if not actual.get("ok") or len(actual.get("pose", [])) != 7:
                    raise CompileFailure(Status.UNKNOWN, f"运动段 {request.id} 的直线 FK 证书缺失", request.id, actual)
                measured = actual["pose"]
                position_error = math.dist(measured[:3], expected_position)
                angle_error = (expected_rotation.inv()*Rotation.from_quat(measured[3:])).magnitude()
                if (position_error > self.linear_position_tolerance
                        or angle_error > self.linear_orientation_tolerance):
                    diagnostic = {"reason": "Cartesian_line_deviation", "sample": [index, sample],
                        "position_error_m": position_error, "orientation_error_rad": float(angle_error),
                        "position_tolerance_m": self.linear_position_tolerance,
                        "orientation_tolerance_rad": self.linear_orientation_tolerance}
                    raise CompileFailure(Status.UNKNOWN,
                        f"运动段 {request.id} 的关节插值偏离直线：{diagnostic}", request.id, diagnostic)
                path.append(q)
        return path

    def _follower_context(self, request):
        if request is not None and self._candidate_session is not None:
            follower = self._candidate_session["followers"].get(request.id)
            if follower is not None:
                return deepcopy(vars(follower))
        return None

    def _adjacent_order(self, request, candidates):
        """Rank the existing pool using an isolated check of its adjacent line."""
        order = list(range(len(candidates)))
        follower = self._candidate_session["followers"].get(request.id)
        if len(order) < 2 or follower is None:
            return order
        started = time.monotonic()
        task_deadline = self.deadline
        deadline = started + 2.
        if task_deadline is not None:
            deadline = min(deadline, task_deadline)
        ordinary_evidence = self.evidence
        bounded = getattr(self.planner, "precheck_deadline", None)
        decision = "no_valid_candidate"
        observations = []
        event_id = next(_PRECHECK_EVENTS)
        try:
            self.deadline = deadline
            self.evidence = []
            with bounded(deadline) if bounded is not None else nullcontext():
                for rank, candidate in enumerate(candidates):
                    now = time.monotonic()
                    if now >= deadline:
                        decision = "timeout"
                        break
                    observation = {"rank": rank, "q": list(candidate["q"])}
                    try:
                        samples = self._line(follower, list(candidate["q"]))
                        observation.update(status="valid", samples=len(samples))
                    except CompileFailure as exc:
                        observation.update(status="not_validated", motion_status=exc.status.value,
                                           detail=exc.detail)
                        if is_transport_failure(exc):
                            decision = "transport_failure"
                            raise
                        if (isinstance(exc.__cause__, TimeoutError)
                                or isinstance(exc.evidence, dict) and exc.evidence.get("timeout")):
                            observation["status"] = decision = "timeout"
                            receipt = getattr(exc.__cause__, "receipt", None)
                            if receipt is not None:
                                observation["deadline_receipt"] = receipt
                        elif not self._finite_line_failure(exc.evidence):
                            observation["status"] = decision = "unknown"
                    finally:
                        observation["seconds"] = time.monotonic() - now
                        observations.append(observation)
                    if time.monotonic() >= deadline:
                        decision = "timeout"
                    if decision in {"timeout", "unknown"}:
                        break
                    if observation["status"] == "valid":
                        if rank:
                            order = [rank, *order[:rank], *order[rank+1:]]
                            decision = "promoted"
                        else:
                            decision = "original_first_passed"
                        break
        finally:
            self.deadline = task_deadline
            self.evidence = ordinary_evidence
            self.evidence.append({"query": "adjacent_linear_precheck", "arm": request.arm,
                "request": request.id, "answer": {
                    "event_id": event_id, "status": decision, "original_ranks": list(range(len(candidates))),
                    "attempt_order": order, "candidates": observations,
                    "elapsed_seconds": time.monotonic() - started, "budget_seconds": 2.,
                    "follower": follower.id}})
        return order

    @staticmethod
    def _finite_line_failure(evidence):
        if isinstance(evidence, list):
            return bool(evidence) and all(isinstance(row, dict) and not row.get("error") for row in evidence)
        return isinstance(evidence, dict) and not evidence.get("error") and bool(
            evidence.get("problem") or evidence.get("reason"))

    def _path(self, request):
        start = list(self.q[request.arm])
        if request.kind == "skill":
            self._validate(request, start)
            return [start]
        if request.kind == "linear":
            if request.target_pose is None:
                raise CompileFailure(Status.INVALID, f"直线运动段 {request.id} 缺少目标位姿", request.id)
            return self._line(request, start)
        if request.kind != "transit":
            raise CompileFailure(Status.INVALID, f"运动段 {request.id} 类型未知：{request.kind}", request.id)
        if request.metadata.get("idle_arm_return"):
            # The rescued action may have installed new geometry or changed
            # the other arm's payload. Certify this exact return goal in that
            # resulting context, then connect it by the ordinary RRT path.
            self._validate(request, request.target_q)
        pool_key = _digest(self._request_context(request))
        session = self._candidate_session
        pool = session["pools"].get(pool_key) if session is not None else None
        if pool is None:
            if request.target_q is not None:
                candidates = [{"q": list(request.target_q), "source": "explicit_target_q"}]
            else:
                remembered = session["ik_failures"].get(pool_key) if session is not None else None
                if remembered is not None:
                    self.evidence.append({"query": "IK_deduplicated", "request": request.id,
                        "arm": request.arm, "answer": {"context_id": pool_key, "scope": "same finite seed set"}})
                    raise CompileFailure(remembered["status"], remembered["detail"],
                                         remembered["segment"], deepcopy(remembered["evidence"]))
                try:
                    self._ik(request, request.target_pose, start)
                except CompileFailure as exc:
                    if session is not None:
                        session["ik_failures"][pool_key] = {"status": exc.status.value,
                            "detail": exc.detail, "segment": exc.segment, "evidence": deepcopy(exc.evidence)}
                    raise
                candidates = deepcopy(self._last_transit_candidates)
            pool = {"key": pool_key, "request": request.id,
                    "candidates": candidates[:self.action_candidate_limit], "rejected": []}
            pool["order"] = (self._adjacent_order(request, pool["candidates"]) if session is not None
                             else list(range(len(pool["candidates"]))))
            if session is not None:
                session["pools"][pool_key] = pool
        options = self._options(request)
        if self.force_rrt:
            options["direct_first"] = False
        other_q = options.pop("other_q")
        exclusions = options.pop("contact_exclusions")
        stages = []
        if self.local_box_margin is not None and "joint_bound_margin" not in options:
            stages.append(("local_box", dict(options, joint_bound_margin=self.local_box_margin),
                           self.local_box_time))
        stages.append(("global", dict(options), None))
        # A request keeps this ledger across full-action backtracking. Charging
        # offered solver slices is conservative even when a solver returns early:
        # no second endpoint/candidate silently receives another ten seconds.
        ledger = ({"allocated": 0., "seconds": 0.} if session is None else
                  session["rrt"].setdefault(request.id, {"allocated": 0., "seconds": 0.}))
        attempts, answer = [], {"ok": False, "error": "No new endpoint within shared RRT budget"}
        order = pool["order"]
        first_rank = self._endpoint_overrides.get(request.id, order[0])
        ranks = [rank for rank in order[order.index(first_rank):] if rank not in pool["rejected"]]
        for position, rank in enumerate(ranks):
            candidate = pool["candidates"][rank]
            goal = list(candidate["q"])
            available = self.rrt_timeout-max(ledger["allocated"], ledger["seconds"])
            if self.deadline is not None:
                available = min(available, self.deadline-time.monotonic())
            if available <= 1e-9:
                break
            # Retain the ordinary local/global stages within each finite slice.
            # Sharing reserves a chance for each still-untried valid endpoint.
            endpoint_remaining = available/(len(ranks)-position)
            self.evidence.append({"query": "transit_endpoint", "request": request.id,
                "arm": request.arm, "answer": {"context_id": pool_key, "rank": rank,
                    **deepcopy(candidate), "change_reason": "initial_ranked_endpoint" if position == 0
                        and first_rank == 0 else "different_valid_IK_endpoint",
                    "rrt_shared_limit_s": self.rrt_timeout,
                    "rrt_allocated_s": ledger["allocated"], "endpoint_slice_s": endpoint_remaining}})
            for stage, stage_options, stage_time in stages:
                remaining = min(endpoint_remaining,
                                self.rrt_timeout-max(ledger["allocated"], ledger["seconds"]))
                if self.deadline is not None:
                    remaining = min(remaining, self.deadline-time.monotonic())
                if remaining <= 1e-9:
                    break
                # A small endpoint slice must still leave a global stage.
                local_time = stage_time
                if local_time is not None and endpoint_remaining <= local_time:
                    local_time = endpoint_remaining*(self.local_box_time/self.rrt_timeout)
                solve_time = remaining if local_time is None else min(local_time, remaining)
                ledger["allocated"] += solve_time
                endpoint_remaining -= solve_time
                began = time.monotonic()
                try:
                    answer = self._call("RRTConnect", request, self.planner.plan,
                        request.arm, start, goal, payload=request.held, placed=request.placed,
                        other_q=other_q, contact_exclusions=exclusions,
                        solve_time=solve_time, options=stage_options)
                finally:
                    ledger["seconds"] += time.monotonic()-began
                    if self.evidence and self.evidence[-1].get("query") == "RRTConnect":
                        self.evidence[-1].update(search_stage=stage, endpoint_rank=rank,
                                               endpoint_q=list(goal), context_id=pool_key)
                attempts.append({"stage": stage, "solve_time_s": solve_time,
                    "endpoint_rank": rank, "endpoint_q": list(goal), "ok": bool(answer.get("ok")),
                    "error": answer.get("error"), "problem": deepcopy(answer.get("problem"))})
                if answer.get("ok") or _failure_status(answer) == Status.INVALID:
                    break
            if answer.get("ok"):
                path = answer.get("path")
                if (not path or any(len(q) != 6 or not all(math.isfinite(v) for v in q) for q in path)
                        or max(abs(a-b) for a, b in zip(path[0], start)) > 1e-5
                        or max(abs(a-b) for a, b in zip(path[-1], goal)) > 1e-5):
                    raise CompileFailure(Status.UNKNOWN,
                        f"运动段 {request.id} 的规划路径缺失或端点不一致", request.id, answer)
                selection = {"request": request.id, "pool_key": pool_key,
                    "rank": rank, "q": list(goal), "source": candidate["source"],
                    "candidates": deepcopy(pool["candidates"]), "attempt_order": list(order)}
                self._selected_endpoints.append(selection)
                return deepcopy(path)
            pool["rejected"].append(rank)
            if _proven(answer):
                raise CompileFailure(Status.INVALID, f"运动段 {request.id} 明确不可行", request.id, answer)
            # Every endpoint uses the same start configuration. Moving the
            # endpoint cannot repair a currently colliding fixed start.
            if str(answer.get("error", "")).lower().startswith("invalid start"):
                break
        tried = "、".join(attempt["stage"] for attempt in attempts)
        raise CompileFailure(Status.UNKNOWN,
            f"运动段 {request.id} 的 RRT-Connect 失败（{tried} 均未找到解）："
            f"{answer.get('error')}; {answer.get('problem')}", request.id,
            {"attempts": attempts, "last_answer": deepcopy(answer), "finite_candidates": True,
             "context_id": pool_key, "rrt_shared_limit_s": self.rrt_timeout, **deepcopy(ledger)})

    def _emit(self, request, local_ids, step_id, external):
        path = self._cached_path(request)
        task_id = f"s{step_id}_{len(self.tasks):04d}_{request.arm}_{request.id}"
        missing = [dep for dep in request.dependencies if dep not in local_ids]
        if missing:
            raise CompileFailure(Status.INVALID, f"运动段 {request.id} 引用了未完成依赖：{missing}", request.id)
        dependencies = [local_ids[dep] for dep in request.dependencies]
        if not request.dependencies:
            dependencies.extend(external)
        if request.arm in self._last_arm:
            dependencies.append(self._last_arm[request.arm])
        metadata = deepcopy(request.metadata)
        metadata.update(action=request.action, source_step=step_id,
                        source_request=request.id, request_kind=request.kind,
                        placed=deepcopy(request.placed), contact_exclusions=deepcopy(request.contacts))
        if request.action in {"release", "install"} and not metadata.get("handover"):
            metadata["goal"] = deepcopy(metadata["target_pose"])
        if request.kind == "linear":
            metadata["target_pose"] = list(request.target_pose)
            metadata["linear_certificate"] = {
                "cartesian_step_m": self.linear_step, "max_joint_step_rad": self.joint_step,
                "max_branch_delta_rad": self.linear_joint_delta,
                "position_tolerance_m": self.linear_position_tolerance,
                "orientation_tolerance_rad": self.linear_orientation_tolerance,
                "fk_checked_states": len(path)-1}
        if request.speed == "contact":
            metadata.update(max_velocity_rad_s=.15, max_acceleration_rad_s2=.30)
        metadata.setdefault("dwell_s", request.duration or .20)
        effects = deepcopy(metadata.get("effects", {}))
        after_placed = metadata.get("placed_after")
        after_payloads = metadata.get("payloads_after")
        if after_placed is not None:
            effects["placed"] = {name: deepcopy(after_placed.get(name))
                for name in set(request.placed) | set(after_placed)
                if request.placed.get(name) != after_placed.get(name)}
        if after_payloads is not None:
            effects["payloads"] = {arm: deepcopy(after_payloads.get(arm))
                for arm in set(request.payloads) | set(after_payloads)
                if request.payloads.get(arm) != after_payloads.get(arm)}
        if effects:
            metadata["effects"] = effects
        duration = max(.08, request.duration or .20) if request.kind == "skill" else 1.
        task = {"id": task_id, "arm": request.arm, "kind": "transit" if request.kind != "skill" else (request.action or "wait"),
                "part": request.part, "duration": duration,
                "q_path": path, "payload": deepcopy(request.held),
                "reference_order": len(self.tasks), "dependencies": _unique(dependencies),
                "metadata": metadata}
        self.tasks.append(task)
        self.q[request.arm] = list(path[-1])
        self.placed = deepcopy(after_placed if after_placed is not None else request.placed)
        self.held = deepcopy(after_payloads if after_payloads is not None else request.payloads)
        self._last_arm[request.arm] = task_id
        local_ids[request.id] = task_id
        return task

    def _cached_path(self, request):
        """Reuse only certificates for the exact geometry and motion mode.

        Cache lifetime is one planning scene. A reset starts a new symbolic
        reference but deliberately keeps validated geometry certificates.
        The accepted two-stage planner and line verifier remain unchanged.
        """
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise CompileFailure(Status.UNKNOWN, "运动编译达到任务墙钟超时", request.id,
                                 {"timeout": True})
        from .reach_certificate import outer_sphere_rejection
        proof = outer_sphere_rejection(self.cfg["arms"].get(request.arm, {}), request.target_pose)
        if proof is not None:
            self.evidence.append({"request": request.id, "arm": request.arm,
                                  "query": "reach_certificate", "answer": deepcopy(proof)})
            problem = proof["problem"]
            raise CompileFailure(Status.INVALID,
                f"运动段 {request.id} 超出实际模型的串行转动链外球上界："
                f"距离 {problem['distance']:.6f} m > {problem['outer_bound']:.6f} m"
                f"（链长 {problem['radius']:.6f} m + 余量 {problem['margin']:.3f} m）",
                request.id, proof)
        contacts = sorted((sorted(pair, key=lambda value: json.dumps(value, sort_keys=True))
                           for pair in request.contacts), key=lambda pair: json.dumps(pair, sort_keys=True))
        context = {
            "scene": [id(self.planner), self.cfg.get("root"), self.cfg.get("task_name")],
            "geometry_version": self.cfg.get("geometry_version"),
            "arm": request.arm, "kind": request.kind, "action": request.action,
            "part": request.part, "target_pose": request.target_pose,
            "target_q": request.target_q, "start_q": self.q,
            "held": request.held, "payloads": request.payloads,
            "placed": request.placed, "contacts": contacts,
            "metadata": request.metadata, "state": self._symbolic_context,
            "candidate_policy": self._candidate_policy(request),
            "limits": [self.ik_seeds, self.rrt_timeout, self.linear_step, self.joint_step,
                       self.force_rrt, self.linear_joint_delta, self.linear_position_tolerance,
                       self.linear_orientation_tolerance, self.local_box_margin, self.local_box_time],
        }
        encoded = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
        key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self._last_cache_key = key
        if key in self.path_cache:
            entry = self.path_cache[key]
            self.cache_hits += 1
            self.evidence.append({"request": request.id, "arm": request.arm,
                "query": "path_cache", "answer": {"ok": True, "cache_key": key,
                    "certificate_queries": len(entry["certificate"]), "cache_hit": True}})
            selection = entry.get("endpoint_selection")
            if selection is not None:
                self._selected_endpoints.append(deepcopy(selection))
                if self._candidate_session is not None:
                    self._candidate_session["pools"].setdefault(selection["pool_key"], {
                        "key": selection["pool_key"], "request": request.id,
                        "candidates": deepcopy(selection["candidates"]),
                        "order": list(selection["attempt_order"]), "rejected": []})
            return deepcopy(entry["path"])
        self.cache_misses += 1
        first = len(self.evidence)
        path = self._path(request)
        self.path_cache[key] = {"context": deepcopy(context), "path": deepcopy(path),
                                "certificate": deepcopy(self.evidence[first:])}
        if request.kind == "transit" and self._selected_endpoints:
            selection = self._selected_endpoints[-1]
            if selection["request"] == request.id:
                self.path_cache[key]["endpoint_selection"] = deepcopy(selection)
                if selection["rank"] != context["candidate_policy"]["endpoint_rank"]:
                    actual_context = deepcopy(context)
                    actual_context["candidate_policy"]["endpoint_rank"] = selection["rank"]
                    self.path_cache[_digest(actual_context)] = {
                        **deepcopy(self.path_cache[key]), "context": actual_context}
        return path

    def _idle_other(self, action, state, requests):
        """Only an unoccupied arm may move before a one-arm action starts."""
        if (len(set(action.occupies)) != 1 or not requests
                or requests[0].kind != "transit" or requests[0].target_pose is None):
            return None
        arm = requests[0].arm
        other = "B" if arm == "A" else "A"
        if (not any(f.predicate == "free" and f.args == (other,) for f in state)
                or other in self.held
                or any(f.predicate in {"held", "supporting", "coheld"} and other in f.args for f in state)):
            return None
        supporting = False
        for task in self.tasks:
            event = task.get("metadata", {}).get("action") or task["kind"]
            if task["arm"] == other and event in {"support_start", "support_end"}:
                supporting = event == "support_start"
        if supporting:
            return None
        home = self.cfg["arms"][other].get("home_q", [])
        if len(home) != 6 or not all(math.isfinite(v) for v in home):
            return None
        if max(abs(math.remainder(value-target, 2*math.pi)) for value, target in zip(self.q[other], home)) <= 1e-5:
            return None
        return other

    @staticmethod
    def _interarm_exhaustion(failure):
        """Require structured inter-arm evidence; never reinterpret UNKNOWN."""
        if failure.status != Status.UNKNOWN or not isinstance(failure.evidence, list) or not failure.evidence:
            return False
        found = False
        for answer in failure.evidence:
            if (not isinstance(answer, dict) or answer.get("error") or answer.get("infeasible")
                    or answer.get("proven_infeasible") or answer.get("status") in ("INVALID", Status.INVALID)):
                return False
            problems = []
            if answer.get("problem"):
                problems.append(answer["problem"])
            stats = answer.get("stats", {})
            if not isinstance(stats, dict):
                return False
            rejected = stats.get("rejections", [])
            if rejected:
                if not isinstance(rejected, list):
                    return False
                problems.extend(rejected)
            for problem in problems:
                if not isinstance(problem, dict) or problem.get("reason") != "interarm_collision":
                    return False
                pair = problem.get("pair", [])
                if not isinstance(pair, list) or len(pair) != 2:
                    return False
                sides = [{side for side in ("A", "B") if side+"_UR5" in str(path).split("/")} for path in pair]
                if {frozenset(side) for side in sides} != {frozenset({"A"}), frozenset({"B"})}:
                    return False
                found = True
        return found

    def _idle_payload_exhaustion(self, failure, request, state, other):
        """Recognize only the current held body against the proven idle arm.

        Seed exhaustion stays UNKNOWN. This permits the existing single
        certified parking attempt, never a collision exclusion or IK proof.
        """
        payload = request.held
        if (other is None or self.ik_seeds != 8 or failure.status != Status.UNKNOWN
                or not isinstance(failure.evidence, list) or len(failure.evidence) != 8
                or not isinstance(payload, dict) or not payload.get("name")
                or self.held.get(request.arm) != payload
                or request.payloads.get(request.arm) != payload):
            return False
        block = payload["name"]
        if (block not in self.frames.blocks or block in request.placed
                or not any(f.predicate == "held" and f.args == (request.arm, block) for f in state)):
            return False
        reason, found = f"block_robot_collision:{block}:{other}", False
        for answer in failure.evidence:
            if (not isinstance(answer, dict) or answer.get("ok") is not False or answer.get("error")
                    or answer.get("infeasible") or answer.get("proven_infeasible")
                    or answer.get("status") not in (None, "UNKNOWN", Status.UNKNOWN)
                    or answer.get("q") or answer.get("solutions")):
                return False
            stats = answer.get("stats")
            if not isinstance(stats, dict):
                return False
            problems = [answer["problem"]] if answer.get("problem") else []
            rejected = stats.get("rejections", [])
            if rejected:
                if not isinstance(rejected, list):
                    return False
                problems.extend(rejected)
            if not problems:
                if stats.get("candidates") != 0:
                    return False
                continue
            for problem in problems:
                if not isinstance(problem, dict) or problem.get("reason") != reason:
                    return False
                pair = problem.get("pair")
                if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(p, str) for p in pair):
                    return False
                paths = [p.split("/") for p in pair]
                robot = [p for p in paths if other+"_UR5" in p and request.arm+"_UR5" not in p]
                body = [p for p in paths if p[-1] == block and not any(side+"_UR5" in p for side in ("A", "B"))]
                if len(robot) != 1 or len(body) != 1 or robot[0] == body[0]:
                    return False
                found = True
        return found

    def _emit_action_requests(self, action, state, requests, step_id, dependencies):
        """Optionally park once, retaining ordinary path and rollback checks.

        Recovery is allowed only before any action primitive has been emitted.
        A receipt is attached to an actual successful parking path, and only
        after the complete retried action succeeds. Scratch/final compilers
        sharing path_cache can replay that decision in the exact same context.
        """
        self._candidate_session["followers"] = {
            head.id: line for head, line in zip(requests, requests[1:])
            if (head.kind == "transit" and head.target_pose is not None and head.target_q is None
                and line.kind == "linear" and line.target_pose is not None
                and head.arm == line.arm and line.dependencies == [head.id]
                and head.action is None and line.action is None
                and head.held == line.held and head.payloads == line.payloads and head.placed == line.placed)
        }
        first = len(self.tasks)
        other = self._idle_other(action, state, requests)
        context_key, receipt = None, None
        if other is not None:
            context = {"scene": [id(self.planner), self.cfg.get("root"), self.cfg.get("task_name")],
                "geometry_version": self.cfg.get("geometry_version"), "action": action.key,
                "state": sorted(map(str, state)), "q": self.q, "held": self.held, "placed": self.placed,
                "touched": sorted(self._touched), "request": vars(requests[0]),
                "homes": {side: row["home_q"] for side, row in self.cfg["arms"].items()},
                "candidate_policy": self._candidate_policy(requests[0]),
                "limits": [self.ik_seeds, self.rrt_timeout, self.linear_step, self.joint_step,
                    self.force_rrt, self.linear_joint_delta, self.linear_position_tolerance,
                    self.linear_orientation_tolerance, self.local_box_margin, self.local_box_time]}
            encoded = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
            context_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            receipt = next((entry.get("idle_parking_recovery") for entry in self.path_cache.values()
                if entry.get("idle_parking_recovery", {}).get("context_key") == context_key), None)

        def emit(external):
            local_ids = {}
            for request in requests:
                self._emit(request, local_ids, step_id, external)

        if receipt is None:
            try:
                emit(dependencies)
                return
            except CompileFailure as failure:
                payload_collision = self._idle_payload_exhaustion(failure, requests[0], state, other)
                if (other is None or len(self.tasks) != first or failure.segment != requests[0].id
                        or not (self._interarm_exhaustion(failure) or payload_collision)):
                    raise
                original = {"status": failure.status.value, "detail": failure.detail,
                            "segment": failure.segment, "evidence": deepcopy(failure.evidence)}
                if payload_collision:
                    original["recovery_trigger"] = "held_payload_idle_arm_collision"
        else:
            original = deepcopy(receipt["original_failure"])
        self.evidence.append({"query": "idle_arm_parking", "request": requests[0].id,
            "arm": requests[0].arm, "answer": {"parked_arm": other, "context_key": context_key,
                "decision_cache_hit": receipt is not None, "original_failure": deepcopy(original)}})
        from .primitives import MotionRequest
        return_q = list(self.q[other])
        parking = MotionRequest("idle_home_"+other, other, "transit",
            target_q=list(self.cfg["arms"][other]["home_q"]), held=None,
            placed=deepcopy(self.placed), payloads=deepcopy(self.held), contacts=self._resting_contacts(),
            metadata={"source_action": action.text, "source_skill": action.name,
                "idle_arm_parking": True, "physical_only": True,
                "parking_return_required": True, "return_reference_q": list(return_q),
                "idle_parking_context": context_key, "recovery_for_request": requests[0].id})
        phase = "parking"
        try:
            parked = self._emit(parking, {}, step_id,
                                _unique([*dependencies, *self._last_arm.values()]))
            phase = "retry"
            emit(_unique([*dependencies, parked["id"]]))
            phase = "return"
            returning = MotionRequest("idle_return_"+other, other, "transit",
                target_q=list(return_q), held=deepcopy(self.held.get(other)),
                placed=deepcopy(self.placed), payloads=deepcopy(self.held), contacts=self._resting_contacts(),
                metadata={"source_action": action.text, "source_skill": action.name,
                    "idle_arm_return": True, "restores_symbolic_reference": True,
                    "idle_parking_context": context_key, "parking_task_id": parked["id"],
                    "return_reference_q": list(return_q)})
            completed = self.tasks[first:]
            referenced = {dep for task in completed for dep in task["dependencies"]}
            leaves = [task["id"] for task in completed if task["id"] not in referenced]
            self._emit(returning, {}, step_id, _unique([*leaves, *self._last_arm.values()]))
        except CompileFailure as recovery:
            # Failure of this optional parking route is not a proof that the
            # original action has no solution. Preserve UNKNOWN and raw proof.
            evidence = {"original_failure": original, "recovery_phase": phase,
                "recovery_status": recovery.status.value, "recovery_evidence": deepcopy(recovery.evidence)}
            raise CompileFailure(Status.UNKNOWN,
                f"原运动验证未知；空闲臂回位恢复的 {phase} 阶段未成功：{recovery.detail}",
                recovery.segment, evidence) from recovery
        for entry in self.path_cache.values():
            if entry.get("context", {}).get("metadata", {}).get("idle_parking_context") == context_key:
                entry["idle_parking_recovery"] = {"context_key": context_key,
                    "parked_arm": other, "return_reference_q": list(return_q),
                    "restores_symbolic_reference": True, "original_failure": deepcopy(original)}

    def compile_action(self, action, state, *, step_id=None, dependencies=()):
        from .primitives import PrimitiveError, expand

        snapshot, first = self._snapshot(), len(self.tasks)
        step_id = step_id if step_id is not None else first+1
        previous = (self._candidate_session, self._endpoint_overrides,
                    self._selected_endpoints, self._action_bindings)
        self._candidate_session = {"pools": {}, "rrt": {}, "ik_failures": {}, "followers": {}}
        self._endpoint_overrides, self._selected_endpoints = {}, []
        self._action_bindings = dict(zip(action.skill.params, action.args))
        trials, failure = [], None
        try:
            self._symbolic_context = tuple(sorted(str(fact) for fact in state))
            self._sync_state(state)
            try:
                requests = expand(action, state, self.frames, self.cfg,
                                  placed=self.placed, payloads=self.held)
            except PrimitiveError as exc:
                raise CompileFailure(exc.status, str(exc)) from exc
            action_start = self._snapshot()
            context = {"action": action.text, "state": self._symbolic_context,
                "q": deepcopy(self.q), "held": deepcopy(self.held), "placed": deepcopy(self.placed),
                "requests": [deepcopy(vars(request)) for request in requests],
                "geometry_version": self.cfg.get("geometry_version"), "policy": self._candidate_policy()}
            context_id = _digest(context)
            accepted = next((entry["complete_action_policies"][context_id]
                for entry in reversed(list(self.path_cache.values()))
                if context_id in entry.get("complete_action_policies", {})), None)
            if accepted is not None:
                self._endpoint_overrides = deepcopy(accepted["endpoint_policy"])
            support_exit_strategy = (accepted or {}).get("support_exit_strategy", "default_lower")
            if support_exit_strategy == "horizontal_only-v1":
                requests = self._horizontal_support_exit(requests)
            for attempt in range(1, self.action_candidate_limit+1):
                self._restore(deepcopy(action_start))
                self._selected_endpoints = []
                trial = {"context_id": context_id, "attempt": attempt,
                    "endpoint_policy": deepcopy(self._endpoint_overrides),
                    "support_exit_strategy": support_exit_strategy,
                    "change_reason": "initial_ranked_endpoints" if attempt == 1 else
                                     "different_endpoint_after_complete_action_rollback"}
                self.evidence.append({"query": "action_candidate", "action": action.text,
                    "answer": {**deepcopy(trial), "context": deepcopy(context)}})
                try:
                    self._emit_action_requests(action, state, requests, step_id, dependencies)
                    if "block" in self._action_bindings:
                        self._touched.add(self._action_bindings["block"])
                    self.path_cache[self._last_cache_key].setdefault("complete_action_policies", {})[context_id] = {
                        "context_id": context_id,
                        "support_exit_strategy": support_exit_strategy,
                        "endpoint_policy": {row["request"]: row["rank"] for row in self._selected_endpoints}}
                    trial.update(status="VALID", endpoints=deepcopy(self._selected_endpoints))
                    self.evidence.append({"query": "action_candidate_result", "action": action.text,
                                          "answer": deepcopy(trial)})
                    return deepcopy(self.tasks[first:])
                except CompileFailure as exc:
                    failure = exc
                    trial.update(status=exc.status.value, failed_segment=exc.segment,
                        endpoints=deepcopy(self._selected_endpoints), detail=exc.detail,
                        failed_tasks=deepcopy(self.tasks[first:]), evidence=deepcopy(exc.evidence))
                    trials.append(trial)
                    self.evidence.append({"query": "action_candidate_result", "action": action.text,
                                          "answer": deepcopy(trial)})
                    if (is_transport_failure(exc) or _proven(exc.evidence)
                            or self.deadline is not None and time.monotonic() >= self.deadline):
                        raise
                    if (action.name == "support_off" and support_exit_strategy == "default_lower"
                            and len(requests) == 3 and exc.segment == requests[-1].id
                            and self._support_lower_collision(exc.evidence)):
                        # The symbolic release and horizontal withdrawal remain
                        # intact. The extra fixed lower is only an exit choice;
                        # retry the entire action from its original checkpoint.
                        requests = self._horizontal_support_exit(requests)
                        support_exit_strategy = "horizontal_only-v1"
                        self._endpoint_overrides = {}
                        continue
                    # Reverse traversal is bounded depth-first backtracking.
                    # Earlier successful prefix segments may be reused, but the
                    # complete failed choice cannot be submitted a second time.
                    changed = False
                    for selection in reversed(self._selected_endpoints):
                        pool = self._candidate_session["pools"][selection["pool_key"]]
                        order = pool["order"]
                        later = next((rank for rank in order[order.index(selection["rank"])+1:]
                                      if rank not in pool["rejected"]), None)
                        if later is not None:
                            self._endpoint_overrides[selection["request"]] = later
                            changed = True
                            break
                        self._endpoint_overrides.pop(selection["request"], None)
                    if not changed:
                        break
            raise CompileFailure(Status.UNKNOWN, failure.detail, failure.segment,
                {**(deepcopy(failure.evidence) if isinstance(failure.evidence, dict) else {}),
                 "finite_candidates": True, "scope": "complete action at this geometric context",
                 "context_id": context_id, "candidate_limit": self.action_candidate_limit,
                 "action_candidate_attempts": trials,
                 "terminal_failure": {"status": failure.status.value, "detail": failure.detail,
                    "segment": failure.segment, "evidence": deepcopy(failure.evidence)}}) from failure
        except BaseException:
            self._restore(deepcopy(snapshot))
            raise
        finally:
            (self._candidate_session, self._endpoint_overrides,
             self._selected_endpoints, self._action_bindings) = previous

    @staticmethod
    def _support_lower_collision(evidence):
        if isinstance(evidence, list):
            return any(MotionCompiler._support_lower_collision(row) for row in evidence)
        if not isinstance(evidence, dict):
            return False
        problem = evidence.get("problem", {})
        if (isinstance(problem, dict) and "collision" in str(problem.get("reason", ""))
                and isinstance(problem.get("pair"), list) and len(problem["pair"]) == 2):
            return True
        stats = evidence.get("stats", {})
        return (isinstance(stats, dict)
                and MotionCompiler._support_lower_collision([
                    {"problem": row} for row in stats.get("rejections", [])
                    if isinstance(row, dict)]))

    @staticmethod
    def _horizontal_support_exit(requests):
        if (len(requests) != 3 or requests[0].action != "support_end"
                or any(row.kind != "linear" for row in requests[1:])):
            raise CompileFailure(Status.UNKNOWN, "托举撤离备选不匹配当前原语结构")
        result = deepcopy(requests[:2])
        for request in result:
            request.metadata["support_exit_strategy"] = "horizontal_only-v1"
        return result

    def compile(self, schedule, graph=None, *, return_home=True):
        from .primitives import MotionRequest

        self.reset()
        steps = sorted(getattr(schedule, "steps", schedule), key=lambda step: step.start)
        edges = getattr(graph, "edges", []) if graph is not None else []
        if isinstance(graph, dict):
            edges = graph.get("edges", [])
        predecessors = {}
        for edge in edges:
            source = edge["source"] if isinstance(edge, dict) else edge.source
            target = edge["target"] if isinstance(edge, dict) else edge.target
            predecessors.setdefault(target, []).append(source)
        completed = {}
        try:
            for number, step in enumerate(steps, 1):
                dependencies = []
                for source in predecessors.get(number, []):
                    if source not in completed:
                        raise CompileFailure(Status.INVALID, f"规划步骤 {number} 的前驱 {source} 尚未编译")
                    dependencies.extend(completed[source])
                tasks = self.compile_action(step.action, step.state_before,
                                            step_id=number, dependencies=dependencies)
                # All per-arm leaves of a two-arm skill must finish before a
                # causal successor can observe that skill's resulting world.
                referenced = {dep for task in tasks for dep in task["dependencies"]}
                completed[number] = [task["id"] for task in tasks if task["id"] not in referenced]
            self._support_event_dependencies(steps)
            if return_home:
                contacts = self._resting_contacts()
                for arm in ("A", "B"):
                    request = MotionRequest(id=f"home_{arm}", arm=arm, kind="transit",
                        target_q=list(self.cfg["arms"][arm]["home_q"]), held=deepcopy(self.held.get(arm)),
                        placed=deepcopy(self.placed), payloads=deepcopy(self.held), contacts=contacts,
                        metadata={"source_action": "home", "final_home": True})
                    self._emit(request, {}, "home", [])
            return deepcopy(self.tasks)
        except BaseException:
            diagnostics = self.evidence
            self.reset()
            self.evidence = diagnostics
            raise

    def _support_event_dependencies(self, steps):
        """Retain load-bearing invariants at their actual skill events.

        An approaching beam must be able to touch the shoe before measured
        readiness. Only its final install/release waits for that readiness;
        support release waits for the anchors that justify the planned state.
        """
        starts = [task for task in self.tasks if task.get("metadata", {}).get("action") == "support_start"]
        stops = [task for task in self.tasks if task.get("metadata", {}).get("action") == "support_end"]
        if not starts and not stops:
            return
        by_id = {task["id"]: task for task in self.tasks}
        outgoing = {ident: [] for ident in by_id}
        for task in self.tasks:
            for previous in task["dependencies"]:
                outgoing[previous].append(task["id"])

        def reaches(source, target):
            pending, seen = [source], {source}
            while pending:
                for following in outgoing[pending.pop()]:
                    if following not in seen:
                        seen.add(following)
                        pending.append(following)
            return target in seen

        def connect(source, target, reason):
            if source["id"] in target["dependencies"]:
                return
            if reaches(target["id"], source["id"]):
                raise CompileFailure(Status.INVALID, "持续支撑依赖会成环："+source["id"]+" -> "+target["id"])
            target["dependencies"].append(source["id"])
            outgoing[source["id"]].append(target["id"])
            target["metadata"].setdefault("support_dependency_evidence", []).append(
                {"source": source["id"], "reason": reason})

        def step(task):
            return steps[task["metadata"]["source_step"]-1]

        installs = [task for task in self.tasks if task.get("metadata", {}).get("action") in {"install", "release"}]
        for stop in stops:
            slot = stop["metadata"]["slot"]
            stop_step = step(stop)
            candidates = [task for task in starts if task["arm"] == stop["arm"]
                          and task["metadata"]["slot"] == slot and step(task).end <= stop_step.start+1e-9]
            if not candidates:
                raise CompileFailure(Status.INVALID, "松开支撑缺少对应的真实ready事件："+slot)
            ready = max(candidates, key=lambda task: step(task).end)
            for installed in installs:
                if (installed["metadata"].get("slot") == slot
                        and step(ready).end <= step(installed).end+1e-9
                        and step(installed).end <= stop_step.start+1e-9):
                    contact = [by_id[previous] for previous in installed["dependencies"]
                               if by_id[previous]["metadata"].get("source_step") == installed["metadata"]["source_step"]
                               and by_id[previous]["metadata"].get("request_kind") == "linear"]
                    if any(reaches(ready["id"], task["id"]) for task in contact):
                        raise CompileFailure(Status.INVALID, "测力就绪依赖不得阻挡载荷接触下降："+slot)
                    ready["metadata"]["support_load_contact_tasks"] = _unique(
                        ready["metadata"].get("support_load_contact_tasks", [])
                        +[task["id"] for task in contact])
                    connect(ready, installed, "measured_support_before_load_install")
            filled = {fact.args[0] for fact in stop_step.state_before if fact.predicate == "filled"}
            definition = self.frames.slots[slot]
            if slot not in filled or not definition.get("overhang"):
                continue
            anchors = definition.get("anchors", [])
            # A second arm/platform may legitimately retain support. Require
            # every anchor only when this planned release relies on anchoring.
            if not all(anchor in filled for anchor in anchors):
                continue
            for anchor in anchors:
                producers = [task for task in installs if task["metadata"].get("slot") == anchor
                             and step(task).end <= stop_step.start+1e-9]
                if producers:
                    connect(max(producers, key=lambda task: step(task).end), stop,
                            "anchor_installed_before_support_end")
                elif not any(row.get("installed_in") == anchor for row in self.frames.blocks.values()):
                    raise CompileFailure(Status.INVALID, "松开支撑缺少锚定安装事件："+anchor)

        # A concurrent support step can appear after the supported insertion in
        # the symbolic list. APEX's reference must place every dependency first;
        # stable topological ordering preserves all existing per-arm chains and
        # unchanged task geometry while permitting that split contact/release.
        degrees = {ident: len(task["dependencies"]) for ident, task in by_id.items()}
        original = {task["id"]: index for index, task in enumerate(self.tasks)}
        pending = []
        for ident, degree in degrees.items():
            if not degree:
                heappush(pending, (original[ident], ident))
        ordered = []
        while pending:
            _, ident = heappop(pending)
            ordered.append(by_id[ident])
            for following in outgoing[ident]:
                degrees[following] -= 1
                if not degrees[following]:
                    heappush(pending, (original[following], following))
        if len(ordered) != len(self.tasks):
            raise CompileFailure(Status.INVALID, "持续支撑参考依赖图含环")
        for index, task in enumerate(ordered):
            if task["reference_order"] != index:
                task["metadata"]["support_original_reference_order"] = task["reference_order"]
            task["reference_order"] = index
        self.tasks = ordered

    def _resting_contacts(self):
        """Exact current bearing pairs; completed grasp exclusions expire."""
        contacts, occupied = [], {}
        candidates = {}
        for block, pose in self.placed.items():
            places = [*self.frames.locations, *self.frames.slots]
            matches = [place for place in places
                       if math.dist(self.frames.block_pose(block, place)[:3], pose[:3]) < 1e-6]
            candidates[block] = matches
            for place in matches:
                if place in self.frames.slots:
                    occupied[place] = block
        for block, places in candidates.items():
            for place in places:
                if place in self.frames.slots:
                    lower = self.frames.slots[place].get("below", [])
                    contacts.extend([[block, occupied[name]] for name in lower if name in occupied])
                    if not lower:
                        contacts.append([block, self.cfg.get("baseplate", "Baseplate")])
                else:
                    kind = self.frames.locations[place]["kind"]
                    if kind in {"stock", "buffer"}:
                        surface = self.cfg.get("pads", {}).get(place, "Pad_"+place)
                    elif kind == "board":
                        surface = self.cfg.get("baseplate", "Baseplate")
                    else:
                        continue
                    contacts.append([block, surface])
        return contacts
