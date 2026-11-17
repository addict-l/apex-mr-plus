"""Compile symbolic skills into transactional, geometry-checked APEX tasks.

Compilation advances a sequential geometric reference while retaining the
symbolic graph's concurrency. A later overlapping symbolic start snapshot
cannot undo an object transition already emitted by this reference compiler.
"""
from copy import deepcopy
from heapq import heappop, heappush
import hashlib
import json
import math
import time

from scipy.spatial.transform import Rotation, Slerp

from skillnet.capability.oracle import Status


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
                 local_box_time=4., path_cache=None):
        if isinstance(ik_seeds, bool) or not isinstance(ik_seeds, int) or ik_seeds < 1:
            raise ValueError("IK 种子数必须为正整数")
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

    def _ik(self, request, pose, seed, *, max_joint_delta=None):
        candidates = [list(seed), *deepcopy(self.cfg.get("ik_seeds", [])),
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
        collect_transit = request.kind == "transit" and max_joint_delta is None
        for attempt, candidate in enumerate(seeds[:self.ik_seeds], 1):
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
                solutions.append(list(q))
                continue
            failures.append(answer)
            if not solutions and _failure_status(answer, ik=True) == Status.INVALID:
                raise CompileFailure(Status.INVALID,
                    f"运动段 {request.id} 明确不可达：{answer.get('problem') or answer.get('error')}", request.id, answer)
        if solutions:
            def elbow_family(configuration):
                value = math.sin(configuration[2])
                return 0 if abs(value) <= 1e-8 else (1 if value > 0 else -1)

            start_family = elbow_family(seed)
            selected = min(solutions, key=lambda q: (
                start_family != 0 and elbow_family(q) != start_family,
                math.dist(q, seed)))
            # Family agreement is a candidate-ranking heuristic only. The
            # selected endpoint still needs the ordinary RRT and dense collision
            # certificate; it is not evidence of a connected free-space path.
            self.evidence.append({"request": request.id, "arm": request.arm,
                "query": "IK_selection", "answer": {
                    "ok": True, "q": list(selected), "start_q": list(seed),
                    "valid_candidates": deepcopy(solutions), "seed_budget": self.ik_seeds,
                    "start_elbow_family": start_family,
                    "selected_elbow_family": elbow_family(selected),
                    "selection_rule": "prefer same sign(sin(q3)), then nearest actual joint configuration; RRT required"}})
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
        goal = (list(request.target_q) if request.target_q is not None
                else self._ik(request, request.target_pose, start))
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
        spent, attempts, answer = 0., [], None
        for stage, stage_options, stage_time in stages:
            remaining = self.rrt_timeout-spent
            if self.deadline is not None:
                remaining = min(remaining, self.deadline-time.monotonic())
            solve_time = max(.001, remaining if stage_time is None else min(stage_time, remaining))
            began = time.monotonic()
            answer = self._call("RRTConnect", request, self.planner.plan,
                request.arm, start, goal, payload=request.held, placed=request.placed,
                other_q=other_q, contact_exclusions=exclusions,
                solve_time=solve_time, options=stage_options)
            spent += time.monotonic()-began
            self.evidence[-1]["search_stage"] = stage
            attempts.append({"stage": stage, "solve_time_s": solve_time, "ok": bool(answer.get("ok")),
                             "error": answer.get("error"), "problem": answer.get("problem")})
            if answer.get("ok") or _failure_status(answer) == Status.INVALID:
                break
        if not answer.get("ok"):
            tried = "、".join(attempt["stage"] for attempt in attempts)
            raise CompileFailure(_failure_status(answer),
                f"运动段 {request.id} 的 RRT-Connect 失败（{tried} 均未找到解）："
                f"{answer.get('error')}; {answer.get('problem')}",
                request.id, {"attempts": attempts, "last_answer": answer})
        answer = dict(answer)
        answer["search_stages"] = attempts
        path = answer.get("path")
        if (not path or any(len(q) != 6 or not all(math.isfinite(v) for v in q) for q in path)
                or max(abs(a-b) for a, b in zip(path[0], start)) > 1e-5
                or max(abs(a-b) for a, b in zip(path[-1], goal)) > 1e-5):
            raise CompileFailure(Status.UNKNOWN, f"运动段 {request.id} 的规划路径缺失或端点不一致", request.id, answer)
        return deepcopy(path)

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
            "limits": [self.ik_seeds, self.rrt_timeout, self.linear_step, self.joint_step,
                       self.force_rrt, self.linear_joint_delta, self.linear_position_tolerance,
                       self.linear_orientation_tolerance, self.local_box_margin, self.local_box_time],
        }
        encoded = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
        key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if key in self.path_cache:
            entry = self.path_cache[key]
            self.cache_hits += 1
            self.evidence.append({"request": request.id, "arm": request.arm,
                "query": "path_cache", "answer": {"ok": True, "cache_key": key,
                    "certificate_queries": len(entry["certificate"]), "cache_hit": True}})
            return deepcopy(entry["path"])
        self.cache_misses += 1
        first = len(self.evidence)
        path = self._path(request)
        self.path_cache[key] = {"context": deepcopy(context), "path": deepcopy(path),
                                "certificate": deepcopy(self.evidence[first:])}
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
            if task["arm"] == other and task["kind"] in {"support_start", "support_end"}:
                supporting = task["kind"] == "support_start"
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
        first = len(self.tasks)
        other = self._idle_other(action, state, requests)
        context_key, receipt = None, None
        if other is not None:
            context = {"scene": [id(self.planner), self.cfg.get("root"), self.cfg.get("task_name")],
                "geometry_version": self.cfg.get("geometry_version"), "action": action.key,
                "state": sorted(map(str, state)), "q": self.q, "held": self.held, "placed": self.placed,
                "touched": sorted(self._touched), "request": vars(requests[0]),
                "homes": {side: row["home_q"] for side, row in self.cfg["arms"].items()},
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
        parking = MotionRequest("idle_home_"+other, other, "transit",
            target_q=list(self.cfg["arms"][other]["home_q"]), held=None,
            placed=deepcopy(self.placed), payloads=deepcopy(self.held), contacts=self._resting_contacts(),
            metadata={"source_action": action.text, "source_skill": action.name,
                "idle_arm_parking": True, "physical_only": True,
                "idle_parking_context": context_key, "recovery_for_request": requests[0].id})
        phase = "parking"
        try:
            parked = self._emit(parking, {}, step_id,
                                _unique([*dependencies, *self._last_arm.values()]))
            phase = "retry"
            emit(_unique([*dependencies, parked["id"]]))
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
                    "parked_arm": other, "original_failure": deepcopy(original)}

    def compile_action(self, action, state, *, step_id=None, dependencies=()):
        from .primitives import PrimitiveError, expand

        snapshot, first = self._snapshot(), len(self.tasks)
        step_id = step_id if step_id is not None else first+1
        try:
            self._symbolic_context = tuple(sorted(str(fact) for fact in state))
            self._sync_state(state)
            try:
                requests = expand(action, state, self.frames, self.cfg,
                                  placed=self.placed, payloads=self.held)
            except PrimitiveError as exc:
                raise CompileFailure(exc.status, str(exc)) from exc
            self._emit_action_requests(action, state, requests, step_id, dependencies)
            bindings = dict(zip(action.skill.params, action.args))
            if "block" in bindings:
                self._touched.add(bindings["block"])
            return deepcopy(self.tasks[first:])
        except BaseException:
            self._restore(snapshot)
            raise

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
