"""Simulator geometry adapter for the existing symbolic feedback protocol."""
from copy import deepcopy
import hashlib
import json
import math

from apexmr.program import parameterize
from apexmr.tpg import build_tpg
from skillnet.capability.oracle import Status
from skillnet.motion.base import MotionResult
from .compiler import CompileFailure, MotionCompiler
from .errors import is_transport_failure
from .primitives import MotionRequest


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def deterministic_ik_seeds(cfg, count):
    """A reproducible, distinct seed bank; never edit the caller's config.

    MotionCompiler prepends the current q and deduplicates its candidates.
    Supplying at least count unique bank entries therefore guarantees count
    distinct actual trials, even when current q equals an entry in the bank.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("IK 种子数必须为正整数")
    rows, seen = [], set()

    def add(row):
        if (not isinstance(row, (list, tuple)) or len(row) != 6
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or abs(v) > 2*math.pi for v in row)):
            raise ValueError("IK 种子必须为关节范围内的六个有限数值")
        key = tuple(round(float(value), 12) for value in row)
        if key not in seen:
            seen.add(key)
            rows.append(list(map(float, row)))

    for row in cfg.get("ik_seeds", []):
        add(row)

    def radical_inverse(index, radix):
        value, factor = 0., 1./radix
        while index:
            index, digit = divmod(index, radix)
            value += digit*factor
            factor /= radix
        return value

    index = 1
    while len(rows) < count:
        add([(2*radical_inverse(index, radix)-1)*math.pi
             for radix in (2, 3, 5, 7, 11, 13)])
        index += 1
    return rows


class _SeedCompiler(MotionCompiler):
    """Per-action escalation also used by the runner's original compiler.

    Keep the accepted compiler implementation and its transactional/cache
    checks. Only its existing seed parameter and private config view differ.
    """

    def __init__(self, *args, seed_overrides=None, **kwargs):
        self.seed_overrides = {} if seed_overrides is None else seed_overrides
        super().__init__(*args, **kwargs)

    def compile_action(self, action, state, **kwargs):
        limit = max(self.ik_seeds, self.seed_overrides.get(action.key, self.ik_seeds))
        if limit == self.ik_seeds:
            return super().compile_action(action, state, **kwargs)
        original_cfg, original_limit = self.cfg, self.ik_seeds
        bank = deterministic_ik_seeds(original_cfg, limit)
        digest = hashlib.sha256(_canonical(bank).encode("utf-8")).hexdigest()
        self.cfg = {**original_cfg, "ik_seeds": bank,
                    "geometry_version": {"original": original_cfg.get("geometry_version"),
                                         "ik_seed_bank_sha256": digest}}
        self.ik_seeds = limit
        self.evidence.append({"query": "ik_seed_configuration", "action": action.text,
            "answer": {"seed_limit": limit, "distinct_bank_entries": len(bank),
                       "seed_bank_sha256": digest, "seed_bank": deepcopy(bank)}})
        try:
            return super().compile_action(action, state, **kwargs)
        finally:
            self.cfg, self.ik_seeds = original_cfg, original_limit


def _failure_classification(failure):
    """Classify terminal IK records, excluding historical RRT/branch pairs."""
    status, evidence = failure.status, failure.evidence
    if status != Status.UNKNOWN:
        return status, {"kind": "backend_"+status.value.lower(), "basis": []}
    path = "failure_evidence"
    # Optional parking retains its original failure for diagnosis; only the
    # terminal recovery's evidence describes the current failure source.
    while isinstance(evidence, dict) and "recovery_evidence" in evidence:
        evidence = evidence["recovery_evidence"]
        path += ".recovery_evidence"
    if isinstance(evidence, dict) and evidence.get("timeout"):
        return status, {"kind": "deadline", "basis": []}
    if isinstance(evidence, dict) and ("attempts" in evidence or "search_stages" in evidence):
        return status, {"kind": "rrt_exhausted", "basis": []}
    basis, branches = [], []

    def problem(row, source):
        if not isinstance(row, dict):
            return
        reason = str(row.get("reason", ""))
        if reason == "IK_branch_discontinuity":
            branches.append({"source": source, **deepcopy(row)})
            return
        pair = row.get("pair")
        collision_pair = (isinstance(pair, (list, tuple)) and len(pair) == 2
                          and all(value is not None and value != "" for value in pair))
        if reason.startswith(("table_collision", "interarm_collision")) or collision_pair:
            basis.append({"source": source, "reason": reason, "pair": deepcopy(pair)})

    records = list(enumerate(evidence)) if isinstance(evidence, list) else [(None, evidence)]
    for index, row in records:
        if not isinstance(row, dict):
            continue
        source = path + ("["+str(index)+"]" if index is not None else "")
        terminal = row.get("problem", row)
        if isinstance(terminal, dict) and terminal.get("reason") == "IK_branch_discontinuity":
            problem(terminal, source+".problem")
            continue  # its nested answer is the rejected IK candidate
        if row.get("ok") is True:
            continue
        problem(terminal, source+".problem")
        rejected = row.get("stats", {}).get("rejections", [])
        if isinstance(rejected, list):
            for rejected_index, rejection in enumerate(rejected):
                problem(rejection, source+f".stats.rejections[{rejected_index}]")
    if basis:
        return Status.INVALID, {"kind": "ik_collision", "basis": basis,
                                "scope": "state/action avoidance; not a global reach proof"}
    return status, {"kind": "ik_branch_discontinuity" if branches else "sampling_unknown",
                    "basis": branches}


class CoppeliaBackend:
    """Validate without changing the planner's cost or symbolic rules.

    Every pipeline round builds a sequential geometric reference. An invalid
    action asks the unchanged pipeline to add its state/action cut; the next
    validation starts a fresh reference, retaining exact-context path caches.
    ``evaluate`` reports a time-parameterized nominal duration, never a cost.
    """

    feedback_round_limit = 12

    def __init__(self, planner, cfg, frames, **compiler_options):
        self.planner, self.cfg, self.frames = planner, cfg, frames
        self.compiler_options = dict(compiler_options)
        self.seed_overrides = {}
        self.compiler = _SeedCompiler(planner, cfg, frames,
                                     seed_overrides=self.seed_overrides, **compiler_options)
        self.validation_records = []
        self.feedback_rounds = []
        self.validation_context = {}
        self.qualification_records = []
        self.action_cache = {}
        self.last_evaluation = None
        self.fatal_error = None
        self._initial_state = None
        self._reset_pending = False
        self._has_prefix = False
        self._pending_escalation = None
        self.payload_limits_kg = dict(cfg.get("payload_limits_kg", {"A": 2., "B": 2.}))
        if any(not math.isfinite(float(limit)) or float(limit) <= 0
               for limit in self.payload_limits_kg.values()):
            raise ValueError("负载上限必须为有限正数")

    def reset(self):
        """Start an independent validation reference without discarding caches."""
        self.compiler.reset()
        self._reset_pending = False
        self._has_prefix = False
        self._pending_escalation = None

    def _scratch_compiler(self):
        options = {**self.compiler_options, "path_cache": self.compiler.path_cache}
        compiler = _SeedCompiler(self.planner, self.cfg, self.frames,
                                 seed_overrides=self.seed_overrides, **options)
        compiler.deadline = self.compiler.deadline
        return compiler

    def validate(self, action, state):
        if self.fatal_error:
            detail = "远程接口已失联，停止后续几何查询："+self.fatal_error
            self.validation_records.append({"action": action.text,
                "status": Status.UNKNOWN.value, "detail": detail, "evidence": [],
                "classification": {"kind": "transport_failure", "basis": []}})
            self.validation_context = {"classification": "transport_failure"}
            return MotionResult(Status.UNKNOWN, detail, [])
        state = frozenset(state)
        if self._initial_state is None:
            self._initial_state = state
        # A second t=0 action may still have both arm_at(home) facts. Only an
        # exact initial state, or an explicit failure, identifies a new round.
        if self._reset_pending or (self._has_prefix and state == self._initial_state):
            self.reset()
        self._pending_escalation = None
        return self._validate_attempt(action, state)

    def _validate_attempt(self, action, state, *, escalated=False):
        first = len(self.compiler.evidence)
        record = {"action": action.text, "state": sorted(map(str, state)),
                  "start_q": deepcopy(self.compiler.q), "held": deepcopy(self.compiler.held),
                  "placed": deepcopy(self.compiler.placed), "escalated": escalated,
                  "ik_seed_limit": self.seed_overrides.get(action.key, self.compiler.ik_seeds)}
        key = _canonical(record)
        try:
            tasks = self.compiler.compile_action(action, state,
                                                  step_id=len(self.validation_records)+1)
            self._has_prefix = True
            record.update(status=Status.VALID.value, detail="真实运动后端：全部原语通过几何验证",
                          tasks=deepcopy(tasks), end_q=deepcopy(self.compiler.q),
                          classification={"kind": "valid", "basis": []})
            self._reset_pending = False
            self._pending_escalation = None
            self.action_cache[key] = deepcopy(tasks)
            result = MotionResult(Status.VALID, record["detail"], [task["id"] for task in tasks])
        except CompileFailure as exc:
            if is_transport_failure(exc):
                self.fatal_error = exc.detail
            status, classification = _failure_classification(exc)
            if self.fatal_error:
                status = Status.UNKNOWN
                classification = {"kind": "transport_failure", "basis": []}
            self._reset_pending = True
            record.update(status=status.value, original_status=exc.status.value, detail=exc.detail,
                          classification=classification,
                          failed_segment=exc.segment, failure_evidence=deepcopy(exc.evidence))
            result = MotionResult(status, exc.detail, [])
            self._pending_escalation = {"action": action.key, "state": state,
                                       "snapshot": self.compiler._snapshot(), "result": result}
        record["evidence"] = deepcopy(self.compiler.evidence[first:])
        failed_arm = next((query.get("arm") for query in reversed(record["evidence"])
                           if query.get("request") == record.get("failed_segment")
                           and query.get("arm") in action.occupies), None)
        record["failed_arm"] = failed_arm
        self.validation_records.append(record)
        self.validation_context = {"classification": record["classification"]["kind"],
            "failed_arm": failed_arm, "ik_seed_limit": record["ik_seed_limit"],
            "validation_index": len(self.validation_records)-1, "escalated": escalated}
        return result

    def validate_escalated(self, action, state):
        """Retry once at the rolled-back start context, retaining its prefix."""
        if self.fatal_error:
            return self.validate(action, state)
        state = frozenset(state)
        pending = self._pending_escalation
        if (pending is None or pending["action"] != action.key or pending["state"] != state
                or pending["snapshot"] != self.compiler._snapshot()):
            raise ValueError("升级验证必须对应上次未知动作及完全相同的几何起点")
        if pending["result"].status != Status.UNKNOWN:
            raise ValueError("只有 UNKNOWN 动作可以升级验证")
        if self.seed_overrides.get(action.key, self.compiler.ik_seeds) >= 24:
            self.validation_context["escalation_skipped"] = "already_at_least_24"
            return pending["result"]
        self.seed_overrides[action.key] = 24
        self._reset_pending = False
        return self._validate_attempt(action, state, escalated=True)

    def record_feedback_round(self, record):
        self.feedback_rounds.append(deepcopy(record))

    def _reach_blocks(self, place):
        """Actual occupying/goal/slot types; shared sites require all types."""
        candidates = [name for name, row in self.frames.blocks.items()
                      if row.get("at") == place or row.get("installed_in") == place]
        if place in self.frames.slots:
            candidates = [name for name, row in self.frames.blocks.items()
                          if row["type"] == self.frames.slots[place]["type"]]
        elif not candidates:
            from skillnet.core.logic import parse_condition
            for text in self.frames.task.get("goal", []):
                condition = parse_condition(text)
                if not condition.negated and condition.fact.predicate == "at" and condition.fact.args[1] == place:
                    if condition.fact.args[0] in self.frames.blocks:
                        candidates.append(condition.fact.args[0])
            if not candidates:
                candidates = [name for name, row in self.frames.blocks.items() if not row.get("two_arms")]
        by_type = {}
        for name in candidates:
            by_type.setdefault(self.frames.blocks[name]["type"], name)
        if not by_type:
            raise CompileFailure(Status.UNKNOWN, f"位置 {place} 没有可核实的对应积木类型")
        return list(by_type.values())

    def _reach_target(self, arm, place, block=None):
        if arm not in self.cfg["arms"] or place not in self.frames.references:
            raise CompileFailure(Status.INVALID, f"未知臂或位置：reach({arm}, {place})")
        block = block or self._reach_blocks(place)[0]
        row = self.frames.locations.get(place, {})
        if row.get("kind") == "handover":
            other = "B" if arm == "A" else "A"
            poses, _, _, _ = self.frames.handover_poses(
                block, place, arm, other, self.frames.payload(block, arm),
                arm_config=self.cfg["arms"])
            return poses[arm]
        return self.frames.grasp_pose(block, place, arm)

    def _qualify_reach(self, fact, record, *, seed_limit=None, pitch_degrees=None):
        from .reach_certificate import outer_sphere_rejection

        arm, place = fact.args
        compiler = self._scratch_compiler()
        if seed_limit is not None:
            bank = deterministic_ik_seeds(self.cfg, seed_limit)
            compiler.cfg = {**deepcopy(self.cfg), "ik_seeds": bank}
            compiler.ik_seeds = seed_limit
            record["seed_bank"] = deepcopy(bank)
            record["seed_bank_sha256"] = hashlib.sha256(_canonical(bank).encode("utf-8")).hexdigest()
        record.update(seed_limit=compiler.ik_seeds,
            pose_mode="catalog" if pitch_degrees is None else f"side_{pitch_degrees:g}",
            qualification_scope="抓取和接近位IK资格；侧姿可达不等于目录顶抓执行已验证")
        raw_results = []
        planner = compiler.planner

        class RecordingPlanner:
            def __getattr__(self, name):
                return getattr(planner, name)

            def solve_ik(self, *args, **kwargs):
                row = {"seed": deepcopy(kwargs.get("options", {}).get("seed"))}
                raw_results.append(row)
                try:
                    row["answer"] = deepcopy(planner.solve_ik(*args, **kwargs))
                    return deepcopy(row["answer"])
                except Exception as exc:
                    row["exception"] = f"{type(exc).__name__}: {exc}"
                    raise

        compiler.planner = RecordingPlanner()
        contexts = []
        try:
            for block in self._reach_blocks(place):
                context = {"block": block, "type": self.frames.blocks[block]["type"],
                           "place": place, "stages": []}
                contexts.append(context)
                try:
                    if self.fatal_error:
                        raise CompileFailure(Status.UNKNOWN, "远程接口已失联，未继续检查该类型")
                    target = (self._reach_target(arm, place, block) if pitch_degrees is None else
                              self.frames.tcp_pose(self.frames.block_pose(block, place),
                                  self.frames.side_payload(block, arm, math.radians(pitch_degrees))))
                    context["target_pose"] = target
                    record.setdefault("target_pose", target)
                    handover = self.frames.locations.get(place, {}).get("kind") == "handover"
                    approach = (self.frames.handover_retreat(target, arm) if handover
                                else self.frames.approach_pose(target))
                    context["approach_rule"] = "handover_minus_tool_z" if handover else "vertical_above"
                    for label, pose in (("grasp", target), ("approach", approach)):
                        stage = {"label": label, "pose": deepcopy(pose), "seed_limit": compiler.ik_seeds}
                        context["stages"].append(stage)
                        proof = outer_sphere_rejection(self.cfg["arms"][arm], pose)
                        if proof is not None:
                            stage["ik_attempts"] = 0
                            raise CompileFailure(Status.INVALID, "目标超出实际模型的已证明外球上界", "qualify_"+label, proof)
                        request = MotionRequest("qualify_"+label, arm, "linear", target_pose=pose,
                            held=deepcopy(compiler.held.get(arm)), payloads=deepcopy(compiler.held),
                            placed=deepcopy(compiler.placed), contacts=compiler._resting_contacts())
                        first = len(compiler.evidence)
                        raw_first = len(raw_results)
                        try:
                            stage["q"] = compiler._ik(request, pose, list(compiler.q[arm]))
                        finally:
                            stage["ik_attempts"] = sum(query["query"] == "IK" for query in compiler.evidence[first:])
                            stage["raw_results"] = deepcopy(raw_results[raw_first:])
                            seeds = [row["seed"] for row in stage["raw_results"]]
                            stage["seed_bank_sha256"] = hashlib.sha256(_canonical(seeds).encode("utf-8")).hexdigest()
                        stage["status"] = Status.VALID.value
                    context.update(status=Status.VALID.value, detail="该类型抓取位与接近位IK均通过")
                except CompileFailure as exc:
                    if is_transport_failure(exc):
                        self.fatal_error = exc.detail
                    context.update(status=exc.status.value, detail=exc.detail,
                                   failure_evidence=deepcopy(exc.evidence))
            statuses = {context["status"] for context in contexts}
            status = Status(next(iter(statuses))) if len(statuses) == 1 else Status.UNKNOWN
            record.update(target_contexts=contexts,
                detail=("所有对应积木类型的抓取/接近IK结论一致" if len(statuses) == 1 else
                        "对应积木类型结论不一致，reach不含类型参数，保留UNKNOWN"))
            return status
        finally:
            record["evidence"] = deepcopy(compiler.evidence)
            record.setdefault("target_contexts", contexts)

    def qualify_candidate(self, fact, *, seed_limit=32, pitch_degrees=None):
        """Explicit bounded D probe, retaining the ordinary 8-seed API.

        Exhaustion is recorded separately from INVALID: only the qualifier's
        complete required pose set may apply FIX v1.3.4's bounded rejection.
        """
        if fact.predicate != "reach" or len(fact.args) != 2:
            raise ValueError("升级候选仅支持 reach(arm, place)")
        if seed_limit != 32 or pitch_degrees not in (None, 30, 45, 60):
            raise ValueError("D候选仅允许32种子与目录/30/45/60度姿态")
        record = {"fact": str(fact)}
        if self.fatal_error:
            record.update(detail=self.fatal_error, candidate_exhausted=False)
            status = Status.UNKNOWN
        else:
            try:
                status = self._qualify_reach(fact, record, seed_limit=seed_limit, pitch_degrees=pitch_degrees)
            except CompileFailure as exc:
                if is_transport_failure(exc):
                    self.fatal_error = exc.detail
                status = exc.status
                record.update(detail=exc.detail, failure_evidence=deepcopy(exc.evidence))

            def exhausted(context):
                proof = context.get("failure_evidence")
                if context.get("status") == "INVALID":
                    return isinstance(proof, dict) and bool(proof.get("proven_infeasible") or proof.get("infeasible"))
                stages = context.get("stages", [])
                rows = stages[-1].get("raw_results", []) if stages else []
                return (context.get("status") == "UNKNOWN" and isinstance(proof, list) and len(proof) == seed_limit
                    and len(rows) == seed_limit
                    and len({tuple(row.get("seed") or []) for row in rows}) == seed_limit
                    and all(isinstance(row.get("answer"), dict) and row["answer"].get("ok") is False
                            and not row["answer"].get("error") and not row.get("exception")
                            and not row["answer"].get("timeout")
                            and row["answer"].get("status") not in ("timeout", "ERROR", "aborted")
                            for row in rows))

            contexts = record.get("target_contexts", [])
            record["candidate_exhausted"] = bool(contexts) and not self.fatal_error and all(exhausted(row) for row in contexts)
        record["status"] = status.value
        self.qualification_records.append(record)
        return status

    def qualify(self, fact):
        """Qualify in a scratch reference; simulator queries are transactional."""
        if self.fatal_error:
            self.qualification_records.append({"fact": str(fact), "status": Status.UNKNOWN.value,
                "detail": "远程接口已失联，停止后续资格查询："+self.fatal_error})
            return Status.UNKNOWN
        record = {"fact": str(fact)}
        compiler = None
        try:
            if fact.predicate == "liftable" and len(fact.args) == 2:
                arm, block = fact.args
                if arm not in self.payload_limits_kg or block not in self.frames.blocks:
                    return None
                mass = self.frames.catalog[self.frames.blocks[block]["type"]].mass
                limit = float(self.payload_limits_kg[arm])
                status = Status.VALID if mass <= limit else Status.INVALID
                record.update(mass_kg=mass, limit_kg=limit, detail="按物理目录质量与配置负载上限比较")
            elif fact.predicate == "reach" and len(fact.args) == 2:
                status = self._qualify_reach(fact, record)
            else:
                return None
        except CompileFailure as exc:
            if is_transport_failure(exc):
                self.fatal_error = exc.detail
            status = exc.status
            record.update(detail=exc.detail, failure_evidence=deepcopy(exc.evidence))
        finally:
            if compiler is not None:
                record["evidence"] = deepcopy(compiler.evidence)
            if "status" not in record:
                record["status"] = locals().get("status", Status.UNKNOWN).value
            self.qualification_records.append(record)
        return status

    def evaluate(self, schedule, graph=None):
        """Return nominal parameterized seconds using a scratch compiler.

        This deliberately does not advance validation state and is not wired
        into DistanceCost. Real inter-arm scheduling/acceptance stays in the
        executor; this value is recorded as a nominal timing observation.
        """
        if self.fatal_error:
            raise ConnectionError("远程接口已失联，停止时间参数化查询："+self.fatal_error)
        compiler = self._scratch_compiler()
        tasks = compiler.compile(schedule, graph)
        tpg = build_tpg(tasks)
        parameterize(tpg, vmax=.60, amax=1.20, jmax=8.)
        seconds = float(tpg["nominal_makespan"])
        self.last_evaluation = {"nominal_seconds": seconds, "tasks": len(tasks),
                                "cache_hits": compiler.cache_hits,
                                "search_cost_changed": False}
        return seconds

    def audit(self):
        return {"validations": self.validation_records, "qualifications": self.qualification_records,
                "feedback_rounds": self.feedback_rounds,
                "fatal_error": self.fatal_error,
                "evaluation": self.last_evaluation,
                "cache": {"entries": len(self.compiler.path_cache), "hits": self.compiler.cache_hits,
                          "misses": self.compiler.cache_misses},
                "path_certificates": self.compiler.path_cache}
