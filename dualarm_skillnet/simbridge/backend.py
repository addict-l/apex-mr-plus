"""Simulator geometry adapter for the existing symbolic feedback protocol."""
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from heapq import heappop, heappush
import hashlib
import json
import math
import time

from apexmr.program import parameterize
from apexmr.tpg import build_tpg
from skillnet.capability.oracle import Status
from skillnet.motion.base import MotionResult
from .compiler import CompileFailure, MotionCompiler
from .errors import is_transport_failure
from .primitives import MotionRequest


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _reported(query, collection, subject=None):
    """Publish operation boundaries without copying the accumulated audit.

    Persistence errors propagate. A caller must not continue an experiment
    whose required local evidence could not be saved. The optional callback
    receives detached JSON-compatible context and the latest new record only.
    """
    def decorate(function):
        @wraps(function)
        def reported(self, *args, **kwargs):
            if self.progress_callback is None:
                return function(self, *args, **kwargs)
            began, first = time.monotonic(), len(getattr(self, collection))
            identifying = {}
            value = args[0] if args else kwargs.get(subject)
            if subject in {"action", "fact"}:
                identifying[subject] = value.text if subject == "action" else str(value)
            round_index = (value.get("round") if query == "feedback_round" and isinstance(value, dict)
                           else self.reference_index)

            def publish(phase, result=None, error=None):
                records = getattr(self, collection)
                event = {"query": query, "phase": phase, "round": round_index, **identifying,
                    "q": deepcopy(self.compiler.q), "held": deepcopy(self.compiler.held),
                    "placed": deepcopy(self.compiler.placed),
                    "elapsed_seconds": time.monotonic()-began,
                    "record_collection": collection,
                    "new_record_indices": list(range(first, len(records))) if phase == "after" else [],
                    "record": deepcopy(records[-1]) if phase == "after" and len(records) > first else None}
                status = getattr(result, "status", result if isinstance(result, Status) else None)
                if status is not None:
                    event["status"] = status.value
                if error is not None:
                    event["exception"] = {"type": type(error).__name__, "message": str(error)}
                self.progress_callback(event)

            publish("before")
            try:
                result = function(self, *args, **kwargs)
            except BaseException as error:
                publish("after", error=error)
                raise
            publish("after", result=result)
            return result
        return reported
    return decorate


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

    def __init__(self, *args, seed_overrides=None, seed_selector=None, **kwargs):
        self.seed_overrides = {} if seed_overrides is None else seed_overrides
        self.seed_selector = seed_selector
        super().__init__(*args, **kwargs)

    def compile_action(self, action, state, **kwargs):
        limit = max(self.ik_seeds, self.seed_overrides.get(action.key, self.ik_seeds))
        if self.seed_selector is not None:
            limit = max(limit, self.seed_selector(self, action, state))
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
    """Keep a finite candidate's last failure separate from its history.

    A collision of one sampled IK configuration or RRT endpoint rejects that
    candidate. Only explicit infeasibility evidence retains INVALID. Earlier
    collisions must not overwrite a later branch or RRT exhaustion reason.
    """
    evidence = failure.evidence
    path = "failure_evidence"
    while isinstance(evidence, dict):
        key = next((name for name in ("recovery_evidence", "terminal_failure", "terminal_evidence")
                    if name in evidence), None)
        if key is None:
            break
        evidence = evidence[key]
        path += "."+key
        if (key == "terminal_failure" and isinstance(evidence, dict)
                and ("evidence" in evidence or "failure_evidence" in evidence)):
            key = "evidence" if "evidence" in evidence else "failure_evidence"
            evidence = evidence[key]
            path += "."+key

    def classify(row, source):
        if not isinstance(row, dict):
            return "sampling_unknown", []
        if row.get("timeout"):
            return "deadline", []
        if "attempts" in row or "search_stages" in row:
            return "rrt_exhausted", []
        terminal = row.get("problem", row)
        if isinstance(terminal, dict) and terminal.get("reason") == "IK_branch_discontinuity":
            return "ik_branch_discontinuity", [{"source": source+".problem", **deepcopy(terminal)}]
        problems = [(terminal, source+".problem")]
        stats = row.get("stats", {})
        rejected = stats.get("rejections", []) if isinstance(stats, dict) else []
        if isinstance(rejected, list):
            problems.extend((item, source+f".stats.rejections[{number}]")
                            for number, item in enumerate(rejected))
        basis = []
        for item, item_source in problems:
            if not isinstance(item, dict):
                continue
            reason, pair = str(item.get("reason", "")), item.get("pair")
            valid_pair = (isinstance(pair, (list, tuple)) and len(pair) == 2
                          and all(value is not None and value != "" for value in pair))
            if "collision" in reason or valid_pair:
                basis.append({"source": item_source, "reason": reason, "pair": deepcopy(pair)})
        return ("ik_collision", basis) if basis else ("sampling_unknown", [])

    rows = evidence if isinstance(evidence, list) else [evidence]
    history = {}
    for number, row in enumerate(rows):
        kind, _ = classify(row, path+f"[{number}]")
        history[kind] = history.get(kind, 0)+1
    terminal = rows[-1] if rows else None
    source = path+(f"[{len(rows)-1}]" if isinstance(evidence, list) and rows else "")
    kind, basis = classify(terminal, source)
    proof = terminal
    if isinstance(proof, dict) and "last_answer" in proof:
        proof = proof["last_answer"]
    proven = (isinstance(proof, dict)
              and (proof.get("proven_infeasible") is True or proof.get("infeasible") is True))
    status = Status.INVALID if proven else Status.UNKNOWN
    scope = "request" if proven else "candidate"
    if proven:
        kind = "proven_infeasible"
    elif kind == "deadline":
        scope = "task"
    elif failure.status == Status.INVALID and kind == "sampling_unknown":
        # Unsupported primitives and malformed requests remain explicit
        # backend INVALID; no global reach conclusion is implied.
        status, kind, scope = Status.INVALID, "backend_invalid", "request"
    return status, {"kind": kind, "basis": basis, "failure_scope": scope,
                    "scope": "current request/candidate only; not a global reach proof",
                    "terminal_source": source, "history_summary": history}


class CoppeliaBackend:
    """Validate without changing the planner's cost or symbolic rules.

    Every pipeline round builds a sequential geometric reference. Finite
    geometric failures only justify temporary state/action avoidance; the next
    reference retains exact-context certificates and finite candidate history.
    ``evaluate`` reports a time-parameterized nominal duration, never a cost.
    """

    feedback_round_limit = 12
    candidate_strategy_version = "motion-context-seeds-v135-3-adjacent-deadline"

    def __init__(self, planner, cfg, frames, *, progress_callback=None, **compiler_options):
        self.planner, self.cfg, self.frames = planner, cfg, frames
        self.progress_callback = progress_callback
        self.compiler_options = dict(compiler_options)
        self.seed_overrides = {}
        self.candidate_ledger = {}
        self.reference_records = []
        self.reference_validations = []
        self._complete_reference = None
        self._finalization_used = False
        self._active_candidate = None
        self.compiler = _SeedCompiler(planner, cfg, frames,
                                     seed_overrides=self.seed_overrides,
                                     seed_selector=self._compiler_seed_limit, **compiler_options)
        self.qualification_hints = deepcopy(getattr(self.compiler, "qualification_hints", []))
        self.qualification_hint_imports = []
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
        self._explicit_reference = False
        self.reference_index = None
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
        self.reference_records = []
        self._complete_reference = None

    def begin_reference(self, round_index, deadline=None):
        """Start one reference without extending its task deadline/history.

        An explicit boundary replaces the legacy initial-state heuristic;
        parallel symbolic snapshots cannot silently reset a live prefix.
        """
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("几何参考轮次必须为正整数")
        if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
            raise ValueError("任务截止时间必须为有限单调时钟值")
        current = self.compiler.deadline
        self.reset()
        self._explicit_reference = True
        self.reference_index = round_index
        self.compiler.deadline = (current if deadline is None else
                                  deadline if current is None else min(current, deadline))

    def import_qualification_hints(self, progress, source=None):
        """Import compatible point IK observations without trusting a path.

        Historical ``qualify_log`` wrappers and current backend records share
        their per-type VALID stages. Their q/pose is only a search candidate:
        the compiler must check FK, collisions and complete motion again in
        the current scene. Explicit alternate-pitch probes remain separate.
        """
        from skillnet.core.logic import parse_fact

        receipt = {"source": str(source) if source is not None else "qualification_records",
                   "imported": 0, "duplicates": 0, "skipped": 0, "skip_reasons": {}}

        def skip(reason):
            receipt["skipped"] += 1
            receipt["skip_reasons"][reason] = receipt["skip_reasons"].get(reason, 0)+1

        def vector(value, size):
            return (isinstance(value, (tuple, list)) and len(value) == size
                    and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                            and math.isfinite(item) for item in value))

        def key(hint):
            return _digest({name: hint.get(name) for name in (
                "arm", "part", "block_type", "place", "pose_mode", "grasp_strategy", "stage", "pose", "q")})

        if isinstance(progress, dict):
            rows = next((progress[name] for name in ("records", "qualification_records", "qualifications")
                         if isinstance(progress.get(name), list)), None)
            if rows is None:
                rows = [progress] if "target_contexts" in progress or "backend_record" in progress else []
        else:
            rows = progress if isinstance(progress, (list, tuple)) else []
        seen = {key(hint) for hint in self.qualification_hints if isinstance(hint, dict)}
        expected_strategy = ("side_opposed" if any(row.get("kind") == "handover"
                            for row in self.frames.locations.values()) else "catalog_top")
        for outer in rows:
            if not isinstance(outer, dict):
                skip("malformed_record")
                continue
            record = outer.get("backend_record", outer)
            if not isinstance(record, dict):
                skip("malformed_backend_record")
                continue
            try:
                fact = parse_fact(record.get("fact", outer.get("fact", "")))
                if fact.predicate != "reach" or len(fact.args) != 2:
                    raise ValueError("not reach")
                arm, fact_place = fact.args
            except (TypeError, ValueError, AttributeError):
                skip("unsupported_fact")
                continue
            if arm not in self.cfg["arms"] or fact_place not in self.frames.references:
                skip("unknown_arm_or_place")
                continue
            contexts = record.get("target_contexts", [])
            if not isinstance(contexts, list):
                skip("malformed_contexts")
                continue
            for context in contexts:
                if not isinstance(context, dict):
                    skip("malformed_context")
                    continue
                part, place = context.get("block"), context.get("place", fact_place)
                if part not in self.frames.blocks or place != fact_place:
                    skip("unknown_part_or_mismatched_place")
                    continue
                block_type = context.get("type")
                if block_type != self.frames.blocks[part]["type"]:
                    skip("mismatched_block_type")
                    continue
                mode = context.get("pose_mode", record.get("pose_mode", outer.get("pose_mode")))
                if mode not in (None, "catalog"):
                    skip("alternate_pose_mode")
                    continue
                strategy = context.get("grasp_strategy", record.get("grasp_strategy", expected_strategy))
                if strategy != expected_strategy:
                    skip("mismatched_grasp_strategy")
                    continue
                stages = context.get("stages", [])
                if not isinstance(stages, list):
                    skip("malformed_stages")
                    continue
                for stage in stages:
                    if not isinstance(stage, dict) or stage.get("status") != Status.VALID.value:
                        skip("nonvalid_stage")
                        continue
                    label, pose, q = stage.get("label"), stage.get("pose"), stage.get("q")
                    if label not in ("grasp", "approach") or not vector(pose, 7) or not vector(q, 6):
                        skip("invalid_point_observation")
                        continue
                    if abs(sum(value*value for value in pose[3:])-1.) > 1e-6:
                        skip("invalid_pose_quaternion")
                        continue
                    if mode is None:
                        # A legacy unlabeled mode is usable only when its
                        # recorded point agrees with today's catalog target.
                        expected = self._reach_target(arm, place, part)
                        if label == "approach":
                            expected = (self.frames.handover_retreat(expected, arm)
                                if self.frames.locations.get(place, {}).get("kind") == "handover"
                                else self.frames.approach_pose(expected))
                        distance = math.sqrt(sum((a-b)**2 for a, b in zip(pose[:3], expected[:3])))
                        dot = abs(sum(a*b for a, b in zip(pose[3:], expected[3:])))
                        angle = 2*math.acos(min(1., max(0., dot)))
                        if (distance > self.compiler.linear_position_tolerance
                                or angle > self.compiler.linear_orientation_tolerance):
                            skip("legacy_pose_mismatch")
                            continue
                    hint = {"arm": arm, "part": part, "block_type": block_type, "place": place,
                        "pose_mode": "catalog", "grasp_strategy": strategy, "stage": label,
                        "pose": list(map(float, pose)), "q": list(map(float, q)),
                        "source": receipt["source"], "legacy_pose_mode_assumed": mode is None,
                        "qualification_scope": "point_ik_only; execution FK/collision/connection revalidation required"}
                    hint_key = key(hint)
                    if hint_key in seen:
                        receipt["duplicates"] += 1
                        continue
                    seen.add(hint_key)
                    self.qualification_hints.append(hint)
                    receipt["imported"] += 1
        self.compiler.qualification_hints = deepcopy(self.qualification_hints)
        receipt["total_hints"] = len(self.qualification_hints)
        self.qualification_hint_imports.append(deepcopy(receipt))
        return receipt

    def _action_context(self, compiler, action, state):
        """Materialize request context without querying or changing the prefix."""
        from .frames import APPROACH, BOARD_TOP, SIDE_GRASP_PITCH
        from .primitives import PrimitiveError, expand

        snapshot = compiler._snapshot()
        try:
            compiler._symbolic_context = tuple(sorted(map(str, state)))
            compiler._sync_state(state)
            try:
                requests = [deepcopy(vars(request)) for request in expand(
                    action, state, self.frames, compiler.cfg,
                    placed=compiler.placed, payloads=compiler.held)]
            except PrimitiveError as exc:
                # Keep malformed/unsupported actions on the ordinary public
                # compile path, with a reproducible context for its evidence.
                requests = [{"primitive_error": str(exc), "status": exc.status.value}]
            active_support = {}
            for task in compiler.tasks:
                event = task.get("metadata", {}).get("action", task["kind"])
                if event in {"support_start", "support_end"}:
                    active_support[task["arm"]] = event == "support_start"
            for fact in state:
                if (fact.predicate == "supporting" and fact.args
                        and fact.args[0] not in active_support):
                    active_support[fact.args[0]] = True
            geometry_cfg = {key: value for key, value in compiler.cfg.items() if key != "ik_seeds"}
            version = geometry_cfg.get("geometry_version")
            if isinstance(version, dict) and "ik_seed_bank_sha256" in version:
                geometry_cfg["geometry_version"] = version.get("original")
            geometry_version = geometry_cfg.get("geometry_version") or "skillnet-ur5-geometry-v135-1"
            context = {"scene": [compiler.cfg.get("root_path"), compiler.cfg.get("task_name")],
                "geometry_version": deepcopy(geometry_version),
                "geometry_sha256": _digest({"cfg": geometry_cfg,
                    "references": self.frames.references,
                    "catalog": {name: asdict(spec) for name, spec in self.frames.catalog.specs.items()},
                    "grasp_parameters": {"side_pitch_rad": SIDE_GRASP_PITCH,
                                         "approach_m": APPROACH, "board_top_m": BOARD_TOP},
                    "task_geometry": {key: self.frames.task.get(key, [])
                                      for key in ("location", "slot", "block")}}),
                "action": action.key, "state": sorted(map(str, state)),
                "q": deepcopy(compiler.q), "held": deepcopy(compiler.held),
                "placed": deepcopy(compiler.placed), "requests": requests,
                "active_support": active_support, "touched": sorted(compiler._touched),
                "motion_limits": [compiler.rrt_timeout, compiler.linear_step, compiler.joint_step,
                    compiler.force_rrt, compiler.linear_joint_delta,
                    compiler.linear_position_tolerance, compiler.linear_orientation_tolerance,
                    compiler.local_box_margin, compiler.local_box_time]}
            return _digest(context), context
        finally:
            compiler._restore(snapshot)

    def _candidate_specs(self, action):
        baseline = max(self.compiler_options.get("ik_seeds", 8),
                       self.seed_overrides.get(action.key, 0))
        counts = [baseline] if baseline >= 24 else [baseline, 24]
        result = []
        for count in counts:
            strategy = {"version": self.candidate_strategy_version, "ik_seed_limit": count,
                "configured_seeds": deepcopy(self.cfg.get("ik_seeds", [])),
                "qualification_hints_sha256": _digest(self.qualification_hints),
                "action_candidate_limit": getattr(self.compiler, "action_candidate_limit", 4)}
            if count > baseline:
                strategy["seed_bank_sha256"] = _digest(deterministic_ik_seeds(self.cfg, count))
            candidate_id = _digest(strategy)
            result.append({"candidate_id": candidate_id,
                           "candidate_version": self.candidate_strategy_version+":"+candidate_id,
                           "ik_seed_limit": count})
        return result

    def _compiler_seed_limit(self, compiler, action, state):
        if compiler is self.compiler and self._active_candidate is not None:
            return self._active_candidate["ik_seed_limit"]
        context_id, _ = self._action_context(compiler, action, frozenset(state))
        entry = self.candidate_ledger.get(context_id, {})
        accepted = entry.get("accepted_candidate")
        if accepted is not None:
            return entry["attempts"][accepted]["record"]["ik_seed_limit"]
        return self.seed_overrides.get((context_id, action.key), compiler.ik_seeds)

    def _scratch_compiler(self):
        options = {**self.compiler_options, "path_cache": self.compiler.path_cache,
                   "qualification_hints": deepcopy(self.qualification_hints)}
        compiler = _SeedCompiler(self.planner, self.cfg, self.frames,
                                 seed_overrides=self.seed_overrides,
                                 seed_selector=self._compiler_seed_limit, **options)
        compiler.deadline = self.compiler.deadline
        return compiler

    @_reported("validate", "validation_records", "action")
    def validate(self, action, state):
        self._complete_reference = None
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
        if not self._explicit_reference and (self._reset_pending
                or (self._has_prefix and state == self._initial_state)):
            self.reset()
        self._pending_escalation = None
        return self._validate_attempt(action, state)

    def _validate_attempt(self, action, state, *, escalated=False, candidate=None):
        context_id, context = self._action_context(self.compiler, action, state)
        entry = self.candidate_ledger.setdefault(context_id,
            {"context": context, "attempts": {}, "accepted_candidate": None})
        specs = self._candidate_specs(action)
        if candidate is None:
            accepted = entry["accepted_candidate"]
            candidate = (next(item for item in specs if item["candidate_id"] == accepted)
                         if accepted in {item["candidate_id"] for item in specs} else specs[0])
        candidate_id = candidate["candidate_id"]
        previous = entry["attempts"].get(candidate_id)
        expired = self.compiler.deadline is not None and time.monotonic() >= self.compiler.deadline
        first = len(self.compiler.evidence)
        record = {"action": action.text, "state": sorted(map(str, state)),
                  "start_q": deepcopy(context["q"]), "held": deepcopy(context["held"]),
                  "placed": deepcopy(context["placed"]), "escalated": escalated,
                  "reference_index": self.reference_index, "context_id": context_id,
                  **candidate, "deduplicated": False}
        before = self.compiler._snapshot()
        try:
            if expired:
                raise CompileFailure(Status.UNKNOWN, "运动验证达到任务墙钟截止时间", evidence={"timeout": True})
            if previous is not None and previous["result"].status != Status.VALID:
                cached = previous["record"]
                record.update({key: deepcopy(cached[key]) for key in (
                    "status", "original_status", "detail", "classification", "failed_segment",
                    "failure_evidence", "failed_arm") if key in cached})
                record.update(deduplicated=True, reused_validation_index=previous["validation_index"])
                result = deepcopy(previous["result"])
                self._reset_pending = True
            else:
                self._active_candidate = candidate
                tasks = self.compiler.compile_action(action, state,
                                                      step_id=len(self.reference_records)+1)
                self._has_prefix = True
                record.update(status=Status.VALID.value, detail="真实运动后端：全部原语通过几何验证",
                              tasks=deepcopy(tasks), end_q=deepcopy(self.compiler.q),
                              classification={"kind": "valid", "basis": [], "failure_scope": "candidate"})
                self._reset_pending = False
                self._pending_escalation = None
                after = self.compiler._snapshot()
                certificate = {"context_id": context_id, **candidate, "action": action.key,
                    "state": state, "tasks": deepcopy(tasks),
                    "start_snapshot": before, "end_snapshot": after}
                self.action_cache[(context_id, candidate_id)] = certificate
                self.reference_records.append(certificate)
                entry["accepted_candidate"] = candidate_id
                result = MotionResult(Status.VALID, record["detail"], [task["id"] for task in tasks])
        except CompileFailure as exc:
            # Guard even a future compiler candidate implementation that
            # raises after partially committing a primitive or parking task.
            self.compiler._restore(before)
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
        finally:
            self._active_candidate = None
        record["evidence"] = deepcopy(self.compiler.evidence[first:])
        failed_arm = record.get("failed_arm") or next((query.get("arm")
            for query in reversed(record["evidence"])
            if query.get("request") == record.get("failed_segment")
            and query.get("arm") in action.occupies), None)
        record["failed_arm"] = failed_arm
        expired = (expired or record["classification"]["kind"] == "deadline"
                   or (self.compiler.deadline is not None and time.monotonic() >= self.compiler.deadline))
        if not expired and not record["deduplicated"]:
            entry["attempts"][candidate_id] = {"result": deepcopy(result), "record": deepcopy(record),
                "validation_index": len(self.validation_records)}
        retryable = (result.status == Status.UNKNOWN and not expired and not self.fatal_error
                     and any(item["candidate_id"] not in entry["attempts"] for item in specs))
        classification = record["classification"]
        scope = classification.get("failure_scope", "candidate")
        record.update(retryable=retryable, failure_scope=scope,
                      temporary_cut=result.status != Status.VALID and scope != "task",
                      attempts_used=len(entry["attempts"]), attempts_limit=len(specs))
        self.validation_records.append(record)
        self.validation_context = {"classification": record["classification"]["kind"],
            "failed_arm": failed_arm, "ik_seed_limit": record["ik_seed_limit"],
            "validation_index": len(self.validation_records)-1, "escalated": escalated,
            "context_id": context_id, **candidate, "deduplicated": record["deduplicated"],
            "failure_scope": scope, "temporary_cut": record["temporary_cut"],
            "retryable": retryable, "attempts_used": record["attempts_used"],
            "attempts_limit": len(specs), "terminal_reason": classification["kind"]}
        if result.status != Status.VALID:
            self._pending_escalation = {"action": action.key, "state": state,
                "snapshot": self.compiler._snapshot(), "result": result,
                "context_id": context_id, "candidate": candidate}
        return result

    @_reported("validate_escalated", "validation_records", "action")
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
        entry = self.candidate_ledger[pending["context_id"]]
        specs = self._candidate_specs(action)
        candidate = next((item for item in specs if item["candidate_id"] not in entry["attempts"]), None)
        if self.compiler.deadline is not None and time.monotonic() >= self.compiler.deadline:
            return self._validate_attempt(action, state, escalated=True, candidate=pending["candidate"])
        if candidate is None:
            self.validation_context.update(escalation_skipped="already_at_least_24",
                                           retryable=False, deduplicated=True)
            return pending["result"]
        self.seed_overrides[(pending["context_id"], action.key)] = candidate["ik_seed_limit"]
        self._reset_pending = False
        return self._validate_attempt(action, state, escalated=True, candidate=candidate)

    def reconsider_cuts(self, records):
        """No unchanged, exhausted seed strategy justifies reopening a cut.

        Future action/prefix strategies may return explicit new candidate
        versions here. Ordinary references never discard history or cuts.
        """
        return []

    @_reported("feedback_round", "feedback_rounds", "record")
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
        if status == Status.VALID:
            self.import_qualification_hints([record], source="backend.qualify_candidate")
        return status

    @_reported("qualify", "qualification_records", "fact")
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
            if record["status"] == Status.VALID.value and fact.predicate == "reach":
                self.import_qualification_hints([record], source="backend.qualify")
        return status

    @staticmethod
    def _reference_schedule(schedule, graph):
        steps = sorted(getattr(schedule, "steps", schedule), key=lambda step: step.start)
        if hasattr(graph, "to_json"):
            graph_data = graph.to_json()
        elif isinstance(graph, dict):
            graph_data = deepcopy(graph)
        else:
            graph_data = {"edges": [asdict(edge) if hasattr(edge, "__dataclass_fields__") else deepcopy(edge)
                                    for edge in getattr(graph, "edges", [])]}
        payload = {"steps": [{"action": step.action.key, "start": step.start, "end": step.end,
                              "state": sorted(map(str, step.state_before))} for step in steps],
                   "graph": graph_data}
        return steps, graph_data, _digest(payload)

    def _reference_context_sha256(self):
        from .frames import APPROACH, BOARD_TOP, SIDE_GRASP_PITCH

        compiler = self.compiler
        context = {"cfg": self.cfg, "compiler_cfg": compiler.cfg,
            "frames": {"task": self.frames.task, "references": self.frames.references,
                       "catalog": {name: asdict(spec) for name, spec in self.frames.catalog.specs.items()}},
            "defaults": {"geometry_version": "skillnet-ur5-geometry-v135-1",
                         "approach_m": APPROACH, "board_top_m": BOARD_TOP, "side_pitch_rad": SIDE_GRASP_PITCH},
            "parameters": {name: getattr(compiler, name) for name in (
                "ik_seeds", "rrt_timeout", "linear_step", "joint_step", "force_rrt",
                "linear_joint_delta", "linear_position_tolerance", "linear_orientation_tolerance",
                "local_box_margin", "local_box_time", "action_candidate_limit")},
            "candidate_policy": compiler._candidate_policy(),
            "backend_hints": self.qualification_hints,
            "validated_candidates": [{"context_id": record["context_id"],
                "candidate_id": record["candidate_id"], "candidate_version": record["candidate_version"]}
                for record in self.reference_records],
            "post_state": {"q": compiler.q, "held": compiler.held, "placed": compiler.placed,
                "last_arm": compiler._last_arm, "touched": sorted(compiler._touched),
                "symbolic_context": compiler._symbolic_context, "tasks_sha256": _digest(compiler.tasks)}}
        return _digest(context)

    def _check_reference_prefix(self, steps):
        """Check the accepted action certificates without replaying geometry."""
        if len(steps) != len(self.reference_records):
            raise CompileFailure(Status.UNKNOWN, "已验证动作前缀与时间表长度不一致",
                                 evidence={"reference_prefix_mismatch": True})
        current = self.compiler._snapshot()
        try:
            for number, (step, record) in enumerate(zip(steps, self.reference_records), 1):
                if step.action.key != record["action"] or frozenset(step.state_before) != record["state"]:
                    raise CompileFailure(Status.UNKNOWN, f"第 {number} 步与已验证动作前缀不一致",
                        evidence={"reference_prefix_mismatch": True, "source_step": number})
                self.compiler._restore(deepcopy(record["start_snapshot"]))
                context_id, _ = self._action_context(self.compiler, step.action, step.state_before)
                candidates = {item["candidate_id"] for item in self._candidate_specs(step.action)}
                if context_id != record["context_id"] or record["candidate_id"] not in candidates:
                    raise CompileFailure(Status.UNKNOWN, f"第 {number} 步的几何或候选策略已改变",
                        evidence={"reference_prefix_mismatch": True, "source_step": number})
        finally:
            self.compiler._restore(current)
        if self.reference_records and current != self.reference_records[-1]["end_snapshot"]:
            raise CompileFailure(Status.UNKNOWN, "实际几何末态与已验证动作前缀不一致",
                                 evidence={"reference_prefix_mismatch": True})

    def _connect_reference_dependencies(self, steps, graph_data):
        tasks = self.compiler.tasks
        groups = {number: [task for task in tasks
                          if task.get("metadata", {}).get("source_step") == number]
                  for number in range(1, len(steps)+1)}
        for edge in graph_data.get("edges", []):
            source = edge["source"] if isinstance(edge, dict) else edge.source
            target = edge["target"] if isinstance(edge, dict) else edge.target
            if source not in groups or target not in groups:
                raise CompileFailure(Status.UNKNOWN, "完整参考的任务图引用了不存在的规划步骤")
            left, right = groups[source], groups[target]
            left_ids, right_ids = {task["id"] for task in left}, {task["id"] for task in right}
            referenced = {dep for task in left for dep in task["dependencies"] if dep in left_ids}
            leaves = [task["id"] for task in left if task["id"] not in referenced]
            roots = [task for task in right if not any(dep in right_ids for dep in task["dependencies"])]
            for task in roots:
                task["dependencies"] = list(dict.fromkeys([*task["dependencies"], *leaves]))
        self.compiler._support_event_dependencies(steps)
        # The support helper has an early return for ordinary tasks, so keep a
        # general stable topological check for the added cross-arm edges too.
        by_id = {task["id"]: task for task in self.compiler.tasks}
        if len(by_id) != len(self.compiler.tasks):
            raise CompileFailure(Status.UNKNOWN, "完整参考含重复任务标识")
        outgoing, degrees = {key: [] for key in by_id}, {}
        original = {task["id"]: index for index, task in enumerate(self.compiler.tasks)}
        for task in self.compiler.tasks:
            dependencies = list(dict.fromkeys(task["dependencies"]))
            if any(dep not in by_id for dep in dependencies):
                raise CompileFailure(Status.UNKNOWN, "完整参考含缺失的任务依赖")
            task["dependencies"] = dependencies
            degrees[task["id"]] = len(dependencies)
            for dep in dependencies:
                outgoing[dep].append(task["id"])
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
        if len(ordered) != len(by_id):
            raise CompileFailure(Status.UNKNOWN, "完整参考的任务依赖含环")
        for index, task in enumerate(ordered):
            task["reference_order"] = index
        self.compiler.tasks = ordered

    def _check_support_load_windows(self, steps):
        """Require a declared load window, not a force-success certificate.

        A force-ready event may precede the beam's arrival, but the contact
        must belong to this ready/stop episode. An unvalidated window only
        rejects this complete reference; it proves no global infeasibility.
        """
        tasks = self.compiler.tasks
        by_id = {task["id"]: task for task in tasks}
        outgoing = {ident: [] for ident in by_id}
        for task in tasks:
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

        for ready in tasks:
            metadata = ready.get("metadata", {})
            if metadata.get("action") != "support_start" or not metadata.get("force_ready_required"):
                continue
            number, slot = metadata["source_step"], metadata["slot"]
            ready_step = steps[number-1]
            stops = [task for task in tasks if task.get("metadata", {}).get("action") == "support_end"
                     and task["arm"] == ready["arm"] and task["metadata"].get("slot") == slot
                     and reaches(ready["id"], task["id"])]
            stop = min(stops, key=lambda task: task["reference_order"]) if stops else None
            stop_step = steps[stop["metadata"]["source_step"]-1] if stop else None
            parts = [part for part in metadata.get("support_parts", []) if part in self.frames.blocks
                     and self.frames.blocks[part]["type"] == self.frames.slots[slot]["type"]]
            placed = metadata.get("placed", {})
            facts = ready_step.state_before
            installed = {fact.args[0] for fact in facts if fact.predicate == "installed"
                         and fact.args[1] == slot}
            filled = any(fact.predicate == "filled" and fact.args == (slot,) for fact in facts)
            present = []
            for part in parts:
                pose = placed.get(part)
                if pose is None or not (part in installed or filled):
                    continue
                expected = self.frames.block_pose(part, slot)
                dot = abs(sum(a*b for a, b in zip(pose[3:], expected[3:])))
                if (math.dist(pose[:3], expected[:3]) <= self.compiler.linear_position_tolerance
                        and 2*math.acos(min(1., max(0., dot))) <= self.compiler.linear_orientation_tolerance):
                    present.append(part)
            contacts, rejected = [], []
            for ident in metadata.get("support_load_contact_tasks", []):
                contact = by_id.get(ident)
                install = next((task for task in tasks
                    if task.get("metadata", {}).get("action") in {"install", "release"}
                    and task["metadata"].get("slot") == slot and task.get("part") in parts
                    and ident in task["dependencies"]
                    and ready_step.end <= steps[task["metadata"]["source_step"]-1].end+1e-9
                    and stop_step is not None
                    and steps[task["metadata"]["source_step"]-1].end <= stop_step.start+1e-9), None)
                if (contact is None or contact.get("metadata", {}).get("request_kind") != "linear"
                        or install is None or contact.get("part") != install.get("part")
                        or contact["metadata"].get("source_step") != install["metadata"].get("source_step")
                        or reaches(ready["id"], ident)):
                    rejected.append({"task": ident, "reason": "missing_or_blocked_same_episode_load_contact"})
                else:
                    contacts.append(ident)
            evidence = {"reason": "unvalidated_force_load_window", "source_step": number,
                "ready_task": ready["id"], "stop_task": stop["id"] if stop else None,
                "slot": slot, "parts": parts, "present_load_parts": present,
                "same_episode_contact_tasks": contacts, "rejected_contact_tasks": rejected,
                "ready_symbolic_end_s": ready_step.end,
                "stop_symbolic_start_s": stop_step.start if stop_step else None,
                "candidate_scope": "this complete reference's declared ready/stop load window only",
                "policy_version": "support-reference-window-v135-1",
                "reference_tasks_sha256": _digest(tasks),
                "geometric_start": {"q": deepcopy(self.reference_records[number-1]["start_snapshot"][0]),
                    "held": deepcopy(self.reference_records[number-1]["start_snapshot"][1]),
                    "placed": deepcopy(placed), "contacts": deepcopy(metadata.get("contact_exclusions", []))}}
            evidence["candidate_id"] = _digest(evidence)
            if rejected or not (present or contacts):
                raise CompileFailure(Status.UNKNOWN,
                    "当前完整参考没有可验证的测力载荷窗口；并非该支撑动作或任务无解："+slot,
                    ready["id"], evidence)
            self.compiler.evidence.append({"query": "support_reference_window", "request": ready["id"],
                "answer": {**evidence, "declared_window_valid": True,
                           "force_verified": False}})

    @staticmethod
    def _terminal_collision_pair(evidence):
        while isinstance(evidence, dict):
            key = next((name for name in ("recovery_evidence", "terminal_failure", "evidence",
                                         "failure_evidence", "last_answer") if name in evidence), None)
            if key is None:
                problem = evidence.get("problem", evidence)
                return deepcopy(problem.get("pair")) if isinstance(problem, dict) else None
            evidence = evidence[key]
        if isinstance(evidence, list) and evidence:
            return CoppeliaBackend._terminal_collision_pair(evidence[-1])
        return None

    def _reference_failed(self, failure, classification, *, started, first, attempts, source_step=None):
        if is_transport_failure(failure):
            self.fatal_error = failure.detail
            classification = "transport_failure"
        scope = "candidate" if source_step is not None else "terminal"
        context = {"phase": "final_reference", "classification": classification,
            "failed_segment": failure.segment, "failure_scope": scope,
            "temporary_cut": source_step is not None, "retryable": False, "no_progress": True,
            "pair": self._terminal_collision_pair(failure.evidence),
            "failure_evidence": deepcopy(failure.evidence), "home_attempts": deepcopy(attempts),
            "context_id": self._reference_context_sha256()}
        if source_step is not None:
            context["source_step"] = source_step
        if isinstance(failure.evidence, dict) and failure.evidence.get("reason") == "unvalidated_force_load_window":
            context.update(candidate_id=failure.evidence["candidate_id"],
                           candidate_version=failure.evidence["policy_version"]+":"+failure.evidence["candidate_id"])
        self.validation_context = context
        detail = f"完整参考验证未通过（{failure.segment or '动作前缀/依赖'}）：{failure.detail}"
        self.reference_validations.append({**deepcopy(context), "status": Status.UNKNOWN.value,
            "detail": detail, "elapsed_seconds": time.monotonic()-started,
            "evidence": deepcopy(self.compiler.evidence[first:])})
        return MotionResult(Status.UNKNOWN, detail, [])

    @_reported("finalize_reference", "reference_validations")
    def finalize_reference(self, schedule, graph=None):
        """Finish the accepted reference transaction, including real homes.

        Existing action trajectories are retained. Dependency completion and
        the two bounded home orders commit together, or restore the entire
        action prefix; a terminal failure cannot invent a symbolic action cut.
        """
        self._finalization_used = True
        started, first = time.monotonic(), len(self.compiler.evidence)
        if self._complete_reference is not None:
            try:
                receipt = self.export_reference(schedule, graph)
                self.validation_context = {"phase": "final_reference", "classification": "valid",
                    "reference_reused": True, "context_id": receipt["context_sha256"], "retryable": False}
                return MotionResult(Status.VALID, "完整参考已验证；复用相同上下文收据",
                                    [task["id"] for task in receipt["tasks"]])
            except ValueError as exc:
                return self._reference_failed(CompileFailure(Status.UNKNOWN, str(exc)),
                    "reference_context_mismatch", started=started, first=first, attempts=[])
        before = self.compiler._snapshot()
        try:
            steps, graph_data, schedule_sha256 = self._reference_schedule(schedule, graph)
            if self.fatal_error:
                raise CompileFailure(Status.UNKNOWN, "远程接口已失联："+self.fatal_error)
            if self.compiler.deadline is not None and time.monotonic() >= self.compiler.deadline:
                raise CompileFailure(Status.UNKNOWN, "完整参考验证达到任务墙钟截止时间", evidence={"timeout": True})
            self._check_reference_prefix(steps)
            self._connect_reference_dependencies(steps, graph_data)
            self._check_support_load_windows(steps)
        except CompileFailure as exc:
            self.compiler._restore(before)
            source = exc.evidence.get("source_step") if isinstance(exc.evidence, dict) else None
            classification = ("deadline" if isinstance(exc.evidence, dict) and exc.evidence.get("timeout") else
                              "unvalidated_force_load_window" if isinstance(exc.evidence, dict)
                              and exc.evidence.get("reason") == "unvalidated_force_load_window" else "reference_prefix_invalid")
            return self._reference_failed(exc, classification, started=started, first=first,
                                          attempts=[], source_step=source)
        except BaseException:
            self.compiler._restore(before)
            raise
        prepared = self.compiler._snapshot()
        attempts, failure, failure_kind = [], None, None
        for order in (("A", "B"), ("B", "A")):
            self.compiler._restore(deepcopy(prepared))
            attempt = {"order": list(order), "homes": []}
            phase = "goal"
            try:
                referenced = {dep for task in self.compiler.tasks for dep in task["dependencies"]}
                dependencies = [task["id"] for task in self.compiler.tasks if task["id"] not in referenced]
                for arm in order:
                    request = MotionRequest("home_"+arm, arm, "transit",
                        target_q=list(self.cfg["arms"][arm]["home_q"]), held=deepcopy(self.compiler.held.get(arm)),
                        placed=deepcopy(self.compiler.placed), payloads=deepcopy(self.compiler.held),
                        contacts=self.compiler._resting_contacts(),
                        metadata={"source_action": "home", "final_home": True,
                                  "home_order": list(order), "complete_reference": True})
                    home = {"arm": arm, "start_q": deepcopy(self.compiler.q),
                            "goal_q": deepcopy(request.target_q), "held": deepcopy(self.compiler.held),
                            "placed": deepcopy(self.compiler.placed), "contacts": deepcopy(request.contacts)}
                    attempt["homes"].append(home)
                    phase, began = "goal", time.monotonic()
                    self.compiler._validate(request, request.target_q)
                    home["goal_check_seconds"] = time.monotonic()-began
                    phase, began = "path", time.monotonic()
                    task = self.compiler._emit(request, {}, "home", dependencies)
                    home["path_seconds"] = time.monotonic()-began
                    home["task_id"] = task["id"]
                    dependencies = [task["id"]]
                attempt["status"] = Status.VALID.value
                attempts.append(attempt)
                tasks = deepcopy(self.compiler.tasks)
                self._complete_reference = {"tasks": tasks, "tasks_sha256": _digest(tasks),
                    "schedule_sha256": schedule_sha256, "context_sha256": self._reference_context_sha256(),
                    "home_order": list(order), "validated_actions": len(steps),
                    "validation_scope": "accepted action paths, event dependencies and sequential final homes; TPG rollout remains required"}
                self.validation_context = {"phase": "final_reference", "classification": "valid",
                    "context_id": self._complete_reference["context_sha256"], "retryable": False,
                    "home_order": list(order), "reference_reused": False}
                self.reference_validations.append({**deepcopy(self.validation_context),
                    "status": Status.VALID.value, "home_attempts": deepcopy(attempts),
                    "elapsed_seconds": time.monotonic()-started,
                    "evidence": deepcopy(self.compiler.evidence[first:]),
                    "tasks_sha256": self._complete_reference["tasks_sha256"],
                    "schedule_sha256": schedule_sha256})
                return MotionResult(Status.VALID, "完整参考及必要回位通过几何验证", [task["id"] for task in tasks])
            except CompileFailure as exc:
                failure = exc
                pair = self._terminal_collision_pair(exc.evidence)
                _, classification = _failure_classification(exc)
                failure_kind = ("home_goal_collision" if phase == "goal" and pair else
                                "home_goal_unknown" if phase == "goal" else "home_path_unknown")
                if classification["kind"] == "deadline":
                    failure_kind = "deadline"
                attempt.update(status=Status.UNKNOWN.value, phase=phase, failed_segment=exc.segment,
                    detail=exc.detail, classification=failure_kind, failure_evidence=deepcopy(exc.evidence), pair=pair)
                attempts.append(attempt)
                self.compiler._restore(deepcopy(prepared))
                if is_transport_failure(exc) or failure_kind == "deadline":
                    break
            except BaseException:
                self.compiler._restore(before)
                raise
        self.compiler._restore(before)
        return self._reference_failed(failure, failure_kind, started=started, first=first, attempts=attempts)

    def export_reference(self, schedule, graph=None):
        """Export a deep copy only while the complete certificate matches."""
        receipt = self._complete_reference
        if receipt is None:
            raise ValueError("当前时间表尚无通过验证的完整参考收据")
        _, _, schedule_sha256 = self._reference_schedule(schedule, graph)
        if schedule_sha256 != receipt["schedule_sha256"]:
            raise ValueError("完整参考与当前时间表或任务依赖图不一致")
        if (_digest(receipt["tasks"]) != receipt["tasks_sha256"]
                or _digest(self.compiler.tasks) != receipt["tasks_sha256"]
                or self._reference_context_sha256() != receipt["context_sha256"]):
            raise ValueError("完整参考的几何、参数、策略或实际末态上下文已改变")
        return deepcopy(receipt)

    def evaluate(self, schedule, graph=None):
        """Return nominal parameterized seconds using a scratch compiler.

        This deliberately does not advance validation state and is not wired
        into DistanceCost. Real inter-arm scheduling/acceptance stays in the
        executor; this value is recorded as a nominal timing observation.
        """
        if self.fatal_error:
            raise ConnectionError("远程接口已失联，停止时间参数化查询："+self.fatal_error)
        reused = self._finalization_used or self._complete_reference is not None
        if reused:
            tasks = self.export_reference(schedule, graph)["tasks"]
            cache_hits = self.compiler.cache_hits
        else:
            compiler = self._scratch_compiler()
            tasks = compiler.compile(schedule, graph)
            cache_hits = compiler.cache_hits
        tpg = build_tpg(tasks)
        parameterize(tpg, vmax=.60, amax=1.20, jmax=8.)
        seconds = float(tpg["nominal_makespan"])
        self.last_evaluation = {"nominal_seconds": seconds, "tasks": len(tasks),
                                "cache_hits": cache_hits, "reference_reused": reused,
                                "search_cost_changed": False}
        return seconds

    def audit(self):
        return {"validations": self.validation_records, "qualifications": self.qualification_records,
                "qualification_hints": deepcopy(self.qualification_hints),
                "qualification_hint_imports": deepcopy(self.qualification_hint_imports),
                "feedback_rounds": self.feedback_rounds,
                "reference_validations": self.reference_validations,
                "complete_reference": ({key: deepcopy(value) for key, value in self._complete_reference.items()
                                        if key != "tasks"} if self._complete_reference is not None else None),
                "fatal_error": self.fatal_error,
                "evaluation": self.last_evaluation,
                "cache": {"entries": len(self.compiler.path_cache), "hits": self.compiler.cache_hits,
                          "misses": self.compiler.cache_misses},
                "path_certificates": self.compiler.path_cache,
                "geometry_contexts": {key: entry["context"]
                                      for key, entry in self.candidate_ledger.items()},
                "candidate_attempts": [{"context_id": context_id, "candidate_id": candidate_id,
                    "candidate_version": attempt["record"]["candidate_version"],
                    "ik_seed_limit": attempt["record"]["ik_seed_limit"],
                    "status": attempt["result"].status.value,
                    "validation_index": attempt["validation_index"],
                    "accepted": entry["accepted_candidate"] == candidate_id}
                    for context_id, entry in self.candidate_ledger.items()
                    for candidate_id, attempt in entry["attempts"].items()]}
