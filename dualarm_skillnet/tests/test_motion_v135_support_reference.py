"""Force-ready windows in complete references, using geometry doubles only."""
from copy import deepcopy
import json
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.planning.taskgraph import build_task_graph
from tests.test_coppelia_backend import backend_fixture
from tests.test_v13_compiler import sequential


class SupportReferenceWindowTests(unittest.TestCase):
    def prepared(self, *, empty_episode=False, existing_load_episode=False):
        engine, geometry, backend = backend_fixture("s3a_cantilever")
        actions = ["pick(A, pillar, stock_l1)", "insert(A, pillar, p1)"]
        if empty_episode:
            actions += ["support_on(A, beam)", "support_off(A, beam)"]
        actions += ["pick(B, beam_block, stock_r1)", "support_on(A, beam)",
                    "insert(B, beam_block, beam)", "pick(B, counterweight, stock_r2)",
                    "insert(B, counterweight, cw)", "support_off(A, beam)"]
        if existing_load_episode:
            actions += ["support_on(A, beam)", "support_off(A, beam)"]
        steps = sequential(engine, actions)
        backend.begin_reference(1)
        for step in steps:
            self.assertEqual(backend.validate(step.action, step.state).status, Status.VALID)
        return engine, geometry, backend, steps

    def test_empty_ready_episode_is_scoped_unknown_and_restores_complete_prefix(self):
        engine, geometry, backend, steps = self.prepared(empty_episode=True)
        before = deepcopy((backend.compiler.q, backend.compiler.held, backend.compiler.placed,
                           backend.compiler.tasks))
        calls = len(geometry.calls)
        result = backend.finalize_reference(steps, build_task_graph(engine, steps))
        self.assertEqual(result.status, Status.UNKNOWN,
                         "A later valid episode must not certify this unloaded ready/stop detour")
        context = backend.validation_context
        self.assertEqual(context["source_step"], 3)
        self.assertEqual(context["failure_scope"], "candidate")
        self.assertTrue(context["temporary_cut"])
        self.assertEqual(context["classification"], "unvalidated_force_load_window")
        self.assertEqual(context["failure_evidence"]["source_step"], 3)
        self.assertIn("beam", str(context["failure_evidence"]))
        self.assertEqual((backend.compiler.q, backend.compiler.held, backend.compiler.placed,
                          backend.compiler.tasks), before)
        self.assertEqual(len(geometry.calls), calls, "Reject the incomplete window before solving homes")
        with self.assertRaises(ValueError):
            backend.export_reference(steps, build_task_graph(engine, steps))

    def test_preload_ready_keeps_its_same_episode_contact_and_required_final_homes(self):
        engine, _, backend, steps = self.prepared()
        ready_step = next(step for step in steps if step.action.name == "support_on")
        self.assertFalse(any(f.predicate == "filled" and f.args == ("beam",) for f in ready_step.state))
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        tasks = backend.export_reference(steps, graph)["tasks"]
        ready = next(task for task in tasks if task["metadata"].get("action") == "support_start")
        contacts = ready["metadata"].get("support_load_contact_tasks", [])
        self.assertEqual(len(contacts), 1)
        contact = next(task for task in tasks if task["id"] == contacts[0])
        self.assertEqual(contact["metadata"]["source_action"], "insert(B, beam_block, beam)")
        self.assertEqual(contact["metadata"]["request_kind"], "linear")
        self.assertEqual(ready["duration"], .3)
        self.assertEqual({task["arm"] for task in tasks if task["metadata"].get("final_home")}, {"A", "B"})
        window = [row["answer"] for row in backend.compiler.evidence
                  if row.get("query") == "support_reference_window"][-1]
        self.assertFalse(window["force_verified"])
        self.assertEqual(window["present_load_parts"], [])
        self.assertEqual(window["same_episode_contact_tasks"], contacts)

    def test_existing_installed_load_is_accepted_without_a_new_contact_episode(self):
        engine, _, backend, steps = self.prepared(existing_load_episode=True)
        graph = build_task_graph(engine, steps)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.VALID)
        tasks = backend.export_reference(steps, graph)["tasks"]
        readies = [task for task in tasks if task["metadata"].get("action") == "support_start"]
        self.assertEqual(len(readies), 2)
        self.assertEqual(readies[-1]["metadata"].get("support_load_contact_tasks", []), [])
        window = [row["answer"] for row in backend.compiler.evidence
                  if row.get("query") == "support_reference_window"][-1]
        self.assertEqual(window["present_load_parts"], ["beam_block"])
        self.assertFalse(window["force_verified"])

    def test_unchanged_incomplete_window_keeps_the_same_candidate_without_motion_retries(self):
        engine, geometry, backend, steps = self.prepared(empty_episode=True)
        graph = build_task_graph(engine, steps)
        calls = len(geometry.calls)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.UNKNOWN)
        first = deepcopy(backend.validation_context)
        self.assertEqual(backend.finalize_reference(steps, graph).status, Status.UNKNOWN)
        self.assertEqual(backend.validation_context["candidate_id"], first["candidate_id"])
        self.assertEqual(backend.validation_context["candidate_version"], first["candidate_version"])
        self.assertEqual(backend.validation_context["context_id"], first["context_id"])
        self.assertEqual(len(geometry.calls), calls)


class MotionProgressEventsTests(unittest.TestCase):
    def test_validate_publishes_exact_before_after_context_and_only_the_new_record(self):
        engine, _, backend = backend_fixture()
        events = []
        backend.progress_callback = events.append
        backend.begin_reference(4)
        before = deepcopy(backend.compiler.q)
        action = engine.parse_action("pick(A, red, stock_l1)")
        self.assertEqual(backend.validate(action, engine.initial).status, Status.VALID)
        self.assertEqual([(e["query"], e["phase"]) for e in events],
                         [("validate", "before"), ("validate", "after")])
        self.assertEqual(events[0]["q"], before)
        self.assertEqual(events[0]["held"], {})
        self.assertEqual(events[-1]["q"], backend.compiler.q)
        self.assertEqual(events[-1]["held"]["A"]["name"], "red")
        self.assertEqual(events[-1]["round"], 4)
        self.assertEqual(events[-1]["action"], action.text)
        self.assertEqual(events[-1]["new_record_indices"], [0])
        self.assertEqual(events[-1]["record"], backend.validation_records[0])
        self.assertNotIn("validations", events[-1])
        self.assertGreaterEqual(events[-1]["elapsed_seconds"], 0.)
        json.dumps(events, allow_nan=False)
        events[-1]["held"]["A"]["offset"][0] += 1.
        self.assertNotEqual(events[-1]["held"], backend.compiler.held)

    def test_failed_motion_publishes_rolled_back_context_and_escalation_record(self):
        engine, geometry, backend = backend_fixture()
        geometry.ik_failure = {"ok": False, "solutions": []}
        events = []
        backend.progress_callback = events.append
        before = deepcopy((backend.compiler.q, backend.compiler.held, backend.compiler.placed))
        action = engine.parse_action("pick(A, red, stock_l1)")
        self.assertEqual(backend.validate(action, engine.initial).status, Status.UNKNOWN)
        self.assertEqual(backend.validate_escalated(action, engine.initial).status, Status.UNKNOWN)
        self.assertEqual([(e["query"], e["phase"]) for e in events],
                         [("validate", "before"), ("validate", "after"),
                          ("validate_escalated", "before"), ("validate_escalated", "after")])
        for event in events:
            self.assertEqual((event["q"], event["held"], event["placed"]), before)
        self.assertEqual(events[-1]["new_record_indices"], [1])
        self.assertEqual(events[-1]["record"]["status"], "UNKNOWN")
        self.assertTrue(events[-1]["record"]["escalated"])

    def test_qualification_and_feedback_publish_serializable_scoped_records(self):
        _, _, backend = backend_fixture()
        events = []
        backend.progress_callback = events.append
        self.assertEqual(backend.qualify(parse_fact("reach(A, stock_l1)")), Status.VALID)
        self.assertEqual(events[-1]["query"], "qualify")
        self.assertEqual(events[-1]["fact"], "reach(A, stock_l1)")
        self.assertEqual(events[-1]["record_collection"], "qualification_records")
        backend.record_feedback_round({"round": 2, "terminal_status": "unknown"})
        self.assertEqual(events[-1]["query"], "feedback_round")
        self.assertEqual(events[-1]["phase"], "after")
        self.assertEqual(events[-1]["round"], 2)
        self.assertEqual(events[-1]["record"], backend.feedback_rounds[-1])
        json.dumps(events, allow_nan=False)

    def test_final_reference_publishes_homes_or_rolled_back_window_failure(self):
        fixture = SupportReferenceWindowTests()
        for empty, expected in ((False, Status.VALID), (True, Status.UNKNOWN)):
            with self.subTest(empty_episode=empty):
                engine, _, backend, steps = fixture.prepared(empty_episode=empty)
                events = []
                backend.progress_callback = events.append
                self.assertEqual(backend.finalize_reference(steps, build_task_graph(engine, steps)).status, expected)
                self.assertEqual([(e["query"], e["phase"]) for e in events],
                                 [("finalize_reference", "before"), ("finalize_reference", "after")])
                self.assertEqual(events[-1]["status"], expected.value)
                self.assertEqual(events[-1]["record"], backend.reference_validations[-1])
                self.assertEqual(events[-1]["q"], backend.compiler.q)
                json.dumps(events, allow_nan=False)

    def test_callback_io_errors_propagate_and_none_keeps_old_constructor_compatible(self):
        engine, geometry, backend = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        def unavailable(_):
            raise OSError("diagnostic disk unavailable")
        backend.progress_callback = unavailable
        with self.assertRaisesRegex(OSError, "diagnostic disk"):
            backend.validate(action, engine.initial)
        self.assertFalse(geometry.calls, "An entry persistence failure cannot silently run motion")
        backend.progress_callback = None
        self.assertEqual(backend.validate(action, engine.initial).status, Status.VALID)

    def test_exit_persistence_error_is_visible_and_exceptional_operations_emit_exit(self):
        engine, _, backend = backend_fixture()
        events = []
        def fail_exit(event):
            events.append(event)
            if event["phase"] == "after":
                raise OSError("exit journal unavailable")
        backend.progress_callback = fail_exit
        with self.assertRaisesRegex(OSError, "exit journal"):
            backend.validate(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(events[-1]["record"]["status"], "VALID")
        # The operation succeeded geometrically, but cannot be reported as
        # successful by the runner when required evidence persistence failed.
        events.clear()
        backend.progress_callback = events.append
        with self.assertRaisesRegex(ValueError, "升级验证"):
            backend.validate_escalated(engine.parse_action("pick(B, blue, stock_r1)"), engine.initial)
        self.assertEqual(events[-1]["query"], "validate_escalated")
        self.assertEqual(events[-1]["phase"], "after")
        self.assertEqual(events[-1]["exception"]["type"], "ValueError")
        self.assertEqual(events[-1]["new_record_indices"], [])


if __name__ == "__main__":
    unittest.main()
