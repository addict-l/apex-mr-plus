"""Deterministic checks for the optional motion precheck transport boundary."""

from collections import deque
from copy import deepcopy
import unittest
from unittest.mock import patch

import zmq

from apexmr.motion import MotionPlanner
from simbridge.compiler import CompileFailure
from simbridge.errors import is_transport_failure
from skillnet.capability.oracle import Status
from tests import test_adjacent_ik_precheck as fixtures


HARD_DEADLINE = 2.0
WORK_DEADLINE = 1.85
ZERO = [0.0] * 6
POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class Socket:
    def __init__(self):
        self.timeouts = {zmq.SNDTIMEO: 20, zmq.RCVTIMEO: 30}

    def getsockopt(self, option):
        return self.timeouts[option]

    def setsockopt(self, option, value):
        self.timeouts[option] = value


class Client:
    """Model the bundled client's REQ send and receive boundaries."""

    def __init__(self, clock):
        self.clock = clock
        self.socket = Socket()
        self.responses = deque()
        self.sends = []
        self.receives = []

    def _send(self, request):
        self.sends.append((deepcopy(request), self.clock(), self.socket.getsockopt(zmq.SNDTIMEO)))

    def _recv(self):
        self.receives.append((self.clock(), self.socket.getsockopt(zmq.RCVTIMEO)))
        request = self.sends[-1][0]
        response = self.responses.popleft()(request)
        return {"ret": [response]}


class Sim:
    simulation_stopped = 0
    scripttype_customization = 1

    def __init__(self, client):
        self.client = client

    def getSimulationState(self):
        return self.simulation_stopped

    def createScript(self, *_):
        return 7

    def setObjectAlias(self, *_):
        pass

    def setObjectParent(self, *_):
        pass

    def initScript(self, *_):
        pass

    def callScriptFunction(self, name, script, *args):
        if name == "mpSetup":
            return {"ok": True, "collision_geometry_policy": "test policy"}
        self.client._send({"func": "sim.callScriptFunction", "args": [name, script, *args]})
        return self.client._recv()["ret"][0]


class DeadlineTransportTests(unittest.TestCase):
    def setUp(self):
        self.clock = fixtures.Clock()
        self.client = Client(self.clock)
        self.planner = MotionPlanner(Sim(self.client), self.client, {"root": 1})
        clock_patch = patch("apexmr.motion.time.monotonic", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def solve(self, options=None):
        return self.planner.solve_ik("A", POSE[:3], POSE[3:], options=options)

    def receipt(self, request, *, restored=True, expired=True, mismatched=False):
        options = request["args"][-1]
        request_id = options["probe_request_id"]
        self.assertIsNotNone(request_id)
        self.assertGreater(options["probe_work_budget_s"], 0)
        self.assertLessEqual(options["probe_work_budget_s"], WORK_DEADLINE - self.client.sends[-1][1] + .001)
        return {
            "request_id": f"other:{request_id}" if mismatched else request_id,
            "server_entered_s": 100.01,
            "work_deadline_s": 100.01 + options["probe_work_budget_s"],
            "work_ended_s": 100.01 + options["probe_work_budget_s"] + (.01 if expired else -.01),
            "restored_s": 100.01 + options["probe_work_budget_s"] + .02,
            "restored": restored,
            "work_expired": expired,
            "boundary": "after_initial_ik",
        }

    def assert_transport_settings_restored(self):
        self.assertEqual(self.client.socket.timeouts, {zmq.SNDTIMEO: 20, zmq.RCVTIMEO: 30})
        self.assertNotIn("_send", vars(self.client))
        self.assertNotIn("_recv", vars(self.client))

    def test_before_send_expiry_preserves_the_same_channel(self):
        self.clock.value = 1.86
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(TimeoutError) as failure:
                self.solve()
        self.assertFalse(is_transport_failure(failure.exception))
        self.assertEqual(self.client.sends, [])
        self.assertEqual(self.client.receives, [])
        self.client.responses.append(lambda _: {"ok": True, "pose": POSE})
        self.assertEqual(self.planner.fk("A", ZERO)["pose"], POSE)
        self.assertEqual(len(self.client.sends), 1)
        self.assert_transport_settings_restored()

    def test_definitive_expired_reply_is_drained_and_same_channel_falls_back(self):
        self.clock.value = 1.2
        native = {}

        def timeout_reply(request):
            native.update(self.receipt(request))
            self.clock.value = 1.94
            return {"ok": False, "probe_timeout": True, "probe_receipt": native}

        self.client.responses.extend([timeout_reply, lambda _: {"ok": True, "solutions": [ZERO]}])
        options = {"seed": ZERO, "multiple": False}
        original = deepcopy(options)
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(TimeoutError) as failure:
                self.solve(options)
        self.assertFalse(is_transport_failure(failure.exception))
        self.assertEqual(failure.exception.receipt["request_id"], native["request_id"])
        self.assertEqual(failure.exception.receipt["native"], native)
        self.assertEqual(failure.exception.receipt["final_reply_s"], 1.94)
        sent_time, send_timeout = self.client.sends[0][1:]
        wait_time, receive_timeout = self.client.receives[0]
        self.assertLessEqual(send_timeout, round((WORK_DEADLINE - sent_time) * 1000))
        self.assertLessEqual(receive_timeout, round((HARD_DEADLINE - wait_time) * 1000))
        self.assertGreater(receive_timeout, round((WORK_DEADLINE - wait_time) * 1000))
        self.assertEqual(self.solve(options)["q"], ZERO)
        self.assertEqual(self.client.sends[-1][0]["args"][-1], original)
        self.assertEqual(options, original)
        self.assert_transport_settings_restored()

    def test_result_received_during_drain_cannot_pass_the_probe(self):
        self.clock.value = 1.2

        def late_success(request):
            self.clock.value = 1.9
            return {"ok": True, "solutions": [ZERO], "probe_receipt": self.receipt(request, expired=False)}

        self.client.responses.extend([late_success, lambda _: {"ok": True, "pose": POSE}])
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(TimeoutError) as failure:
                self.solve()
        self.assertFalse(is_transport_failure(failure.exception))
        self.assertEqual(self.planner.fk("A", ZERO)["pose"], POSE)
        self.assertEqual(len(self.client.receives), 2)
        self.assert_transport_settings_restored()

    def assert_poisoned(self):
        old_counts = len(self.client.sends), len(self.client.receives)
        with self.assertRaises(ConnectionError) as failure:
            self.planner.fk("A", ZERO)
        self.assertTrue(is_transport_failure(failure.exception))
        self.assertEqual((len(self.client.sends), len(self.client.receives)), old_counts)
        self.assertEqual(len(self.client.responses), 1)
        self.assert_transport_settings_restored()

    def test_missing_hard_reply_poisoned_without_consuming_a_later_message(self):
        def no_reply(_):
            self.clock.value = HARD_DEADLINE
            raise zmq.Again()

        self.client.responses.extend([no_reply, lambda _: {"ok": True, "pose": POSE}])
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(ConnectionError) as failure:
                self.solve()
        self.assertTrue(is_transport_failure(failure.exception))
        self.assert_poisoned()

    def test_wrong_receipt_id_poisoned(self):
        self.client.responses.extend([
            lambda request: {"ok": False, "probe_timeout": True,
                             "probe_receipt": self.receipt(request, mismatched=True)},
            lambda _: {"ok": True, "pose": POSE},
        ])
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(ConnectionError):
                self.solve()
        self.assert_poisoned()

    def test_failed_transaction_restoration_poisoned(self):
        self.client.responses.extend([
            lambda request: {"ok": False, "probe_timeout": True,
                             "probe_receipt": self.receipt(request, restored=False)},
            lambda _: {"ok": True, "pose": POSE},
        ])
        with self.planner.precheck_deadline(HARD_DEADLINE):
            with self.assertRaises(ConnectionError):
                self.solve()
        self.assert_poisoned()


class CompilerDeadlineTests(unittest.TestCase):
    def test_safe_probe_timeout_preserves_order_and_rechecks_native_action(self):
        compiler, action, initial, home, plans, _, follower = fixtures.AdjacentIKTests().configure(failure=None)
        original_line = compiler._line
        original_deadline = compiler.deadline

        def line(request, start):
            if not plans and request.id == follower.id:
                timeout = TimeoutError("Native probe reached its work deadline; restored")
                timeout.receipt = {"request_id": "probe", "restored": True, "work_expired": True}
                raise CompileFailure(Status.UNKNOWN, "optional probe expired", request.id) from timeout
            return original_line(request, start)

        compiler._line = line
        tasks = compiler.compile_action(action, initial)
        event = fixtures.AdjacentIKTests().observation(compiler)
        self.assertEqual(event["status"], "timeout")
        self.assertEqual(event["attempt_order"], [0, 1])
        self.assertTrue(event["candidates"][0]["deadline_receipt"]["restored"])
        self.assertEqual(plans[0][0], home["A"])
        self.assertGreater(plans[0][1][2], 0)
        self.assertEqual([task["metadata"]["source_request"] for task in tasks], ["r1", "r2", "r3", "r4"])
        self.assertEqual(compiler.deadline, original_deadline)


if __name__ == "__main__":
    unittest.main()
