"""Adjacent-line ordering through the ordinary grounded action compiler."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import zmq

from apexmr.motion import MotionPlanner
from simbridge.compiler import CompileFailure
from simbridge.errors import is_transport_failure
from simbridge.primitives import expand
from skillnet.capability.oracle import Status
from tests.test_v13_compiler import fixture


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class AdjacentIKTests(unittest.TestCase):
    def configure(self, *, failure="positive", rrt_failure=False, clock=None):
        engine, geometry, compiler = fixture(ik_seeds=2)
        action = engine.parse_action("pick(A, red, stock_l1)")
        initial = engine.initial
        approach, descent = expand(action, initial, compiler.frames, compiler.cfg)[:2]
        home = deepcopy(compiler.q)
        base_ik, base_fk, base_plan = geometry.solve_ik, geometry.fk, geometry.plan
        branch_queries = []
        plans = []

        def ik(arm, position, quaternion, **kwargs):
            seed = kwargs["options"]["seed"]
            if position == approach.target_pose[:3] and not kwargs.get("payload"):
                result = base_ik(arm, position, quaternion, **kwargs)
                if seed != home[arm]:
                    result["q"][2] *= -1
                return result
            negative = seed[2] < 0
            if position[2] < approach.target_pose[2] - 1e-4 and not kwargs.get("payload"):
                branch_queries.append((negative, deepcopy(kwargs), len(plans)))
                if clock is not None and not plans:
                    clock.value = 2.0
                invalid = failure == "both" or (failure == "positive" and not negative)
                if invalid and (not rrt_failure or not plans):
                    return {"ok": False, "problem": {"reason": "table_collision:A",
                            "pair": ["Link2", "Workbench"]}}
            result = base_ik(arm, position, quaternion, **kwargs)
            if negative:
                result["q"][2] *= -1
            return result

        def fk(arm, q):
            equivalent = list(q)
            equivalent[2] = abs(q[2])
            return base_fk(arm, equivalent)

        def plan(arm, start, goal, **kwargs):
            plans.append((list(start), list(goal), deepcopy(kwargs)))
            if rrt_failure and goal[2] < 0:
                return {"ok": False, "error": "No exact collision-free solution within budget"}
            return base_plan(arm, start, goal, **kwargs)

        geometry.solve_ik, geometry.fk, geometry.plan = ik, fk, plan
        return compiler, action, initial, home, plans, branch_queries, descent

    def observation(self, compiler):
        return next(row["answer"] for row in compiler.evidence
                    if row["query"] == "adjacent_linear_precheck")

    def test_promotes_first_passing_continuation_and_checks_it_again_after_rrt(self):
        comp, action, state, home, plans, queries, descent = self.configure()
        tasks = comp.compile_action(action, state)
        self.assertLess(plans[0][1][2], 0)
        self.assertEqual(plans[0][0], home["A"])
        self.assertEqual([task["metadata"]["source_request"] for task in tasks], ["r1", "r2", "r3", "r4"])
        for negative, kwargs, plan_count in queries:
            self.assertEqual(kwargs["options"]["other_q"], home["B"])
            self.assertEqual(kwargs["options"]["contact_exclusions"], descent.contacts)
            self.assertEqual(kwargs["placed"], descent.placed)
        self.assertTrue(any(negative and count == 0 for negative, _, count in queries))
        self.assertTrue(any(negative and count > 0 for negative, _, count in queries))
        self.assertEqual(self.observation(comp)["attempt_order"], [1, 0])

    def test_stops_when_original_first_passes(self):
        comp, action, state, _, plans, _, _ = self.configure(failure=None)
        self.assertEqual(len(comp.compile_action(action, state)), 4)
        self.assertGreater(plans[0][1][2], 0)
        result = self.observation(comp)
        self.assertEqual(result["status"], "original_first_passed")
        self.assertEqual([row["rank"] for row in result["candidates"]], [0])

    def test_cooperative_timeout_restores_original_order_context_and_task_deadline(self):
        clock = Clock()
        comp, action, state, home, plans, queries, _ = self.configure(failure=None, clock=clock)
        comp.deadline = 20.0
        with patch("simbridge.compiler.time.monotonic", clock):
            self.assertEqual(len(comp.compile_action(action, state)), 4)
        self.assertEqual(comp.deadline, 20.0)
        self.assertEqual(plans[0][0], home["A"])
        self.assertEqual(plans[0][2]["other_q"], home["B"])
        self.assertGreater(plans[0][1][2], 0)
        result = self.observation(comp)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["attempt_order"], [0, 1])
        self.assertEqual([row["rank"] for row in result["candidates"]], [0])

    def test_unconnectable_promotion_falls_back_to_original_rank(self):
        comp, action, state, home, plans, _, _ = self.configure(rrt_failure=True)
        self.assertEqual(len(comp.compile_action(action, state)), 4)
        self.assertLess(plans[0][1][2], 0)
        self.assertGreater(plans[-1][1][2], 0)
        self.assertTrue(all(plan[0]==home["A"] for plan in plans))
        self.assertLessEqual(sum(plan[2]["solve_time"] for plan in plans), 10.0)
        ranks = [row["endpoint_rank"] for row in comp.evidence if row["query"] == "RRTConnect"]
        self.assertEqual(list(dict.fromkeys(ranks)), [1, 0])
        self.assertEqual(sum(row["query"]=="adjacent_linear_precheck" for row in comp.evidence), 1)


    def test_all_finite_nonpasses_preserve_order_and_original_collision_checks(self):
        comp, action, state, home, plans, queries, _ = self.configure(failure="both")
        placed = deepcopy(comp.placed)
        with self.assertRaises(CompileFailure) as failed:
            comp.compile_action(action, state)
        self.assertEqual(failed.exception.status, Status.UNKNOWN)
        self.assertGreater(plans[0][1][2], 0)
        self.assertEqual(comp.q, home)
        self.assertEqual(comp.held, {})
        self.assertEqual(comp.placed, placed)
        self.assertEqual(comp.tasks, [])
        event = self.observation(comp)
        self.assertEqual(event["status"], "no_valid_candidate")
        self.assertEqual(event["attempt_order"], [0, 1])
        self.assertEqual([row["rank"] for row in event["candidates"]], [0, 1])
        self.assertTrue(any(count > 0 for _, _, count in queries))


class PrecheckTransportTests(unittest.TestCase):
    class Socket:
        def __init__(self):
            self.timeouts = {zmq.SNDTIMEO: 20, zmq.RCVTIMEO: 30}

        def getsockopt(self, option):
            return self.timeouts[option]

        def setsockopt(self, option, value):
            self.timeouts[option] = value

    def make_planner(self, clock, receive_failure=False):
        transport = self.Socket()
        calls = []

        class Client:
            socket = transport

            def _send(self, data):
                calls.append(("send", self.socket.getsockopt(zmq.SNDTIMEO)))
                clock.value += .25

            def _recv(self):
                calls.append(("recv", self.socket.getsockopt(zmq.RCVTIMEO)))
                if receive_failure:
                    raise zmq.Again()
                return {"ok": True, "pose": [0, 0, 0, 0, 0, 0, 1]}

        class Sim:
            def callScriptFunction(self, function, script, *args):
                client._send((function, script, *args))
                return client._recv()

        client = Client()
        planner = MotionPlanner.__new__(MotionPlanner)
        planner.sim, planner.client, planner.script = Sim(), client, 1
        planner._precheck_deadline = None
        return planner, calls

    def test_absolute_deadline_bounds_each_rpc_wait_and_restores_socket(self):
        clock = Clock()
        planner, calls = self.make_planner(clock)
        with patch("apexmr.motion.time.monotonic", clock), planner.precheck_deadline(2.0):
            self.assertTrue(planner.fk("A", [0.]*6)["ok"])
            clock.value = 1.98
            with self.assertRaises(TimeoutError):
                planner.fk("A", [0.]*6)
        self.assertEqual(calls, [("send", 1850), ("recv", 1750)])
        self.assertEqual(planner.client.socket.timeouts, {zmq.SNDTIMEO: 20, zmq.RCVTIMEO: 30})
        self.assertNotIn("_send", vars(planner.client))
        self.assertIsNone(planner._precheck_deadline)

    def test_in_flight_transport_error_is_not_a_safe_geometric_fallback(self):
        clock = Clock()
        planner, _ = self.make_planner(clock, receive_failure=True)
        with patch("apexmr.motion.time.monotonic", clock), planner.precheck_deadline(2.0):
            with self.assertRaises(ConnectionError) as failed:
                planner.fk("A", [0.]*6)
        self.assertTrue(is_transport_failure(failed.exception))
        self.assertEqual(planner.client.socket.timeouts, {zmq.SNDTIMEO: 20, zmq.RCVTIMEO: 30})
        self.assertIsNone(planner._precheck_deadline)


if __name__ == "__main__":
    unittest.main()
