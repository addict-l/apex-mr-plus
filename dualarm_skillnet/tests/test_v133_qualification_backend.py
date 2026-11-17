"""Typed public qualification calls use fake geometry, never a simulator."""
from copy import deepcopy
import math
from pathlib import Path
import tomllib
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from simbridge.backend import CoppeliaBackend
from simbridge.frames import BOARD_TOP, Frames
from simbridge.primitives import expand
from tests.test_v13_compiler import fixture
from tests.test_reach_certificate import attach_certificate


ROOT = Path(__file__).resolve().parents[1]


def typed_backend(*, goal=True):
    _, geometry, compiler = fixture("sim/s3a_cantilever_sim")
    raw = tomllib.loads((ROOT / "tasks/sim/s3a_cantilever_sim.toml").read_text(encoding="utf-8-sig"))
    raw["location"].append({"id": "typed_goal", "kind": "buffer", "pos": [.1, -.2, 0.]})
    if goal:
        raw["goal"].append("at(beam_block, typed_goal)")
    return geometry, CoppeliaBackend(geometry, compiler.cfg, Frames(raw))


class TypedQualificationTests(unittest.TestCase):
    def test_handover_qualification_matches_actual_giver_receiver_tcp_and_retreat(self):
        from scipy.spatial.transform import Rotation
        engine, geometry, compiler = fixture("s1_reach_handover")
        state = engine.execute(engine.parse_action("pick(B, red, stock_r1)"), engine.initial)
        requests = expand(engine.parse_action("handover(B, A, red, hand_mid)"), state,
                          compiler.frames, compiler.cfg)
        backend = CoppeliaBackend(geometry, compiler.cfg, compiler.frames)
        for arm in ("A", "B"):
            self.assertEqual(backend.qualify(parse_fact(f"reach({arm}, hand_mid)")), Status.VALID)
            stages = backend.qualification_records[-1]["target_contexts"][0]["stages"]
            actual = [next(req.target_pose for req in requests if req.arm == arm and req.kind == kind)
                      for kind in ("transit", "linear")]
            for stage, expected in zip(stages, actual):
                for value, target in zip(stage["pose"][:3], expected[:3]):
                    self.assertAlmostEqual(value, target, places=10)
                self.assertLess((Rotation.from_quat(stage["pose"][3:]).inv()
                                 * Rotation.from_quat(expected[3:])).magnitude(), 1e-10)

    def test_empty_goal_uses_goal_beam_type_instead_of_first_pillar(self):
        _, backend = typed_backend()
        self.assertEqual(backend.qualify(parse_fact("reach(B, typed_goal)")), Status.VALID)
        record = backend.qualification_records[-1]
        self.assertAlmostEqual(record["target_pose"][2], BOARD_TOP + .02 + .026)
        self.assertEqual(record["target_contexts"][0]["type"], "beam_type")
        self.assertEqual(record["target_contexts"][0]["block"], "beam_block")

    def test_shared_multitype_location_does_not_generalize_one_type_success(self):
        geometry, backend = typed_backend(goal=False)
        original = geometry.solve_ik
        def fail_beam(arm, position, quaternion, **kwargs):
            if position[2] < BOARD_TOP + .06:
                return {"ok": False, "infeasible": True, "problem": {"reason": "mock certified beam infeasibility"}}
            return original(arm, position, quaternion, **kwargs)
        geometry.solve_ik = fail_beam
        self.assertEqual(backend.qualify(parse_fact("reach(B, typed_goal)")), Status.UNKNOWN)
        contexts = backend.qualification_records[-1]["target_contexts"]
        self.assertEqual({context["type"] for context in contexts}, {row["type"] for row in backend.frames.blocks.values()})
        self.assertEqual({context["status"] for context in contexts}, {"VALID", "INVALID"})

    def test_slot_uses_matching_type_and_leaves_main_reference_unchanged(self):
        _, backend = typed_backend()
        before = deepcopy((backend.compiler.q, backend.compiler.placed, backend.compiler.held))
        self.assertEqual(backend.qualify(parse_fact("reach(B, beam)")), Status.VALID)
        self.assertEqual(backend.qualification_records[-1]["target_contexts"][0]["type"], "beam_type")
        self.assertAlmostEqual(backend.qualification_records[-1]["target_pose"][2], BOARD_TOP+.152+.02+.026)
        self.assertEqual([stage["ik_attempts"] for stage in backend.qualification_records[-1]["target_contexts"][0]["stages"]], [1, 1])
        self.assertEqual((backend.compiler.q, backend.compiler.placed, backend.compiler.held), before)

    def test_outer_sphere_proof_is_invalid_but_finite_candidate_collisions_remain_unknown(self):
        _, backend = typed_backend()
        attach_certificate(backend.compiler, "B", radius=.1)
        self.assertEqual(backend.qualify(parse_fact("reach(B, typed_goal)")), Status.INVALID)
        self.assertTrue(backend.qualification_records[-1]["target_contexts"][0]["failure_evidence"]["proven_infeasible"])
        geometry, backend = typed_backend()
        geometry.ik_failure = {"ok": False, "problem": {"reason": "block_robot_collision", "pair": ["wrist", "beam"]}}
        self.assertEqual(backend.qualify(parse_fact("reach(B, typed_goal)")), Status.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
