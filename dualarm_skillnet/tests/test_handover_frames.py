"""Base-relative transfer yaw and reachable retreat geometry, without a simulator."""
import math
from pathlib import Path
import unittest

from scipy.spatial.transform import Rotation

from simbridge.frames import BOARD_TOP, Frames


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/s1_reach_handover.toml"


def transfer(arms):
    frames = Frames(TASK)
    frames.arm_config = arms
    result = frames.handover_poses("red", "hand_mid", "B", "A", frames.payload("red", "B"))
    poses, body = result[:2]
    return frames, poses, body


def horizontal_angle(direction, reference):
    length = math.hypot(*direction[:2])*math.hypot(*reference[:2])
    cosine = sum(a*b for a, b in zip(direction[:2], reference[:2]))/length
    return math.acos(max(-1., min(1., cosine)))


class HandoverFramesTests(unittest.TestCase):
    def test_side_grasp_leaves_six_mm_for_original_shoe_front_plane(self):
        # The original 8 mm shoe is centred 4 mm behind TCP; its front face
        # reaches TCP. A cuboid's support projection gives an independent
        # separating-plane check, without treating tool contact as exempt.
        frames = Frames(ROOT / "tasks/s1_reach_handover.toml")
        size = frames.catalog[frames.blocks["red"]["type"]].size
        for side in ("A", "B"):
            pose = frames.grasp_pose("red", "stock_r1", side)
            direction = Rotation.from_quat(pose[3:]).apply([0., 0., 1.])
            extent = sum(abs(float(value))*length/2 for value, length in zip(direction, size))
            self.assertGreaterEqual(size[1]/2+.006-extent, .006-1e-9)

    def test_east_west_bases_give_inward_retreat_and_wrist_clearance(self):
        arms = {"A": {"base_pose": [-.70, .13, .783, 0, 0, 0, 1]},
                "B": {"base_pose": [.70, .13, .783, 0, 0, 1, 0]}}
        frames, poses, body = transfer(arms)
        for side in ("B", "A"):
            with self.subTest(arm=side):
                pose, base = poses[side], arms[side]["base_pose"]
                direction = Rotation.from_quat(pose[3:]).apply([0., 0., -1.])
                toward_base = [base[i]-body[i] for i in range(3)]
                self.assertLessEqual(horizontal_angle(direction, toward_base), math.pi/4)
                wrist = [pose[i]+.20*direction[i] for i in range(3)]
                self.assertLessEqual(math.dist(wrist[:2], base[:2]), .65)
                retreat = frames.handover_retreat(pose, side)
                self.assertAlmostEqual(math.dist(retreat[:3], pose[:3]), .10)
                retreat_wrist = [retreat[i]+.20*direction[i] for i in range(3)]
                self.assertLessEqual(math.dist(retreat_wrist[:2], base[:2]), .65)
        self.assertAlmostEqual(Rotation.from_quat(body[3:]).as_euler("xyz")[2], math.pi/2)

    def test_rotating_base_layout_changes_yaw_instead_of_fixing_ninety_degrees(self):
        east_west = {"A": {"base_pose": [-.70, .13, .783, 0, 0, 0, 1]},
                     "B": {"base_pose": [.70, .13, .783, 0, 0, 1, 0]}}
        north_south = {"A": {"base_pose": [0, .70, .783, 0, 0, 0, 1]},
                       "B": {"base_pose": [0, -.70, .783, 0, 0, 0, 1]}}
        _, _, first_body = transfer(east_west)
        frames, poses, second_body = transfer(north_south)
        first_yaw = Rotation.from_quat(first_body[3:]).as_euler("xyz")[2]
        second_yaw = Rotation.from_quat(second_body[3:]).as_euler("xyz")[2]
        self.assertNotAlmostEqual(first_yaw, second_yaw)
        self.assertTrue(abs(second_yaw) < 1e-9 or abs(abs(second_yaw)-math.pi) < 1e-9)
        for side in ("B", "A"):
            direction = Rotation.from_quat(poses[side][3:]).apply([0., 0., -1.])
            base = north_south[side]["base_pose"]
            toward_base = [base[i]-second_body[i] for i in range(3)]
            self.assertLessEqual(horizontal_angle(direction, toward_base), math.pi/4)

    def test_side_grasp_wrist_clearance_at_pick_place_transfer_and_retreat(self):
        arms = {"A": {"base_pose": [-.70, .13, .783, 0, 0, 0, 1]},
                "B": {"base_pose": [.70, .13, .783, 0, 0, 1, 0]}}
        frames, poses, _ = transfer(arms)
        # The five pose categories include both arms at the shared handover.
        samples = [
            ("pick", "B", frames.grasp_pose("red", "stock_r1", "B")),
            ("place", "A", frames.grasp_pose("red", "board_l1", "A")),
            ("handover", "B", poses["B"]),
            ("handover", "A", poses["A"]),
            ("giver_retreat", "B", frames.handover_retreat(poses["B"], "B")),
            ("taker_retreat", "A", frames.handover_retreat(poses["A"], "A")),
        ]
        for category, side, tcp in samples:
            tool_z = Rotation.from_quat(tcp[3:]).apply([0., 0., 1.])
            wrist = [tcp[i]-.20*tool_z[i] for i in range(3)]
            with self.subTest(category=category, arm=side, check="height"):
                self.assertGreaterEqual(wrist[2], BOARD_TOP+.12)
            with self.subTest(category=category, arm=side, check="base_distance"):
                self.assertLessEqual(math.dist(wrist[:2], arms[side]["base_pose"][:2]), .65)

    def test_side_payload_preserves_standoff_and_body_orientation_round_trip(self):
        arms = {"A": {"base_pose": [-.70, .13, .783, 0, 0, 0, 1]},
                "B": {"base_pose": [.70, .13, .783, 0, 0, 1, 0]}}
        frames, handover, handover_body = transfer(arms)
        samples = [
            ("pick", "B", frames.block_pose("red", "stock_r1"),
             frames.grasp_pose("red", "stock_r1", "B")),
            ("place", "A", frames.block_pose("red", "board_l1"),
             frames.grasp_pose("red", "board_l1", "A")),
            ("handover", "B", handover_body, handover["B"]),
            ("handover", "A", handover_body, handover["A"]),
        ]
        standoff = frames.catalog[frames.blocks["red"]["type"]].size[1]/2+.006
        for category, side, body, tcp in samples:
            with self.subTest(category=category, arm=side):
                binding = frames.side_payload("red", side)
                self.assertLessEqual(abs(math.dist(tcp[:3], body[:3])-standoff), 1e-9)
                recovered_tcp = frames.tcp_pose(body, binding)
                self.assertLessEqual(math.dist(recovered_tcp[:3], tcp[:3]), 1e-9)
                tcp_rotation = Rotation.from_quat(tcp[3:])
                body_rotation = Rotation.from_quat(body[3:])
                self.assertLessEqual((tcp_rotation.inv()*Rotation.from_quat(
                    recovered_tcp[3:])).magnitude(), 1e-9)
                recovered_center = [tcp[i]+value for i, value in
                                    enumerate(tcp_rotation.apply(binding["offset"]))]
                self.assertLessEqual(math.dist(recovered_center, body[:3]), 1e-9)
                self.assertLessEqual((body_rotation.inv()*tcp_rotation*Rotation.from_quat(
                    binding["quaternion"])).magnitude(), 1e-9)
                measured_binding = frames.payload("red", side, tcp, body)
                self.assertLessEqual(math.dist(measured_binding["offset"], binding["offset"]), 1e-9)
                self.assertLessEqual((Rotation.from_quat(binding["quaternion"]).inv()
                                     *Rotation.from_quat(measured_binding["quaternion"])).magnitude(), 1e-9)


if __name__ == "__main__":
    unittest.main()
