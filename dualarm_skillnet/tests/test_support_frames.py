"""Offline geometry of a horizontal shoe underneath a raised cantilever."""
import math
import unittest

from scipy.spatial.transform import Rotation

from simbridge.frames import BOARD_TOP, Frames
from simbridge.workcell import configure_support_tools


def raised_beam():
    data = {"name": "raised_support", "block": [], "location": [], "slot": [
        {"id": "p0", "type": "pillar_type", "pos": [.12, .12, .03], "below": []},
        {"id": "p1", "type": "pillar_type", "pos": [.12, .12, .06], "below": ["p0"]},
        {"id": "beam", "type": "beam_type", "pos": [0., .12, .09], "below": ["p1"],
         "overhang": True, "support_point": True}]}
    arms = {"A": {"base_pose": [-.7, .13, .783, 0, 0, 0, 1]},
            "B": {"base_pose": [.7, .13, .783, 0, 0, 1, 0]}}
    return Frames(data, arm_config=arms)


class SupportFrameTests(unittest.TestCase):
    def test_both_rotated_shoes_are_horizontal_and_touch_with_palm_clearance(self):
        class Recorder:
            def __init__(self):
                self.poses = {}
            def setObjectPose(self, handle, pose, parent):
                self.poses[handle] = (pose, parent)
        sim = Recorder()
        arms = {side: {"palm": side+"palm", "support_sensor": side+"sensor",
                       "support_pad": side+"shoe"} for side in ("A", "B")}
        configure_support_tools(sim, arms)
        frames = raised_beam()
        bottom = frames.support_point("beam")[2]
        for side, arm in arms.items():
            tcp = frames.support_pose("beam", side)
            tool = Rotation.from_quat(tcp[3:])
            sensor, parent = sim.poses[arm["support_sensor"]]
            self.assertEqual(parent, arm["palm"])
            normal = (tool*Rotation.from_quat(sensor[3:])).apply([0, 0, 1])
            self.assertAlmostEqual(abs(normal[2]), 1.)
            center = tool.apply([sensor[0], sensor[1], sensor[2]-.071])+tcp[:3]
            self.assertAlmostEqual(center[2]+.004, bottom)
            self.assertLessEqual(tcp[2]+.016, bottom-.002+1e-10)

    def test_both_arms_enter_horizontally_at_the_free_end_with_wrist_clearance(self):
        frames = raised_beam()
        for side in ("A", "B"):
            with self.subTest(arm=side):
                pose = frames.support_pose("beam", side)
                tool_z = Rotation.from_quat(pose[3:]).apply([0, 0, 1])
                self.assertAlmostEqual(tool_z[2], 0., places=9)
                self.assertLess(pose[0], -.16)
                base = frames.arm_config[side]["base_pose"]
                toward = [pose[0]-base[0], pose[1]-base[1]]
                self.assertGreater(sum(tool_z[i]*toward[i] for i in range(2)), 0.)
                wrist = [pose[i]-.20*tool_z[i] for i in range(3)]
                self.assertGreaterEqual(wrist[2], BOARD_TOP+.12)

    def test_retreat_withdraws_horizontally_before_a_later_lowering(self):
        frames = raised_beam()
        for side in ("A", "B"):
            pose = frames.support_pose("beam", side)
            withdrawn = frames.support_retreat(pose)
            tool_z = Rotation.from_quat(pose[3:]).apply([0, 0, 1])
            self.assertAlmostEqual(withdrawn[2], pose[2], places=9)
            for index in range(3):
                self.assertAlmostEqual(withdrawn[index]-pose[index], -.10*tool_z[index])


if __name__ == "__main__":
    unittest.main()
