"""All nine grounded skill expansions and object-mode transitions."""
from copy import deepcopy
import math
from pathlib import Path
import unittest

from scipy.spatial.transform import Rotation

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem, problem_from_data
from simbridge.frames import SIDE_GRASP_PITCH, SUPPORT_PAD_TOP_OFFSET, Frames, read_task
from simbridge.primitives import PrimitiveError, expand, world_state


ROOT = Path(__file__).resolve().parents[1]


def setup_task(name):
    path = ROOT / "tasks" / (name+".toml")
    problem = load_problem(path, ROOT / "skills/blocks_v1.toml")
    arms = {"A": {"base_pose": [-.70, .13, .783, 0, 0, 0, 1]},
            "B": {"base_pose": [.70, .13, .783, 0, 0, 1, 0]}}
    return problem, Engine(problem), Frames(path, arm_config=arms)


class PrimitiveTests(unittest.TestCase):
    def test_pick_approach_grasp_lift_changes_mode(self):
        problem, engine, frames = setup_task("s1_parallel_place")
        requests = expand(engine.parse_action("pick(A, red, stock_l1)"), problem.initial, frames)
        self.assertEqual([request.kind for request in requests], ["transit", "linear", "skill", "linear"])
        self.assertEqual(requests[2].action, "grasp")
        self.assertIn("red", requests[1].placed)
        self.assertNotIn("red", requests[3].placed)
        self.assertEqual(requests[3].held["name"], "red")
        self.assertAlmostEqual(requests[0].target_pose[2]-requests[1].target_pose[2], .10)
        self.assertAlmostEqual(requests[3].target_pose[2]-requests[1].target_pose[2], .10)
        self.assertEqual(requests[3].dependencies, [requests[2].id])

    def test_pick_lift_keeps_both_registered_coplanar_bearing_contacts(self):
        problem, engine, frames = setup_task("s1_parallel_place")
        cfg = {"pads": {"stock_r1": 101}, "baseplate": 102,
               "support_contact_surfaces": {"blue": [101, 102]}}
        requests = expand(engine.parse_action("pick(B, blue, stock_r1)"), problem.initial, frames, cfg)
        # The right stock body overlaps the board footprint by 10 mm at its
        # edge. Both coplanar surfaces must be valid at the lift's first point.
        lift = requests[-1]
        self.assertEqual(lift.kind, "linear")
        self.assertEqual(lift.held["name"], "blue")
        self.assertIn(["blue", 101], lift.contacts)
        self.assertIn(["blue", 102], lift.contacts)
        self.assertNotIn(["blue", "Workbench"], lift.contacts)
        self.assertEqual(lift.contacts.count(["blue", 101]), 1)

    def test_place_releases_without_rigid_snap(self):
        problem, engine, frames = setup_task("s1_parallel_place")
        state = {fact for fact in problem.initial if fact != Fact("at", ("red", "stock_l1"))}
        state.add(Fact("held", ("A", "red")))
        requests = expand(engine.parse_action("place(A, red, board_l1)"), state, frames)
        release = requests[2]
        self.assertEqual(release.action, "release")
        self.assertFalse(release.metadata["rigid_snap"])
        self.assertIsNone(requests[3].held)
        self.assertEqual(requests[3].placed["red"], frames.block_pose("red", "board_l1"))

    def test_insert_installs_at_physical_slot_height(self):
        problem, engine, frames = setup_task("s2_tower")
        state = {Fact("held", ("A", "b1"))}
        requests = expand(engine.parse_action("insert(A, b1, t1)"), state, frames)
        install = requests[2]
        self.assertEqual(install.action, "install")
        self.assertTrue(install.metadata["rigid_snap"])
        self.assertEqual(install.metadata["slot"], "t1")
        self.assertEqual(install.metadata["target_pose"], frames.slot_pose("t1"))
        self.assertIsNone(requests[3].held)

    def test_detach_grasps_unfixes_then_lifts(self):
        problem, engine, frames = setup_task("s2_tower")
        state = {Fact("installed", ("b1", "t1")), Fact("filled", ("t1",))}
        requests = expand(engine.parse_action("detach(A, b1, t1)"), state, frames)
        self.assertEqual([request.action for request in requests], [None, None, "grasp", "detach", None])
        self.assertTrue(requests[2].metadata["installed_grasp"])
        self.assertTrue(requests[3].metadata["remove_attachment"])
        self.assertEqual(requests[-1].held["name"], "b1")
        self.assertNotIn("b1", requests[-1].placed)

    def test_handover_explicit_cross_arm_dependencies_and_preserved_binding(self):
        problem, engine, frames = setup_task("s1_reach_handover")
        state = {Fact("held", ("B", "red"))}
        requests = expand(engine.parse_action("handover(B, A, red, hand_mid)"), state, frames)
        self.assertEqual([request.arm for request in requests], ["B", "A", "A", "B", "B", "A"])
        take, give = requests[2:4]
        self.assertEqual(set(take.dependencies), {requests[0].id, requests[1].id})
        self.assertIn(take.id, give.dependencies)
        self.assertIn(give.id, requests[-1].dependencies)
        self.assertEqual(take.action, "grasp")
        self.assertEqual(give.action, "release")
        self.assertFalse(give.metadata["rigid_snap"])
        self.assertEqual(requests[0].held, world_state(state, frames)[1]["B"])
        self.assertIsNone(requests[-2].held)
        self.assertEqual(requests[-1].held["name"], "red")
        # The two tool-z approach axes are horizontally opposed and both
        # pitched downward by SIDE_GRASP_PITCH (v1.3.2): a horizontal side
        # grasp would leave each wrist at block height.
        z0 = Rotation.from_quat(requests[0].target_pose[3:]).apply([0, 0, 1])
        z1 = Rotation.from_quat(requests[1].target_pose[3:]).apply([0, 0, 1])
        self.assertAlmostEqual(float(z0[0]*z1[0]+z0[1]*z1[1]), -math.cos(SIDE_GRASP_PITCH)**2)
        self.assertAlmostEqual(float(z0[2]), -math.sin(SIDE_GRASP_PITCH))
        self.assertAlmostEqual(float(z1[2]), -math.sin(SIDE_GRASP_PITCH))
        # The airborne block turns90deg; both fixed side bindings now approach
        # from their own base side, and retreat along negative tool-z.
        self.assertLess(z0[0], 0.)
        self.assertGreater(z1[0], 0.)
        self.assertAlmostEqual(float(z0[1]), 0.)
        self.assertAlmostEqual(float(z1[1]), 0.)
        center_x = frames.references["hand_mid"][0]
        self.assertGreater(requests[0].target_pose[0], center_x)
        self.assertLess(requests[1].target_pose[0], center_x)
        # Retreats run 0.10 m along negative tool-z: outward toward each base
        # and upward by the grasp pitch.
        outward, upward = .10*math.cos(SIDE_GRASP_PITCH), .10*math.sin(SIDE_GRASP_PITCH)
        self.assertAlmostEqual(requests[-2].target_pose[0], requests[0].target_pose[0]+outward)
        self.assertAlmostEqual(requests[-1].target_pose[0], requests[1].target_pose[0]-outward)
        self.assertAlmostEqual(requests[-2].target_pose[2], requests[0].target_pose[2]+upward)
        self.assertAlmostEqual(requests[-1].target_pose[2], requests[1].target_pose[2]+upward)
        for request in requests:
            self.assertAlmostEqual(request.metadata["yaw_rad"], 1.5707963267948966)
            self.assertTrue(request.metadata["yaw_selection_reason"])
            self.assertEqual(request.metadata["handover_frame_selection"]["base_positions"]["B"],
                             [.70, .13, .783])
        self.assertAlmostEqual(requests[0].held["offset"][2], .14/2+.006)
        # New owner places with its actual side binding and identity block pose.
        after = {Fact("held", ("A", "red"))}
        place = expand(engine.parse_action("place(A, red, board_l1)"), after, frames,
                       placed=requests[-1].placed, payloads=requests[-1].payloads)
        self.assertEqual(place[0].held, requests[-1].held)
        self.assertEqual(place[2].metadata["target_pose"], frames.block_pose("red", "board_l1"))

    def test_transfer_side_pick_keeps_wrist_clearance_and_body_transform(self):
        problem, engine, frames = setup_task("s1_reach_handover")
        requests = expand(engine.parse_action("pick(B, red, stock_r1)"), problem.initial, frames)
        grasp_pose = requests[1].target_pose
        lift = requests[-1]
        # A south-face TCP stands off by half catalog depth +6mm along the
        # pitched approach axis: outside the face and above the block top, so
        # the wrist stays clear of the pad. The block keeps identity orientation.
        self.assertAlmostEqual(grasp_pose[0], .4)
        self.assertAlmostEqual(grasp_pose[1], -.2-.076*math.cos(SIDE_GRASP_PITCH))
        self.assertAlmostEqual(grasp_pose[2], frames.block_pose("red")[2]+.076*math.sin(SIDE_GRASP_PITCH))
        self.assertGreater(grasp_pose[2], frames.block_pose("red")[2]+.14/2*0+.038)
        binding = lift.held
        rotation = Rotation.from_quat(grasp_pose[3:])
        body_position = rotation.apply(binding["offset"])+grasp_pose[:3]
        body_rotation = rotation*Rotation.from_quat(binding["quaternion"])
        self.assertLess(max(abs(a-b) for a, b in zip(body_position,
                                                   frames.block_pose("red")[:3])), 1e-9)
        self.assertLess(body_rotation.magnitude(), 1e-9)

    def test_support_on_requires_force_ready_without_rigid_attachment(self):
        problem, engine, frames = setup_task("s3a_cantilever")
        requests = expand(engine.parse_action("support_on(B, beam)"), problem.initial, frames)
        self.assertEqual([request.kind for request in requests], ["transit", "linear", "skill"])
        self.assertEqual(requests[-1].action, "support_start")
        self.assertTrue(requests[-1].metadata["force_ready_required"])
        self.assertFalse(requests[-1].metadata["rigid_snap"])
        self.assertIn(["B_SupportShoe", "beam_block"], requests[-1].contacts)
        bottom = frames.slot_pose("beam")[2]-.02
        self.assertAlmostEqual(requests[1].target_pose[2]+SUPPORT_PAD_TOP_OFFSET, bottom)
        self.assertAlmostEqual(requests[0].target_pose[2], requests[1].target_pose[2])

    def test_support_off_ends_before_retreat_and_keeps_empty_hands(self):
        problem, engine, frames = setup_task("s3a_cantilever")
        state = {Fact("supporting", ("B", "beam"))}
        requests = expand(engine.parse_action("support_off(B, beam)"), state, frames)
        self.assertEqual(requests[0].action, "support_end")
        self.assertEqual(requests[1].dependencies, [requests[0].id])
        self.assertAlmostEqual(requests[1].target_pose[2], frames.support_pose("beam", "B")[2])
        self.assertLess(requests[2].target_pose[2], requests[1].target_pose[2])
        self.assertEqual(requests[2].dependencies, [requests[1].id])
        self.assertFalse(requests[-1].payloads)

    def _co_action(self, text):
        data = deepcopy(read_task(ROOT / "tasks/s2_tower.toml"))
        data["block"][0]["two_arms"] = True
        problem = problem_from_data(data)
        return Engine(problem).parse_action(text), problem.initial, Frames(data)

    def test_co_pick_is_explicit_invalid(self):
        action, state, frames = self._co_action("co_pick(A, B, b1, stock_l1)")
        with self.assertRaisesRegex(PrimitiveError, "^共持尚未接入运动后端$") as caught:
            expand(action, state, frames)
        self.assertEqual(caught.exception.status, Status.INVALID)

    def test_co_insert_is_explicit_invalid(self):
        action, state, frames = self._co_action("co_insert(A, B, b1, t1)")
        with self.assertRaisesRegex(PrimitiveError, "^共持尚未接入运动后端$") as caught:
            expand(action, state, frames)
        self.assertEqual(caught.exception.status, Status.INVALID)

    def test_override_geometry_survives_stale_symbolic_parallel_snapshot(self):
        problem, engine, frames = setup_task("s1_parallel_place")
        action = engine.parse_action("pick(B, blue, stock_r1)")
        moved_red = frames.block_pose("red", "board_l1")
        placed, payloads = world_state(problem.initial, frames)
        placed["red"] = moved_red
        requests = expand(action, problem.initial, frames, placed=placed, payloads=payloads)
        self.assertTrue(all(request.placed["red"] == moved_red for request in requests))
        self.assertEqual(placed["red"], moved_red)


if __name__ == "__main__":
    unittest.main()
