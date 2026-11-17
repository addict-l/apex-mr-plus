"""Workcell physical mapping and output tests without a simulator process."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skillnet.domains.blocks import load_problem
from simbridge.catalog import BlockCatalog, load_catalog
from simbridge.frames import BOARD_TOP, Frames, task_to_world, world_to_task
from simbridge.workcell import build
from simbridge.workcell import _collision_free_homes


ROOT = Path(__file__).resolve().parents[1]


class FakeSim:
    handle_all = -1
    handle_scene = -2
    primitiveshape_cuboid = 0
    objintparam_visibility_layer = 1
    shapeintparam_static = 2
    shapeintparam_respondable = 3
    colorcomponent_ambient_diffuse = 4
    objectspecialproperty_renderable = 1
    objectspecialproperty_collidable = 2
    objectspecialproperty_measurable = 4
    arrayparam_background_color1 = 10
    arrayparam_background_color2 = 11
    arrayparam_ambient_light = 12
    intparam_dynamic_engine = 20
    physics_bullet = 0
    floatparam_simulation_time_step = 30
    boolparam_realtime_simulation = 40

    def __init__(self):
        self.aliases, self.poses, self.parents = {}, {}, {}
        self.next_handle = 100

    def _create(self, *args):
        self.next_handle += 1
        self.poses[self.next_handle] = [0., 0., 0., 0., 0., 0., 1.]
        return self.next_handle

    createDummy = createPrimitiveShape = createVisionSensor = createForceSensor = _create

    def setObjectAlias(self, handle, alias):
        self.aliases[handle] = alias

    def getObjectAlias(self, handle):
        return self.aliases[handle]

    def setObjectParent(self, handle, parent, keep):
        self.parents[handle] = parent

    def setObjectPosition(self, handle, relative, position):
        self.poses[handle][:3] = position

    def setObjectPose(self, handle, pose, relative):
        self.poses[handle] = list(pose)

    def getObjectPose(self, handle, relative):
        return list(self.poses[handle])

    def getObjectsInTree(self, root, *args):
        return list(self.aliases)

    def saveScene(self, path):
        Path(path).write_bytes(b"FAKE_SCENE")

    def __getattr__(self, name):
        if name.startswith(("set", "write", "close", "reset")):
            return lambda *args: None
        raise AttributeError(name)


class CatalogAndFrameTests(unittest.TestCase):
    def test_catalog_covers_every_existing_task_type(self):
        catalog = load_catalog()
        for task in (ROOT / "tasks").glob("*.toml"):
            with self.subTest(task=task.stem):
                frames = Frames(task, catalog)
                self.assertTrue(frames.blocks)

    def test_missing_type_fails_instead_of_guessing_dimensions(self):
        with self.assertRaisesRegex(ValueError, "缺少积木类型"):
            Frames(ROOT / "tasks/s1_parallel_place.toml", BlockCatalog({}))

    def test_reference_coordinates_and_block_half_height(self):
        frames = Frames(ROOT / "tasks/s1_parallel_place.toml")
        self.assertEqual(frames.references["stock_l1"], [-.4, -.2, .772])
        self.assertAlmostEqual(frames.block_pose("red")[2], BOARD_TOP + .038)
        self.assertEqual(task_to_world([.1, .2, .3]), [.1, .2, 1.072])
        for got, wanted in zip(world_to_task(task_to_world([.1, .2, .3])), [.1, .2, .3]):
            self.assertAlmostEqual(got, wanted)

    def test_slot_heights_follow_below_and_physical_dimensions(self):
        frames = Frames(ROOT / "tasks/s2_tower.toml")
        self.assertAlmostEqual(frames.slot_pose("t1")[2], .772+.038)
        self.assertAlmostEqual(frames.slot_pose("t2")[2], .772+.076+.038)
        self.assertAlmostEqual(frames.slot_pose("t3")[2], .772+.152+.038)
        self.assertAlmostEqual(frames.references["t1"][2], .802)

    def test_loaded_problem_keeps_physical_task_rows(self):
        problem = load_problem(ROOT / "tasks/s1_parallel_place.toml",
                               ROOT / "skills/blocks_v1.toml")
        self.assertEqual(Frames(problem).block_pose("red"),
                         Frames(ROOT / "tasks/s1_parallel_place.toml").block_pose("red"))

    def test_grasp_and_approach_pose_are_above_body(self):
        frames = Frames(ROOT / "tasks/s1_parallel_place.toml")
        grasp = frames.grasp_pose("red")
        self.assertAlmostEqual(grasp[2], .772+.076+.006)
        self.assertAlmostEqual(frames.approach_pose(grasp)[2]-grasp[2], .10)


class WorkcellOutputTests(unittest.TestCase):
    def test_home_branch_selection_rejects_table_collision(self):
        sim = FakeSim()
        cfg = {"arms": {side: {"joints": list(range(6))} for side in ("A", "B")},
               "free_q": {}, "initial": {}}
        candidate = lambda value: {"q": [value]*6, "error_m": .0001, "error_rad": .00001}
        candidates = {"A": [candidate(0), candidate(1)], "B": [candidate(2)]}
        with patch("simbridge.workcell.MotionPlanner") as factory:
            planner = factory.return_value.__enter__.return_value
            planner.validate_pair.side_effect = [
                {"ok": False, "problem": {"reason": "table_collision:A"}}, {"ok": True}]
            evidence = _collision_free_homes(sim, None, cfg, candidates)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(cfg["arms"]["A"]["home_q"], [1]*6)
        self.assertEqual(cfg["free_q"]["B"], [2]*6)

    def test_no_clear_home_branch_fails_without_collision_exemption(self):
        candidate = {"q": [0]*6, "error_m": 0., "error_rad": 0.}
        cfg = {"arms": {}, "free_q": {}, "initial": {}}
        with patch("simbridge.workcell.MotionPlanner") as factory:
            planner = factory.return_value.__enter__.return_value
            planner.validate_pair.return_value = {
                "ok": False, "problem": {"reason": "table_collision:A"}}
            with self.assertRaisesRegex(RuntimeError, "均存在碰撞"):
                _collision_free_homes(FakeSim(), None, cfg,
                                      {"A": [candidate], "B": [candidate]})
            self.assertNotIn("contact_exclusions", planner.validate_pair.call_args.kwargs)

    def test_save_scene_and_json_poses_match_task_with_real_body_height(self):
        (ROOT / "out").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as directory:
            sim = FakeSim()

            def fake_arms(sim, install, root, shape, homes):
                return {side: {"home_q": [0]*6, "model": root,
                               "joints": list(range(6)), "tip": root}
                        for side in ("A", "B")}

            with (patch("simbridge.workcell.build_arms", side_effect=fake_arms),
                  patch("simbridge.workcell.configure_support_tools") as support_tools,
                  patch("simbridge.workcell._home_joints"),
                  patch("simbridge.workcell._collision_free_homes", return_value=[]),
                  patch("simbridge.workcell.make_label", return_value=-1)):
                cfg = build(sim, None, ROOT / "tasks/s1_parallel_place.toml", ROOT, directory)
            support_tools.assert_not_called()
            self.assertEqual(set(cfg["blocks"]), {"red", "blue"})
            self.assertTrue((Path(directory)/"workcell.ttt").is_file())
            report = json.loads((Path(directory)/"workcell.json").read_text(encoding="utf-8"))
            self.assertEqual(report["coordinate_mapping"]["task_reference_world"]["stock_r1"],
                             [.4, -.2, .772])
            self.assertEqual(report["rigid_bodies"]["red"]["world_pose"][:2], [-.4, -.2])
            self.assertAlmostEqual(report["rigid_bodies"]["red"]["world_pose"][2], .810)
            self.assertGreater(report["rigid_bodies"]["red"]["mass"], 0)
            self.assertTrue(report["rigid_bodies"]["red"]["dynamic"])

    def test_output_outside_project_is_rejected_before_scene_mutation(self):
        with self.assertRaisesRegex(ValueError, "项目根目录"):
            build(None, None, ROOT / "tasks/s1_parallel_place.toml", ROOT, ROOT.parent / "forbidden")


if __name__ == "__main__":
    unittest.main()
