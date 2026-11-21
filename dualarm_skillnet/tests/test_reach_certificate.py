"""Measured-chain reach proofs and public compiler behavior, without a sim."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from simbridge.compiler import CompileFailure
from simbridge.reach_certificate import (MARGIN, measure_reach_certificate,
                                         outer_sphere_rejection, verified_sphere)
from tests.test_v13_compiler import fixture


ROOT = Path(__file__).resolve().parents[1]


class ChainModel:
    """Read-only simulator double with the actual parent/type/pose API."""
    object_joint_type = 1
    object_shape_type = 0
    shapeintparam_static = 2
    joint_revolute = 10

    def __init__(self, positions=None, *, origin=(0., 0., 0.), radius=.7):
        self.model, self.tip, self.joints = 10, 30, list(range(20, 26))
        self.parents, self.types, self.aliases, self.poses = {}, {}, {}, {}
        self.tree_joints, self.joint_types = list(self.joints), {joint: self.joint_revolute for joint in self.joints}
        self.q, self.static = {joint: 0. for joint in self.joints}, 1
        self.parents[99], self.types[99], self.aliases[99] = -1, self.object_shape_type, "scene_root"
        points = positions or [[origin[0]+radius*index/7, origin[1], origin[2]] for index in range(8)]
        handles = [self.model, *self.joints, self.tip]
        for index, (handle, position) in enumerate(zip(handles, points)):
            self.types[handle] = self.object_joint_type if handle in self.joints else self.object_shape_type
            self.aliases[handle] = "object_" + str(handle)
            self.poses[handle] = list(position)+[0., 0., 0., 1.]
            if index == 0:
                self.parents[handle] = 99
            else:
                intermediate = 40 + index
                self.types[intermediate] = self.object_shape_type
                self.aliases[intermediate] = "rigid_link_" + str(index)
                self.parents[intermediate] = handles[index-1]
                self.parents[handle] = intermediate
        self.arm = {"model": self.model, "tip": self.tip, "joints": self.joints,
                    "base_pose": self.poses[self.model]}

    def getObjectType(self, handle):
        return self.types[handle]

    def getObjectParent(self, handle):
        return self.parents[handle]

    def getObjectAlias(self, handle):
        return self.aliases[handle]

    def getObjectsInTree(self, model, kind, options):
        return self.tree_joints

    def getJointType(self, handle):
        return self.joint_types[handle]

    def getJointPosition(self, handle):
        return self.q[handle]

    def getObjectInt32Param(self, handle, parameter):
        return self.static

    def getObjectPose(self, handle, relative):
        return list(self.poses[handle])


def attach_certificate(compiler, side="A", radius=.2):
    arm = compiler.cfg["arms"][side]
    origin = arm["base_pose"][:3]
    model = ChainModel(origin=origin, radius=radius)
    arm.update(deepcopy(model.arm))
    arm["reach_certificate"] = measure_reach_certificate(model, arm)
    assert arm["reach_certificate"]["certified"], arm["reach_certificate"]
    return arm["reach_certificate"]


class MeasuredChainTests(unittest.TestCase):
    def test_literal_orthogonal_chain_uses_all_seven_segment_norms(self):
        positions = [[0, 0, 0], [0, 0, .1], [.2, 0, .1], [.2, .3, .1],
                     [.2, .3, .5], [.7, .3, .5], [.7, .9, .5], [.7, .9, 1.2]]
        model = ChainModel(positions)
        proof = measure_reach_certificate(model, model.arm)
        self.assertTrue(proof["certified"])
        self.assertAlmostEqual(proof["radius"], .1+.2+.3+.4+.5+.6+.7)
        self.assertEqual(proof["origin"], [0, 0, 0])
        self.assertEqual(len(proof["segments"]), 7)
        self.assertEqual([node["handle"] for node in proof["actual_topology"] if node["is_joint"]], model.joints)
        self.assertGreater(len(proof["actual_topology"]), 8)
        self.assertEqual(proof["margin"], .01)
        self.assertIsNotNone(verified_sphere(model.arm, proof))

    def test_literal_diagonal_segment_is_euclidean_length_not_component_sum(self):
        positions = [[0, 0, 0], [3, 4, 0], [3, 4, .1], [3.2, 4, .1],
                     [3.2, 4.3, .1], [3.2, 4.3, .5], [3.7, 4.3, .5], [3.7, 4.9, .5]]
        model = ChainModel(positions)
        proof = measure_reach_certificate(model, model.arm)
        self.assertTrue(proof["certified"])
        self.assertEqual(proof["segments"][0]["length"], 5)
        self.assertAlmostEqual(proof["radius"], 5+.1+.2+.3+.4+.5+.6)

    def test_prismatic_and_spherical_joint_types_are_never_certified(self):
        for joint_type in (11, 12):
            with self.subTest(joint_type=joint_type):
                model = ChainModel()
                model.joint_types[model.joints[2]] = joint_type
                proof = measure_reach_certificate(model, model.arm)
                self.assertFalse(proof["certified"])
                self.assertIsNone(proof["radius"])
                self.assertIn("非转动", proof["reason"])

    def test_extra_or_missing_joint_in_actual_tree_refuses_guess(self):
        for tree in ([20, 21, 22, 23, 24], [20, 21, 22, 23, 24, 25, 999]):
            model = ChainModel()
            model.tree_joints = tree
            proof = measure_reach_certificate(model, model.arm)
            self.assertFalse(proof["certified"])
            self.assertIsNone(proof["radius"])

    def test_six_tree_joints_but_missing_ancestor_is_uncertified(self):
        model = ChainModel()
        model.parents[23] = 21
        proof = measure_reach_certificate(model, model.arm)
        self.assertFalse(proof["certified"])
        self.assertIn("拓扑顺序", proof["reason"])

    def test_extra_active_ancestor_is_uncertified_even_if_arm_list_has_six(self):
        model = ChainModel()
        model.types[47] = model.object_joint_type
        model.joint_types[47] = model.joint_revolute
        proof = measure_reach_certificate(model, model.arm)
        self.assertFalse(proof["certified"])
        self.assertIn("拓扑顺序", proof["reason"])

    def test_cycle_disconnected_chain_nonfinite_pose_and_unfixed_root_refuse_guess(self):
        for defect in ("cycle", "disconnected", "nonfinite", "dynamic_root"):
            with self.subTest(defect=defect):
                model = ChainModel()
                if defect == "cycle":
                    model.parents[20] = model.tip
                elif defect == "disconnected":
                    model.parents[20] = -1
                elif defect == "nonfinite":
                    model.poses[22][0] = float("nan")
                else:
                    model.static = 0
                proof = measure_reach_certificate(model, model.arm)
                self.assertFalse(proof["certified"])
                self.assertIsNone(proof["radius"])

    def test_unknown_api_returns_uncertified_without_catalog_radius(self):
        proof = measure_reach_certificate(object(), {"model": 1, "tip": 2, "joints": list(range(3, 9))})
        self.assertFalse(proof["certified"])
        self.assertIsNone(proof["radius"])
        self.assertIn("AttributeError", proof["reason"])

    def test_malformed_arm_input_is_uncertified_without_crashing_or_guessing(self):
        for arm in (None, {}, {"model": 1, "tip": 2, "joints": None},
                    {"model": 1, "tip": 2, "joints": [3, 3, 4, 5, 6, 7]}):
            with self.subTest(arm=arm):
                proof = measure_reach_certificate(object(), arm)
                self.assertFalse(proof["certified"])
                self.assertIsNone(proof["radius"])

    def test_changing_joint_snapshot_is_not_a_certificate(self):
        model = ChainModel()
        calls = 0
        original = model.getJointPosition
        def moving(handle):
            nonlocal calls
            calls += 1
            return original(handle) + (.1 if calls > 6 else 0.)
        model.getJointPosition = moving
        proof = measure_reach_certificate(model, model.arm)
        self.assertFalse(proof["certified"])
        self.assertIn("状态发生变化", proof["reason"])

    def test_active_joint_above_model_root_invalidates_fixed_base_premise(self):
        model = ChainModel()
        model.types[99] = model.object_joint_type
        proof = measure_reach_certificate(model, model.arm)
        self.assertFalse(proof["certified"])
        self.assertIn("固定基座", proof["reason"])

    def test_boundary_and_inside_are_not_proofs_but_outside_preserves_full_evidence(self):
        model = ChainModel()
        proof = measure_reach_certificate(model, model.arm)
        model.arm["reach_certificate"] = proof
        bound = proof["radius"] + MARGIN
        self.assertIsNone(outer_sphere_rejection(model.arm, [bound, 0, 0, 0, 0, 0, 1]))
        self.assertIsNone(outer_sphere_rejection(model.arm, [bound-.001, 0, 0, 0, 0, 0, 1]))
        failure = outer_sphere_rejection(model.arm, [bound+.001, 0, 0, 0, 0, 0, 1])
        self.assertTrue(failure["proven_infeasible"])
        self.assertEqual(failure["reach_certificate"], proof)
        self.assertEqual(failure["status"], "INVALID")

    def test_tampered_length_missing_evidence_or_moved_origin_is_not_trusted(self):
        model = ChainModel()
        for defect in ("radius", "segments", "topology", "origin", "handles"):
            proof = measure_reach_certificate(model, model.arm)
            if defect == "radius":
                proof["radius"] = .01
            elif defect == "segments":
                proof["segments"][0]["length"] = .01
            elif defect == "topology":
                proof["actual_topology"][1]["parent"] = 999
            elif defect == "origin":
                proof["origin"][0] += .1
            else:
                proof["joint_handles"] = list(reversed(model.joints))
            model.arm["reach_certificate"] = proof
            with self.subTest(defect=defect):
                self.assertIsNone(outer_sphere_rejection(model.arm, [20, 0, 0, 0, 0, 0, 1]))


class CompilerReachCertificateTests(unittest.TestCase):
    def test_compile_action_rejects_proven_outer_sphere_before_any_ik(self):
        engine, geometry, compiler = fixture()
        certificate = attach_certificate(compiler)
        geometry.ik_failure = {"ok": False, "error": "finite seeds exhausted"}
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(caught.exception.status, Status.INVALID)
        self.assertTrue(caught.exception.evidence["proven_infeasible"])
        self.assertEqual(caught.exception.evidence["reach_certificate"], certificate)
        self.assertFalse(geometry.calls)
        self.assertFalse(compiler.tasks)
        self.assertFalse(compiler.held)
        self.assertEqual(compiler.evidence[-1]["query"], "reach_certificate")

    def test_new_proof_guard_runs_before_preexisting_successful_path_cache(self):
        engine, geometry, compiler = fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        compiler.compile_action(action, engine.initial)
        self.assertTrue(compiler.path_cache)
        compiler.reset()
        calls = len(geometry.calls)
        attach_certificate(compiler)
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(action, engine.initial)
        self.assertEqual(caught.exception.status, Status.INVALID)
        self.assertEqual(len(geometry.calls), calls)

    def test_uncertified_or_absent_proof_preserves_finite_ik_unknown(self):
        for certificate in (None, {"certified": False, "reason": "no topology"},
                            {"certified": True, "radius": .1, "origin": [0, 0, 0]}):
            with self.subTest(certificate=certificate):
                engine, geometry, compiler = fixture()
                compiler.cfg["arms"]["A"]["reach_certificate"] = certificate
                geometry.ik_failure = {"ok": False, "error": "finite seeds exhausted"}
                with self.assertRaises(CompileFailure) as caught:
                    compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
                self.assertEqual(caught.exception.status, Status.UNKNOWN)
                self.assertEqual(sum(call[0] == "ik" for call in geometry.calls), compiler.ik_seeds)

    def test_certified_inside_target_preserves_ordinary_ik_unknown(self):
        engine, geometry, compiler = fixture()
        attach_certificate(compiler, radius=2.)
        geometry.ik_failure = {"ok": False, "error": "finite seeds exhausted"}
        with self.assertRaises(CompileFailure) as caught:
            compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(sum(call[0] == "ik" for call in geometry.calls), compiler.ik_seeds)

    def test_target_exactly_on_margin_boundary_retains_ordinary_ik_unknown(self):
        engine, geometry, compiler = fixture()
        model = ChainModel()
        arm = compiler.cfg["arms"]["A"]
        arm.update(deepcopy(model.arm))
        arm["reach_certificate"] = measure_reach_certificate(model, arm)
        boundary = arm["reach_certificate"]["radius"] + MARGIN
        target = [boundary, 0, 0, 0, 0, 0, 1]
        geometry.ik_failure = {"ok": False, "error": "finite seeds exhausted"}
        with patch.object(compiler.frames, "approach_pose", return_value=target):
            with self.assertRaises(CompileFailure) as caught:
                compiler.compile_action(engine.parse_action("pick(A, red, stock_l1)"), engine.initial)
        self.assertEqual(caught.exception.status, Status.UNKNOWN)
        self.assertEqual(sum(call[0] == "ik" for call in geometry.calls), compiler.ik_seeds)


class WorkcellCertificateTests(unittest.TestCase):
    def test_certificate_is_measured_after_home_selection_and_saved_for_both_arms(self):
        from simbridge.workcell import build
        from tests.test_v13_workcell import FakeSim

        events, sim = [], FakeSim()
        def fake_arms(sim, install, root, shape, homes):
            return {side: {"home_q": [0.]*6, "model": root,
                           "joints": list(range(6)), "tip": root} for side in ("A", "B")}
        def measured(simulator, arm):
            events.append("measure")
            return {"certified": False, "radius": None, "reason": "mock has no verifiable topology"}
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as output:
            with (patch("simbridge.workcell.build_arms", side_effect=fake_arms),
                  patch("simbridge.workcell._home_joints"),
                  patch("simbridge.workcell._collision_free_homes", side_effect=lambda *args: events.append("home") or []),
                  patch("simbridge.workcell.measure_reach_certificate", side_effect=measured) as measure,
                  patch("simbridge.workcell.make_label", return_value=-1),
                  patch.object(sim, "writeCustomTableData", side_effect=lambda *args: events.append("persist"))):
                cfg = build(sim, None, ROOT / "tasks/s1_parallel_place.toml", ROOT, output)
            self.assertEqual(events, ["home", "measure", "measure", "persist"])
            self.assertEqual(measure.call_count, 2)
            report = json.loads((Path(output) / "workcell.json").read_text(encoding="utf-8"))
        for side in ("A", "B"):
            self.assertIn("reach_certificate", cfg["arms"][side])
            self.assertEqual(report["cfg"]["arms"][side]["reach_certificate"], cfg["arms"][side]["reach_certificate"])


if __name__ == "__main__":
    unittest.main()
