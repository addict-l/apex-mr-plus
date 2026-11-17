"""Generate a dynamic dual-UR5 workcell solely from a task and catalog."""
from dataclasses import asdict
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from apexmr.scene_helpers import build_arms, make_label, mass_inertia
from apexmr.motion import MotionPlanner
from .catalog import load_catalog
from .frames import (BOARD_TOP, TABLE_TOP, SUPPORT_SENSOR_RISE, SUPPORT_PAD_SIZE,
                     SUPPORT_SENSOR_PALM_Z, Frames, read_task)
from .reach_certificate import measure_reach_certificate


ROOT_ALIAS = "APEX_MR_Scene3"
ROOT_PATH = "/" + ROOT_ALIAS


def configure_support_tools(sim, arms):
    """Mirror horizontal shoe frames for both arms, leaving TCPs unchanged."""
    for side, arm in arms.items():
        quaternion = Rotation.from_euler("x", math.pi/2 if side == "A" else -math.pi/2).as_quat().tolist()
        pose = [0, -SUPPORT_SENSOR_RISE, SUPPORT_SENSOR_PALM_Z]+quaternion
        sim.setObjectPose(arm["support_sensor"], pose, arm["palm"])
        sim.setObjectPose(arm["support_pad"], [0, 0, 0, 0, 0, 0, 1], arm["support_sensor"])
        arm["support_sensor_pose_in_palm"] = pose
        arm["support_pad_size"] = list(SUPPORT_PAD_SIZE)


def _label_asset(path, text):
    image = Image.new("RGB", (800, 160), "#eff4f3")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = (ImageFont.truetype(str(font_path), 48)
            if font_path.exists() else ImageFont.load_default())
    ImageDraw.Draw(image).text((16, 40), text, font=font, fill="#174b52")
    image.save(path)


def _home_joints(sim, client, arms, frames):
    """Real IK for task-specified TCP home positions, with bounded seeds."""
    ik = client.require("simIK")
    seeds = [[0, -1.15, 1.7, -2.1, -1.57, 0],
             [-2.35, -.25, 1.57, .25, -1.57, -.78],
             [-1.37, 1.24, -1.57, 1.90, -1.57, .20],
             [math.pi, -1.15, 1.7, -2.1, -1.57, 0]]
    candidates = {}
    for side, arm in arms.items():
        environment = ik.createEnvironment()
        group = ik.createGroup(environment)
        try:
            ik.setGroupCalculation(environment, group, ik.method_damped_least_squares, .02, 200)
            ik.addElementFromScene(environment, group, arm["model"], arm["tip"],
                                   arm["target"], ik.constraint_pose)
            desired = frames.home_pose(side)
            sim.setObjectPose(arm["target"], desired, -1)
            candidates[side] = []
            for seed in seeds:
                for joint, q in zip(arm["joints"], seed):
                    sim.setJointPosition(joint, q)
                ik.handleGroup(environment, group, {"syncWorlds": True, "allowError": False})
                measured = sim.getObjectPose(arm["tip"], -1)
                error = math.dist(measured[:3], desired[:3])
                angle = (Rotation.from_quat(measured[3:]).inv()
                         * Rotation.from_quat(desired[3:])).magnitude()
                if error <= .001 and angle <= .02:
                    q = [sim.getJointPosition(joint) for joint in arm["joints"]]
                    candidates[side].append({"q": q, "error_m": error, "error_rad": float(angle)})
            if not candidates[side]:
                raise RuntimeError(f"臂 {side} 的初始 home 逆运动学失败：{error:.6f} m, {angle:.6f} rad")
            selected = candidates[side][0]
            arm["home_q"] = selected["q"]
            arm["home_pose"] = desired
            arm["home_error_m"] = selected["error_m"]
            arm["home_error_rad"] = selected["error_rad"]
            for joint, q in zip(arm["joints"], arm["home_q"]):
                sim.setJointPosition(joint, q)
                sim.setJointTargetPosition(joint, q)
        finally:
            ik.eraseEnvironment(environment)
    return candidates


def _collision_free_homes(sim, client, cfg, candidates):
    """Select a paired home branch using the complete collision geometry."""
    evidence = []
    with MotionPlanner(sim, client, cfg) as planner:
        for left in candidates["A"]:
            for right in candidates["B"]:
                answer = planner.validate_pair(left["q"], right["q"], placed=cfg["initial"])
                evidence.append({"A": left["q"], "B": right["q"], "answer": answer})
                if not answer.get("ok"):
                    continue
                for side, selected in (("A", left), ("B", right)):
                    arm = cfg["arms"][side]
                    arm["home_q"] = selected["q"]
                    arm["home_error_m"] = selected["error_m"]
                    arm["home_error_rad"] = selected["error_rad"]
                    cfg["free_q"][side] = selected["q"]
                    for joint, q in zip(arm["joints"], selected["q"]):
                        sim.setJointPosition(joint, q)
                        sim.setJointTargetPosition(joint, q)
                return evidence
    raise RuntimeError("初始 home 的有限 IK 分支均存在碰撞：" + json.dumps(evidence, ensure_ascii=False))


def _rigid_attachment(sim, parent, body, alias, anchor):
    sensor = sim.createForceSensor(0, [0, 1, 5, 0, 0], [.006, 500, 10., 0, 0])
    sim.setObjectAlias(sensor, alias)
    sim.setObjectParent(sensor, parent, True)
    sim.setObjectPose(sensor, list(anchor), -1)
    sim.setObjectInt32Param(sensor, sim.objintparam_visibility_layer, 0)
    sim.setObjectParent(body, sensor, True)
    sim.resetDynamicObject(body)
    return sensor


def build(sim, client, task, install, outdir, catalog=None):
    """Return the planner/runtime cfg and save workcell.ttt/workcell.json.

    ``task`` accepts a task TOML path, raw task dict, or loaded Problem.
    A fresh scene is built from installed UR5 models. Archive scenes are never
    loaded, and all generated files are constrained to this project's root.
    """
    project = Path(__file__).resolve().parents[1]
    output = Path(outdir).resolve()
    if not output.is_relative_to(project):
        raise ValueError("工作站输出必须位于项目根目录内")
    output.mkdir(parents=True, exist_ok=True)
    data = read_task(task)
    catalog = catalog or load_catalog()
    frames = Frames(data, catalog)
    sim.closeScene()
    root = sim.createDummy(.001)
    sim.setObjectAlias(root, ROOT_ALIAS)
    sim.setObjectInt32Param(root, sim.objintparam_visibility_layer, 0)
    sim.setArrayParam(sim.arrayparam_background_color1, [.90, .94, .96])
    sim.setArrayParam(sim.arrayparam_background_color2, [.97, .98, .99])
    sim.setArrayParam(sim.arrayparam_ambient_light, [.45, .45, .45])
    objects = []

    def shape(name, size, pos, color, static=True, parent=None, respondable=True):
        parent = root if parent is None else parent
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(size), 3)
        sim.setObjectAlias(handle, name)
        sim.setObjectParent(handle, parent, True)
        sim.setObjectPosition(handle, -1, list(pos))
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, list(color))
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, int(static))
        sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, int(respondable))
        sim.setObjectSpecialProperty(handle, sim.objectspecialproperty_renderable
                                     | sim.objectspecialproperty_collidable
                                     | sim.objectspecialproperty_measurable)
        objects.append({"alias": name, "handle": handle, "size": list(size),
                        "static": static, "respondable": respondable})
        return handle

    table = shape("Workbench", [1.90, 1.27, .065], [0, 0, TABLE_TOP-.0325], [.70, .77, .80])
    baseplate = shape("Baseplate", [.70, .72, .012], [0, .12, BOARD_TOP-.006], [.25, .33, .38])
    obstacles = [baseplate]
    surfaces = [table, baseplate]
    pads = {}
    labels = {}
    for name, row in frames.locations.items():
        if row["kind"] not in {"stock", "buffer", "handover"}:
            continue
        x, y, z = frames.references[name]
        pad = shape("Pad_"+name, [.18, .18, .012], [x, y, z-.006], [.23, .62, .63],
                    respondable=row["kind"] != "handover")
        if row["kind"] != "handover":
            obstacles.append(pad)
            surfaces.append(pad)
        else:
            sim.setObjectSpecialProperty(pad, sim.objectspecialproperty_renderable)
        pads[name] = pad
        asset = output / ("label_"+name+".png")
        _label_asset(asset, name)
        labels[name] = make_label(sim, root, "Label_"+name, asset,
                                 [x, y-.07, z+.0003], [.16, .032])
    for side, x in (("A", -.70), ("B", .70)):
        shape(side+"_Mount", [.23, .24, .022], [x, .13, .771], [.20, .25, .30])

    blocks, parts, initial, sizes, attachments = {}, [], {}, {}, {}
    for name, row in frames.blocks.items():
        spec = catalog[row["type"]]
        body_pose = frames.block_pose(name)
        # Semantic colour names in the original flat task are physically visible.
        color = {"red": [.80, .12, .08], "blue": [.035, .40, .85]}.get(name, spec.color)
        body = shape(name, spec.size, body_pose[:3], color, False)
        sim.setObjectPose(body, body_pose, -1)
        mass_inertia(sim, body, spec.size, spec.mass)
        blocks[name], initial[name], sizes[name] = body, body_pose, list(spec.size)
        location = row.get("installed_in") or row.get("at")
        parts.append({"id": name, "alias": name, "type": row["type"], "size": list(spec.size),
                      "mass": spec.mass, "source": body_pose[:3], "dependencies": [],
                      "initial_place": location,
                      "support_contact_surfaces": (["Pad_"+location] if location in pads
                                                   else ["Baseplate"])})
    # Initial rigid connections are created only after all lower bodies exist.
    installed_by_slot = {row["installed_in"]: name for name, row in frames.blocks.items()
                         if row.get("installed_in")}
    for slot, name in installed_by_slot.items():
        lower = frames.slots[slot].get("below", [])
        parent = blocks[installed_by_slot[lower[0]]] if lower else baseplate
        anchor = list(initial[name])
        anchor[2] -= sizes[name][2] / 2
        attachments[name] = _rigid_attachment(sim, parent, blocks[name], "InitialSnap_"+name, anchor)

    homes = {side+"_home": [0, -1.15, 1.7, -2.1, -1.57, 0] for side in ("A", "B")}
    arms = build_arms(sim, Path(install), root, shape, homes)
    # A horizontal shoe protrudes into a top-grasped brick. Keep the original
    # tool assembly for tasks with no manual-support affordance; support tasks
    # still configure both mirrored shoes as specified by the handoff.
    horizontal_support = any(row.get("support_point") for row in frames.slots.values())
    if horizontal_support:
        configure_support_tools(sim, arms)
    # Exact shoe/beam contact pairs must also obey the backend's bearing-face
    # tolerance; registering the horizontal shoes prevents deep penetration
    # from being accepted as a generic tool contact.
    surfaces.extend(arm["support_pad"] for arm in arms.values() if "support_pad" in arm)
    for arm in arms.values():
        arm["base_pose"] = sim.getObjectPose(arm["model"], -1)
    home_candidates = _home_joints(sim, client, arms, frames)
    sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_bullet)
    sim.setFloatProperty(sim.handle_scene, "dynamicsStepSize", .002)
    sim.setIntProperty(sim.handle_scene, "bullet.iterations", 150)
    sim.setVector3Property(sim.handle_scene, "gravity", [0, 0, -9.81])
    sim.setFloatParam(sim.floatparam_simulation_time_step, .02)
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    for handle in [table, *obstacles, *blocks.values()]:
        sim.setFloatProperty(handle, "bullet.friction", .8)
        sim.setFloatProperty(handle, "bullet.frictionOld", .8)
        sim.setFloatProperty(handle, "bullet.restitution", 0.)
        sim.setFloatProperty(handle, "bullet.linearDamping", .04)
        sim.setFloatProperty(handle, "bullet.angularDamping", .04)

    eye, target = [1.40, -2.95, 2.55], [0, .02, 1.01]
    import numpy as np
    z_axis = np.array(target)-eye
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0., 0., 1.], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    camera_pose = eye + Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat().tolist()
    camera = sim.createVisionSensor(1+2+4+128, [1280, 880, 0, 0],
                                   [.025, 10, math.radians(48), .01, 0, 0, 0, .91, .94, .97, 0, 0])
    sim.setObjectAlias(camera, "OverviewRGB")
    sim.setObjectParent(camera, root, True)
    sim.setObjectPose(camera, camera_pose, -1)
    sim.setObjectInt32Param(camera, sim.objintparam_visibility_layer, 0)
    cfg = {"root": root, "root_path": ROOT_PATH, "task_name": data["name"],
           "support_tool_configuration": "horizontal" if horizontal_support else "catalog_grasp",
           "blocks": blocks, "block_shapes": {name: [handle] for name, handle in blocks.items()},
           "parts": parts, "arms": arms, "table": table, "baseplate": baseplate,
           "camera": camera, "obstacles": obstacles,
           "obstacle_aliases": [sim.getObjectAlias(handle) for handle in obstacles],
           "support_surface_handles": surfaces, "initial": initial, "sizes": sizes,
           "support_contact_surfaces": {
               part["id"]: list(dict.fromkeys(part["support_contact_surfaces"]+["Baseplate"]))
               for part in parts},
           "initial_attachments": attachments, "initial_installed": installed_by_slot,
           "articulations": {}, "initial_articulations": {}, "initial_support_contacts": {},
           "pads": pads, "labels": labels, "dt": .02, "program": {},
           "free_q": {side: arm["home_q"] for side, arm in arms.items()},
           "runtime": {"pose_tolerance_m": .005, "orientation_tolerance_rad": .02},
           "physics": {"engine": "Bullet", "gravity": [0, 0, -9.81],
                       "grasp": "ideal dynamic force-sensor constraint",
                       "connections": "ideal rigid snap at measured pose",
                       "platform_contact": "contact only; no rigid beam attachment"}}
    home_evidence = _collision_free_homes(sim, client, cfg, home_candidates)
    for arm in arms.values():
        arm["reach_certificate"] = measure_reach_certificate(sim, arm)
    sim.writeCustomTableData(root, "Scene3APEX", cfg)
    sim.saveScene(str(output / "workcell.ttt"))
    all_handles = sim.getObjectsInTree(root, sim.handle_all, 0)
    report = {"task": data["name"], "scene": str(output / "workcell.ttt"),
              "coordinate_mapping": {"board_top_m": BOARD_TOP, "table_top_m": TABLE_TOP,
                 "task_reference_world": frames.references,
                 "slot_body_world": {name: frames.slot_pose(name) for name in frames.slots},
                 "rule": "任务位置是支承参考面；积木中心上移半高；槽位按 below 与目录层高堆叠，保留原始参考点"},
              "objects": [{"alias": sim.getObjectAlias(handle), "handle": handle,
                           "world_pose": sim.getObjectPose(handle, -1)} for handle in all_handles],
              "rigid_bodies": {name: {**asdict(catalog[frames.blocks[name]["type"]]),
                                       "handle": body, "world_pose": initial[name],
                                       "dynamic": True, "initial_attachment": attachments.get(name)}
                               for name, body in blocks.items()},
              "arms": arms, "home_collision_checks": home_evidence,
              "physics": cfg["physics"], "cfg": cfg}
    (output / "workcell.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                          encoding="utf-8")
    return cfg
