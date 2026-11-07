"""Compile ILP-selected grasp/support modes into genuine joint-path tasks.

Assembly order is supplied as a valid part order. The ILP decides installer,
grasp, and support mode from geometry-qualified candidates; it does not invent
the assembly. All transit/contact motion polylines come from simOMPL's actual
RRTConnect planner. A later TPG may relax the global serial reference order.
"""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .allocation import solve_allocation, support_requirement
from .motion import MotionPlanner


MAX_VELOCITY = .60
MAX_ACCELERATION = 1.20
APPROACH = .10
LIFT = .13


def support_contact_geometry(tcp_position, quaternion, supported_part, pad_size, minimum_overlap=.002):
    """Qualify upward support by projected OBB contact, not absence of collision.

    TCP is the pad's local +Z top-face centre in this tool model. A finite XY
    contact patch and a touching/preloaded vertical plane are both required.
    The intersection is clipped as convex polygons, including rotated pads.
    """
    rotation = Rotation.from_quat(quaternion)
    normal = rotation.apply([0,0,1])
    beam_rotation = Rotation.from_quat(supported_part.get("goal_quaternion", [0,0,0,1]))
    beam_normal = beam_rotation.apply([0,0,1])
    top = np.asarray(tcp_position, dtype=float)
    goal = np.asarray(supported_part["goal"], dtype=float)
    bx, by, bz = supported_part["size"]
    sx, sy, _ = pad_size
    pad = [(top+rotation.apply([x*sx/2,y*sy/2,0]))[:2].tolist()
           for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    underside = [goal+beam_rotation.apply([x*bx/2,y*by/2,-bz/2])
                 for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    clip = [p[:2].tolist() for p in underside]
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    polygon = pad
    for a,b in zip(clip,clip[1:]+clip[:1]):
        out=[]
        edge=np.asarray(b)-a
        for p,q in zip(polygon,polygon[1:]+polygon[:1]):
            p,q=np.asarray(p),np.asarray(q)
            ip=cross(edge,p-a)>=-1e-12; iq=cross(edge,q-a)>=-1e-12
            if ip:out.append(p.tolist())
            if ip != iq:
                direction=q-p;denom=cross(edge,direction)
                if abs(denom)>1e-15:
                    out.append((p+direction*(-cross(edge,p-a)/denom)).tolist())
        polygon=out
        if not polygon:break
    if polygon:
        overlap=np.ptp(np.asarray(polygon),axis=0).tolist()
        area=abs(sum(cross(a,b) for a,b in zip(polygon,polygon[1:]+polygon[:1])))/2
    else:overlap=[0.,0.];area=0.
    gap=float(np.mean([p[2] for p in underside])-top[2])
    aligned=bool(normal[2]>.99 and beam_normal[2]>.99)
    qualified=bool(aligned and min(overlap)>=minimum_overlap-1e-9
                   and area>=minimum_overlap**2-1e-12 and -.003<=gap<=.0001)
    return {"qualified":qualified,"projected_overlap_m":overlap,"overlap_area_m2":area,
            "vertical_gap_m":gap,"pad_normal":normal.tolist(),
            "minimum_overlap_m":minimum_overlap,"polygon_xy":polygon}


def _save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))


def trajectory_duration(path, dwell=0.):
    """Conservative rest-to-rest bound for the unchanged joint polyline.

    Each original segment is treated as a rest-to-rest trapezoidal move. This
    avoids a velocity/acceleration claim based only on total Euclidean length.
    The runtime must still time-parameterize each segment consistently.
    """
    total = 0.
    for a, b in zip(path, path[1:]):
        delta = max(abs(x-y) for x, y in zip(a, b))
        if delta <= MAX_VELOCITY**2/MAX_ACCELERATION:
            total += 2*math.sqrt(delta/MAX_ACCELERATION)
        else:
            total += delta/MAX_VELOCITY + MAX_VELOCITY/MAX_ACCELERATION
    return max(.08, total + dwell)


def _payload(part, grasp, arm):
    quat = grasp["quaternion_by_arm"][arm]
    rotation = Rotation.from_quat(quat)
    world_offset = np.asarray(grasp["tcp_offset"], dtype=float)
    return {
        "part": part["id"],
        "offset": rotation.inv().apply(-world_offset).tolist(),
        "quaternion": rotation.inv().as_quat().tolist(),
    }


def _position(part, grasp, location, dz=0.):
    p = np.asarray(part[location], dtype=float) + np.asarray(grasp["tcp_offset"], dtype=float)
    p[2] += dz
    return p.tolist()


class _CompileFailure(RuntimeError):
    def __init__(self, choice, message, evidence=None):
        super().__init__(message)
        self.choice = choice
        self.evidence = evidence or {}


class _Compiler:
    def __init__(self, sim, planner, cfg):
        self.sim, self.planner, self.cfg = sim, planner, cfg
        self.home = {a: list(cfg["arms"][a]["home_q"]) for a in ["A", "B"]}
        self.parts = {p["id"]: p for p in cfg["parts"]}
        self.calls = []
        self.qualification = []
        self.path_cache = {}
        root = Path(__file__).resolve().parents[1]
        self.path_journal = root/"results"/"normal"/"rrt_paths"
        self.path_journal.mkdir(parents=True, exist_ok=True)
        scene = root/"scenes"/"10_Normal_Assembly.ttt"
        scene_hash = hashlib.sha256(scene.read_bytes()).hexdigest() if scene.is_file() else "unsaved"
        self.geometry_key = json.dumps({"cache_version":3,"scene_sha256":scene_hash,
            "parts":cfg["parts"],"home":self.home,
            "collision_geometry_policy":planner.info.get("collision_geometry_policy")},sort_keys=True)

    def path_name(self, h):
        return self.sim.getObjectAlias(h, 1)

    def tool_contacts(self, arm, part):
        contacts = []
        for h in self.cfg["arms"][arm]["shapes"]:
            alias = self.sim.getObjectAlias(h)
            if alias.startswith(arm+"_") and any(s in alias for s in ["Palm", "Finger", "SupportShoe", "Tool"]):
                contacts.append([self.path_name(h), part])
        return contacts

    def support_contacts(self, arm, support_part):
        # The dynamic shoe is the designated supporting surface. Do not
        # exempt an entire arm, forearm, or all structure collisions.
        contacts = []
        for h in self.cfg["arms"][arm]["shapes"]:
            alias = self.sim.getObjectAlias(h)
            if alias.startswith(arm+"_") and "SupportShoe" in alias:
                contacts.append([self.path_name(h), support_part])
        return contacts

    def resting_contacts(self, part):
        names = list(part.get("dependencies", []))
        contacts = [[part["id"], name] for name in names]
        if not names:
            for h in self.cfg.get("obstacles", []):
                contacts.append([part["id"], self.path_name(h)])
        contacts.append([part["id"], self.path_name(self.cfg["table"])])
        return contacts

    def ik(self, arm, position, quat, seed, placed, payload=None, exclusions=None, other_q=None, other_payload=None):
        options = {"seed": seed, "ik_time": .10, "max_solutions": 8,
                   "other_q": other_q if other_q is not None else self.home["B" if arm == "A" else "A"],
                   "contact_exclusions": exclusions or [], "contact_tolerance": .003}
        if other_payload:
            options["other_payload"] = other_payload
        answer = self.planner.solve_ik(arm, position, quat, payload, placed, options)
        self.calls.append({"type": "IK", "arm": arm, "position": position,
                           "ok": bool(answer.get("ok")), "stats": answer.get("stats", {}),
                           "problem": answer.get("problem"), "error": answer.get("error")})
        return answer

    def qualify(self):
        candidates = []
        placed = {}
        support_cache = {}
        for part in self.cfg["parts"]:
            pid = part["id"]
            required = support_requirement(part)["support_required"]
            for arm in ["A", "B"]:
                other = "B" if arm == "A" else "A"
                for grasp in part["grasps"]:
                    payload = _payload(part, grasp, arm)
                    quat = grasp["quaternion_by_arm"][arm]
                    q = self.home[arm]
                    geometry = {}
                    failure = None
                    specs = [
                        ("source_approach", _position(part, grasp, "source", APPROACH), None, []),
                        ("source_grasp", _position(part, grasp, "source"), None,
                         self.tool_contacts(arm, pid) + self.resting_contacts(part)),
                        ("source_lift", _position(part, grasp, "source", LIFT), payload,
                         self.resting_contacts(part)),
                        ("goal_approach", _position(part, grasp, "goal", APPROACH), payload, []),
                        ("goal_place", _position(part, grasp, "goal"), payload,
                         self.resting_contacts(part)),
                    ]
                    for key, position, held, contacts in specs:
                        seed = self.home[arm] if required and key == "goal_approach" else q
                        answer = self.ik(arm, position, quat, seed, placed, held, contacts)
                        if not answer.get("ok"):
                            failure = {"stage": key, "position": position,
                                       "problem": answer.get("problem"), "error": answer.get("error")}
                            break
                        q = answer["q"]
                        geometry[key] = q
                    candidate = {
                        "part": pid, "arm": arm, "grasp": grasp["id"],
                        "feasible": failure is None, "cost": 1., "support_candidates": [],
                        "endpoint_evidence": geometry, "endpoint_failure": failure,
                        "payload": payload, "support_failures": [],
                    }
                    if failure is None:
                        sequence = [self.home[arm]] + list(geometry.values()) + [self.home[arm]]
                        candidate["cost"] = sum(_norm(a, b) for a, b in zip(sequence, sequence[1:]))/MAX_VELOCITY + 1.
                        if required:
                            support_part = part.get("support_part", part["dependencies"][-1])
                            supported = self.parts[support_part]
                            bottom = supported["goal"][2] - supported["size"][2]/2
                            # Candidate contact frames, not a part-to-arm lookup.
                            for x, y in [(.03, .160), (.06, .160), (.03, .17), (.06, .17), (-.03, .160)]:
                                for yaw in [0., math.pi/2, math.pi, -math.pi/2]:
                                    position = [x, y, bottom + .0005]
                                    quat_support = Rotation.from_euler("z", yaw).as_quat().tolist()
                                    shoe = self.cfg["arms"][other].get("support_pad")
                                    if shoe is None:
                                        shoe = next(h for h in self.cfg["arms"][other]["shapes"]
                                                    if "SupportShoe" in self.sim.getObjectAlias(h))
                                    pad_size = [self.sim.getObjectFloatParam(shoe,getattr(self.sim,"objfloatparam_objbbox_max_"+axis))-
                                                self.sim.getObjectFloatParam(shoe,getattr(self.sim,"objfloatparam_objbbox_min_"+axis))
                                                for axis in ["x","y","z"]]
                                    contact_geometry = support_contact_geometry(position,quat_support,supported,pad_size)
                                    if not contact_geometry["qualified"]:
                                        candidate["support_failures"].append({"stage":"contact_patch",
                                            "position":position,"quaternion":quat_support,"geometry":contact_geometry})
                                        continue
                                    contacts = self.support_contacts(other, support_part)
                                    answer = self.ik(other, position, quat_support, self.home[other], placed,
                                                     exclusions=contacts, other_q=geometry["goal_approach"],
                                                     other_payload=payload)
                                    if not answer.get("ok"):
                                        candidate["support_failures"].append({"stage":"contact", "position":position,
                                            "quaternion":quat_support, "problem":answer.get("problem")})
                                        continue
                                    q_contact = answer["q"]
                                    achieved = self.planner.fk(other,q_contact)["pose"]
                                    contact_geometry = support_contact_geometry(
                                        achieved[:3],achieved[3:],supported,pad_size)
                                    if not contact_geometry["qualified"]:
                                        candidate["support_failures"].append({"stage":"actual_contact_patch",
                                            "position":achieved[:3],"geometry":contact_geometry})
                                        continue
                                    before = None
                                    for pre in [[x,y,bottom-.004],[x,.12,bottom-.004],[x,.11,bottom+.0005]]:
                                        before = self.ik(other, pre, quat_support, q_contact, placed,
                                                        other_q=geometry["goal_approach"], other_payload=payload)
                                        if before.get("ok"):
                                            break
                                        candidate["support_failures"].append({"stage":"precontact", "position":pre,
                                            "quaternion":quat_support, "problem":before.get("problem")})
                                    if not before or not before.get("ok"):
                                        continue
                                    candidate["support_candidates"].append({
                                        "arm": other, "feasible": True,
                                        "cost": 2*_norm(self.home[other], q_contact)/MAX_VELOCITY + 1.2,
                                        "tcp_position": position, "quaternion": quat_support,
                                        "precontact_position": pre, "q_contact": q_contact,
                                        "q_precontact": before["q"], "support_part": support_part,
                                        "contact_exclusions": contacts,
                                        "contact_geometry": contact_geometry,
                                    })
                                    if len(candidate["support_candidates"]) >= 3:
                                        break
                                if len(candidate["support_candidates"]) >= 3:
                                    break
                    candidates.append(candidate)
                    print(json.dumps({"candidate": pid, "arm": arm, "grasp": grasp["id"],
                                      "feasible": candidate["feasible"],
                                      "support_candidates": len(candidate["support_candidates"]),
                                      "failure": failure}), flush=True)
            placed[pid] = list(part["goal"])
        return candidates

    def plan(self, choice, arm, start, goal, placed, payload=None, exclusions=None, other_q=None, other_payload=None,
             motion_options=None, solve_time=2.):
        request = {"arm":arm,"start":start,"goal":goal,"placed":placed,
                               "payload":payload,"contacts":exclusions,"other_q":other_q,
                               "other_payload":other_payload}
        if motion_options:
            request["motion_options"] = motion_options
        identity = json.dumps(request,sort_keys=True,separators=(",",":"))
        journal = self.path_journal/(hashlib.sha256((self.geometry_key+identity).encode()).hexdigest()+".json")
        if identity not in self.path_cache and journal.is_file():
            old = json.loads(journal.read_text(encoding="utf-8"))
            if old.get("geometry_key") == self.geometry_key and old.get("request") == identity:
                self.path_cache[identity] = old["path"]
        if identity in self.path_cache:
            self.calls.append({"type":"RRTConnect","part":choice["part"],"arm":arm,
                               "ok":True,"cache_reused":True})
            return deepcopy(self.path_cache[identity])
        print(json.dumps({"planning":choice["part"],"arm":arm,
                          "joint_displacement_rad":_norm(start,goal)}),flush=True)
        options = {"contact_tolerance": .003, "joint_bound_margin": 1.1,
                   "other_payload": other_payload or {}}
        options.update(motion_options or {})
        answer = self.planner.plan(arm, start, goal, payload, placed, other_q=other_q,
                                   contact_exclusions=exclusions or [], solve_time=solve_time, options=options)
        if not answer.get("ok"):
            # A bounded local search may fail even when a wider route exists.
            if motion_options:
                options["joint_bound_margin"] = .40
            else:
                options.pop("joint_bound_margin", None)
            answer = self.planner.plan(arm, start, goal, payload, placed, other_q=other_q,
                                       contact_exclusions=exclusions or [], solve_time=max(1.4,2*solve_time), options=options)
        self.calls.append({"type": "RRTConnect", "part": choice["part"], "arm": arm,
                           "ok": bool(answer.get("ok")), "stats": answer.get("stats", {}),
                           "problem": answer.get("problem"), "error": answer.get("error")})
        if not answer.get("ok"):
            raise _CompileFailure(choice, f"{choice['part']} {arm}: {answer.get('error', 'motion planning failed')}", answer)
        self.path_cache[identity] = deepcopy(answer["path"])
        _save(journal, {"geometry_key":self.geometry_key,"request":identity,
                        "path":answer["path"],"stats":answer.get("stats",{})})
        print(json.dumps({"planned":choice["part"],"arm":arm,"states":len(answer["path"]),
                          "wall_s":answer.get("stats",{}).get("wall_time_s")}),flush=True)
        return answer["path"]

    def staged_support_path(self, choice, support, placed, other_q, other_payload=None, reverse=False,
                            persistent_contacts=None):
        """Enter a support contact from below, with no sideways loaded sliding.

        Lower/translate stages retain every tool/structure collision test. Only
        vertical engagement and initial withdrawal exempt the exact shoe/part
        pair. Every segment is an actual RRTConnect query; dense FK additionally
        certifies that the contact segment remains on its vertical TCP axis.
        """
        arm = support["arm"]
        persistent_contacts = list(persistent_contacts or [])
        contact = support["tcp_position"]
        pre = support["precontact_position"]
        bottom = self.parts[support["support_part"]]["goal"][2] - self.parts[support["support_part"]]["size"][2]/2
        if not support.get("staging_geometry"):
            failures = []
            for clearance in [.005, .003]:
                positions = [[pre[0],pre[1],bottom-clearance],
                             [contact[0],contact[1],bottom-clearance]]
                answers = []
                for position,seed in zip(positions,[support["q_precontact"],support["q_contact"]]):
                    answer = self.planner.solve_ik(arm,position,support["quaternion"],placed=placed,
                        options={"seed":seed,"multiple":False,"other_q":other_q,
                                 "other_payload":other_payload or {},"contact_exclusions":persistent_contacts,
                                 "contact_tolerance":.003})
                    self.calls.append({"type":"IK","arm":arm,"position":position,
                        "ok":bool(answer.get("ok")),"stats":answer.get("stats",{}),
                        "problem":answer.get("problem"),"stage":"under_support"})
                    answers.append(answer)
                if all(answer.get("ok") for answer in answers):
                    support["staging_geometry"] = {"clearance_m":clearance,
                        "outside_low":positions[0],"inside_low":positions[1],
                        "q_outside_low":answers[0]["q"],"q_inside_low":answers[1]["q"]}
                    break
                failures.append({"clearance_m":clearance,"answers":answers})
            else:
                raise _CompileFailure(choice,"No collision-free under-support staging IK",{"stages":failures})
        geometry = support["staging_geometry"]
        if reverse:
            stages = [("vertical_disengage",support["q_contact"],geometry["q_inside_low"],True),
                      ("outside_translate",geometry["q_inside_low"],geometry["q_outside_low"],False),
                      ("outside_raise",geometry["q_outside_low"],support["q_precontact"],False)]
        else:
            stages = [("outside_lower",support["q_precontact"],geometry["q_outside_low"],False),
                      ("under_beam_translate",geometry["q_outside_low"],geometry["q_inside_low"],False),
                      ("vertical_engage",geometry["q_inside_low"],support["q_contact"],True)]
        result, markers = [], []
        for name,start,goal,contact_allowed in stages:
            path = self.plan(choice,arm,start,goal,placed,other_q=other_q,other_payload=other_payload,
                exclusions=persistent_contacts+(support["contact_exclusions"] if contact_allowed else []),solve_time=.6,
                motion_options={"joint_bound_margin":.06 if _norm(start,goal)<.08 else .16,
                                "max_joint_step":.015})
            marker = {"name":name,"q_start_index":max(0,len(result)-1),"contact_allowed":contact_allowed}
            if contact_allowed:
                max_xy = 0.
                for a,b in zip(path,path[1:]):
                    count = max(1,math.ceil(_norm(a,b)/.003))
                    for i in range(count+1):
                        sample = [x+(y-x)*i/count for x,y in zip(a,b)]
                        pose = self.planner.fk(arm,sample)["pose"]
                        max_xy = max(max_xy,_norm(pose[:2],contact[:2]))
                if max_xy >= .0003:
                    raise _CompileFailure(choice,"Support contact path leaves its vertical TCP axis",
                                          {"stage":name,"max_xy_axis_deviation_m":max_xy})
                marker["max_xy_axis_deviation_m"] = max_xy
            result.extend(deepcopy(path if not result else path[1:]))
            marker["q_end_index"] = len(result)-1
            markers.append(marker)
        return result, {"contact_stages":markers,"under_beam_clearance_m":geometry["clearance_m"],
                        "max_velocity_rad_s":.025,"max_acceleration_rad_s2":.12,
                        "contact_skill":"lower_outside_then_translate_under_then_vertical_engage"}

    def compile(self, allocation, candidates):
        tasks = []
        placed = {}
        q = deepcopy(self.home)
        completed_placements = {}
        choices = {c["part"]: c for c in allocation["choices"]}

        def emit(arm, kind, pid, path, *, deps=None, payload=None, action=None,
                 contacts=None, support_part=None, force=0., dwell=0., extra=None):
            identity = f"{pid}_{arm}_{kind}_{len(tasks):03d}"
            metadata = {"action": action, "part": pid, "support_part": support_part,
                        "force_N": force, "contact_exclusions": deepcopy(contacts or []),
                        "placed": deepcopy(placed), "max_velocity_rad_s": MAX_VELOCITY,
                        "max_acceleration_rad_s2": MAX_ACCELERATION,
                        "time_parameterization": "rest_to_rest_per_polyline_segment",
                        "dwell_s": dwell}
            if extra:
                metadata.update(deepcopy(extra))
            task = {"id": identity, "arm": arm, "kind": kind, "part": pid,
                    "duration": trajectory_duration(path, dwell), "q_path": deepcopy(path),
                    "dependencies": list(deps or []), "reference_order": len(tasks),
                    "payload": deepcopy(payload), "metadata": metadata}
            tasks.append(task)
            q[arm] = list(path[-1])
            return identity

        for part in self.cfg["parts"]:
            pid = part["id"]
            choice = choices[pid]
            candidate = candidates[choice["candidate_index"]]
            arm = choice["arm"]
            other = "B" if arm == "A" else "A"
            geometry = candidate["endpoint_evidence"]
            payload = candidate["payload"]
            tool_contacts = self.tool_contacts(arm, pid)
            surface_contacts = self.resting_contacts(part)
            dependencies = [completed_placements[d] for d in part.get("dependencies", [])]
            path = self.plan(choice, arm, q[arm], geometry["source_approach"], placed, other_q=q[other])
            emit(arm, "transit", pid, path, deps=dependencies)
            path = self.plan(choice, arm, q[arm], geometry["source_grasp"], placed,
                             exclusions=tool_contacts+surface_contacts, other_q=q[other])
            emit(arm, "pick", pid, path, action="grasp", contacts=tool_contacts+surface_contacts,
                 dwell=.25, extra={"payload_after": payload, "grasp": choice["grasp"]})
            path = self.plan(choice, arm, q[arm], geometry["source_lift"], placed, payload,
                             surface_contacts, q[other])
            emit(arm, "transit", pid, path, payload=payload, contacts=surface_contacts)
            path = self.plan(choice, arm, q[arm], geometry["goal_approach"], placed, payload,
                             other_q=q[other])
            carry_id = emit(arm, "transit", pid, path, payload=payload)
            support = None
            support_start = None
            support_contacts = []
            support_part = None
            if choice["support_arm"]:
                support = candidate["support_candidates"][choice["support_candidate_index"]]
                helper = support["arm"]
                support_part = support["support_part"]
                support_contacts = support["contact_exclusions"]
                path = self.plan(choice, helper, q[helper], support["q_precontact"], placed,
                                 other_q=q[arm], other_payload=payload)
                helper_dependencies = list(dict.fromkeys(
                    dependencies + [completed_placements[support_part], carry_id]))
                emit(helper, "transit", pid, path, deps=helper_dependencies, support_part=support_part)
                path, skill_metadata = self.staged_support_path(choice,support,placed,q[arm],payload)
                support_start = emit(helper, "support_start", pid, path, action="support_start",
                                     support_part=support_part, contacts=support_contacts, dwell=.35,
                                     extra={"contact_position": support["tcp_position"],
                                            "precontact_position": support["precontact_position"],
                                            **skill_metadata})
            path = self.plan(choice, arm, q[arm], geometry["goal_place"], placed, payload,
                             surface_contacts+support_contacts, q[other])
            place_id = emit(arm, "place", pid, path, deps=[support_start] if support_start else [],
                            payload=payload, action="release", contacts=surface_contacts+support_contacts,
                            support_part=support_part, force=part.get("insertion_force", 0.), dwell=.40,
                            extra={"goal": list(part["goal"]), "payload_after": None,
                                   "force_application_point": list(part["goal"]),
                                   "support_required": choice["support_required"]})
            completed_placements[pid] = place_id
            placed[pid] = list(part["goal"])
            # The installer returns to HOME while a helper still holds its
            # certified support pose. The later TPG can relax this serial order.
            path = self.plan(choice, arm, q[arm], geometry["goal_approach"], placed,
                             exclusions=tool_contacts+surface_contacts+support_contacts, other_q=q[other])
            emit(arm, "transit", pid, path, contacts=tool_contacts+surface_contacts+support_contacts)
            path = self.plan(choice, arm, q[arm], self.home[arm], placed,
                             exclusions=support_contacts, other_q=q[other])
            emit(arm, "transit", pid, path, contacts=support_contacts)
            if support is not None:
                helper = support["arm"]
                path, skill_metadata = self.staged_support_path(choice,support,placed,q[arm],reverse=True)
                emit(helper, "support_end", pid, path, deps=[place_id], action="support_end",
                     support_part=support_part, contacts=support_contacts, dwell=.15,extra=skill_metadata)
                path = self.plan(choice, helper, q[helper], self.home[helper], placed, other_q=q[arm])
                emit(helper, "transit", pid, path)
            if self.cfg.get("_compile_part_limit") and len(completed_placements) >= self.cfg["_compile_part_limit"]:
                break
        return tasks


def compile_reference(sim, client, cfg, output_dir):
    """Return an audited, ILP-selected sequential reference task plan.

    A finite candidate may pass endpoint IK but fail its connecting path.
    Such a selected mode is cut and the ILP is solved again; no failed path is
    silently replaced by Cartesian interpolation or an unchecked animation.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with MotionPlanner(sim, client, cfg) as planner:
        compiler = _Compiler(sim, planner, cfg)
        candidates = compiler.qualify()
        _save(output/"candidate_qualification.json", candidates)
        problem = {"arms": ["A", "B"], "parts": deepcopy(cfg["parts"]),
                   "candidates": candidates, "load_balance_weight": .15,
                   "cost_unit": "estimated_joint_travel_seconds", "mip_rel_gap": 0.,
                   "initial_joint_configurations": deepcopy(compiler.home)}
        cuts = []
        allocation = None
        for attempt in range(1, 33):
            allocation = solve_allocation(problem)
            _save(output/"allocation.json", allocation)
            _save(output/"allocation_problem.json", problem)
            if not allocation.get("solved"):
                _save(output/"planning_failures.json", cuts)
                raise RuntimeError(f"No feasible ILP mode assignment: {allocation.get('status')}")
            try:
                tasks = compiler.compile(allocation, candidates)
                break
            except _CompileFailure as exc:
                choice = exc.choice
                candidate = candidates[choice["candidate_index"]]
                if choice.get("support_candidate_index") is not None:
                    candidate["support_candidates"][choice["support_candidate_index"]]["feasible"] = False
                else:
                    candidate["feasible"] = False
                cuts.append({"attempt": attempt, "choice": choice, "reason": str(exc), "evidence": exc.evidence})
                print(json.dumps({"motion_mode_cut": choice["part"], "arm": choice["arm"],
                                  "grasp": choice["grasp"], "reason": str(exc)}), flush=True)
        else:
            raise RuntimeError("Reference compilation exhausted its mode-repair budget")
        result = {"allocation": allocation, "tasks": tasks,
                  "motion_stats": {"calls": compiler.calls, "mode_cuts": cuts,
                                   "collision_geometry_policy": planner.info["collision_geometry_policy"],
                                   "wall_time_s": time.perf_counter()-started,
                                   "backend": "simIK + simOMPL RRTConnect",
                                   "path_count": sum(c["type"] == "RRTConnect" for c in compiler.calls),
                                   "reference_motion_tasks": len(tasks),
                                   "planner_invocations": sum(c["type"] == "RRTConnect" and not c.get("cache_reused") for c in compiler.calls),
                                   "path_cache_hits": sum(bool(c.get("cache_reused")) for c in compiler.calls),
                                   "max_velocity_rad_s": MAX_VELOCITY,
                                   "max_acceleration_rad_s2": MAX_ACCELERATION},
                  "problem": problem}
        _save(output/"reference_tasks.json", tasks)
        _save(output/"tasks.json", tasks)
        _save(output/"motion_stats.json", result["motion_stats"])
        _save(output/"planning_failures.json", cuts)
        _save(output/"alloc_and_reference.json", result)
        return result
