"""Stopped-scene kinematics and exact native mesh predicates for the showcase."""
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from bridge_runtime import HERE, transform, set_transform
from bridge_native import BATCH_GEOMETRY


GEOMETRY = r"""
local sim=require('sim')
local meshes,pairsToCheck,seats,internalPairs={},{},{},{}
local function bounds(h,item)
    local m=sim.getObjectMatrix(h,sim.handle_world)
    local c,r={},{}
    local center,extent=item[1],item[2]
    for axis=0,2 do
        local i=4*axis+1
        c[axis+1]=m[i+3]+m[i]*center[1]+m[i+1]*center[2]+m[i+2]*center[3]
        r[axis+1]=math.abs(m[i])*extent[1]+math.abs(m[i+1])*extent[2]+math.abs(m[i+2])*extent[3]+2e-8
    end
    return c,r,m
end
local function separation(a,b)
    local square=0
    for i=1,3 do
        local gap=math.max(0,math.abs(a[1][i]-b[1][i])-a[2][i]-b[2][i])
        square=square+gap*gap
    end
    return math.sqrt(square)
end
local function equalSeat(a,b)
    local square=0
    local trace=0
    for i=1,12 do
        if i%4==0 then square=square+(a[i]-b[i])^2 else trace=trace+a[i]*b[i] end
    end
    return square<=1e-10 and (trace-1)*.5>=math.cos(2e-4)
end
local function key(a,b)
    if a<b then return tostring(a)..':'..tostring(b) end
    return tostring(b)..':'..tostring(a)
end
function bridgeConfigure(meshRecords,pairs,contacts,distanceExcluded)
    for _,row in ipairs(meshRecords) do meshes[row[1]]={row[2],row[3]} end
    pairsToCheck=pairs
    seats=contacts
    for _,handles in ipairs(distanceExcluded or {}) do
        internalPairs[key(handles[1],handles[2])]=true
    end
    return {shape_count=#meshRecords,pair_count=#pairs,seat_count=#contacts,
            distance_excluded_pair_count=#(distanceExcluded or {})}
end
function bridgeReadJoints(handles)
    local values={}
    for i,h in ipairs(handles) do values[i]=sim.getJointPosition(h) end
    return values
end
function bridgeWriteJoints(handles,values)
    for i,h in ipairs(handles) do sim.setJointPosition(h,values[i]) end
    return bridgeReadJoints(handles)
end
function bridgeWriteJaw(master,follower,q,direct)
    sim.setJointPosition(master,q)
    if direct then sim.setJointPosition(follower,q) end
    return {sim.getJointPosition(master),sim.getJointPosition(follower)}
end
function bridgeSample(contacts,limit,measure,timed,filtered,exhaustive)
    local world,allowed={},{}
    measure=measure~=false
    local phase={aabb_seconds=0,collision_seconds=0,distance_seconds=0}
    local function now() return timed and sim.getSystemTime() or 0 end
    local begun=now()
    local aabbPassed,collisions,distances=0,0,0
    local allowedCollisions,rejected={},{}
    for h,record in pairs(meshes) do world[h]={bounds(h,record)} end
    for _,row in ipairs(contacts) do allowed[key(row[1],row[2])]='declared_contact' end
    for _,row in ipairs(seats) do
        local a,b,am,bm=row[1],row[2],row[3],row[4]
        if equalSeat(world[a][3],am) and equalSeat(world[b][3],bm) then
            allowed[key(a,b)]='nominal_seat'
        end
    end
    phase.aabb_seconds=now()-begun
    local best,nearest=limit,nil
    local rawBest,rawNearest=limit,nil
    local scope=not measure and 'not_measured' or
                        (filtered and 'external_filtered' or 'all_active_pairs')
    for _,pair in ipairs(pairsToCheck) do
        local a,b=pair[1],pair[2]
        begun=now()
        local lower=separation(world[a],world[b])
        phase.aabb_seconds=phase.aabb_seconds+now()-begun
        local permit=allowed[key(a,b)]
        if lower<=1e-10 then
            collisions=collisions+1
            begun=now()
            local hit,objects=sim.checkCollision(a,b)
            phase.collision_seconds=phase.collision_seconds+now()-begun
            if hit>0 then
                if permit then
                    allowedCollisions[#allowedCollisions+1]={a,b,permit}
                else
                    rejected[#rejected+1]={pair_handles=objects or pair}
                    if not exhaustive then break end
                end
            end
        else
            aabbPassed=aabbPassed+1
        end
        if measure and not permit then
            local internal=filtered and internalPairs[key(a,b)]
            local searchLimit=internal and rawBest or best
            if lower<searchLimit then
                distances=distances+1
                begun=now()
                local found,data,objects=sim.checkDistance(a,b,searchLimit)
                phase.distance_seconds=phase.distance_seconds+now()-begun
                if found>0 then
                    if not internal then best,nearest=data[7],objects or pair end
                    if filtered and data[7]<rawBest then rawBest,rawNearest=data[7],objects or pair end
                end
            end
        end
    end
    if #rejected>0 then
        local result={ok=false,minimum_distance_m=measure and 0 or 'not_measured',
                clearance_scope=scope,pair_handles=rejected[1].pair_handles,kind='native_collision',
                coll_queries=collisions,dist_queries=distances,aabb_separated=aabbPassed,
                collisions=exhaustive and rejected or nil,
                allowed_collisions=allowedCollisions,timing=timed and phase or nil}
        if measure and filtered then
            result.unfiltered_minimum_distance_m=0
            result.unfiltered_pair_handles=result.pair_handles
            result.unfiltered_distance_is_lower_bound=false
        end
        return result
    end
    local result={ok=true,minimum_distance_m=measure and best or 'not_measured',pair_handles=nearest,
            distance_is_lower_bound=measure and not nearest or false,clearance_scope=scope,
            coll_queries=collisions,dist_queries=distances,aabb_separated=aabbPassed,
            allowed_collisions=allowedCollisions,timing=timed and phase or nil}
    if measure and filtered then
        result.unfiltered_minimum_distance_m=rawBest
        result.unfiltered_pair_handles=rawNearest
        result.unfiltered_distance_is_lower_bound=not rawNearest
    end
    return result
end
"""


def _matrix(value):
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        return array.copy()
    if array.size == 7:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(array[3:]).as_matrix()
        matrix[:3, 3] = array[:3]
        return matrix
    if array.size == 12:
        matrix = np.eye(4)
        matrix[:3] = array.reshape(3, 4)
        return matrix
    raise ValueError("Expected a rigid pose")


def _handle(ref):
    return int(ref["handle"] if isinstance(ref, dict) else ref)


def _flat(matrix):
    return matrix[:3].ravel().tolist()


class CalibratedChain:
    def __init__(self, world, side):
        self.world, self.side = world, side
        arm = world.arms[side]
        joints = [_handle(ref) for ref in arm["joints"]]
        intervals = [world.sim.getJointInterval(h) for h in joints]
        self.lower = np.array([(bounds[0] if not cyclic else -2*math.pi) for cyclic,bounds in intervals])
        self.upper = np.array([(sum(bounds) if not cyclic else 2*math.pi) for cyclic,bounds in intervals])
        origin_q = world.q(side)
        try:
            world.setq(side, np.zeros(6))
            matrices = [transform(world.sim, h) for h in joints]
            self.points = np.asarray([m[:3, 3] for m in matrices])
            self.axes = np.asarray([m[:3, 2] / np.linalg.norm(m[:3, 2]) for m in matrices])
            self.tip_zero = world.tcp(side)
        finally:
            world.setq(side, origin_q)
        self.error = self.pose_error(self.fk(origin_q)[0], world.tcp(side))
        if self.error[0] > 5e-5 or self.error[1] > 2e-4:
            raise ValueError(f"{side} measured revolute-axis model FK mismatch {self.error}")

    @staticmethod
    def pose_error(actual, desired):
        return (float(np.linalg.norm(actual[:3, 3]-desired[:3, 3])),
                float(Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).magnitude()))

    def fk(self, q):
        matrix = np.eye(4)
        origins, directions = [], []
        for point, axis, angle in zip(self.points, self.axes, q):
            relative = Rotation.from_rotvec(axis * angle).as_matrix()
            current = matrix[:3, :3]
            origins.append(current @ point + matrix[:3, 3])
            directions.append(current @ axis)
            matrix[:3, 3] += current @ (point - relative @ point)
            matrix[:3, :3] = current @ relative
        pose = matrix @ self.tip_zero
        jacobian = np.vstack((np.cross(directions, pose[:3, 3]-origins).T, np.asarray(directions).T))
        return pose, jacobian

    def solve(self, target, seed, max_solutions, seed_count):
        rng = np.random.default_rng(721 if self.side == "A" else 722)
        seeds = [seed]
        seeds.extend(seed + np.array([0., d2, d3, d4, d5, 0.])
                     for d2,d3,d4,d5 in ((.6,-1.4,.8,0.),(-.6,1.4,-.8,0.),(0.,0.,math.pi,-1.5)))
        seeds.extend(rng.uniform(-math.pi, math.pi, (seed_count, 6)))
        results = []
        for start in seeds[:seed_count]:
            q = np.clip(start, self.lower, self.upper)
            damping = 1e-5
            for _ in range(100 if seed_count > 1 else 45):
                actual, jac = self.fk(q)
                dx = target[:3, 3] - actual[:3, 3]
                dw = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
                if np.linalg.norm(dx) < 5e-7 and np.linalg.norm(dw) < 2e-6:
                    # Choose the nearest equivalent within the measured limits.
                    winding = np.rint((seed-q)/(2*math.pi))
                    winding = np.clip(winding, np.ceil((self.lower-q)/(2*math.pi)),
                                      np.floor((self.upper-q)/(2*math.pi)))
                    candidate = q + 2*math.pi*winding
                    if not any(np.max(np.abs(candidate-old)) < .001 for old in results):
                        results.append(candidate)
                    break
                error = np.r_[dx, dw]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
                if np.linalg.norm(delta, np.inf) > .35:
                    delta *= .35 / np.linalg.norm(delta, np.inf)
                improved = False
                for scale in (1., .5, .25):
                    nxt = np.clip(q + scale*delta, self.lower, self.upper)
                    test = self.fk(nxt)[0]
                    dp, dr = self.pose_error(test, target)
                    if dp*dp + dr*dr < float(error @ error):
                        q = nxt
                        improved = True
                        break
                if improved:
                    damping = max(1e-8, damping*.6)
                else:
                    damping *= 10.
                    if damping > .1:
                        break
            if len(results) >= max_solutions:
                break
        return sorted(results,key=lambda q: float(np.linalg.norm(q-seed)))


class BridgeWorld:
    def __init__(self, sim, client, *, batch_joints=False):
        self.batch_joints = batch_joints
        self.script = None
        self.sim, self.client = sim, client
        self.manifest = json.loads((HERE/"assets"/"scene_manifest.json").read_text(encoding="utf-8-sig"))
        self.arms, self.parts = self.manifest["arms"], self.manifest["parts"]
        scene = HERE/"assets"/"lego_bridge_kinematic_v1.ttt"
        if sha256(scene.read_bytes()).hexdigest().lower() != self.manifest["scene_sha256"].lower():
            raise RuntimeError("The saved candidate scene SHA-256 changed")
        sim.loadScene(str(scene))
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("Phase 0 requires simulation stopped")
        if sim.getBoolParam(sim.boolparam_dynamics_handling_enabled):
            raise RuntimeError("Candidate scene dynamics are enabled")
        remap = {}

        def resolve_refs(node):
            if isinstance(node, dict):
                if "handle" in node and "path" in node:
                    original = node["handle"]
                    actual = sim.getObject(node["path"])
                    previous = remap.setdefault(original, actual)
                    if previous != actual:
                        raise RuntimeError(f"Object identifier disagreement {node}")
                    node["handle"] = actual
                else:
                    for value in node.values():
                        resolve_refs(value)
            elif isinstance(node, list):
                for value in node:
                    resolve_refs(value)

        resolve_refs(self.manifest)
        self.rebound_handles = {old:new for old,new in remap.items() if old != new}
        for row in self.manifest.get("mechanical_contacts", []):
            if isinstance(row, dict):
                row["pair"] = [remap.get(h,h) if isinstance(h,int) else h for h in row["pair"]]
        self.home_parents = {name: sim.getObjectParent(_handle(row)) for name,row in self.parts.items()}
        self.offsets = {}
        for name, row in self.parts.items():
            mesh = transform(sim, _handle(row))
            base = np.eye(4)
            base[:3, 3] = row["initial_bottom_m"]
            self.offsets[name] = np.linalg.inv(base) @ mesh
        self.chains = {}
        self._calibrate()
        self.script = sim.createScript(sim.scripttype_customization, GEOMETRY + BATCH_GEOMETRY, 0, "lua")
        sim.setObjectAlias(self.script, "LEGO_Bridge_Phase0_Geometry")
        sim.initScript(self.script)
        self._configure_query()
        self._configure_batch()

    def _calibrate(self):
        for side in self.arms:
            self.chains[side] = CalibratedChain(self, side)

    def q(self, side):
        joints = [_handle(ref) for ref in self.arms[side]["joints"]]
        if self.script is not None and self.batch_joints:
            return np.asarray(self.sim.callScriptFunction("bridgeReadJoints",self.script,joints))
        return np.asarray([self.sim.getJointPosition(h) for h in joints])

    def setq(self, side, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint positions")
        chain = self.chains.get(side)
        if chain is not None and (np.any(q < chain.lower-1e-8) or np.any(q > chain.upper+1e-8)):
            raise ValueError(f"{side} joint configuration exceeds native joint limits")
        if self.script is not None and self.batch_joints:
            handles = [_handle(ref) for ref in self.arms[side]["joints"]]
            observed = self.sim.callScriptFunction("bridgeWriteJoints",self.script,handles,q.tolist())
            if np.max(np.abs(np.asarray(observed)-q)) > 1e-7:
                raise RuntimeError(f"Native joint write did not preserve {side} command")
            return
        for ref, angle in zip(self.arms[side]["joints"], q):
            self.sim.setJointPosition(_handle(ref), float(angle))

    def tcp(self, side):
        return transform(self.sim, _handle(self.arms[side]["tcp"]))

    def jaw(self, side, q):
        value = float(q)
        if not 0. <= value <= .017 + 1.e-10:
            raise ValueError("Jaw position exceeds declared native interval")
        arm = self.arms[side]
        if self.script is not None and self.batch_joints:
            observed = self.sim.callScriptFunction("bridgeWriteJaw",self.script,
                _handle(arm["jaw_opening"]),_handle(arm["jaw_follower"]),value,
                arm.get("jaw_driver_mode") == "direct_both")
            if np.max(np.abs(np.asarray(observed)-value)) > 1e-7:
                raise RuntimeError(f"Native jaw write did not preserve {side} command")
            return
        self.sim.setJointPosition(_handle(arm["jaw_opening"]),value)
        if arm.get("jaw_driver_mode") == "direct_both":
            self.sim.setJointPosition(_handle(arm["jaw_follower"]),value)
        follower = self.sim.getJointPosition(_handle(arm["jaw_follower"]))
        if abs(follower - value) > 1.e-7:
            raise ValueError(f"{side} follower did not track the master: {follower} != {value}")

    def bottom(self, name):
        return transform(self.sim, _handle(self.parts[name])) @ np.linalg.inv(self.offsets[name])

    def set_bottom(self, name, world_bottom):
        if np.shape(world_bottom)==(3,):
            tmp = np.eye(4)
            tmp[:3, 3] = world_bottom
            world_bottom = tmp
        set_transform(self.sim, _handle(self.parts[name]), _matrix(world_bottom) @ self.offsets[name])

    def attach(self, name, side):
        handle = _handle(self.parts[name])
        initial = self.bottom(name)
        parent = _handle(self.arms[side]["tcp"]) if side is not None else self.home_parents[name]
        self.sim.setObjectParent(handle, parent, True)
        drift, angle = CalibratedChain.pose_error(self.bottom(name), initial)
        if drift > 1e-5 or angle > 2e-4 or self.sim.getObjectParent(handle) != parent:
            raise RuntimeError(f"Invalid {name} attachment transform {drift} m, {angle} rad")
        return drift

    def ik(self, side, target, seed=None, max_solutions=1, seed_count=1):
        target = _matrix(target)
        seed = self.q(side) if seed is None else np.asarray(seed, dtype=float)
        return self.chains[side].solve(target,seed,max_solutions,seed_count)

    def _group(self, h, joints):
        while h >= 0:
            if h in joints:
                return joints[h]
            h = self.sim.getObjectParent(h)
        return 0

    def _configure_query(self):
        sim = self.sim
        roles = {}
        for side, arm in self.arms.items():
            joints = {_handle(ref):idx for idx,ref in enumerate(arm["joints"],1)}
            joints.update({_handle(arm["jaw_opening"]):7, _handle(arm["jaw_follower"]):8})
            tool = {_handle(ref) for ref in arm["tool_shapes"]}
            for ref in arm["shapes"]:
                h = _handle(ref)
                role = "tool" if h in tool else "link"
                roles[h] = (role,side,self._group(h,joints))
        for ref in self.manifest["environment_shapes"]:
            h = _handle(ref)
            roles.setdefault(h,("environment",None,None))
        for row in self.parts.values():
            roles[_handle(row)] = ("part",None,None)
        self.handle_roles = {h:{"role":role,"side":side,"joint_group":joint_group}
                             for h,(role,side,joint_group) in roles.items()}
        # Use every original collidable render/assembly mesh. Noncollidable
        # inertia shells remain in the scene and manifest with original flags.
        self._names = {h: f"{sim.getObjectAlias(h, 1)} [#{h}]" for h in roles}
        active = {h:value for h,value in roles.items()
                  if sim.getObjectSpecialProperty(h) & sim.objectspecialproperty_collidable}
        self.inactive_collision_handles = sorted(set(roles)-set(active))
        mesh_data = []
        for h in active:
            dims, local_pose = sim.getShapeBB(h)
            box = _matrix(local_pose)
            extent = np.abs(box[:3,:3]) @ (np.asarray(dims)*.5)
            mesh_data.append([h, box[:3,3].tolist(), extent.tolist()])
        fixed = {}
        for row in self.manifest.get("mechanical_contacts", []):
            fixed[tuple(sorted(_handle(x) for x in (row["pair"] if isinstance(row,dict) else row)))] = row
        for arm in self.arms.values():
            jaw_proof = arm.get("jaw_evidence", {})
            if jaw_proof.get("passed"):
                # Only the plane and bearing overlap proven across the travel
                # qualify. The other moving-tool pairs are still queried.
                rail = _handle(arm["shape_roles"]["Rail_cover"][0])
                for ref in arm["shape_roles"]["Slide"]:
                    fixed[tuple(sorted((_handle(ref), rail)))] = "proven slide bearing plane"
        excluded = []
        pairs = []
        for a,b in combinations(sorted(active),2):
            ka,sa,ia = active[a]
            kb,sb,ib = active[b]
            reason = None
            if ka == kb == "environment":
                reason = "fixed environment to fixed environment"
            elif a in active and b in active and (a,b) in fixed:
                reason = "declared mechanical attachment contact"
            elif sa is not None and sa == sb:
                if ia==ib and (ka==kb=="link" or ka==kb=="tool" or ia==6):
                    reason = "rigid assembly"
                elif ka==kb=="link" and abs(ia-ib)==1:
                    reason = "OEM joint interface"
            elif kb=="environment" and ka=="link" and ia==0 and sim.getObjectAlias(b)==f"Bench_mount_{sa}":
                reason = "OEM arm base to mounting pedestal"
            elif ka=="environment" and kb=="link" and ib==0 and sim.getObjectAlias(a)==f"Bench_mount_{sb}":
                reason = "OEM arm base to mounting pedestal"
            if reason:
                excluded.append([a,b,reason])
            else:
                pairs.append([a,b])
        self.geometry_scope = {"active":len(active), "tested_pairs":len(pairs),
                               "inactive_original_flags":self.inactive_collision_handles, "structural":excluded}
        board = _handle(self.manifest["board"])
        board_matrix = _flat(transform(sim,board))
        seats = []

        def target(name, field):
            pose = np.eye(4)
            pose[:3,3] = self.parts[name][field]
            return _flat(pose @ self.offsets[name])

        for name,row in self.parts.items():
            h = _handle(row)
            seats.append([h,board,target(name,"initial_bottom_m"),board_matrix])
            if abs(row["target_bottom_m"][2]-.002) < 1e-5:
                seats.append([h,board,target(name,"target_bottom_m"),board_matrix])
        for upper, lower in (("Orange","Blue"),("Green","Red")):
            for i in (1,2):
                n1,n2=f"{upper}_{i:02}",f"{lower}_{i:02}"
                seats.append([_handle(self.parts[n1]),_handle(self.parts[n2]),
                              target(n1,"target_bottom_m"),target(n2,"target_bottom_m")])
                deck="Bridge_deck_12x4"
                seats.append([_handle(self.parts[deck]),_handle(self.parts[n1]),
                              target(deck,"target_bottom_m"),target(n1,"target_bottom_m")])
        # Only the distance presentation metric omits these pairs. The same
        # collision candidate list is used for every query and scope.
        verified_tool_sets = [
            {_handle(ref) for ref in arm["tool_shapes"]}
            for arm in self.arms.values()
            if arm.get("jaw_evidence",{}).get("passed") is True
            and not arm["jaw_evidence"].get("unexpected_collisions")
        ]
        distance_excluded = [pair for pair in pairs
                             if any(set(pair)<=handles for handles in verified_tool_sets)]
        self.geometry_scope["clearance_only_excluded_pairs"] = distance_excluded
        self.query_registration = sim.callScriptFunction(
            "bridgeConfigure",self.script,mesh_data,pairs,seats,distance_excluded)

    def check(self, allow=(), requested_distance_m=.025, *, measure=True, filtered_clearance=False,
              all_collisions=False):
        contacts = [[_handle(a),_handle(b)] for a,b in allow]
        result = self.sim.callScriptFunction(
            "bridgeSample",self.script,contacts,float(requested_distance_m),
            bool(measure),False,bool(filtered_clearance),bool(all_collisions))
        self._label_pairs(result)
        return result

    def _label_pairs(self, result):
        for row in (result, *result.get("collisions", ())):
            raw = row.get("pair_handles")
            row["pair"] = [self._names.get(h, str(h)) for h in raw] if raw else None
        if "unfiltered_minimum_distance_m" in result:
            raw = result.get("unfiltered_pair_handles")
            result["unfiltered_pair"] = [self._names.get(h, str(h)) for h in raw] if raw else None


    def _configure_batch(self):
        arms = {}
        for side, row in self.arms.items():
            arms[side] = {"tcp":_handle(row["tcp"]), "joints":[_handle(h) for h in row["joints"]],
                          "master":_handle(row["jaw_opening"]), "follower":_handle(row["jaw_follower"]),
                          "direct":row.get("jaw_driver_mode")=="direct_both",
                          "pads":[_handle(h) for h in row["shape_roles"]["Grip_pad"]]}
        parts = {}
        for name, row in self.parts.items():
            span = np.asarray(row["bounds_max_m"]) - np.asarray(row["bounds_min_m"])
            body = .0032 if name=="Bridge_deck_12x4" else .016
            parts[name] = {"handle":_handle(row), "parent":self.home_parents[name],
                           "mesh_to_bottom":_flat(np.linalg.inv(self.offsets[name])),
                           "dimensions":[float(span[0]),float(span[1]),body]}
        self.sim.callScriptFunction("bridgeBatchConfigure",self.script,arms,parts)

    def _prepare_state(self, state):
        q,jaw,parts = state["q"],state["jaw"],state["parts"]
        if set(q)!=set(self.arms) or set(jaw)!=set(self.arms) or set(parts)!=set(self.parts):
            raise ValueError("Full sampled state requires both arms, both jaws and every original part")
        out = {"q":{}, "jaw":{}, "parts":{}, "measure_distance":bool(state.get("measure_distance", True))}
        for side,chain in self.chains.items():
            joints=np.asarray(q[side],dtype=float)
            value=float(jaw[side])
            if (joints.shape!=(6,) or not np.isfinite(joints).all()
                    or np.any(joints < chain.lower-1e-8) or np.any(joints > chain.upper+1e-8)):
                raise ValueError(f"{side} sample exceeds joint limits")
            if not 0. <= value <= .017+1e-10:
                raise ValueError(f"{side} sample exceeds jaw limits")
            out["q"][side]=joints.tolist()
            out["jaw"][side]=value
        for name,row in parts.items():
            owner = row.get("owner")
            if owner is None:
                out["parts"][name]={"pose":self._flat_valid(row["pose"])}
            elif owner in self.arms:
                out["parts"][name]={"owner":owner,"binding":self._flat_valid(row["binding"])}
            else:
                raise ValueError(f"Unknown owner {owner} for {name}")
        if state.get("allow"):
            contacts = [[_handle(a),_handle(b)] for a,b in state["allow"]]
            if any(a==b or a not in self._names or b not in self._names for a,b in contacts):
                raise ValueError("Invalid collision-contact pair")
            out["allow"] = contacts
        if state.get("faces"):
            out["faces"] = [{"side":row["side"],"part":row["part"],"axis":int(row["axis"])}
                            for row in state["faces"]]
        expected_tcp = state.get("tcp_targets") or state.get("expected_tcp")
        if expected_tcp:
            out["tcp_targets"]={side:self._flat_valid(pose) for side,pose in expected_tcp.items()}
        if state.get("follower"):
            out["follower"] = {side:float(value) for side,value in state["follower"].items()}
        return out

    @staticmethod
    def _flat_valid(pose):
        m = _matrix(pose)
        if (not np.isfinite(m).all()
                or np.max(np.abs(m[:3,:3].T @ m[:3,:3] - np.eye(3)))>1e-5
                or abs(np.linalg.det(m[:3,:3])-1.)>1e-5):
            raise ValueError("Invalid sampled rigid transform")
        return _flat(m)

    def state(self):
        """Capture both actual arms, matched jaws, all brick transforms and owners."""
        state=self.sim.callScriptFunction("bridgeSnapshot",self.script)
        for row in state["parts"].values():
            row["pose"]=_matrix(row["pose"]).tolist()
            row["owner"]=row.get("owner")
            if row.get("binding"):
                row["binding"]=_matrix(row["binding"]).tolist()
        state["allow"]=[]
        state["measure_distance"]=True
        return state

    def apply_state(self,state,*,enforce_transfer=False):
        """Place a measured state. With enforcement, emit only zero-drift ownership changes."""
        return self.sim.callScriptFunction("bridgeApplyState",self.script,
                                           self._prepare_state(state),bool(enforce_transfer))

    def batch_check(self,states,*,requested_distance_m=.025,stop_at_failure=True,heartbeat=None,
                    batch_size=48,deadline=None,measure=None,filtered_clearance=False,all_collisions=False):
        """Sample exact native geometry in cancellable-size, restoring RPC batches."""
        if not 1 <= batch_size <= 64 or not 0 < requested_distance_m:
            raise ValueError("Invalid native batch bounds")
        clock=time.monotonic
        rows=[]
        first_failure=None
        timing={"rpc_seconds":0.,"serialization_seconds":0.,"log_seconds":0.}
        for start in range(0,len(states),batch_size):
            position=start
            stop=min(start+batch_size,len(states))
            while position<stop:
                if deadline is not None and clock()>=deadline:
                    return {"ok":False,"rows":rows,"first_failure":first_failure,
                            "error":"native_batch_deadline","timing":timing,"restored":True}
                begun=clock()
                payload=[self._prepare_state(row) for row in states[position:stop]]
                for row in payload:
                    if measure is not None:
                        row["measure_distance"]=bool(measure)
                    if filtered_clearance:
                        row["filtered_clearance"]=True
                    if all_collisions:
                        row["all_collisions"]=True
                timing["serialization_seconds"]+=clock()-begun
                begun=clock()
                work_seconds= min(6., max(1e-6, deadline-begun)) if deadline is not None else 6.
                result=self.sim.callScriptFunction("bridgeBatchSamples",self.script,payload,
                                                   float(requested_distance_m),bool(stop_at_failure),work_seconds)
                timing["rpc_seconds"]+=clock()-begun
                if not result.get("restored") or result.get("error"):
                    return {"ok":False,"rows":rows,"first_failure":first_failure,
                            "error":result.get("error","unverified_native_restore"),"timing":timing,
                            "restored":bool(result.get("restored"))}
                for name,value in result["timing"].items():
                    timing[name]=timing.get(name,0.)+value
                sampled=result["rows"]
                if not sampled:
                    raise RuntimeError("Native batch returned no rows")
                for row in sampled:
                    self._label_pairs(row)
                    rows.append(row)
                position+=len(sampled)
                if result.get("first_failure") and first_failure is None:
                    first_failure=position-len(sampled)+result["first_failure"]-1
                if heartbeat:
                    begun=clock()
                    heartbeat({"sampled":len(rows),"total":len(states),"first_failure":first_failure})
                    timing["log_seconds"]+=clock()-begun
                if first_failure is not None and stop_at_failure:
                    break
            if first_failure is not None and stop_at_failure:
                break
        expired = deadline is not None and clock()>=deadline
        result = {"ok":not expired and first_failure is None and len(rows)==len(states),"rows":rows,
                  "first_failure":first_failure,"timing":timing,"restored":True}
        if expired:
            result["error"]="native_batch_deadline"
        return result
