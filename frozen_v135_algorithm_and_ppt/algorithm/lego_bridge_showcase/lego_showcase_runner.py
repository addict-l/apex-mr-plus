"""Native stopped-scene motion, TPG and independent playback for the LEGO showcase."""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import math
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]/"diagnostics"/"tpg_broadphase_sim20_a1"/"candidate"
sys.path.insert(0,str(PARENT))

import skillnet
from apexmr import tpg as parent_tpg
from bridge_runtime import atomic_json, scene_session, transform
from bridge_geometry import BridgeWorld, CalibratedChain
from lego_grasp_model import deck_grasp, jaw_q, linear_targets, matrix, pier_taker_grasp, translate
from p0_2_height_taker import giver_pose, taker_pose
from p0_2_handover import part_sizes


DECK = "Bridge_deck_12x4"
DECK_HEIGHT_M = .002300
DECK_OPEN_M = .031840
PIER_OPEN_M = .021
DECK_JAW_OPEN_M = .040
HOVER_NEAR_M = .052
HOVER_FAR_M = .108
RRT_SECONDS = 60.
SIDES = ("A","B")
TCP_POSITION_M = 5e-4
TCP_ORIENTATION_RAD = 2e-3
CONTACT_POINT_STEP_M = .001
PARKING_DIRECTORY = HERE/"outboard_park_a1"
PARKING_POLICY = "v1-home-outboard-shared-corridor"


class NativeBlocker(RuntimeError):
    def __init__(self, message, **detail):
        super().__init__(message)
        self.detail = detail


def demand(condition, message, **detail):
    if not condition:
        raise NativeBlocker(message, **detail)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_digest(data):
    return sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def module_receipt(directory, plan):
    source = PARENT.parent/"isolation_receipt.json"
    pin = read(source)
    modules = {}
    for name, module in (("skillnet",skillnet),("apexmr.tpg",parent_tpg)):
        path = Path(module.__file__).resolve()
        demand(path.is_relative_to(PARENT), "loaded module escaped the frozen parent", module=name)
        rel = path.relative_to(PARENT).as_posix()
        data = {"path":str(path),"sha256":digest(path)}
        demand(pin["files"].get(rel)==data["sha256"],"loaded module SHA does not match parent",module=name)
        modules[name]=data
    demand(Path(sys.path[0]).resolve()==PARENT,"sys.path[0] must be the frozen parent",path=sys.path[0])
    input_path=Path(plan.get("task_path",directory/"task.toml")).resolve()
    demand(input_path.is_relative_to(HERE) and input_path.is_file() and digest(input_path)==plan.get("input_sha256"),
           "frozen mode input does not match the symbolic plan",task=str(input_path))
    plan_hash = digest(directory/"plan.json")
    atomic_json(directory/"motion_import_receipt.json",
        {"parent_fingerprint":pin["candidate_fingerprint"],"imports":modules,
         "path0":sys.path[0],"plan_sha256":plan_hash,"plan_input_sha256":plan.get("input_sha256"),
         "runner_sha256":digest(__file__)})
    return plan_hash


def parse_actions(plan):
    raw = plan.get("actions",plan.get("steps",[]))
    result = []
    for index,row in enumerate(raw):
        if not isinstance(row,dict):
            raise ValueError("plan actions must be dictionaries")
        token = row.get("action",row.get("text",row.get("label","")))
        skill = row.get("kind",row.get("skill",row.get("name")))
        if not isinstance(skill,str):
            skill = token.split("(",1)[0]
        params = row.get("args") or row.get("arguments")
        if not params and "(" in token:
            params = [s.strip() for s in token.split("(",1)[1][:-1].split(",")]
        params = list(params or ())
        part = row.get("part") or row.get("block")
        arm = row.get("arm")
        giver = row.get("giver")
        taker = row.get("taker")
        if skill in ("pick","place","insert"):
            arm = arm or (params[0] if len(params)>=1 else None)
            part = part or (params[1] if len(params)>=2 else None)
        elif skill == "handover":
            giver = giver or (params[0] if len(params)>=1 else None)
            taker = taker or (params[1] if len(params)>=2 else None)
            part = part or (params[2] if len(params)>=3 else None)
            arm = giver
        if skill=="insert":
            skill="place"
        result.append({"id":str(row.get("id",f"s{index+1:02d}")),"kind":skill,"part":part,
                       "arm":arm,"giver":giver,"taker":taker,
                       "dependencies":[str(v) for v in row.get("dependencies",[])],"symbolic":row})
    demand(result and all(r["kind"] in {"pick","place","handover"} and r["arm"] in SIDES
                          and isinstance(r["part"],str) for r in result),"unsupported symbolic action",
           actions=result)
    return result


def handover_settings(input_sha):
    from bridge_handover import HANDOVER_REVISION, require_handoff_bottom

    path = HERE/"mode_b"/"handover_gate.json"
    data = read(path)
    demand(data.get("passed") is True and data.get("handover_revision")==HANDOVER_REVISION
           and data.get("task_sha256")==input_sha,
           "Mode B gate did not qualify the same shared handover implementation")
    result = {}
    rows = data.get("directions",[])
    if isinstance(rows,dict):
        rows = list(rows.values())
    for row in rows:
        demand(row.get("passed") is True,"handover direction failed",direction=row)
        pick = row.get("candidate") or row.get("selected") or row.get("selected_candidate")
        if pick is None and row.get("route_attempts"):
            pick = next((attempt["candidate"] for attempt in row["route_attempts"] if attempt.get("passed")),None)
        key = row.get("id",row.get("direction"))
        demand(pick and key in {"A_to_B","B_to_A"},"missing passed native handover candidate",direction=key)
        try:
            require_handoff_bottom(matrix(pick["handover_bottom_m"]))
        except ValueError as error:
            raise NativeBlocker("handover candidate is outside the common hand_mid",direction=key) from error
        result[key]=pick
    demand(set(result)=={"A_to_B","B_to_A"},"Mode B requires two passed directions")
    return result,digest(path)


def part_owner(state, name):
    return state["parts"][name].get("owner")


def apply(world, state, *, transfer=False):
    result = world.apply_state(state,enforce_transfer=transfer)
    demand(not isinstance(result,dict) or result.get("ok",True),"native state application failed",state=state,native=result)
    return result


def state_for_step(state, side, q=None, jaw=None, *, distance=True, faces=()):
    snap = deepcopy(state)
    snap.pop("follower",None)
    snap.pop("expected_tcp",None)
    snap.pop("tcp_targets",None)
    if q is not None:
        snap["q"][side]=np.asarray(q,dtype=float).tolist()
    if jaw is not None:
        snap["jaw"][side]=float(jaw)
    snap["measure_distance"]=distance
    snap["faces"]=list(faces)
    snap["allow"]=[]
    return snap


def upward_elbow(chain, q):
    """Classify using the measured shoulder, elbow and first wrist axes."""
    joint_origins = []
    transform = np.eye(4)
    for point, axis, angle in zip(chain.points[:4], chain.axes[:4], q[:4]):
        basis = transform[:3, :3].copy()
        joint_origins.append(basis @ point + transform[:3, 3])
        turn = Rotation.from_rotvec(axis * angle).as_matrix()
        transform[:3, 3] += basis @ (point - turn @ point)
        transform[:3, :3] = basis @ turn
    shoulder, elbow, wrist = joint_origins[1:]
    span = wrist - shoulder
    length_squared = float(span @ span)
    if length_squared <= 1e-12:
        return False
    projection = np.clip(((elbow - shoulder) @ span) / length_squared, 0., 1.)
    return bool(elbow[2] - (shoulder + projection * span)[2] > 1e-6)


def pair_failure(reply):
    fail = reply.get("first_failure")
    if isinstance(fail,int):
        rows=reply.get("rows",[])
        return rows[fail] if 0 <= fail < len(rows) else fail
    return fail or reply


class SharedPassage:
    """Classify a physical ingress in the frozen board frame."""
    def __init__(self, world):
        board=world.manifest["board"]["handle"]
        dimensions,box=world.sim.getShapeBB(board)
        self.plate_from_world=np.linalg.inv(transform(world.sim,board) @ matrix(box))
        self.half=np.asarray(dimensions[:2],dtype=float)*.5
        self.arm_sign={}
        for side in SIDES:
            mount=world.arms[side]["mount"]["handle"]
            self.arm_sign[side]=int(np.sign(self.local(transform(world.sim,mount)[:3,3])[0]))
        demand(set(self.arm_sign.values())=={-1,1},"arm mounts do not straddle the native plate")

        positive=[]
        negative=[]
        for support in world.manifest["support_regions"]:
            center=self.local(support["center_m"])
            low,high=support["bounds_xy_m"]
            corners=[self.local([x,y,support["center_m"][2]])[0]
                     for x,y in product((low[0],high[0]),(low[1],high[1]))]
            if center[0]>0.:
                positive.append(min(corners))
            else:
                negative.append(max(corners))
        demand(positive and negative and max(negative)<0.<min(positive),
               "the supports do not define an open central passage in the native plate frame")
        self.open_interval=(max(negative),min(positive))

    def local(self, point):
        return (self.plate_from_world @ np.r_[point,1.])[:3]

    def classify(self, side, target):
        x,y,_=self.local(matrix(target)[:3,3])
        inside=abs(x)<=self.half[0] and abs(y)<=self.half[1]
        crosses=inside and x*self.arm_sign[side]<0.
        shared=self.open_interval[0]<=x<=self.open_interval[1]
        if not shared and not crosses:
            return None
        return {"kind":"cross_table" if crosses else "shared_passage",
                "plate_target_xy_m":[float(x),float(y)],
                "shared_interval_x_m":list(self.open_interval)}


class MotionTrace:
    def __init__(self, world, mode, directory, actions, plan_hash, *,
                 parking_calibration, deck_direction=None, deck_precheck_sha=None,
                 deck_binding=None, fixture_context=False, handover_task_sha=None):
        self.world,self.mode,self.directory=world,mode,directory
        self.actions,self.plan_hash=actions,plan_hash
        self.trace=directory/"reference.json"
        self.since=time.monotonic()
        self.action_id=None
        self.probes=[]
        self.parking_attempts=set()
        self.initial=world.state()
        demand(parking_calibration.get("passed") is True
               and parking_calibration.get("scene_sha256")==world.manifest["scene_sha256"],
               "outboard parking calibration is absent or uses another scene")
        home=parking_calibration["home_q"]
        parks=parking_calibration["park_q"]
        for side in SIDES:
            initial=np.asarray(self.initial["q"][side],dtype=float)
            park=np.asarray(parks[side],dtype=float)
            chain=world.chains[side]
            demand(initial.shape==(6,) and np.max(np.abs(initial-home[side]))<=1e-6
                   and park.shape==(6,) and np.isfinite(park).all()
                   and np.all(park>=chain.lower) and np.all(park<=chain.upper)
                   and np.max(np.abs(initial-park))>1e-6,
                   "calibrated outboard target does not match this arm or its original home",arm=side)
        self.park_policy={"version":PARKING_POLICY,"home_q":deepcopy(home),"outboard_park_q":deepcopy(parks),
                          "calibration_sha256":parking_calibration.get("_receipt_sha256")
                          or json_digest(parking_calibration),"deck_direction":deck_direction,
                          "deck_precheck_sha256":deck_precheck_sha,"deck_binding":deck_binding,
                          "fixture_context":bool(fixture_context)}
        demand(not fixture_context or deck_direction in {-1,1} and deck_precheck_sha is None,
               "fixture allowance is limited to private deck qualification")
        self.deck_direction=deck_direction
        self.deck_binding=deck_binding
        self.fixture_context=fixture_context
        self.passage=SharedPassage(world)
        self.state=deepcopy(self.initial)
        self.tasks=[]
        self.committed=[]
        self.bridge_started=False
        self.variants={}
        self.assists={}
        if mode=="mode_b":
            self.assists,handover_receipt_sha=handover_settings(handover_task_sha)
            self.park_policy["handover_gate_sha256"]=handover_receipt_sha
        self.assist_actions={r["part"]:r for r in actions if r["kind"]=="handover"}
        self.dims={name:part_sizes(world,name, dimensions=(.0958,.0318,.0032) if name==DECK else (.0158,.0158,.016))
                   for name in world.parts}

    def receipt(self, status_name, **extra):
        atomic_json(self.trace,{"status":status_name,"plan_sha256":self.plan_hash,
                    "scene_sha256":self.world.manifest["scene_sha256"],"park_policy":self.park_policy,
                    "initial":self.initial,"current":self.state,"tasks":self.tasks,
                    "actions_committed":self.committed,"probes":self.probes,
                    "grasp_variants":self.variants,"compile_wall_seconds":time.monotonic()-self.since,**extra})

    def ping(self, **extra):
        atomic_json(self.directory/"motion_progress.json",{"at":time.time(),"steps":len(self.committed),
                    "action_id":self.action_id,"primitive_count":len(self.tasks),**extra})

    def query(self, states, label):
        if not states:
            return {"ok":True,"rows":[]}
        out=self.world.batch_check(states,stop_at_failure=True,measure=False,
                     heartbeat=lambda *args,**kwargs:self.ping(phase=label,samples=len(states)))
        fail_index=out.get("first_failure")
        failed_state=states[fail_index] if isinstance(fail_index,int) and 0 <= fail_index < len(states) else None
        demand(out["ok"],"native trajectory rejected",primitive=label,native=pair_failure(out),
               first_failure=fail_index,failed_state=failed_state,last_safe_state=self.state)
        return out

    def add(self, side, part, label, states, *, effect=None, near=True):
        demand(bool(states),"cannot add empty motion")
        before=deepcopy(self.state)
        demand(states[0]["q"][side]==before["q"][side] and states[0]["jaw"][side]==before["jaw"][side],
               "primitive starts from a different arm or jaw state",primitive=label)
        witness=self.query(states,label)
        after=deepcopy(states[-1] if effect is None else effect)
        self.query([after],label+"_final")
        apply(self.world,after,transfer=effect is not None)
        self.state=after
        identifier=f"m{len(self.tasks)+1:04d}"
        q_path=[row["q"][side] for row in states]
        jaw_path=[row["jaw"][side] for row in states]
        points=[{"q":q,"jaw":jaw,"faces":row.get("faces",[]),"measure_distance":row.get("measure_distance",near)}
                for q,jaw,row in zip(q_path,jaw_path,states)]
        phase_time=[]
        for l,r in zip(points,points[1:]):
            q_delta=max(abs(a-b) for a,b in zip(l["q"],r["q"]))
            jaw_delta=abs(l["jaw"]-r["jaw"])
            phase_time.append(max(.004,q_delta/.85,jaw_delta/.012))
        if not phase_time:
            phase_time=[.04];points.append(deepcopy(points[0]))
        times=np.r_[0.,np.cumsum(phase_time)].tolist()
        deps=[]
        if self.action_id is not None and not any(t["metadata"]["symbolic_id"]==self.action_id for t in self.tasks):
            logical=next(r for r in self.actions if r["id"]==self.action_id)
            for needs in logical["dependencies"]:
                earlier=[t for t in self.tasks if t["metadata"]["symbolic_id"]==needs]
                demand(earlier,"symbolic dependency missing from serial plan",action=logical)
                anchor=earlier[-1]
                if (anchor["metadata"]["primitive"]=="post_place_park" and anchor["arm"]!=side
                        and label!="idle_endpoint_park"):
                    # The symbolic place already completed at retreat. Its
                    # separate park still owns its arm and gets geometric edges.
                    anchor=earlier[-2]
                deps.append(anchor["id"])
        if self.tasks and self.tasks[-1]["metadata"]["symbolic_id"]==self.action_id:
            deps.append(self.tasks[-1]["id"])
        effect_record = None
        if effect is not None:
            effect_record = {"part":part,"from":part_owner(before,part),"to":part_owner(after,part),
                             "before":before["parts"][part],"after":after["parts"][part]}
        self.tasks.append({"id":identifier,"arm":side,"part":part,"kind":"transit" if any(
                       q!=q_path[0] for q in q_path) else label, "duration":times[-1],
                       "dependencies":list(dict.fromkeys(deps)),"reference_order":len(self.tasks),
                       "q_path":q_path, "metadata":{"primitive":label,"symbolic_id":self.action_id,
                       "state_before":before,"state_after":after,"samples":points,"times":times,
                       "jaw_width_start_m":.006+2*jaw_path[0],"jaw_width_end_m":.006+2*jaw_path[-1],
                       "follower_mode":self.world.arms[side].get("jaw_driver_mode"),
                       "effect":effect_record,"distance_mode":"not_measured",
                       "witness_timing":witness.get("timing")}})
        self.ping(phase=label,status="passed")

    @staticmethod
    def faces(side,part,axis):
        return [{"side":side,"part":part,"axis":axis}]

    def holding_faces(self, state=None):
        current=state or self.state
        return [face for part,row in current["parts"].items() if row.get("owner") in SIDES
                for face in self.faces(row["owner"],part,self.axis(part,row["owner"]))]

    def axis(self,part,side):
        assist=self.assist_actions.get(part)
        return 1 if part==DECK or assist and side==assist["giver"] else 0

    def width(self,part,axis):
        return DECK_OPEN_M if part==DECK else float(self.dims[part][axis])+4e-6

    def open_width(self,part):
        return DECK_JAW_OPEN_M if part==DECK else PIER_OPEN_M

    def tcp_target(self,side,part,bottom,direction):
        if part==DECK:
            return deck_grasp(bottom,direction=direction,height_m=DECK_HEIGHT_M)
        assist=self.assist_actions.get(part)
        if assist:
            cfg=self.assists[f'{assist["giver"]}_to_{assist["taker"]}']
            if side==assist["giver"]:
                return giver_pose(bottom,cfg["tilt_deg"], 1 if side=="A" else -1,
                                  cfg["giver_height_m"],cfg["giver_outboard_m"])
            pose=taker_pose(bottom,cfg["taker_tilt_deg"],cfg["taker_offset_m"],cfg["taker_height_m"])
            tb=np.asarray(self.world.manifest["parts"][part]["target_bottom_m"],dtype=float)
            for other,other_state in self.state["parts"].items():
                if other==part or other_state.get("owner") is not None:
                    continue
                ob=np.asarray(self.world.manifest["parts"][other]["target_bottom_m"],dtype=float)
                seated=np.linalg.norm(np.asarray(other_state["pose"],dtype=float)[:3,3]-ob)<0.005
                if (seated and abs(ob[0]-tb[0])<0.01 and 0.005<(ob[1]-tb[1])<0.025
                        and abs(ob[2]-tb[2])<0.01):
                    bm=matrix(bottom)
                    pose=bm@np.diag([-1.,-1.,1.,1.])@(np.linalg.inv(bm)@pose)
                    break
            return pose
        return pier_taker_grasp(bottom,direction=direction)

    def jaw_path(self,side,part,to_q,label,*,contact_at_end=False,contact_at_start=False,axis=None):
        before=self.state["jaw"][side]
        n=max(1,math.ceil(abs(to_q-before)/.00022))
        rows=[]
        for i in range(n+1):
            value=before+(to_q-before)*i/n
            faces=self.holding_faces()
            if (i==0 and contact_at_start) or (i==n and contact_at_end):
                faces=[*faces,*self.faces(side,part,self.axis(part,side) if axis is None else axis)]
            rows.append(state_for_step(self.state,side,jaw=value,faces=faces))
        self.add(side,part,label,rows)

    def switch_owner(self, side, part, owner, label, *, extra_faces=()):
        before=deepcopy(self.state)
        part_pos=matrix(self.world.state()["parts"][part]["pose"])
        new=deepcopy(self.state)
        new["parts"][part]={"pose":part_pos.tolist(),"owner":owner}
        if owner is not None:
            tcp=self.world.tcp(owner)
            new["parts"][part]["binding"]=(np.linalg.inv(tcp)@part_pos).tolist()
        faces=self.holding_faces(new)
        new["faces"]=merge_faces(faces,extra_faces)
        new["measure_distance"]=True
        before["faces"]=merge_faces(self.holding_faces(),extra_faces)
        before["measure_distance"]=True
        self.add(side,part,label,[before],effect=new)
        error=CalibratedChain.pose_error(matrix(self.world.state()["parts"][part]["pose"]),part_pos)
        demand(error[0]<=1e-5 and error[1]<=2e-4,"ownership change moved a part",part=part,error=error)
        self.tasks[-1]["metadata"]["measured_transfer_error"]={"m":error[0],"rad":error[1]}

    def precise_q(self,side,q,target):
        chain=self.world.chains[side]
        q=np.asarray(q,dtype=float).copy()
        for _ in range(4):
            actual,jacobian=chain.fk(q)
            error=np.r_[target[:3,3]-actual[:3,3],
                Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).as_rotvec()]
            if np.linalg.norm(error) <= 1e-11:
                break
            correction=np.linalg.lstsq(jacobian,error,rcond=1e-9)[0]
            if not np.isfinite(correction).all() or np.max(np.abs(correction)) > .005:
                break
            nxt=q+correction
            if np.any(nxt<chain.lower) or np.any(nxt>chain.upper):
                break
            q=nxt
        return q

    def near_qs(self,side,start_q,start_pose,target):
        # Precheck TCP and conservatively budget rotation of 220 mm offset
        # tooling/held corners. Native FK is separately required per sample.
        q=np.asarray(start_q,dtype=float)
        out=[q.tolist()]
        poses=[start_pose]
        for at in linear_targets(start_pose,target,step_m=.004,angle_step_rad=.02):
            sols=self.world.ik(side,at,seed=q,max_solutions=1,seed_count=1)
            demand(bool(sols),"continuous native-pose IK has no branch",arm=side,target=at.tolist())
            following=self.precise_q(side,sols[0],at)
            demand(np.max(np.abs(following-q))<=.25,"near motion has a joint branch jump",
                   arm=side,q=following.tolist(),q_before=q.tolist())
            out.append(following.tolist())
            poses.append(at)
            q=following
        return out,poses

    def rows_for(self,side,qs,*,near=True,poses=None,state=None):
        given=state or self.state
        faces=self.holding_faces(given)
        rows=[]
        for i,q in enumerate(qs):
            row=state_for_step(given,side,q=q,distance=near,faces=faces)
            if poses is not None:
                row["expected_tcp"]={side:np.asarray(poses[i]).tolist()}
            rows.append(row)
        return rows

    def free(self,side,part,goals,label,*,corridor=None):
        from bridge_motion import MAX_JOINT_SAMPLE_RAD, connect_free
        plan=connect_free(self.world,side,self.state,goals,timeout_s=RRT_SECONDS,
                          seed=len(self.tasks)+721,heartbeat=lambda *a,**k:self.ping(phase=label),
                          via_qs=(self.park_q(side,"home"),self.park_q(side,"outboard")))
        if not plan["ok"]:
            self.probes.append({"action":self.action_id,"primitive":label,"result":plan})
        demand(plan["ok"],"bounded native-mesh free-space RRT failed",arm=side,part=part,primitive=label,
               native=plan,last_safe_state=self.state)
        qs=[list(q) for q in plan["q_path"]]
        if qs[0]!=self.state["q"][side]:
            demand(np.max(np.abs(np.asarray(qs[0])-self.state["q"][side]))<1e-7,
                   "free-space certificate begins at another state",q0=qs[0])
            qs[0]=self.state["q"][side]
        demand(all(max(abs(a-b) for a,b in zip(l,r))<=MAX_JOINT_SAMPLE_RAD+1e-7 for l,r in zip(qs,qs[1:])),
               "free-space certificate exceeds 0.02 rad sampling")
        self.add(side,part,label,self.rows_for(side,qs,near=False),near=False)
        if corridor is not None:
            task=self.tasks[-1]
            task["metadata"]["corridor"]=corridor
            anchor=corridor.get("yield_primitive_id")
            if anchor is not None and anchor not in task["dependencies"]:
                task["dependencies"].append(anchor)

    def park_q(self, side, target):
        demand(target in {"home","outboard"},"unknown parking target",arm=side,target=target)
        key="home_q" if target=="home" else "outboard_park_q"
        return self.park_policy[key][side]

    def at_park(self, side, target):
        return bool(np.max(np.abs(np.asarray(self.state["q"][side])-self.park_q(side,target)))<=1e-6)

    def park(self, side, target, label, *, requested_by=None):
        demand(not any(piece.get("owner") == side for piece in self.state["parts"].values()),
               "an arm cannot park while holding a part", arm=side, primitive=label)
        joints=self.park_q(side,target)
        self.free(side,None,[joints],label)
        demand(self.at_park(side,target),"park path did not reach its fixed target",
               arm=side,target=target,primitive=label)
        self.tasks[-1]["metadata"]["parking"]={"target":target,"target_q":joints,
                                                  "requested_by":requested_by}

    def prepare_shared_entry(self, side, target, *, permit_yield=True):
        corridor=self.passage.classify(side,target)
        if corridor is None:
            return None
        other="B" if side=="A" else "A"
        corridor["yield_arm"]=other
        if not permit_yield or any(piece.get("owner")==other for piece in self.state["parts"].values()):
            corridor["yield_primitive_id"]=None
            corridor["yield_condition"]="opposite_arm_participating_or_holding"
            return corridor
        # The first ingress cannot treat calibration as a placed robot. The
        # idle arm at original home must traverse a certified live path.
        if self.at_park(other,"home"):
            self.park(other,"outboard","shared_ingress_idle_park",requested_by=side)
        if self.at_park(other,"outboard"):
            anchor=next((task["id"] for task in reversed(self.tasks) if task["arm"]==other
                         and task["metadata"].get("parking",{}).get("target")=="outboard"),None)
            demand(anchor is not None or self.fixture_context,
                   "outboard pose has no preceding verified primitive",arm=other,ingress_arm=side)
            corridor["yield_primitive_id"]=anchor
            if anchor is None:
                corridor["yield_condition"]="native_verified_deck_fixture"
        else:
            corridor["yield_primitive_id"]=None
            corridor["yield_condition"]="opposite_arm_not_at_home_or_outboard"
        return corridor

    def recover_idle_endpoint(self, side, label, failure):
        other = "B" if side == "A" else "A"
        recovery_key = (self.action_id, side, label)
        if recovery_key in self.parking_attempts:
            return False
        if any(piece.get("owner") == other for piece in self.state["parts"].values()):
            return False
        if self.at_park(other,"outboard"):
            return False
        eligible = [row for row in failure.detail.get("endpoint_candidates", []) if row["upward_elbow"]]
        if not eligible:
            return False
        actual = self.world.state()
        if (any(piece.get("owner") == other for piece in actual["parts"].values())
                or np.max(np.abs(np.asarray(actual["q"][other]) - self.state["q"][other])) > 1e-7):
            return False
        states = [state_for_step(self.state, side, row["q"], distance=False,
                                 faces=self.holding_faces()) for row in eligible]
        answer = self.world.batch_check(states, stop_at_failure=False, measure=False, all_collisions=True)
        rows = answer.get("rows", [])
        if answer.get("error") or answer.get("restored") is not True or len(rows) != len(states):
            return False

        def is_interarm(collision):
            pair = collision.get("pair_handles", [])
            if len(pair) != 2:
                return False
            roles = [self.world.handle_roles.get(handle, {}) for handle in pair]
            return ({item.get("side") for item in roles} == {side, other}
                    and all(item.get("role") in {"link", "tool"} for item in roles))

        exclusive = all(row.get("kind") == "native_collision" and row.get("collisions")
                        and all(is_interarm(pair) for pair in row["collisions"]) for row in rows)
        probe = {"action":self.action_id,"primitive":label,"parking_arm":other,
                 "eligible_endpoint_states":states,"native_endpoints":rows,
                 "exclusively_blocked_by_idle_arm":bool(exclusive)}
        self.probes.append(probe)
        if not exclusive:
            return False
        self.parking_attempts.add(recovery_key)
        self.park(other,"outboard","idle_endpoint_park",requested_by=side)
        probe["parking_primitive_id"] = self.tasks[-1]["id"]
        return True

    def choose_contact_column(self,side,part,target,label,*,heights=(HOVER_NEAR_M,HOVER_FAR_M),extra_seed=None):
        # Solve from the exact closed-pose target outward, then reverse the
        # checked dense column for descent. Try both elevations and multiple
        # IK windings from the same actual assembled world.
        here=np.asarray(self.state["q"][side],dtype=float)
        options=self.world.ik(side,target,seed=here,max_solutions=8,seed_count=30)
        if extra_seed is not None:
            exact=self.world.ik(side,target,seed=np.asarray(extra_seed,dtype=float),
                                max_solutions=1,seed_count=1)
            if exact and not any(float(np.max(np.abs(exact[0]-q)))<.001 for q in options):
                options=[exact[0],*options]
        ascending=[]
        endpoint_candidates=[]
        for index,q in enumerate(options):
            q=self.precise_q(side,q,target)
            endpoint_candidates.append({"q":q.tolist(),"upward_elbow":upward_elbow(self.world.chains[side],q)})
            for height in heights:
                try:
                    qs,poses=self.near_qs(side,q,target,translate(target,[0.,0.,height]))
                    checks=self.rows_for(side,qs,poses=poses)
                    native=self.query(checks,label+"_probe")
                except NativeBlocker as exc:
                    self.probes.append({"action":self.action_id,"primitive":label,"candidate":index,
                          "height_m":height,"q_contact":q.tolist(),"failure":str(exc),"detail":exc.detail})
                    continue
                ascending.append({"qs":qs,"poses":poses,"rank":index,"height_m":height,
                                  "native_cleared":len(native.get("rows",[]))})
                break
            if len(ascending)>=3:
                break
        demand(ascending,"all exact contact columns fail native geometry/IK",action=self.action_id,
               primitive=label,part=part,arm=side,ik_candidates=len(options),
               endpoint_candidates=endpoint_candidates,probes=self.probes[-10:])
        return ascending

    def enter_contact(self,side,part,target,label,*,heights=(HOVER_NEAR_M,HOVER_FAR_M),extra_seed=None):
        corridor=self.prepare_shared_entry(side,target)
        try:
            columns=self.choose_contact_column(side,part,target,label,heights=heights,extra_seed=extra_seed)
        except NativeBlocker as error:
            if not self.recover_idle_endpoint(side,label,error):
                raise
            corridor=self.prepare_shared_entry(side,target)
            columns=self.choose_contact_column(side,part,target,label,heights=heights,extra_seed=extra_seed)
        self.free(side,part,[col["qs"][-1] for col in columns],label+"_free_join",corridor=corridor)
        selected=min(columns,key=lambda col: np.linalg.norm(np.asarray(self.state["q"][side])-col["qs"][-1]))
        demand(np.max(np.abs(np.asarray(self.state["q"][side])-selected["qs"][-1]))<1e-7,
               "free-space connector did not end at a checked contact column",phase=label)
        qs=[self.state["q"][side],*reversed(selected["qs"][:-1])]
        poses=list(reversed(selected["poses"]))
        self.add(side,part,label+"_vertical",self.rows_for(side,qs,poses=poses))
        observed=self.world.tcp(side)
        ep,er=CalibratedChain.pose_error(observed,target)
        demand(ep<=TCP_POSITION_M and er<=TCP_ORIENTATION_RAD,"native endpoint misses exact contact target",
               arm=side,primitive=label,position_error_m=ep,orientation_error_rad=er)
        return selected

    def retreat_column(self,side,part,column,label):
        qs=[self.state["q"][side],*column["qs"][1:]]
        self.add(side,part,label,self.rows_for(side,qs,poses=column["poses"]))

    def direct_near(self,side,part,target,label,*,extra_faces=()):
        start=self.world.tcp(side)
        qs,poses=self.near_qs(side,self.state["q"][side],start,target)
        rows=self.rows_for(side,qs,poses=poses)
        if extra_faces:
            for row in rows:
                row["faces"]=[*row["faces"],*extra_faces]
        self.add(side,part,label,rows)

    def pick(self, action):
        part,side=action["part"],action["arm"]
        demand(part_owner(self.state,part) is None,"picked an already-held part",part=part)
        self.jaw_path(side,part,jaw_q(self.open_width(part)),"prepare_open")
        original=self.world.bottom(part)
        start_tasks,start_state=len(self.tasks),deepcopy(self.state)
        directions=(self.deck_direction,) if part==DECK else (1,-1)
        demand(part!=DECK or self.deck_direction in {-1,1},
               "deck action requires the signed booth-side binding",arm=side)
        for direction in directions:
            try:
                grasp=self.tcp_target(side,part,original,direction)
                column=self.enter_contact(side,part,grasp,"approach_pick")
                axis=self.axis(part,side)
                self.jaw_path(side,part,jaw_q(self.width(part,axis)),"close_pick",contact_at_end=True)
                self.switch_owner(side,part,side,"attach_at_pick",extra_faces=self.faces(side,part,axis))
                self.require_deck_binding(part)
                self.retreat_column(side,part,column,"lift_held")
                self.variants[part]=direction
                return
            except NativeBlocker as error:
                self.probes.append({"action":self.action_id,"primitive":"pick_variant","direction":direction,
                                    "failure":str(error),"detail":error.detail,
                                    "checked_prefix":self.tasks[start_tasks:]})
                del self.tasks[start_tasks:]
                self.state=deepcopy(start_state)
                apply(self.world,self.state)
                if direction==directions[-1]:
                    raise

    def require_deck_binding(self, part):
        if part!=DECK or self.deck_binding is None:
            return
        current=matrix(self.state["parts"][part]["binding"])
        error=CalibratedChain.pose_error(current,matrix(self.deck_binding))
        demand(error[0]<=1e-5 and error[1]<=TCP_ORIENTATION_RAD,
               "active deck binding differs from the qualified booth-side binding",
               position_error_m=error[0],orientation_error_rad=error[1])

    def place(self,action):
        part,side=action["part"],action["arm"]
        demand(part_owner(self.state,part)==side,"symbolic placing arm does not own the piece",
               part=part,arm=side,owner=part_owner(self.state,part))
        self.require_deck_binding(part)
        target_bottom=matrix(self.world.manifest["parts"][part]["target_bottom_m"])
        target_pose=target_bottom @ self.world.offsets[part] @ np.linalg.inv(matrix(self.state["parts"][part]["binding"]))
        column=self.enter_contact(side,part,target_pose,"approach_place")
        pos,rot=CalibratedChain.pose_error(self.world.bottom(part),target_bottom)
        demand(pos<=1e-5 and rot<=TCP_ORIENTATION_RAD,"native seating misses completed geometry",
               part=part,position_error_m=pos,orientation_error_rad=rot)
        self.switch_owner(side,part,None,"detach_at_place",extra_faces=self.faces(side,part,self.axis(part,side)))
        self.jaw_path(side,part,jaw_q(self.open_width(part)),"open_placed",contact_at_start=True)
        self.retreat_column(side,part,column,"retreat_open")
        self.park(side,"outboard","post_place_park")

    def handover(self,action):
        from bridge_handover import execute_handoff, verify_handoff_withdrawal

        part,giver,taker=action["part"],action["giver"],action["taker"]
        demand(part_owner(self.state,part)==giver,"handover giver owns no part",action=action)
        cfg=self.assists[f"{giver}_to_{taker}"]
        bottom=matrix(cfg["handover_bottom_m"])
        target=bottom@self.world.offsets[part]@np.linalg.inv(matrix(self.state["parts"][part]["binding"]))
        destination=self.tcp_target(taker,part,bottom,1)
        plain_taker=taker_pose(bottom,cfg["taker_tilt_deg"],cfg["taker_offset_m"],cfg["taker_height_m"])
        taker_seed=None if not np.allclose(destination,plain_taker) else cfg["qs_rad"][taker]

        def arrive_taker(hover, grip):
            offset=hover[:3,3]-grip[:3,3]
            demand(abs(offset[0])+abs(offset[1])<=1e-10 and abs(offset[2]-.025)<=1e-10,
                   "handover helper did not specify the required 25 mm vertical entry")
            self.jaw_path(taker,part,jaw_q(PIER_OPEN_M),"taker_open")
            self.enter_contact(taker,part,grip,"taker_handoff_insert",heights=(float(offset[2]),),
                               extra_seed=taker_seed)

        def transfer():
            self.jaw_path(taker,part,jaw_q(self.width(part,0)),"taker_close",contact_at_end=True,axis=0)
            faces=[*self.faces(giver,part,1),*self.faces(taker,part,0)]
            self.switch_owner(taker,part,taker,"handover_atomic",extra_faces=faces)

        try:
            clearance=execute_handoff(
                bottom,target,destination,
                giver_arrive=lambda pose:self.enter_contact(giver,part,pose,"giver_to_handoff",
                                                            extra_seed=cfg["qs_rad"][giver]),
                taker_arrive=arrive_taker,transfer=transfer,
                giver_open=lambda:self.jaw_path(giver,part,jaw_q(PIER_OPEN_M),"giver_open",
                                               contact_at_start=True,axis=1),
                giver_retreat=lambda pose,label:self.direct_near(giver,part,pose,label),
                verify_withdrawal=lambda:verify_handoff_withdrawal(self.world),
                actual_bottom=lambda:self.world.bottom(part))
        except ValueError as error:
            raise NativeBlocker("shared handover failed strict position or withdrawal checks",
                                action=self.action_id,part=part,detail=str(error)) from error
        self.tasks[-1]["metadata"]["handover_withdrawal"]=clearance

    def compile(self):
        if self.trace.is_file():
            previous=read(self.trace)
            demand(previous.get("scene_sha256")==self.world.manifest["scene_sha256"],
                   "checkpoint uses another physical scene")
            demand(previous["plan_sha256"]==self.plan_hash,
                   "checkpoint has a different symbolic plan")
            demand(previous.get("park_policy")==self.park_policy,
                   "checkpoint uses a different home/outboard or calibrated deck policy")
            self.state=previous["current"]
            self.tasks=previous["tasks"]
            self.committed=previous["actions_committed"]
            self.probes=previous.get("probes",[])
            self.variants=previous.get("grasp_variants",{})
            apply(self.world,self.state)
            if previous.get("status")=="complete":
                return previous
        self.receipt("planning")
        for action in self.actions[len(self.committed):]:
            self.action_id=action["id"]
            demand(action["part"] in self.world.parts,"symbolic part does not occur in scene",action=action)
            print(f'{self.mode}: {len(self.committed)+1}/{len(self.actions)} {action["kind"]} '
                  f'{action["arm"]} {action["part"]}',flush=True)
            before=deepcopy(self.state)
            start=len(self.tasks)
            try:
                getattr(self,action["kind"])(action)
            except NativeBlocker as error:
                prefix=self.tasks[start:]
                del self.tasks[start:]
                self.state=before
                failure={"message":str(error),"detail":error.detail,"action":action,
                         "replayable_prefix":prefix}
                try:
                    apply(self.world,before)
                except Exception as rollback_error:
                    failure["rollback_error"]=str(rollback_error)
                    self.receipt("blocked",failure=failure)
                    raise rollback_error from error
                failure["rollback_restored"]=True
                self.receipt("blocked",failure=failure)
                raise
            self.committed.append({"action":action,"trajectory_ids":[t["id"] for t in self.tasks[start:]],
                                   "native_state":self.world.state()})
            self.receipt("planning")
        self.action_id=None
        for side in SIDES:
            if self.at_park(side,"home"):
                continue
            try:
                self.park(side,"home","return_to_park")
            except NativeBlocker as error:
                self.receipt("blocked",failure={"message":str(error),"detail":error.detail,"stage":"return_to_park"})
                raise
        self.receipt("complete")
        return read(self.trace)



def _stride_samples(samples,max_n=8):
    if len(samples)<=max_n:
        return samples
    indices=[int(round(i*(len(samples)-1)/(max_n-1))) for i in range(max_n)]
    return [samples[index] for index in indices]


class RealConflict:
    """Sample complete paired configurations, using Type-2 for unknown worlds."""
    def __init__(self,world,folder,tasks=None,forced_conflicts=None):
        self.world,self.folder=world,folder
        self.cache={}
        self.requests=0
        self.expired=0
        self.samples=0
        self.began=time.monotonic()
        self.journal=folder/"tpg_oracle.jsonl"
        self.journal.write_text("",encoding="utf-8")
        self.forced_conflicts=set(forced_conflicts or ())
        self.task_order={task["id"]:index for index,task in enumerate(tasks or ())}
        self.part_epoch={}
        part_movers={}
        part_handover={}
        for index,task in enumerate(tasks or ()):
            effect=task.get("metadata",{}).get("effect")
            if not effect:
                continue
            part=effect.get("part",task.get("part"))
            part_movers.setdefault(part,set()).update(
                arm for arm in (effect.get("from"),effect.get("to")) if arm in SIDES)
            if task.get("metadata",{}).get("primitive")=="handover_atomic":
                part_handover[part]=(index,effect.get("from"),effect.get("to"))
        for part,movers in part_movers.items():
            if len(movers)==1:
                self.part_epoch[part]=("single",next(iter(movers)))
            elif len(movers)==2 and part in part_handover:
                h_idx,epoch_giver,epoch_taker=part_handover[part]
                self.part_epoch[part]=("handover",h_idx,epoch_giver,epoch_taker)

    def ping(self,*_args,**_kw):
        atomic_json(self.folder/"tpg_progress.json",{"at":time.time(),"oracle_calls":self.requests,
            "geometry_samples":self.samples,"conservative_timeout_calls":self.expired})

    def __call__(self,a,b):
        if ((a["id"],b["id"]) in self.forced_conflicts
                or (b["id"],a["id"]) in self.forced_conflicts):
            return self.finish(a,b,None,{"conflict":True,"evidence":{"scope":"cegar_replay_conflict"}})
        am,bm=a["metadata"],b["metadata"]
        sa,sb=am["state_before"],bm["state_before"]
        key=json_digest({"A":a["arm"],"B":b["arm"],"left":am,"right":bm})
        if key in self.cache:
            return self.cache[key]
        self.requests+=1
        self.ping()
        combined=deepcopy(sa)
        for name in sa["parts"]:
            p,r=sa["parts"][name],sb["parts"][name]
            if p==r:
                continue
            if name==a["part"]==b["part"]:
                return self.finish(a,b,key,{"conflict":True,"evidence":{"scope":"shared part can only have one owner"}})
            if p.get("owner")==a["arm"] and r.get("owner")!=b["arm"]:
                combined["parts"][name]=deepcopy(p)
            elif r.get("owner")==b["arm"] and p.get("owner")!=a["arm"]:
                combined["parts"][name]=deepcopy(r)
            elif p.get("owner") is None and r.get("owner") is None and np.max(
                    np.abs(matrix(p["pose"])-matrix(r["pose"])))<1e-8:
                combined["parts"][name]=deepcopy(p)
            elif p.get("owner") is None and r.get("owner") is None and name in self.part_epoch:
                epoch=self.part_epoch[name]
                chosen=None
                if epoch[0]=="single":
                    if epoch[1]==a["arm"]:
                        chosen=p
                    elif epoch[1]==b["arm"]:
                        chosen=r
                else:
                    _,h_idx,epoch_giver,epoch_taker=epoch
                    ia=self.task_order.get(a.get("task_id"))
                    ib=self.task_order.get(b.get("task_id"))
                    if ia is not None and ib is not None:
                        if ia<h_idx and ib<h_idx:
                            chosen=p if a["arm"]==epoch_giver else r if b["arm"]==epoch_giver else None
                        elif ia>h_idx and ib>h_idx:
                            chosen=p if a["arm"]==epoch_taker else r if b["arm"]==epoch_taker else None
                if chosen is None:
                    return self.finish(a,b,key,{"conflict":True,"evidence":{"scope":"different inactive part worlds",
                                                                           "part":name}})
                combined["parts"][name]=deepcopy(chosen)
            else:
                # Different inactive part locations have not been certified
                # for overlap. Order the pair without calling it a collision.
                return self.finish(a,b,key,{"conflict":True,"evidence":{"scope":"different inactive part worlds",
                                                                       "part":name}})
        checks=[]
        count=0
        deadline=self.began+2400.
        for l,r in product(_stride_samples(am["samples"],4),_stride_samples(bm["samples"],4)):
            if time.monotonic()>=deadline:
                self.expired+=1
                return self.finish(a,b,None,{"conflict":True,"evidence":{"scope":"uncertified overlap timeout",
                                                                      "completed_samples":count}})
            value=state_for_step(combined,a["arm"],l["q"],l["jaw"],
                distance=bool(l["measure_distance"] or r["measure_distance"]),
                faces=merge_faces(l["faces"],r["faces"]))
            value["q"][b["arm"]]=r["q"]
            value["jaw"][b["arm"]]=r["jaw"]
            checks.append(value)
            if len(checks)>=48:
                answer=self.world.batch_check(checks,stop_at_failure=True,heartbeat=self.ping,deadline=deadline,measure=False)
                count+=len(answer.get("rows",[]))
                self.samples+=len(answer.get("rows",[]))
                if not answer["ok"]:
                    if answer.get("error")=="native_batch_deadline":
                        self.expired+=1
                        return self.finish(a,b,None,{"conflict":True,"evidence":{"scope":"uncertified overlap timeout",
                                                                              "completed_samples":count}})
                    return self.finish(a,b,key,{"conflict":True,"evidence":{"scope":"native complete state",
                                        "samples":count,"first_failure":pair_failure(answer)}})
                checks=[]
        if checks:
            answer=self.world.batch_check(checks,stop_at_failure=True,heartbeat=self.ping,deadline=deadline,measure=False)
            count+=len(answer.get("rows",[]))
            self.samples+=len(answer.get("rows",[]))
            if not answer["ok"]:
                if answer.get("error")=="native_batch_deadline":
                    self.expired+=1
                    return self.finish(a,b,None,{"conflict":True,"evidence":{"scope":"uncertified overlap timeout",
                                                                          "completed_samples":count}})
                return self.finish(a,b,key,{"conflict":True,"evidence":{"scope":"native complete state",
                                        "samples":count,"first_failure":pair_failure(answer)}})
        return self.finish(a,b,key,{"conflict":False,"evidence":{"scope":"native complete state Cartesian product",
                                                               "samples":count}})

    def finish(self,a,b,key,answer):
        with self.journal.open("a",encoding="utf-8") as output:
            output.write(json.dumps({"left":a["id"],"right":b["id"],**answer},ensure_ascii=False,
                                    allow_nan=False)+"\n")
        if key is not None:
            self.cache[key]=answer
        return answer


def merge_faces(*sequences):
    out={}
    for values in sequences:
        for face in values:
            out[(face["side"],face["part"],face["axis"])]=face
    return list(out.values())


def intersect_faces(a,b):
    right={(row["side"],row["part"],row["axis"]) for row in b}
    return [row for row in a if (row["side"],row["part"],row["axis"]) in right]


def node_sample(row,now):
    meta=row["metadata"]
    knots=meta["times"]
    duration=row["end"]-row["start"]
    completed=duration<=1e-12 or now>=row["end"]-1e-9
    if completed:
        return meta["samples"][-1]
    at=(now-row["start"])/duration*knots[-1]
    index=max(0,min(len(knots)-2,int(np.searchsorted(knots,at,side="right")-1)))
    start,end=meta["samples"][index:index+2]
    alpha=max(0.,min(1.,(at-knots[index])/(knots[index+1]-knots[index])))
    faces=(start["faces"] if alpha<=1e-7 else end["faces"] if 1-alpha<=1e-7
           else intersect_faces(start["faces"],end["faces"]))
    return {"q":[x+(y-x)*alpha for x,y in zip(start["q"],end["q"])],
            "jaw":start["jaw"]+(end["jaw"]-start["jaw"])*alpha,"faces":faces,
            "measure_distance":start["measure_distance"] or end["measure_distance"]}


def timeline(tpg,world=None):
    from bridge_motion import MAX_JOINT_SAMPLE_RAD
    stamps={0.,float(tpg["nominal_makespan"])}
    for node in tpg["nodes"]:
        start,end,duration=node["start"],node["end"],node["end"]-node["start"]
        knots=node["metadata"]["times"]
        stamps.update(start+duration*value/knots[-1] for value in knots)
    unique=[]
    for value in sorted(stamps):
        if not unique or value-unique[-1]>1e-8:
            unique.append(value)
    output=[unique[0]]
    for value in unique[1:]:
        before=output[-1]
        subdivision=math.ceil((value-before)/.05)
        if world is not None:
            active=[node for node in tpg["nodes"] if node["end"]>before+1e-9 and node["start"]<value-1e-9]
            relative_bound=0.
            near=False
            for node in active:
                first=node_sample(node,max(before,node["start"]))
                last=node_sample(node,min(value,node["end"]))
                dq=np.max(np.abs(np.asarray(first["q"])-last["q"]))
                subdivision=max(subdivision,math.ceil(dq/MAX_JOINT_SAMPLE_RAD))
                near=near or first["measure_distance"] or last["measure_distance"]
                chain=world.chains[node["arm"]]
                pa,pb=chain.fk(first["q"])[0],chain.fk(last["q"])[0]
                dx,dr=CalibratedChain.pose_error(pa,pb)
                # Tool rails and the carried pieces lie within this TCP
                # radius. Budget opposing corner motion of both arms.
                relative_bound+=dx+.220*dr+abs(first["jaw"]-last["jaw"])
            if near:
                subdivision=max(subdivision,math.ceil(relative_bound/.00089))
        output.extend(before+(value-before)*i/subdivision for i in range(1,subdivision+1))
    return output


def validate_support(world,part):
    piece=world.parts[part]["handle"]
    initial=world.bottom(part)
    evidence=[]
    try:
        world.set_bottom(part,translate(initial,[0.,0.,.00002]))
        for contact in world.manifest["support_regions"]:
            lower=world.parts[contact["part"]]["handle"]
            collision,pair=world.sim.checkCollision(piece,lower)
            measured,data,_=world.sim.checkDistance(piece,lower,.001)
            separation=float(data[6]) if measured>0 else None
            passed=collision==0 and separation is not None and abs(separation-.00002)<=1e-5
            evidence.append({"part":contact["part"],"slot":contact["slot"],"clearance_m":separation,
                             "collision":bool(collision),"native_pair":pair,"passed":passed})
    finally:
        world.set_bottom(part,initial)
    demand(len(evidence)==4 and all(row["passed"] for row in evidence),
           "four independent deck support interfaces did not match",supports=evidence)
    return evidence


class NativeReplay:
    def __init__(self,world,reference,tpg,folder):
        self.world,self.reference,self.tpg,self.folder=world,reference,tpg,folder
        self.state=deepcopy(reference["initial"])
        self.samples=[]
        self.events=[]
        self.transfer=[]
        self.handover_withdrawals=[]
        self.key_clearances=[]
        self.pending=[]
        self.began=time.monotonic()
        self.ended=set()
        self.started=set()

    def ping(self,*_args,**_kwargs):
        atomic_json(self.folder/"replay_progress.json",{"at":time.time(),"samples_done":len(self.samples),
                            "events_done":len(self.events),"transfers_done":len(self.transfer)})

    def flush(self):
        if not self.pending:
            return
        states=[row["state"] for row in self.pending]
        out=self.world.batch_check(states,stop_at_failure=True,heartbeat=self.ping,measure=False)
        if not out["ok"]:
            index=out.get("first_failure")
            mark=self.pending[index] if isinstance(index,int) and 0<=index<len(self.pending) else self.pending[-1]
            demand(False,"independent replay rejected native scene",first_failure=pair_failure(out),
                   suspected_frame=mark,validated_sample_count=len(self.samples))
        apply(self.world,self.pending[-1]["state"])
        self.samples.extend(self.pending)
        self.pending=[]
        self.ping()

    def event(self,label,node,when):
        self.events.append({"event":label,"node_id":node["id"],"part":node.get("part"),
                            "arm":node["arm"],"kind":node["metadata"]["primitive"],"time_s":when})

    def clearance_kind(self, node, state):
        meta = node["metadata"]
        kind = meta["primitive"]
        if kind == "handover_atomic":
            return "handover_handshake"
        if kind != "detach_at_place":
            return None
        name = meta["effect"]["part"]
        if name == DECK:
            return "deck_final_seat"
        parts = self.world.manifest["parts"]
        part = parts[name]
        center = np.asarray(part["target_bottom_m"])
        width = np.asarray(part["bounds_max_m"]) - part["bounds_min_m"]
        for other, other_state in state["parts"].items():
            if other == name or other == DECK or other_state.get("owner") is not None:
                continue
            descriptor = parts[other]
            target = np.asarray(descriptor["target_bottom_m"])
            if abs(center[2] - target[2]) > 1e-5:
                continue
            bottom = matrix(other_state["pose"]) @ np.linalg.inv(self.world.offsets[other])
            if np.linalg.norm(bottom[:3, 3] - target) > 1e-5:
                continue
            other_width = np.asarray(descriptor["bounds_max_m"]) - descriptor["bounds_min_m"]
            gap = np.sort(np.abs(center[:2] - target[:2]) - .5 * (width[:2] + other_width[:2]))
            if gap[0] < 0. and 0. <= gap[1] <= .001:
                return "neighbor_final_seat"
        return None

    def switch(self,node,when,current):
        self.flush()
        effect=node["metadata"]["effect"]
        name=effect["part"]
        demand(part_owner(self.state,name)==effect["from"],"scheduled ownership differs from reference",
               node=node["id"],expected=effect["from"],actual=part_owner(self.state,name))
        # Bring both arms to the actual event boundary before touching parent.
        verified=self.world.batch_check([current],stop_at_failure=True,measure=False)
        demand(verified["ok"],"invalid exact pre-transfer state",node=node["id"],native=pair_failure(verified))
        key = self.clearance_kind(node, current)
        if key is not None:
            measured = self.world.batch_check([current], stop_at_failure=True,
                                              measure=True, filtered_clearance=True)
            demand(measured["ok"], "key frame failed native clearance query", node=node["id"],
                   clearance_kind=key, native=pair_failure(measured))
            self.key_clearances.append({"kind":key,"node":node["id"],"part":name,"time_s":when,
                                       "native":measured["rows"][0],"timing":measured["timing"]})
        apply(self.world,current)
        old=matrix(self.world.state()["parts"][name]["pose"])
        handover=node["metadata"]["primitive"]=="handover_atomic"
        if handover:
            from bridge_handover import require_handoff_bottom
            try:
                hand_miss=require_handoff_bottom(self.world.bottom(name))
            except ValueError as error:
                raise NativeBlocker("independent handover transfer is outside fixed hand_mid",
                                    node=node["id"],detail=str(error)) from error
        candidate=deepcopy(current)
        candidate["parts"][name]=deepcopy(effect["after"])
        candidate["faces"]=merge_faces(current.get("faces",[]),node["metadata"]["state_after"].get("faces",[]))
        apply(self.world,candidate,transfer=True)
        actual=matrix(self.world.state()["parts"][name]["pose"])
        ep,er=CalibratedChain.pose_error(actual,old)
        demand(ep<=1e-5 and er<=TCP_ORIENTATION_RAD,"native playback switch moved the brick",node=node["id"],
               position_error_m=ep,orientation_error_rad=er)
        post=self.world.batch_check([candidate],stop_at_failure=True,measure=False)
        demand(post["ok"],"invalid exact post-transfer state",node=node["id"],native=pair_failure(post))
        self.state=candidate
        self.transfer.append({"node":node["id"],"time_s":when,"part":name,"giver":effect["from"],
            "receiver":effect["to"],"position_error_m":ep,"orientation_error_rad":er,
            "kind":node["metadata"]["primitive"]})
        if handover:
            self.transfer[-1]["hand_mid_error_m"]=hand_miss
        return candidate

    def verify_withdrawal(self, node, when, state):
        from bridge_handover import verify_handoff_withdrawal
        self.flush()
        raw=self.world.batch_check([state],stop_at_failure=True,measure=False)
        demand(raw["ok"],"independent handover withdrawal collides",node=node["id"],native=pair_failure(raw))
        apply(self.world,state)
        try:
            native=verify_handoff_withdrawal(self.world)
        except ValueError as error:
            raise NativeBlocker("independent giver withdrawal does not leave positive clearance",
                                node=node["id"],detail=str(error)) from error
        self.handover_withdrawals.append({"node":node["id"],"time_s":when,"part":node["part"],
                                          "native":native})

    def run(self):
        sequence=sorted(self.tpg["nodes"],key=lambda row:row["reference_index"])
        apply(self.world,self.state)
        by_side={side:[node for node in sequence if node["arm"]==side] for side in SIDES}
        indices={side:0 for side in SIDES}
        ts=timeline(self.tpg,self.world)
        for now in ts:
            planned=deepcopy(self.state)
            planned.pop("follower",None)
            policies=[]
            endings=[]
            for side in SIDES:
                offset=indices[side]
                while offset<len(by_side[side]):
                    node=by_side[side][offset]
                    if now+1e-8 < node["start"]:
                        break
                    if node["id"] not in self.started:
                        self.event("start",node,node["start"])
                        self.started.add(node["id"])
                    state=node_sample(node,now)
                    if abs(now-node["start"])<=1e-8:
                        current_q=planned["q"][side]
                        current_jaw=planned["jaw"][side]
                        demand(np.max(np.abs(np.asarray(state["q"])-current_q))<1e-6
                               and abs(current_jaw-state["jaw"])<1e-8,
                               "scheduled motion does not join halted arm",node=node["id"])
                    planned["q"][side]=state["q"]
                    planned["jaw"][side]=state["jaw"]
                    policies.append(state)
                    if now+1e-8<node["end"]:
                        break
                    if node["id"] not in self.ended:
                        self.event("finish",node,node["end"])
                        self.ended.add(node["id"])
                        endings.append(node)
                    offset+=1
                indices[side]=offset
            planned["faces"]=merge_faces(*(row["faces"] for row in policies))
            # A waiting arm still owns its brick and must meet its hold mode.
            faces=deepcopy(planned["faces"])
            for part,mode in planned["parts"].items():
                side=mode.get("owner")
                if side not in SIDES:
                    continue
                axis=1 if part==DECK else next((r["axis"] for r in self.state.get("faces",[])
                                              if r["side"]==side and r["part"]==part),0)
                faces=merge_faces(faces,[{"side":side,"part":part,"axis":axis}])
            planned["faces"]=faces
            planned["measure_distance"]=not policies or any(row["measure_distance"] for row in policies)
            planned["allow"]=[]
            for node in sorted(endings,key=lambda row:row["reference_index"]):
                if node["metadata"]["effect"]:
                    planned=self.switch(node,now,planned)
                if node["metadata"]["primitive"]=="giver_retreat_vertical":
                    self.verify_withdrawal(node,now,planned)
            self.state=planned
            self.pending.append({"time_s":now,"state":planned})
            if len(self.pending)>=48:
                self.flush()
        self.flush()
        demand(len(self.ended)==len(sequence),"independent replay did not complete every node",
               expected=len(sequence),actual=len(self.ended))
        final=self.world.state()
        positions={}
        for name,part in self.world.manifest["parts"].items():
            wanted=matrix(part["target_bottom_m"])
            error=CalibratedChain.pose_error(self.world.bottom(name),wanted)
            positions[name]={"position_error_m":error[0],"orientation_error_rad":error[1],
                             "target_bottom_m":part["target_bottom_m"],
                             "observed_bottom":self.world.bottom(name).tolist()}
            demand(final["parts"][name].get("owner") is None and error[0]<=.0001 and error[1]<=TCP_ORIENTATION_RAD,
                   "final part does not match independent completed scene",part=name,result=positions[name])
        final_home=self.reference["park_policy"]["home_q"]
        parking={side:float(np.max(np.abs(np.asarray(final["q"][side])-final_home[side])))
                 for side in SIDES}
        demand(max(parking.values())<=1e-6,"arm did not return to the original parking configuration",errors=parking)
        supports=validate_support(self.world,DECK)
        handover_count=sum(row["kind"]=="handover_atomic" for row in self.transfer)
        demand(len(self.handover_withdrawals)==handover_count,
               "independent handovers lack a matching giver withdrawal check",
               handovers=handover_count,withdrawals=len(self.handover_withdrawals))
        maximum=max(edge["position_error_m"] for edge in positions.values())
        actual_seconds=self.tpg["nominal_makespan"]
        validation={"passed":True,"collisions":0,"unapproved_collisions":0,"part_positions":positions,
                    "maximum_part_position_error_m":maximum,"support_zones":supports,"parking_errors_rad":parking,
                    "home_errors_rad":parking,"handover_withdrawals":self.handover_withdrawals,
                    "handover_events":[entry for entry in self.transfer if entry["kind"]=="handover_atomic"],
                    "all_attachment_events":self.transfer,"native_samples":len(self.samples),
                    "sweep_clearance":"not_measured","key_clearances":self.key_clearances,
                    "maximum_time_step_s":max(np.diff(ts),default=0.),"reference_plan_sha256":self.reference["plan_sha256"],
                    "replay_wall_s":time.monotonic()-self.began,
                    "waiting_pose_collision_check":"passed on sampled synchronous timeline",
                    "scope":"native sampled stopped-scene synchronous validation"}
        replay={"passed":True,"initial":self.reference["initial"],"sample_states":self.samples,
                "events":sorted(self.events,key=lambda row:row["time_s"]),"seconds":actual_seconds,
                "scene_sha256":self.world.manifest["scene_sha256"],"validation":validation}
        atomic_json(self.folder/"replay.json",replay)
        atomic_json(self.folder/"validation.json",validation)
        return validation


def locked_parking_inputs(actions):
    path=PARKING_DIRECTORY/"parking_calibration.json"
    calibration=read(path)
    calibration["_receipt_sha256"]=digest(path)
    deck_path=PARKING_DIRECTORY/"precheck_deck.json"
    deck=read(deck_path)
    arms={row["arm"] for row in actions if row["part"]==DECK}
    arm=deck.get("selected_arm")
    direction=deck.get("selected_direction")
    demand(calibration.get("passed") is True,"outboard park native qualification has no pass")
    demand(deck.get("passed") is True and deck.get("scene_sha256")==calibration.get("scene_sha256")
           and arm in SIDES and arms=={arm} and type(direction) is int and direction in {-1,1},
           "symbolic deck arm or orientation does not match the sole native precheck selection",
           planned_arms=sorted(arms),qualified_arm=arm,qualified_direction=direction)
    binding=matrix(deck["selected_binding"])
    return {"parking_calibration":calibration,"deck_direction":direction,
            "deck_precheck_sha":digest(deck_path),"deck_binding":binding.tolist()}


def corridor_dag_evidence(reference, tpg, mode):
    tasks={task["id"]:task for task in reference["tasks"]}
    nodes={node["id"]:node["task_id"] for node in tpg["nodes"]}
    actual={(nodes[edge["source"]],nodes[edge["target"]])
            for edge in tpg["edges"] if edge["type"]=="type2" and edge["reason"]=="task_dependency"}
    directions=set()
    verified=[]
    for target in tasks.values():
        detail=target["metadata"].get("corridor")
        if not detail or detail.get("yield_primitive_id") is None:
            continue
        anchor=detail["yield_primitive_id"]
        source=tasks.get(anchor)
        demand(source is not None and source["arm"]!=target["arm"]
               and source["metadata"].get("parking",{}).get("target")=="outboard"
               and anchor in target["dependencies"] and (anchor,target["id"]) in actual,
               "shared ingress has no matching Type-2 outboard egress edge",target=target["id"],anchor=anchor)
        parked_arm=source["arm"]
        calibrated=reference["park_policy"]["outboard_park_q"][parked_arm]
        demand(np.max(np.abs(np.asarray(source["q_path"][-1])-calibrated))<=1e-6
               and np.max(np.abs(np.asarray(target["metadata"]["state_before"]["q"][parked_arm])-calibrated))<=1e-6,
               "corridor dependency does not match the calibrated yielding pose",
               target=target["id"],anchor=anchor,arm=parked_arm)
        directions.add((source["arm"],target["arm"]))
        verified.append({"source_primitive":anchor,"target_primitive":target["id"],
                         "source_arm":source["arm"],"target_arm":target["arm"],"kind":detail["kind"]})
    demand(tpg.get("graph_acyclic") is True,"shared passage TPG is cyclic")
    demand(mode!="mode_a" or directions=={("A","B"),("B","A")},
           "Mode A did not record both reciprocal corridor dependencies",directions=sorted(directions))
    return {"acyclic":True,"verified_type2":verified,"directions":[list(pair) for pair in sorted(directions)],
            "waiting_pose_collision_check":"independent synchronous replay required"}


def run(mode,phase):
    folder=HERE/mode
    folder.mkdir(exist_ok=True)
    plan=read(folder/"plan.json")
    plan_hash=module_receipt(folder,plan)
    actions=parse_actions(plan)
    demand(len(actions)==(18 if mode=="mode_a" else 22),"unexpected symbolic action count",count=len(actions))
    if mode=="mode_a":
        demand(not any(row["kind"]=="handover" for row in actions),"free reach plan has handover")
    else:
        demand(sum(row["kind"]=="handover" for row in actions)==4,"strict zones must have four handovers")
    planning_inputs=locked_parking_inputs(actions)
    started=time.monotonic()
    print(f"{mode}: native showcase {phase}",flush=True)
    label=mode if phase=="all" else f"{mode}_{phase}"
    session_callback=scene_session
    with session_callback(label) as (sim,client):
        world=BridgeWorld(sim,client,batch_joints=True)
        if phase in {"compile","plan","all"}:
            trace=MotionTrace(world,mode,folder,actions,plan_hash,
                              handover_task_sha=plan["input_sha256"],**planning_inputs)
            reference=trace.compile()
        else:
            reference=read(folder/"reference.json")
        demand(reference.get("status")=="complete","reference is incomplete")
        demand(reference["plan_sha256"]==plan_hash and reference["scene_sha256"]==world.manifest["scene_sha256"],"reference scene or symbolic file differs")
        policy=reference["park_policy"]
        demand(policy.get("version")==PARKING_POLICY and policy.get("fixture_context") is False
               and policy.get("calibration_sha256")==planning_inputs["parking_calibration"]["_receipt_sha256"]
               and policy.get("deck_precheck_sha256")==planning_inputs["deck_precheck_sha"],
               "reference and stage use different park or deck qualifications")
        if phase in {"compile","plan"}:
            return 0
        if phase in {"tpg","all"}:
            began=time.monotonic()
            oracle=RealConflict(world,folder,tasks=reference["tasks"])
            attempts=6 if phase=="all" else 1
            for attempt in range(attempts):
                tpg=parent_tpg.build_tpg(reference["tasks"],oracle,max_joint_span=100.,skip_ordered_pairs=True)
                tpg["kinematic_tpg_wall_s"]=time.monotonic()-began
                tpg["plan_sha256"]=plan_hash
                tpg["counts"]["native_state_samples"]=oracle.samples
                tpg["counts"]["conservative_oracle_timeouts"]=oracle.expired
                tpg["certificate"]["physical_rollout_required"]=True
                tpg["corridor_dag"]=corridor_dag_evidence(reference,tpg,mode)
                atomic_json(folder/"tpg.json",tpg)
                if phase=="tpg":
                    break
                demand(tpg["plan_sha256"]==plan_hash,"TPG uses another symbolic input")
                demand(tpg.get("corridor_dag")==corridor_dag_evidence(reference,tpg,mode),
                       "TPG corridor dependency audit was not reproducible")
                # Force a fresh scene load before independent native revalidation.
                world=BridgeWorld(sim,client,batch_joints=True)
                motion=NativeReplay(world,reference,tpg,folder)
                try:
                    validation=motion.run()
                except NativeBlocker as failure:
                    atomic_json(folder/"validation.json",{"passed":False,"reason":str(failure),
                            "detail":failure.detail,"validated_sample_count":len(motion.samples),
                            "completed_events":motion.events,"native_transfers":motion.transfer,
                            "key_clearances":motion.key_clearances,"handover_withdrawals":motion.handover_withdrawals})
                    atomic_json(folder/"replay_prefix.json",{"passed":False,"initial":motion.reference["initial"],
                            "sample_states":motion.samples,"events":motion.events,"reason":str(failure)})
                    detail=failure.detail or {}
                    frame=detail.get("suspected_frame") or {}
                    native=detail.get("first_failure") or {}
                    if (attempt+1<attempts and frame.get("time_s") is not None
                            and native.get("kind")=="native_collision"):
                        t_fail=float(frame["time_s"])
                        live={"A":[],"B":[]}
                        for node in tpg["nodes"]:
                            if node["start"]<=t_fail+1e-9 and node["end"]>=t_fail-1e-9:
                                if node["arm"] in live:
                                    live[node["arm"]].append(node)
                        if live["A"] and live["B"]:
                            first,second=sorted([live["A"][0],live["B"][0]],
                                                key=lambda row:row["reference_index"])
                            key=(first["id"],second["id"])
                            if key not in oracle.forced_conflicts and key[::-1] not in oracle.forced_conflicts:
                                oracle.forced_conflicts.add(key)
                                print(f"CEGAR retry {attempt+1}: forced conflict {key[0]} -> {key[1]} at {t_fail:.6f}s",flush=True)
                                oracle.world=world
                                oracle.began=time.monotonic()
                                continue
                    raise
                break
        else:
            tpg=read(folder/"tpg.json")
        if phase=="tpg":
            return 0
        if phase!="all":
            demand(tpg["plan_sha256"]==plan_hash,"TPG uses another symbolic input")
            demand(tpg.get("corridor_dag")==corridor_dag_evidence(reference,tpg,mode),
                   "TPG corridor dependency audit was not reproducible")
            # Force a fresh scene load before independent native revalidation.
            world=BridgeWorld(sim,client,batch_joints=True)
            motion=NativeReplay(world,reference,tpg,folder)
            try:
                validation=motion.run()
            except NativeBlocker as failure:
                atomic_json(folder/"validation.json",{"passed":False,"reason":str(failure),
                        "detail":failure.detail,"validated_sample_count":len(motion.samples),
                        "completed_events":motion.events,"native_transfers":motion.transfer,
                        "key_clearances":motion.key_clearances,"handover_withdrawals":motion.handover_withdrawals})
                atomic_json(folder/"replay_prefix.json",{"passed":False,"initial":motion.reference["initial"],
                        "sample_states":motion.samples,"events":motion.events,"reason":str(failure)})
                raise
    ref_s=tpg["reference_makespan"]
    parallel_s=tpg["nominal_makespan"]
    summary={"passed":True,"mode":mode,"symbolic_steps":len(actions),
             "handover_count":sum(r["kind"]=="handover" for r in actions),
             "tpg_nodes":tpg["counts"]["nodes"],"type1":tpg["counts"]["type1_edges"],
             "type2":tpg["counts"]["task_type2_edges"]+tpg["counts"]["motion_type2_edges"],
             "serial_reference_s":ref_s,"parallel_schedule_s":parallel_s,"speedup":ref_s/parallel_s,
             "planning_wall_s":reference["compile_wall_seconds"]+tpg["kinematic_tpg_wall_s"],
             "native_rollout_s":validation["replay_wall_s"],"total_stage_wall_s":time.monotonic()-started,
             "final_position_error_m":validation["maximum_part_position_error_m"],"collisions":0,
             "proof_paths":{name:str(folder/name) for name in ("plan.json","tpg.json","reference.json",
                                                              "validation.json","replay.json")},
             "scene_sha256":world.manifest["scene_sha256"],"park_policy":policy,
             "corridor_dag":tpg["corridor_dag"]}
    atomic_json(folder/"summary.json",summary)
    print(f"{mode}: fully passed independent native validation",flush=True)
    return 0


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("mode_a","mode_b"),required=True)
    parser.add_argument("--phase",choices=("compile","plan","tpg","validate","all"),default="all")
    ns=parser.parse_args()
    try:
        return run(ns.mode,ns.phase)
    except Exception as error:
        folder=HERE/ns.mode
        value={"mode":ns.mode,"phase":ns.phase,"passed":False,"reason":str(error),
               "traceback":traceback.format_exc()}
        if isinstance(error,NativeBlocker):
            value["detail"]=error.detail
        atomic_json(folder/"stage_failure.json",value)
        print(value["traceback"],flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
