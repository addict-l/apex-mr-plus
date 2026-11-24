
"""Deadline-bounded connection search and Cartesian interpolation for the LEGO fixture."""
from dataclasses import dataclass
from hashlib import sha256
from collections import OrderedDict
from copy import deepcopy
import json
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MAX_JOINT_SAMPLE_RAD=.08
MAX_NEAR_POINT_STEP_M=.001
CONNECT_EXTENSION_RAD=.20
TOOL_AND_PAYLOAD_RADIUS_M=.20


def _plain(value):
    return value.tolist() if isinstance(value,np.ndarray) else value


def _for_q(state,side,q):
    sample={**state, "q":{**state["q"], side:list(q)}}
    sample.pop("tcp_targets",None)
    sample.pop("expected_tcp",None)
    return sample


def edge_samples(a,b,max_step=MAX_JOINT_SAMPLE_RAD):
    start,goal=np.asarray(a),np.asarray(b)
    n=max(1,math.ceil(float(np.max(np.abs(goal-start)))/max_step))
    return [goal.copy() if k==n else start+(goal-start)*(k/n) for k in range(1,n+1)]


@dataclass
class Tree:
    q: list
    parents: list
    start_side: bool

    def branch(self, i):
        result=[]
        while i!=-1:
            result.append(self.q[i])
            i=self.parents[i]
        return result


def connect_free(world,side,state,goal_qs,*,timeout_s=10.,seed=721,heartbeat=None,
                 start_q=None,max_iterations=1500,via_qs=()):
    """Return one exact-geometry free-space path, or bounded failure evidence."""
    started=time.monotonic()
    deadline=started+float(timeout_s)
    initial=np.asarray(state["q"][side] if start_q is None else start_q,dtype=float)
    candidates=np.asarray(goal_qs,dtype=float)
    if candidates.shape==(6,):
        candidates=candidates.reshape(1,6)
    if not candidates.size:
        return {"ok":False,"reason":"empty_goal_pool","q_path":[],"wall_seconds":0.}
    if candidates.ndim!=2 or candidates.shape[1]!=6:
        raise ValueError("Expected a list of six-joint goals")
    if not np.isfinite(candidates).all() or timeout_s<=0:
        raise ValueError("Invalid free-space search parameters")
    spatial=world.chains[side]
    base={**state,"measure_distance":False}
    base.pop("tcp_targets",None)
    base.pop("expected_tcp",None)
    context=deepcopy(base)
    del context["q"][side]
    digest=sha256(json.dumps(context,default=_plain,sort_keys=True,separators=(",",":"),
                              allow_nan=False).encode()).hexdigest()
    memo=getattr(world,"_failed_bridge_edges",None)
    if memo is None:
        memo=world._failed_bridge_edges=OrderedDict()
    stats={"samples_checked":0,"rpc_seconds":0.,"native_seconds":0.,"deduplicated_collisions":0}
    first=None
    last=None

    def cert(samples):
        nonlocal first,last
        if time.monotonic()>=deadline:
            return False,None
        result=world.batch_check([_for_q(base,side,q) for q in samples],
                                 stop_at_failure=True,deadline=deadline)
        stats["samples_checked"]+=len(result["rows"])
        for key in ("rpc_seconds","native_seconds","collision_seconds","distance_seconds",
                    "state_fk_seconds","serialization_seconds","log_seconds"):
            stats[key]=stats.get(key,0.)+result.get("timing",{}).get(key,0.)
        if not result["ok"]:
            slot=result.get("first_failure")
            if slot is None:
                last={"kind":result.get("error","incomplete_native_query")}
            else:
                last={**result["rows"][slot],"q":np.asarray(samples[slot]).tolist()}
                if first is None:
                    first=last
            return False,last
        return True,None

    def edge(a,b):
        nonlocal first,last
        key=(digest,tuple(a),tuple(b))
        recheck=memo.get(key)
        if recheck is not None:
            stats["deduplicated_collisions"]+=1
            last=recheck
            if first is None:
                first=last
            return False
        ok,rejected=cert(edge_samples(a,b))
        if not ok and rejected and rejected.get("kind")=="native_collision":
            memo[key]=rejected
            if len(memo)>1024:
                memo.popitem(last=False)
        return ok

    def answer(path,method,iterations,status="solved"):
        return {"ok":bool(path),"reason":status,"method":method,
                "q_path":[np.asarray(q).tolist() for q in path],"iterations":iterations,
                "wall_seconds":time.monotonic()-started,"timeout_s":timeout_s,
                "first_failure":first,"last_failure":last,"timing":stats,"context_sha256":digest,
                "max_joint_sample_rad":MAX_JOINT_SAMPLE_RAD,"minimum_distance_m":"not_measured"}

    if not cert([initial])[0]:
        return answer([], "endpoint",0,"invalid_start")
    valid=[]
    ordered=sorted(candidates,key=lambda q:float(np.linalg.norm(q-initial)))
    for goal in ordered:
        if not cert([goal])[0]:
            continue
        if edge(initial,goal):
            return answer([initial,*edge_samples(initial,goal)],"direct",0)
        valid.append(goal)
    if not valid:
        return answer([],"endpoint",0,"budget" if time.monotonic()>=deadline else "no_valid_goal")

    for via in via_qs:
        via=np.asarray(via,dtype=float)
        if float(np.linalg.norm(via-initial))<=1e-6:
            continue
        if not cert([via])[0] or not edge(initial,via):
            continue
        for goal in valid:
            if not edge(via,goal):
                continue
            samples=[initial,*edge_samples(initial,via),*edge_samples(via,goal)]
            if cert(samples)[0]:
                return answer(samples,"via",0)

    left,right=Tree([initial],[-1],True),Tree(valid,[-1]*len(valid),False)
    rng=np.random.default_rng(seed)

    def extend(tree,target):
        distances=np.max(np.abs(np.asarray(tree.q)-target),axis=1)
        i=int(np.argmin(distances))
        distance=float(distances[i])
        if distance<=1e-10:
            return i,True
        next_q=target if distance<=CONNECT_EXTENSION_RAD else tree.q[i]+(
            (target-tree.q[i])*CONNECT_EXTENSION_RAD/distance)
        if not edge(tree.q[i],next_q):
            return None,False
        tree.q.append(np.asarray(next_q))
        tree.parents.append(i)
        return len(tree.q)-1,distance<=CONNECT_EXTENSION_RAD

    iterations=0
    while iterations<max_iterations and time.monotonic()<deadline:
        iterations+=1
        if heartbeat and iterations%16==0:
            heartbeat({"algorithm":"RRTConnect","iteration":iterations,**stats})
        if rng.random()<.28:
            target=right.q[int(rng.integers(0,len(right.q)))]
        elif rng.random()<.65:
            lower=np.maximum(spatial.lower,np.minimum(initial,valid[0])-.75)
            upper=np.minimum(spatial.upper,np.maximum(initial,valid[0])+.75)
            target=rng.uniform(lower,upper)
        else:
            target=rng.uniform(spatial.lower,spatial.upper)
        index,_=extend(left,target)
        if index is not None:
            for _ in range(25):
                if time.monotonic()>=deadline:
                    break
                other,reached=extend(right,left.q[index])
                if other is None:
                    break
                if reached:
                    a,b=left.branch(index),right.branch(other)
                    if not left.start_side:
                        a,b=b,a
                    vertices=list(reversed(a))+b[1:]
                    samples=[vertices[0]]
                    for previous,next_q in zip(vertices,vertices[1:]):
                        samples.extend(edge_samples(previous,next_q))
                    # Materialize and certify the exact returned ordering in
                    # the same collision context. No cached validation credit.
                    if cert(samples)[0]:
                        return answer(samples,"RRTConnect",iterations)
                    break
        left,right=right,left
    reason="budget" if time.monotonic()>=deadline else "iteration_limit"
    return answer([],"RRTConnect",iterations,reason)


def near_waypoints(start,goal,*,radius_m=TOOL_AND_PAYLOAD_RADIUS_M,jaw_change_m=0.):
    """Subdivide by a conservative bound on tool and rigid payload corner travel."""
    a,b=np.asarray(start,dtype=float),np.asarray(goal,dtype=float)
    ar,br=Rotation.from_matrix(a[:3,:3]),Rotation.from_matrix(b[:3,:3])
    length=float(np.linalg.norm(b[:3,3]-a[:3,3]))
    angle=(ar.inv()*br).magnitude()
    bound=length+radius_m*angle+abs(jaw_change_m)
    count=max(1,math.ceil(bound/MAX_NEAR_POINT_STEP_M),math.ceil(angle/math.radians(.5)))
    interp=Slerp([0.,1.],Rotation.concatenate([ar,br]))
    for index in range(1,count+1):
        fraction=index/count
        pose=np.eye(4)
        pose[:3,:3]=interp(fraction).as_matrix()
        pose[:3,3]=a[:3,3]+fraction*(b[:3,3]-a[:3,3])
        yield pose


def continuous_ik(world,side,state,targets,*,heartbeat=None,joint_limit_rad=.25,deadline=None):
    """Solve and test already subdivided Cartesian states in order."""
    base={**state,"measure_distance":True}
    previous=np.asarray(state["q"][side],dtype=float)
    samples=[]
    path=[previous.tolist()]
    began=time.monotonic()
    for index,pose in enumerate(targets):
        if deadline is not None and time.monotonic()>=deadline:
            return {"ok":False,"reason":"deadline","q_path":path,"first_failure":index}
        answer=world.ik(side,pose,seed=previous,max_solutions=1,seed_count=1)
        if not answer or np.max(np.abs(answer[0]-previous))>joint_limit_rad:
            return {"ok":False,"reason":"continuous_ik","q_path":path,"first_failure":index,
                    "requested_tcp":np.asarray(pose).tolist(),"ik_seconds":time.monotonic()-began}
        previous=np.asarray(answer[0])
        path.append(previous.tolist())
        row=_for_q(base,side,previous)
        row["tcp_targets"]={side:np.asarray(pose).tolist()}
        samples.append(row)
    ik_elapsed=time.monotonic()-began
    checked=world.batch_check(samples,stop_at_failure=True,heartbeat=heartbeat,deadline=deadline)
    checked.update(q_path=path,ik_seconds=ik_elapsed)
    return checked
