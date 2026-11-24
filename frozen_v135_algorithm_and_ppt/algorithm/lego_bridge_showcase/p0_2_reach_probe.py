from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bridge_geometry import CalibratedChain
from lego_grasp_model import matrix, pier_giver_grasp, pier_taker_grasp, pose_error
from p0_2_handover import CASES, TILT_ANGLES_DEG, HANDOVER_CENTERS_M

rows=json.loads(Path("native/01_-_Initial_inventory.json").read_bytes())
chains = {}
for side in "AB":
    joints = sorted((row for row in rows if row["alias"]=="joint" and "/UR5_"+side+"/" in row["path"]),
                    key=lambda row: row["path"].count("/joint"))
    tip=[r for r in rows if r["alias"]=="TCP_between_fingertips" and "UR5_"+side+"/" in r["path"]][0]
    q0=np.array([row["jpos"] for row in joints])
    motion=np.eye(4)
    points, axes=[],[]
    for row, q in zip(joints,q0):
        zero=np.linalg.inv(motion) @ matrix(row["pose"])
        origin, axis=zero[:3,3], zero[:3,2]
        points.append(origin)
        axes.append(axis)
        spin=Rotation.from_rotvec(axis*q).as_matrix()
        rel=np.eye(4)
        rel[:3,:3], rel[:3,3]=spin, origin-spin @ origin
        motion=motion @ rel
    chain=CalibratedChain.__new__(CalibratedChain)
    chain.side=side
    chain.lower=np.asarray([row["jinterval"][1][0] for row in joints])
    chain.upper=np.asarray([sum(row["jinterval"][1]) for row in joints])
    chain.points, chain.axes = np.array(points),np.array(axes)
    raw=matrix(tip["pose"])
    chain.tip_zero=np.linalg.inv(motion) @ raw
    assert max(pose_error(chain.fk(q0)[0],raw))<=1e-6
    chains[side]=(chain,q0)
for case in CASES:
    g,t=case["giver"],case["taker"]
    gc,g0=chains[g]
    tc,t0=chains[t]
    max_pairs=0
    for p, angle, direction in product(HANDOVER_CENTERS_M,TILT_ANGLES_DEG,(1,-1)):
        ga=gc.solve(pier_giver_grasp(p,angle,case["incline"]),g0,2,9)
        ta=tc.solve(pier_taker_grasp(p,direction),t0,2,9)
        max_pairs+=bool(ga and ta)
    bad= pier_taker_grasp(HANDOVER_CENTERS_M[0])
    print(case["id"], "both endpoint IK pairs", max_pairs, "same-face bad IK exist",
          bool(gc.solve(bad,g0,1,9) and tc.solve(bad,t0,1,9)), flush=True)
