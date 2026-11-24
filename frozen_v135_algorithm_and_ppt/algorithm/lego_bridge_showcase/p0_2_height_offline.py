from pathlib import Path
from itertools import product
import ast
import numpy as np
from lego_grasp_model import linear_targets, matrix
from p0_2_height_taker import TAKER_TILTS_DEG, OUTBOARD_PAIRS_M, TILT_ANGLES_DEG, HEIGHT_PAIRS_M, taker_pose, giver_pose
ast.parse(Path("p0_2_height_taker.py").read_text(encoding="utf-8-sig"))
grid=list(product(TILT_ANGLES_DEG,TAKER_TILTS_DEG,HEIGHT_PAIRS_M,OUTBOARD_PAIRS_M))
assert len(grid)==len(set(grid))==180
for ga,ta,(gh,th),(gx,ty) in grid:
    p=matrix([.04,.024,.018])
    g=giver_pose(p,ga,1,gh,gx);t=taker_pose(p,ta,ty,th)
    assert np.allclose(g[:3,0],[0,1,0])
    assert np.allclose(t[:3,0],[1,0,0])
    assert t[2,3]>g[2,3]
    assert np.isclose(g[0,3],p[0,3]-gx)
    assert np.isclose(t[1,3],p[1,3]+ty)
    assert np.isclose(np.linalg.det(g[:3,:3]),1.) and np.isclose(np.linalg.det(t[:3,:3]),1.)
    end=t.copy();end[2,3]+=.031
    for pt in linear_targets(t,end,step_m=.00095):
        assert np.allclose(pt[:3,:3],t[:3,:3])
        assert np.allclose(pt[:2,3],t[:2,3])
print("180 distinct height-separated fixtures retain signed axes and vertical seating")
