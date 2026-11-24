from pathlib import Path
import ast
import numpy as np
from lego_grasp_model import linear_targets, matrix
from p0_2_tilted_taker import TAKER_TILTS_DEG, TAKER_OFFSETS_M, taker_pose
ast.parse(Path("p0_2_tilted_taker.py").read_text(encoding="utf-8-sig"))
for angle in TAKER_TILTS_DEG:
    for offset in TAKER_OFFSETS_M:
        root=matrix([.04,.024,.018])
        p=taker_pose(root,angle,offset)
        assert np.allclose(p[:3,0],[1,0,0])
        assert p[1,2] < 0 and p[2,2] < 0
        assert np.isclose(np.linalg.det(p[:3,:3]),1.)
        target=p.copy();target[2,3]+=.03
        for item in linear_targets(p,target,step_m=.00095):
            assert np.allclose(item[:3,:3],p[:3,:3])
            assert np.allclose(item[:2,3],p[:2,3])
print("tilted taker signed jaw and unchanged vertical sweep verified")
