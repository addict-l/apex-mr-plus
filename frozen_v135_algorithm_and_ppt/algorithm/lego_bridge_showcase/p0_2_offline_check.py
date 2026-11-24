import ast
from pathlib import Path
for name in ('lego_grasp_model.py', 'p0_2_handover.py'):
    ast.parse(Path(name).read_text(encoding="utf-8-sig"))
import numpy as np
from lego_grasp_model import deck_grasp, jaw_q, linear_targets, pier_giver_grasp, pier_taker_grasp
bottom = [0., 0., .002]
for pose in (deck_grasp(bottom), pier_giver_grasp(bottom, 65., 1), pier_taker_grasp(bottom)):
    assert np.allclose(pose[:3,:3] @ pose[:3,:3].T, np.eye(3))
    assert np.isclose(np.linalg.det(pose[:3,:3]), 1.)
    assert pose[2,2] < 0.
assert np.isclose(jaw_q(.040), .017)
assert np.isclose(jaw_q(.0158), .0049)
path=list(linear_targets(deck_grasp(bottom),deck_grasp([.1001, 0., .002])))
assert len(path)==101
print("role transforms, aperture and 1 mm sampling verified")
