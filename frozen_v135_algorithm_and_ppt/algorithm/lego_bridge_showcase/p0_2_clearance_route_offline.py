import ast
from pathlib import Path
import numpy as np
from p0_2_clearance_route import joint_waypoints, SOURCE

ast.parse(Path("p0_2_clearance_route.py").read_text(encoding="utf-8-sig"))
class Chain:
    def fk(self, q):
        m=np.eye(4)
        m[0,3]=q[0]
        return (m,None)
points=joint_waypoints(Chain(),np.zeros(6),np.array([.01,0,0,0,0,0]))
assert len(points)>10
assert all(np.linalg.norm(q-r)<=.00086 for q,r in zip([np.zeros(6),*points],points))
assert np.isclose(points[-1][0],.01)
assert SOURCE.is_file()
print("separate connector source and measured-stock input parse; adaptive join displacement tested")
