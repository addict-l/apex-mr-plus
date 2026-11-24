"""Offline checks for the isolated parking precheck helpers."""
import ctypes
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from bridge_outboard_precheck import PROJECTION, candidates, maximum_open_state, path_proof, same_physical_state
from bridge_motion import MAX_JOINT_SAMPLE_RAD


class OutboardPrecheckTests(unittest.TestCase):
    def test_lua_transforms_full_geometry_and_preserves_native_joints(self):
        dll = Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\lua.dll")
        lua = ctypes.CDLL(str(dll))
        lua.luaL_newstate.restype = ctypes.c_void_p
        lua.luaL_openlibs.argtypes = [ctypes.c_void_p]
        lua.lua_close.argtypes = [ctypes.c_void_p]
        lua.luaL_loadbufferx.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.c_char_p, ctypes.c_char_p]
        lua.lua_pcallk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_ssize_t, ctypes.c_void_p]
        lua.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lua.lua_tolstring.restype = ctypes.c_char_p
        mock = """
local q=.1
package.preload['sim']=function()
    return {handle_world=-1, getJointPosition=function() return q end,
            setJointPosition=function(_,v) q=v end,
            getObjectMatrix=function(h)
                if h==20 then return {1,0,0,0, 0,1,0,0, 0,0,1,0} end
                return {0,1,0,q, -1,0,0,0, 0,0,1,0}
            end}
end
"""
        assertions = """
local identity={1,0,0,0,1,0,0,0,1}
local plate={20,{0,0,0},{.2,.1,.01},identity}
local complete={{10,{.01,0,0},{.01,.09,.02},identity}}
local right=projectionScan({1},complete,{{.4}},plate,1)
assert(math.abs(right.rows[1].gap_m-(.11-4e-8))<1e-9 and q==.1)
assert(math.abs(right.rows[1].individual[1][2]-(.31-2e-8))<1e-9)
local left=projectionScan({1},complete,{{-.4},{-.25}},plate,-1)
assert(math.abs(left.rows[1].gap_m-(.11-4e-8))<1e-9 and left.rows[2].gap_m<0 and q==.1)
local rotated={{10,{0,0,0},{.01,.09,.02},{0,-1,0,1,0,0,0,0,1}}}
local accurate=projectionScan({1},rotated,{{.4}},plate,1)
assert(math.abs(accurate.rows[1].gap_m-(.19-4e-8))<1e-9 and q==.1)
"""
        source = (mock + PROJECTION + assertions).encode()
        state = lua.luaL_newstate()
        try:
            lua.luaL_openlibs(state)
            status = lua.luaL_loadbufferx(state, source, len(source), b"projection", None)
            if status == 0:
                status = lua.lua_pcallk(state, 0, 0, 0, 0, None)
            self.assertEqual(status, 0, None if status == 0 else lua.lua_tolstring(state, -1, None))
        finally:
            lua.lua_close(state)

    def test_candidates_respect_frozen_joint_limits_and_symmetric_offsets(self):
        chain = SimpleNamespace(lower=np.full(6, -1.0), upper=np.full(6, 1.0))
        qs = np.asarray(candidates(chain, [0.0] * 6))
        self.assertTrue(np.all(np.abs(qs) <= 1.0))
        self.assertTrue(np.any(qs[:, 0] > 0) and np.any(qs[:, 0] < 0))

    def test_native_identity_detects_unrelated_owner_and_joint_changes(self):
        before = {"q": {"A": [0.0] * 6, "B": [0.0] * 6}, "jaw": {"A": 0.0, "B": 0.0},
                  "parts": {"brick": {"owner": None, "parent": -1, "pose": np.eye(4).tolist()}}}
        moved = {**before, "q": {**before["q"], "B": [0.1] * 6}}
        owned = {**before, "parts": {"brick": {**before["parts"]["brick"], "owner": "A"}}}
        self.assertTrue(same_physical_state(before, before))
        self.assertFalse(same_physical_state(before, moved))
        self.assertFalse(same_physical_state(before, owned))

    def test_path_proof_demands_matching_endpoint_and_dense_certificate(self):
        class NativeDouble:
            chains = {"A": SimpleNamespace(lower=np.full(6, -1.0), upper=np.full(6, 1.0))}

            @staticmethod
            def batch_check(states, **_):
                return {"ok": True, "rows": [{"ok": True}] * len(states)}

        before = {"q": {"A": [0.0] * 6, "B": [0.0] * 6}, "jaw": {"A": 0.0, "B": 0.0}, "parts": {}}
        path = path_proof(NativeDouble(), "A", before, [0.05, 0., 0., 0., 0., 0.], 721)
        self.assertTrue(path["endpoints_and_spacing_passed"])
        self.assertEqual(len(path["q_path"]), math.ceil(0.05 / MAX_JOINT_SAMPLE_RAD) + 1)

    def test_maximum_open_snapshot_carries_matching_follower_command(self):
        world = SimpleNamespace(arms={side: {"jaw_interval_m": [0., .017]} for side in ("A", "B")})
        parked = {"jaw": {"A": .003, "B": .004}, "follower": {"A": .003, "B": .004}}
        opened = maximum_open_state(world, parked)
        self.assertEqual(opened["jaw"], {"A": .017, "B": .017})
        self.assertEqual(opened["follower"], opened["jaw"])
        self.assertEqual(parked["follower"], {"A": .003, "B": .004})


if __name__ == "__main__":
    unittest.main()
