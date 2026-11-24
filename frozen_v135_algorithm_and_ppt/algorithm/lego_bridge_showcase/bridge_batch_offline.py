
"""Fast offline guards for the showcase batch and motion entry points."""
import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bridge_geometry import GEOMETRY
from bridge_native import BATCH_GEOMETRY
from bridge_motion import connect_free, near_waypoints


def validate_lua_chunk():
    path=Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\lua.dll")
    lua=ctypes.CDLL(str(path))
    lua.luaL_newstate.restype=ctypes.c_void_p
    lua.lua_close.argtypes=[ctypes.c_void_p]
    lua.luaL_openlibs.argtypes=[ctypes.c_void_p]
    lua.luaL_loadbufferx.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_char_p]
    lua.lua_pcallk.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                            ctypes.c_ssize_t,ctypes.c_void_p]
    lua.lua_tolstring.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    lua.lua_tolstring.restype=ctypes.c_char_p
    state=lua.luaL_newstate()
    try:
        lua.luaL_openlibs(state)
        source=(GEOMETRY_MOCK+GEOMETRY+BATCH_GEOMETRY+GEOMETRY_ASSERTIONS).encode()
        status=lua.luaL_loadbufferx(state,source,len(source),b"bridge_geometry",None)
        if status==0:
            status=lua.lua_pcallk(state,0,0,0,0,None)
        if status!=0:
            raise ValueError(lua.lua_tolstring(state,-1,None))
    finally:
        lua.lua_close(state)


GEOMETRY_MOCK = r"""
local mockCollision, mockDistance, collCalls, distCalls, clock = {},{},0,0,0
local function mockPair(a,b) return a..':'..b end
package.preload['sim']=function()
    return {handle_world=-1,
        getObjectMatrix=function() return {1,0,0,0, 0,1,0,0, 0,0,1,0} end,
        getSystemTime=function() clock=clock+.001; return clock end,
        checkCollision=function(a,b)
            collCalls=collCalls+1
            return mockCollision[mockPair(a,b)] and 1 or 0, {a,b}
        end,
        checkDistance=function(a,b,limit)
            distCalls=distCalls+1
            local distance=mockDistance[mockPair(a,b)]
            return distance<limit and 1 or 0, {0,0,0,0,0,0,distance}, {a,b}
        end}
end
"""


GEOMETRY_ASSERTIONS = r"""
local pairlist={{1,2},{1,3},{2,3}}
local meshlist={}
for id=1,3 do meshlist[id]={id,{0,0,0},{.01,.01,.01}} end
local registration=bridgeConfigure(meshlist,pairlist,{},{{1,2}})
assert(registration.pair_count==3 and registration.distance_excluded_pair_count==1)
mockDistance={['1:2']=.001,['1:3']=.01,['2:3']=.012}

local all=bridgeSample({},.025,true,true,false,false)
assert(all.ok and all.minimum_distance_m==.001 and all.coll_queries==3)
local filtered=bridgeSample({},.025,true,true,true,false)
assert(filtered.ok and filtered.minimum_distance_m==.01 and filtered.coll_queries==3)
assert(filtered.dist_queries==3 and filtered.clearance_scope=='external_filtered')
assert(filtered.unfiltered_minimum_distance_m==.001 and filtered.unfiltered_pair_handles[2]==2)
distCalls=0
local fast=bridgeSample({},.025,false,true,false,false)
assert(fast.ok and fast.minimum_distance_m=='not_measured' and fast.coll_queries==3)
assert(fast.dist_queries==0 and fast.timing.distance_seconds==0 and distCalls==0)
assert(fast.unfiltered_minimum_distance_m==nil)

mockCollision={['1:2']=true,['2:3']=true}
local first=bridgeSample({},.025,false,true,true,false)
assert(not first.ok and first.pair_handles[2]==2 and first.coll_queries==1)
assert(first.minimum_distance_m=='not_measured' and first.collisions==nil)
local both=bridgeSample({},.025,false,true,true,true)
assert(not both.ok and #both.collisions==2 and both.collisions[2].pair_handles[1]==2)
assert(both.coll_queries==3 and both.dist_queries==0 and both.timing.distance_seconds==0)
local declared=bridgeSample({{1,2}},.025,false,true,true,true)
assert(not declared.ok and #declared.allowed_collisions==1 and #declared.collisions==1)
assert(declared.pair_handles[1]==2 and declared.pair_handles[2]==3)
"""


class GeometryDouble:
    chains={"A":SimpleNamespace(lower=np.full(6,-1.),upper=np.full(6,1.))}

    @staticmethod
    def batch_check(states,**_):
        rows=[]
        for row in states:
            a,b,*_=row["q"]["A"]
            clear=not (-.14<=a<=.15 and -.36<=b<=.37)
            rows.append({"ok":clear,"kind":"native_free" if clear else "native_collision"})
            if not clear:
                return {"ok":False,"rows":rows,"first_failure":len(rows)-1,"timing":{}}
        return {"ok":True,"rows":rows,"first_failure":None,"timing":{}}


def validate_motion():
    s={"q":{"A":[-.65,0.,0.,0.,0.,0.],"B":[0.]*6},"jaw":{"A":.006,"B":.006},"parts":{}}
    solved=connect_free(GeometryDouble(),"A",s,[[.66,0.,0.,0.,0.,0.]],timeout_s=2.)
    assert solved["ok"] and solved["method"]=="RRTConnect",solved
    assert np.max(np.abs(np.diff(solved["q_path"],axis=0)))<=.02+1e-10
    a=np.eye(4); b=np.eye(4); b[0,3]=.0032
    points=list(near_waypoints(a,b))
    assert len(points)==4
    assert np.array_equal(points[-1],b)


if __name__=="__main__":
    validate_lua_chunk()
    validate_motion()
    print("Lua collision-only, filtered clearance and exhaustive checks passed; RRT-Connect retained <=0.02 rad samples",flush=True)
