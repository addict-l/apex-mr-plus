"""Recorded long-profile geometry with an explicit along-path servo lag mock.

This uses an isolated Lua VM. The 50 ms lag is a controlled model, not a claim
to reconstruct unrecorded pre-dynamics joint values from a 100 ms trace.
"""
import unittest

from tests import test_v13_controller as controller_support


@unittest.skipUnless((controller_support.INSTALL/"lua.dll").is_file(), "CoppeliaSim 自带 Lua DLL 不可用")
class LuaMotionProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller_support.LuaControllerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        controller_support.LuaControllerTests.tearDownClass.__func__(cls)

    lua_run = controller_support.LuaControllerTests.lua_run

    @staticmethod
    def short_profile(perpendicular_error):
        return r'''
local duration=2.
local function point(t)
 local u=math.max(0,math.min(1,t/duration))
 return u*u*u*(10+u*(-15+6*u))
end
function sim.getJointPosition(joint)
 if joint==201 then return point(math.max(0,clock-.02)) end
 if joint==202 then return PERPENDICULAR end
 return 0
end
cfg.program.nodes={};cfg.program.edges={{source='first',target='last'}}
for i,id in ipairs({'first','last'}) do
 local meta={motion_run_id='first',motion_run_offset=i-1,motion_run_duration=duration}
 if i==1 then meta.continuation_to='last' else meta.continuation_from='first' end
 local n=node(id,'B',nil,meta);n.node_type='pose';n.kind='transit';n.part=nil
 n.duration=1.;n.q_start={point(i-1),0,0,0,0,0};n.q_end={point(i),0,0,0,0,0}
 n.q_samples={n.q_start,n.q_end}
 n.motion_segments={{q0={0,0,0,0,0,0},q1={1,0,0,0,0,0},duration=1.,
   profile_duration=duration,profile_offset=i-1}}
 cfg.program.nodes[i]=n
end
sysCall_init();clock=.02;sysCall_actuation()
'''.replace("PERPENDICULAR", str(perpendicular_error))

    def test_continuation_rejects_cross_path_error_above_endpoint_tolerance(self):
        self.lua_run(self.short_profile(.009)+r'''
for _=1,60 do clock=clock+.05;sysCall_actuation() end
assert(getResult().node_states.first~='done','A 0.009 rad transverse error cannot pass a 0.008 rad corridor')
for _=1,180 do clock=clock+.05;sysCall_actuation() end
local result=getResult()
assert(result.failure and result.failure:find('guard timeout',1,true))
assert(#result.tracking_failures==0,'This is an endpoint corridor violation, not a tracking-limit violation')
''')

    def test_continuation_keeps_the_true_tracking_error_abort(self):
        self.lua_run(self.short_profile(.16)+r'''
for _=1,20 do clock=clock+.05;sysCall_actuation() end
local result=getResult()
assert(result.failure and result.failure:find('Servo tracking limit',1,true))
assert(#result.tracking_failures==1 and result.tracking_failures[1].error_rad>.15)
''')

    def test_terminal_motion_keeps_endpoint_tolerance_and_original_dwell(self):
        self.lua_run(r'''
local offset=.009
function sim.getJointPosition(joint)
 if joint~=201 then return 0 end
 local u=math.max(0,math.min(1,(clock-.02)))
 return u*u*u*(10+u*(-15+6*u))-offset
end
local n=node('terminal','B',nil)
n.node_type='pose';n.kind='transit';n.part=nil;n.duration=1.
n.q_start={0,0,0,0,0,0};n.q_end={1,0,0,0,0,0};n.q_samples={n.q_start,n.q_end}
n.motion_segments={{q0=n.q_start,q1=n.q_end,duration=1.}}
cfg.program.nodes={n};cfg.program.edges={}
sysCall_init();clock=.02;sysCall_actuation()
for _=1,30 do clock=clock+.05;sysCall_actuation() end
assert(getResult().node_states.terminal~='done','The terminal endpoint still requires error below 0.008 rad')
offset=.006;clock=2.;sysCall_actuation()
clock=2.02;sysCall_actuation()
assert(getResult().node_states.terminal~='done','Terminal motion still needs its existing 0.04 second dwell')
clock=2.06;sysCall_actuation()
assert(getResult().node_states.terminal=='done')
assert(not getResult().failure,getResult().failure)
''')

    def test_real_long_collinear_profile_accepts_actual_progress_with_small_along_path_lag(self):
        self.lua_run(r'''
-- Native attempt 2, s6_0020_B_r1 p0057/p0058 shared profile.
local q0={-2.1365289208122684,-.9896149740928923,1.9595775951750967,
 2.9165444801436657,-1.0010771137027445,5.775300970981984}
local q1={-1.7196019018123587,-.5671412148151027,1.2396175479378093,
 -2.0520718208408937,-.5053532076540392,6.113494257255217}
local duration=15.526925940576746
local boundaries={0,1.8340491933567689,2.3538263217850246,duration}
local function point(t)
 local u=math.max(0,math.min(1,t/duration))
 local s=u*u*u*(10+u*(-15+6*u));local q={}
 for i=1,6 do q[i]=q0[i]+(q1[i]-q0[i])*s end
 return q
end
function sim.getJointPosition(joint)
 if joint<201 or joint>206 then return 0 end
 -- A deterministic 50 ms servo lag stays exactly on the validated line.
 -- Coppelia's cyclic joints expose the representative in [-pi,pi].
 local q=point(math.max(0,clock-.02-.05))
 return math.atan(math.sin(q[joint-200]),math.cos(q[joint-200]))
end
cfg.program.nodes={};cfg.program.edges={}
for i=1,3 do
 local meta={motion_run_id='move1',motion_run_offset=boundaries[i],motion_run_duration=duration}
 if i<3 then meta.continuation_to='move'..(i+1) end
 if i>1 then meta.continuation_from='move'..(i-1) end
 local n=node('move'..i,'B',nil,meta)
 n.part=nil;n.node_type='pose';n.kind='transit'
 n.q_start=point(boundaries[i]);n.q_end=point(boundaries[i+1]);n.q_samples={n.q_start,n.q_end}
 n.duration=boundaries[i+1]-boundaries[i]
 n.motion_segments={{q0=q0,q1=q1,duration=n.duration,
  profile_duration=duration,profile_offset=boundaries[i]}}
 cfg.program.nodes[i]=n
 if i>1 then cfg.program.edges[#cfg.program.edges+1]={source='move'..(i-1),target='move'..i} end
end
sysCall_init();clock=.02;sysCall_actuation()
for _=1,400 do clock=clock+.05;sysCall_actuation() end
local result=getResult()
assert(not result.failure,result.failure)
assert(result.done,'Actual progress along the certified profile must advance its milestones')
assert(#result.tracking_failures==0,'The existing tracking guard remains satisfied')
''')


if __name__ == "__main__":
    unittest.main()
