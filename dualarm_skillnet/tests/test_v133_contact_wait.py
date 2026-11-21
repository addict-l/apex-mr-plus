"""Actual contact monitoring at an acquired endpoint, in an isolated Lua VM.

The API mocks model the platform attempt-2 gap: the carrying arm still holds
its part after the certified descent, while final install waits for support.
No simulator or physical collision exemption is introduced by these tests.
"""
import unittest

from tests import test_v13_controller as controller_support


def endpoint_wait_scenario(*, depth=.0001, certified=True, new_motion=False,
                           joint_drift=0., release_held=False, lower_change=None, colliding=True):
    pairs = "{{'red','pillar'}}" if certified else "{{'red','Baseplate'}}"
    source = r'''
aliases[51]='pillar';parents[51]=1;all[#all+1]=51
cfg.blocks.pillar=51
cfg.parts[1].size={.52,.16,.04}
cfg.parts[#cfg.parts+1]={id='pillar',size={.12,.14,.076},type='pillar_type'}
cfg.initial_installed={p1='pillar'}
cfg.slot_below={beam={'p1'}}
cfg.slot_types={beam='beam_type',p1='pillar_type'}
sim.objfloatparam_objbbox_min_x=101;sim.objfloatparam_objbbox_max_x=102
sim.objfloatparam_objbbox_min_y=103;sim.objfloatparam_objbbox_max_y=104
sim.objfloatparam_objbbox_min_z=105;sim.objfloatparam_objbbox_max_z=106
local beamPose={0,.12,.944-DEPTH,0,0,0,1}
local pillarPose={.12,.12,.886,0,0,0,1}
local originalPose=sim.getObjectPose
function sim.getObjectPose(h,relative)
 if relative==sim.handle_world then
  if h==50 then return {table.unpack(beamPose)} end
  if h==51 then return {table.unpack(pillarPose)} end
 end
 return originalPose(h,relative)
end
function sim.getObjectFloatParam(h,key)
 local size=h==50 and {.52,.16,.04} or {.12,.14,.076}
 local axis=math.floor((key-101)/2)+1
 return size[axis]/2*(key%2==1 and -1 or 1)
end
local originalMatrix=sim.getObjectMatrix
function sim.getObjectMatrix(h,relative)
 if h==50 then return {1,0,0,0,0,1,0,.12,0,0,1,.944-DEPTH} end
 if h==51 then return {1,0,0,.12,0,1,0,.12,0,0,1,.886} end
 return originalMatrix(h,relative)
end
function sim.checkCollision(a,b)
 if COLLIDING and clock>.5 and ((a==50 and b==51) or (a==51 and b==50)) then return 1,{a,b} end
 return 0,{}
end
local originalJoint=sim.getJointPosition
function sim.getJointPosition(joint)
 if joint==206 and clock>.3 then return JOINT_DRIFT end
 return originalJoint(joint)
end
local support=node('support::skill','A','support_start',
 {slot='beam',support_load_contact_tasks={'contact'}});support.part=nil
local pick=node('pick::skill','B','grasp',{payload_after=binding})
local contact=node('contact::p0000','B',nil,{request_kind='linear',contact_exclusions=PAIRS})
contact.task_id='contact';contact.node_type='pose';contact.kind='transit'
local install=node('install::skill','B','install',{slot='beam',below={'p1'},target_pose=beamPose})
cfg.program.nodes={support,pick,contact,install}
cfg.program.edges={{source='pick::skill',target='contact::p0000'},
 {source='contact::p0000',target='install::skill'},
 {source='support::skill',target='install::skill'}}
NEW_MOTION
RELEASE_HELD
LOWER_CHANGE
sysCall_init();clock=.02;sysCall_actuation()
for _=1,14 do clock=clock+.05;sysCall_actuation();sysCall_sensing() end
local result=getResult()
assert(result.node_states['contact::p0000']=='done','The certified contact path must finish')
assert(result.node_states['install::skill']=='pending','Install must still await measured support')
HOLDER_ASSERT
'''
    new = ""
    if new_motion:
        new = r'''
local depart=node('depart::p0000','B',nil);depart.part='red'
depart.node_type='pose';depart.kind='transit';depart.duration=10.
cfg.program.nodes[#cfg.program.nodes+1]=depart
cfg.program.edges[#cfg.program.edges+1]={source='contact::p0000',target='depart::p0000'}
'''
    release = ""
    if release_held:
        release = r'''
local release=node('release::skill','B','release',{location='board',target_pose=beamPose})
cfg.program.nodes[#cfg.program.nodes+1]=release
cfg.program.edges[#cfg.program.edges+1]={source='contact::p0000',target='release::skill'}
'''
    lower = ""
    if lower_change:
        lower = r'''
cfg.initial_attachments.pillar=700
support.metadata.action='wait';support.duration=100.
local lowerPick=node('lower_pick::skill','A','grasp',
 {payload_after={name='pillar',offset={0,0,.044},quaternion={1,0,0,0}}})
lowerPick.part='pillar'
cfg.program.nodes[#cfg.program.nodes+1]=lowerPick
cfg.program.edges[#cfg.program.edges+1]={source='contact::p0000',target='lower_pick::skill'}
LOWER_FINISH
'''
        lower_finish = "cfg.program.edges[#cfg.program.edges+1]={source='lower_pick::skill',target='support::skill'}"
        if lower_change == "detach":
            lower_finish = r'''
local lowerDetach=node('lower_detach::skill','A','detach',{slot='p1'});lowerDetach.part='pillar'
cfg.program.nodes[#cfg.program.nodes+1]=lowerDetach
cfg.program.edges[#cfg.program.edges+1]={source='lower_pick::skill',target='lower_detach::skill'}
cfg.program.edges[#cfg.program.edges+1]={source='lower_detach::skill',target='support::skill'}
'''
        lower = lower.replace("LOWER_FINISH", lower_finish)
    holder_assert = ("assert(result.final_state.held.B==false,'The completed release must clear the held part')"
                     if release_held else
                     "assert(result.final_state.held.B=='red','The part must remain in the carrying gripper')")
    return (source.replace("DEPTH", repr(depth)).replace("PAIRS", pairs)
            .replace("JOINT_DRIFT", repr(joint_drift)).replace("NEW_MOTION", new)
            .replace("COLLIDING", "true" if colliding else "false")
            .replace("RELEASE_HELD", release).replace("LOWER_CHANGE", lower)
            .replace("HOLDER_ASSERT", holder_assert))


@unittest.skipUnless((controller_support.INSTALL/"lua.dll").is_file(), "CoppeliaSim 自带 Lua DLL 不可用")
class EndpointContactWaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller_support.LuaControllerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        controller_support.LuaControllerTests.tearDownClass.__func__(cls)

    lua_run = controller_support.LuaControllerTests.lua_run

    def test_certified_shallow_bearing_contact_survives_held_endpoint_wait(self):
        self.lua_run(endpoint_wait_scenario()+r'''
assert(not result.failure,result.failure)
assert(#result.collision_failures==0 and result.collision_monitor.checks>0,
 'Waiting must retain the measured collision checks and the certified bearing contact')
for _=1,10 do clock=clock+.02;sysCall_sensing() end
local receipts={}
for _,event in ipairs(getResult().events) do
 if event.kind=='endpoint_bearing_contact_accepted' then receipts[#receipts+1]=event end
end
assert(#receipts==1,'Only an actual accepted collision may emit one receipt per exact endpoint pair')
local details=receipts[1].details
assert(details.source_task=='contact' and details.held=='red')
assert(details.collision_measured==true and details.depth_gate_verified==true)
assert(details.endpoint_error_rad<.008 and details.face_gap_m<=.003)
''')

    def test_unknown_pair_is_still_an_actual_collision(self):
        self.lua_run(endpoint_wait_scenario(certified=False)+r'''
assert(result.failure and #result.collision_failures==1,'An uncertified pair must fail')
''')

    def test_deep_bearing_overlap_still_exceeds_the_original_three_mm_gate(self):
        self.lua_run(endpoint_wait_scenario(depth=.004)+r'''
assert(result.failure and #result.collision_failures==1,'Four mm overlap must fail')
''')

    def test_new_motion_cannot_inherit_the_old_endpoint_contact_permission(self):
        self.lua_run(endpoint_wait_scenario(new_motion=True)+r'''
assert(result.active_B=='depart::p0000','A new physical motion must have begun')
assert(result.failure and #result.collision_failures==1,'A new node must expire the old endpoint permission')
''')

    def test_drift_from_the_acquired_joint_endpoint_expires_contact_permission(self):
        self.lua_run(endpoint_wait_scenario(joint_drift=.009)+r'''
assert(result.failure and #result.collision_failures==1,'A nine mrad endpoint drift must expire the contact')
''')

    def test_equivalent_cyclic_joint_position_is_the_same_endpoint(self):
        self.lua_run(endpoint_wait_scenario(joint_drift=6.283185307179586)+r'''
assert(not result.failure,result.failure)
assert(#result.collision_failures==0,'A full turn representative must preserve the acquired joint endpoint')
''')

    def test_release_of_the_carried_part_expires_contact_permission(self):
        self.lua_run(endpoint_wait_scenario(release_held=True)+r'''
assert(result.node_states['release::skill']=='done','The carrying grip must actually release')
assert(result.failure and #result.collision_failures==1,'A released part must not retain a held-endpoint permission')
''')

    def test_grasp_of_the_lower_part_expires_contact_permission(self):
        self.lua_run(endpoint_wait_scenario(lower_change="grasp")+r'''
assert(result.node_states['lower_pick::skill']=='done','The other gripper must acquire the lower part')
assert(result.final_state.installed.pillar=='p1','Grasp alone has not detached the lower part yet')
assert(result.failure and #result.collision_failures==1,'An acquired lower part must invalidate the upper waiting contact')
''')

    def test_detach_of_the_lower_part_expires_contact_permission(self):
        self.lua_run(endpoint_wait_scenario(lower_change="detach")+r'''
assert(result.node_states['lower_detach::skill']=='done','The lower part must actually detach')
assert(not result.final_state.installed.pillar,'The lower slot registration must be removed')
assert(result.failure and #result.collision_failures==1,'A detached lower part must invalidate the upper waiting contact')
''')

    def test_no_collision_cannot_emit_a_bearing_acceptance_receipt(self):
        self.lua_run(endpoint_wait_scenario(colliding=False)+r'''
assert(not result.failure,result.failure)
for _,event in ipairs(result.events) do
 assert(event.kind~='endpoint_bearing_contact_accepted','No-collision samples cannot invent an acceptance receipt')
end
''')


if __name__ == "__main__":
    unittest.main()
