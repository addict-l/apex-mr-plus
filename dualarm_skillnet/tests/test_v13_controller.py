"""Compile and exercise controller state transitions with an isolated Lua VM.

The installed simulator's Lua DLL is reused without starting a simulator or
claiming physical execution. Its API is replaced with a deterministic mock.
"""
import ctypes
import os
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALL = Path(os.environ.get("COPPELIASIM_ROOT", r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu"))
BASE = (ROOT / "apexmr/controller.lua").read_text(encoding="utf-8")
EXTENSION = (ROOT / "simbridge/controller_ext.lua").read_text(encoding="utf-8")

MOCK = r'''
math.atan2=math.atan
local clock=0
local ids=1000
local aliases={[1]='APEX_MR_Scene3',[2]='Baseplate',[3]='Workbench',[50]='red',
 [10]='A_UR5',[11]='A_TCP',[12]='A_GripperPalm',[13]='A_SupportShoe',
 [14]='A_PadForceSensor',[15]='A_ToolForceSensor',
 [20]='B_UR5',[21]='B_TCP',[22]='B_GripperPalm',[23]='B_SupportShoe',
 [24]='B_PadForceSensor',[25]='B_ToolForceSensor'}
local parents={[10]=1,[20]=1,[12]=106,[13]=12,[22]=206,[23]=22,[50]=1}
local pose={0,0,.810,0,0,0,1}
local binding={name='red',offset={0,0,.044},quaternion={1,0,0,0}}
local zero={0,0,0,0,0,0}
local all={} for h in pairs(aliases) do all[#all+1]=h end
local writes=0
local collide=false
local collide_shape=10
sim={handle_world=-1,handle_all=-2,object_shape_type=0,shapeintparam_static=10,
 shapeintparam_respondable=11,objintparam_visibility_layer=12,verbosity_errors=0}
function sim.getObject(path) return 1 end
function sim.getSimulationTime() return clock end
function sim.getObjectsInTree(root,kind)
 if root==10 then return {10,12,13} elseif root==20 then return {20,22,23} end return all
end
function sim.getObjectAlias(h) return aliases[h] or ('mock_'..h) end
function sim.getObjectParent(h) return parents[h] or -1 end
function sim.getObjectInt32Param(h,key) return key==10 and 0 or 1 end
function sim.setObjectInt32Param() end
function sim.isDynamicallyEnabled() return true end
function sim.getJointPosition() return 0 end
function sim.getJointVelocity() return 0 end
function sim.getJointForce() return 0 end
function sim.setJointTargetPosition() writes=writes+1 end
function sim.createCollection() ids=ids+1;return ids end
function sim.addItemToCollection() end
function sim.destroyCollection() end
function sim.createForceSensor() ids=ids+1;return ids end
function sim.setObjectAlias(h,name) aliases[h]=name end
function sim.setObjectParent(h,parent,keep) assert(keep);parents[h]=parent end
function sim.getObjectPose(h,relative)
 if relative==11 or relative==21 then return {0,0,.044,1,0,0,0} end
 return {table.unpack(pose)}
end
function sim.setObjectPose(h,value,relative)
 assert(h>=1000,'Task body was teleported')
end
function sim.getObjectMatrix() return {1,0,0,0,0,1,0,0,0,0,1,0} end
function sim.readForceSensor() return 1,{0,0,-.2943},{0,0,0} end
function sim.removeObjects(handles) for _,h in ipairs(handles) do parents[h]=nil end end
function sim.resetDynamicObject() end
function sim.addLog() end
function sim.checkCollision(a,b)
 if collide and ((a==collide_shape and b==50) or (a==50 and b==collide_shape)) then return 1,{a,b} end
 return 0,{}
end
local cfg={profile='skillnet_v1_3',root=1,table=3,baseplate=2,parts={{id='red',size={.12,.14,.076},type='2x4'}},
 blocks={red=50},obstacles={},arms={},initial_attachments={},initial_installed={},
 runtime={initial_settle_s=.01,final_settle_s=.01},program={nodes={},edges={}}}
for index,side in ipairs({'A','B'}) do
 local base=index*10
 cfg.arms[side]={model=base,joints={},tip=base+1,palm=base+2,support_pad=base+3,
  support_sensor=base+4,sensor=base+5,home_q=zero}
 for j=1,6 do cfg.arms[side].joints[j]=index*100+j end
end
function sim.readCustomTableData() return cfg end
mockCfg=cfg
package.preload.sim=function() return sim end
local function node(id,arm,action,meta,q)
 meta=meta or {};meta.action=action;meta.dwell_s=.02
 return {id=id,task_id=id,arm=arm,node_type='skill',kind=action,part='red',duration=.02,
  metadata=meta,q_start=q or zero,q_end=q or zero,q_samples={q or zero}}
end
local function run()
 sysCall_init()
 for _=1,200 do clock=clock+.05;sysCall_actuation();if getState().done or getState().failure then break end end
 return getResult()
end
'''


class ControllerSourceGuards(unittest.TestCase):
    def test_generic_runtime_never_teleports_bodies_or_joints(self):
        code = re.sub(r"--[^\n]*", "", EXTENSION)
        self.assertNotRegex(code, r"sim\.(?:setObjectPose|setObjectPosition|setJointPosition)\s*\(")
        self.assertIn("sim.setJointTargetPosition", code)
        self.assertIn("sim.checkCollision", code)
        self.assertIn("sim.readCustomTableData", BASE)

    def test_profile_dispatch_preserves_archive_default(self):
        self.assertIn("profileCfg.profile=='skillnet_v1_3'", BASE)
        for method in ("init", "actuation", "dyn", "sensing", "cleanup", "state", "result"):
            self.assertIn("legacyCallbacks."+method+"(", BASE)


@unittest.skipUnless((INSTALL / "lua.dll").is_file(), "CoppeliaSim 自带 Lua DLL 不可用")
class LuaControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = os.add_dll_directory(str(INSTALL))
        cls.lua = ctypes.CDLL(str(INSTALL / "lua.dll"))
        cls.lua.luaL_newstate.restype = ctypes.c_void_p
        cls.lua.luaL_openlibs.argtypes = [ctypes.c_void_p]
        cls.lua.luaL_loadbufferx.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                           ctypes.c_char_p, ctypes.c_char_p]
        cls.lua.lua_pcallk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]
        cls.lua.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_size_t)]
        cls.lua.lua_tolstring.restype = ctypes.c_char_p
        cls.lua.lua_close.argtypes = [ctypes.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.directory.close()

    def lua_run(self, assertions, *, execute=True):
        # The copied controller has its own chunk-local cfg. Keep test inputs
        # outside that namespace while the mock API retains its closure.
        assertions = re.sub(r"\bcfg\b", "mockCfg", assertions)
        source = (MOCK+"\n"+BASE+"\n"+EXTENSION+"\n"+assertions).encode("utf-8")
        state = self.lua.luaL_newstate()
        try:
            self.lua.luaL_openlibs(state)
            result = self.lua.luaL_loadbufferx(state, source, len(source), b"mock_controller", None)
            if result == 0 and execute:
                result = self.lua.lua_pcallk(state, 0, 0, 0, 0, None)
            self.assertEqual(result, 0,
                (self.lua.lua_tolstring(state, -1, None) or b"Lua error").decode("utf-8"))
        finally:
            self.lua.lua_close(state)

    def test_combined_source_compiles_with_installed_lua(self):
        self.lua_run("", execute=False)

    def test_handover_requires_receive_before_giver_release(self):
        self.lua_run(r'''
cfg.program.nodes={
 node('pick','A','grasp',{payload_after=binding}),
 node('take','B','grasp',{handover=true,giver='A',taker='B',transfer_payload=binding}),
 node('give','A','release',{handover=true,giver='A',taker='B'}),
 node('place','B','release',{target_pose=pose,location='board'})}
cfg.program.edges={{source='pick',target='take'},{source='take',target='give'},{source='give',target='place'}}
local result=run();assert(result.done,result.failure);assert(writes>0)
assert(result.final_state.held.A==false and result.final_state.held.B==false)
local received,released
for _,event in ipairs(result.events) do
 if event.kind=='handover_received' then received=event.time end
 if event.kind=='handover_released' then released=event.time end
end
assert(received and released and received<released)
assert(parents[50]==1,'Final place must release free body to scene root')
''')

    def test_install_and_detach_preserve_gripper_then_release_dynamic_body(self):
        self.lua_run(r'''
cfg.initial_attachments.red=700;cfg.initial_installed.base='red';parents[50]=700
cfg.program.nodes={
 node('pick','A','grasp',{payload_after=binding,installed_grasp=true}),
 node('remove','A','detach',{slot='base'}),
 node('insert','A','install',{slot='base',target_pose=pose,below={}}),
 node('pick_again','A','grasp',{payload_after=binding,installed_grasp=true}),
 node('remove_again','A','detach',{slot='base'}),
 node('place','A','release',{target_pose=pose,location='stock'})}
for i=1,5 do cfg.program.edges[#cfg.program.edges+1]={source=cfg.program.nodes[i].id,target=cfg.program.nodes[i+1].id} end
local result=run();assert(result.done,result.failure)
assert(not result.final_state.installed.red);assert(not result.snap_dynamic.red)
local count=0
for _,event in ipairs(result.events) do
 if event.kind=='detach' then count=count+1;assert(event.details.gripper_retained and event.details.dynamic) end
end
assert(count==2);assert(parents[50]==1)
''')

    def test_giver_release_without_receiver_grasp_fails(self):
        self.lua_run(r'''
cfg.program.nodes={node('give','A','release',{handover=true,giver='A',taker='B'})}
local result=run();assert(not result.done)
assert(result.failure:find('before receiver grasp',1,true))
''')

    def test_actual_collision_check_records_pair_and_stops(self):
        self.lua_run(r'''
cfg.program.nodes={node('wait','A','wait')};sysCall_init();collide=true
for _=1,5 do clock=clock+.02;sysCall_sensing() end
local result=getResult();assert(not result.done)
assert(#result.collision_failures==1);assert(result.collision_monitor.enabled)
assert(result.collision_monitor.checks>0)
assert(result.collision_failures[1].first=='A_UR5')
assert(result.collision_failures[1].second=='red')
''')

    def test_tracking_guard_uses_measured_joint_error(self):
        self.lua_run(r'''
cfg.program.nodes={node('wait','A','wait',nil,{.5,.5,.5,.5,.5,.5})}
sysCall_init();clock=.02;sysCall_actuation();clock=.70;sysCall_actuation()
local result=getResult();assert(not result.done);assert(#result.tracking_failures==1)
assert(result.tracking_failures[1].error_rad>.49)
assert(result.tracking_monitor.samples>0 and result.tracking_monitor.enabled)
''')

    def test_exact_gripper_contact_pair_accepts_side_grasp_contact(self):
        self.lua_run(r'''
cfg.program.nodes={node('wait','A','wait',{contact_exclusions={{'A_GripperPalm','red'}}})}
cfg.program.nodes[1].duration=10
sysCall_init();clock=.02;sysCall_actuation();collide=true;collide_shape=12
for _=1,5 do clock=clock+.02;sysCall_sensing() end
local result=getResult();assert(not result.failure,result.failure)
assert(#result.collision_failures==0 and result.collision_monitor.checks>0)
''')


if __name__ == "__main__":
    unittest.main()
