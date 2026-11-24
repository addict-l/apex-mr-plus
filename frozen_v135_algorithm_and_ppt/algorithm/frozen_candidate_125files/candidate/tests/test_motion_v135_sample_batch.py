"""Public batch-query semantics in the bundled Lua VM; no simulator startup.

The API mock is stateful: scalar mpValidatePair calls are the independent
reference for each batch sample. Every mesh query is retained; only redundant
world writes may be removed. Performance assertions count writes, never time.
"""
import unittest
from pathlib import Path

from tests import test_v13_controller as controller_support

ROOT = Path(__file__).resolve().parents[1]
BACKEND = (ROOT / "apexmr/motion_backend.lua").read_text(encoding="utf-8")

MOCK = r"""
local function clone(value)
 if type(value)~='table' then return value end
 local result={} for key,item in pairs(value) do result[key]=clone(item) end return result
end
local function equal(a,b)
 if type(a)~=type(b) then return false end
 if type(a)~='table' then return a==b end
 for k,v in pairs(a) do if not equal(v,b[k]) then return false end end
 for k in pairs(b) do if a[k]==nil then return false end end
 return true
end
mock={poses={},joints={},collections={},nextCollection=1000,collision_calls=0,
      writes={joint=0,block=0,target=0,articulation=0}}
sim={handle_world=-1,handle_all=-2,handle_single=-3,object_shape_type=1,
     objectspecialproperty_collidable=1,shapeintparam_static=1,shapeintparam_respondable=2}
local aliases={[1]='APEX_MR_Scene3',[2]='Workbench',[3]='GateObstacle',
 [10]='A_UR5',[11]='A_tip',[12]='A_target',[13]='A_link',[14]='A_Finger',
 [20]='B_UR5',[21]='B_tip',[22]='B_target',[23]='B_link',[24]='B_Finger',
 [50]='brick',[51]='other',[70]='gate'}
for h in pairs(aliases) do mock.poses[h]={h*.1,0,1,0,0,0,1} end
for _,base in ipairs({100,200}) do for j=1,6 do mock.joints[base+j]=0 end end
mock.joints[70]=0
function sim.getSystemTime() return 0 end
function sim.getJointPosition(h) return assert(mock.joints[h],'Unknown joint') end
function sim.setJointPosition(h,q)
 mock.joints[h]=q
 local kind=h==70 and 'articulation' or 'joint';mock.writes[kind]=mock.writes[kind]+1
end
function sim.getObjectPose(h)
 if h==11 then return {-1+mock.joints[101],mock.joints[102],1,0,0,0,1} end
 if h==21 then return {1+mock.joints[201],mock.joints[202],1,0,0,0,1} end
 return clone(assert(mock.poses[h],'Unknown pose '..tostring(h)))
end
function sim.setObjectPose(h,p)
 mock.poses[h]=clone(p)
 local kind=(h==50 or h==51) and 'block' or 'target';mock.writes[kind]=mock.writes[kind]+1
end
function sim.setObjectPosition(h,frame,p)
 local pose=sim.getObjectPose(h);for i=1,3 do pose[i]=p[i] end;sim.setObjectPose(h,pose)
end
function sim.multiplyPoses(a,b)
 return {a[1]+b[1],a[2]+b[2],a[3]+b[3],b[4],b[5],b[6],b[7]}
end
function sim.getPoseInverse(p) return {-p[1],-p[2],-p[3],p[4],p[5],p[6],p[7]} end
function sim.getObjectAlias(h) return aliases[h] or 'joint_'..tostring(h) end
function sim.getObject(path)
 local name=path:match('[^/]+$');for h,alias in pairs(aliases) do if alias==name then return h end end
 error('Unknown path '..path)
end
function sim.getObjectsInTree(h,kind)
 if h==10 then return {10,13,14} end
 if h==20 then return {20,23,24} end
 if h==50 or h==51 then return {h} end
 local result={} for k in pairs(aliases) do result[#result+1]=k end return result
end
function sim.getObjectParent(h)
 if h==13 then return 101 elseif h==14 then return 106 end
 if h==23 then return 201 elseif h==24 then return 206 end
 return 1
end
function sim.getObjectSpecialProperty() return 1 end
function sim.getObjectInt32Param(h,param) return param==sim.shapeintparam_static and 0 or 1 end
function sim.createCollection()
 local h=mock.nextCollection;mock.nextCollection=h+1;mock.collections[h]={};return h
end
function sim.addItemToCollection(h,kind,item) table.insert(mock.collections[h],item) end
function sim.getCollectionObjects(h) return clone(mock.collections[h]) end
local function contains(h,item)
 if h==item then return true end
 for _,value in ipairs(mock.collections[h] or {}) do if value==item then return true end end return false
end
local function match(a,b,x,y) return (contains(a,x) and contains(b,y)) or (contains(a,y) and contains(b,x)) end
function sim.checkCollision(a,b)
 mock.collision_calls=mock.collision_calls+1
 if mock.inject_error and mock.joints[101]==.777 then error('injected_mesh_failure') end
 if match(a,b,13,23) and mock.joints[101]+mock.joints[201]>1 then return 1,{13,23} end
 if match(a,b,13,50) and mock.joints[101]>.5 then
  local x=mock.poses[50][1]
  if x==9 or (mock.sensitive_x and x==mock.sensitive_x) then return 1,{13,50} end
 end
 if match(a,b,13,3) and mock.joints[70]>.5 then return 1,{13,3} end
 return 0,{}
end
simIK={method_damped_least_squares=1,constraint_pose=1}
function simIK.createEnvironment() return 1 end
function simIK.createGroup() return 1 end
function simIK.setGroupCalculation() end
function simIK.addElementFromScene()
 local mapping={} for _,base in ipairs({100,200}) do for j=1,6 do mapping[base+j]=base+j end end
 return 1,mapping
end
function simIK.setElementPrecision() end
package.preload.sim=function() return sim end
package.preload.simIK=function() return simIK end
package.preload.simOMPL=function() return {} end
local cfg={root=1,root_path='/APEX_MR_Scene3',table=2,blocks={brick=50,other=51},
 block_shapes={brick={50},other={51}},obstacles={3},articulations={gate=70},arms={}}
for index,side in ipairs({'A','B'}) do
 local base=index*10;local jointBase=index*100
 cfg.arms[side]={model=base,tip=base+1,target=base+2,joints={}}
 for j=1,6 do cfg.arms[side].joints[j]=jointBase+j end
end
function mock.setup()
 local answer=mpSetup(cfg);assert(answer.ok,answer.error)
 mock.initial={poses=clone(mock.poses),joints=clone(mock.joints)}
end
function mock.resetCounts()
 mock.writes={joint=0,block=0,target=0,articulation=0};mock.collision_calls=0
end
function mock.restored()
 assert(equal(mock.poses,mock.initial.poses),'object/target pose escaped transaction')
 assert(equal(mock.joints,mock.initial.joints),'joint/articulation escaped transaction')
end
function mock.q(first) return {first,0,0,0,0,0} end
function mock.pose(x) return {x,0,1,0,0,0,1} end
function mock.sample(a,b,placed,options,payloads)
 return {A=a and mock.q(a) or nil,B=b and mock.q(b) or nil,
  placed=placed or {brick=mock.pose(2),other=mock.pose(3)},
  options=options or {},payloads=payloads or {}}
end
function mock.compare(samples,outerOptions)
 local expected={};mock.resetCounts()
 for _,s in ipairs(samples) do
  expected[#expected+1]=mpValidatePair(s.A or mock.q(0),s.B or mock.q(0),
   s.payloads,s.placed,s.options or outerOptions)
  mock.restored()
 end
 local referenceChecks=mock.collision_calls
 mock.resetCounts();local answer=mpValidateSamples(samples,outerOptions)
 assert(equal(answer,expected),'batch differs from independently restored scalar queries')
 assert(mock.collision_calls==referenceChecks,'batch removed or repeated full mesh predicates')
 mock.restored();return answer
end
"""


@unittest.skipUnless((controller_support.INSTALL / "lua.dll").is_file(), "CoppeliaSim 自带 Lua DLL 不可用")
class SampleBatchSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller_support.LuaControllerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        controller_support.LuaControllerTests.tearDownClass.__func__(cls)

    def lua_run(self, assertions):
        source = (MOCK + "\n" + BACKEND + "\nmock.setup()\n" + assertions).encode("utf-8")
        state = self.lua.luaL_newstate()
        try:
            self.lua.luaL_openlibs(state)
            code = self.lua.luaL_loadbufferx(state, source, len(source), b"sample_batch_boundary", None)
            if code == 0:
                code = self.lua.lua_pcallk(state, 0, 0, 0, 0, None)
            self.assertEqual(code, 0, (self.lua.lua_tolstring(state, -1, None) or b"Lua error").decode("utf-8"))
        finally:
            self.lua.lua_close(state)

    def test_all_samples_match_scalar_full_geometry_and_first_collision_pair(self):
        self.lua_run(r"""
local rows={mock.sample(.1,.2),mock.sample(.8,.4),mock.sample(.2,.1),mock.sample(.9,.8)}
local answer=mock.compare(rows)
assert(answer[1].ok and not answer[2].ok and answer[3].ok and not answer[4].ok)
assert(answer[2].problem.reason=='interarm_collision')
assert(equal(answer[2].problem.pair,{'A_link','B_link'}))
""")

    def test_placed_contact_and_articulation_changes_are_not_reused_as_old_world(self):
        self.lua_run(r"""
local near={brick=mock.pose(9),other=mock.pose(3)}
local rows={mock.sample(.8,0),mock.sample(.8,0,near),
 mock.sample(.8,0,near,{contact_exclusions={{13,50}}}),mock.sample(.8,0,near),
 mock.sample(.1,0,nil,{articulations={gate=.9}}),mock.sample(.1,0,nil,{articulations={gate=.1}})}
local answer=mock.compare(rows)
assert(answer[1].ok and not answer[2].ok and answer[3].ok and not answer[4].ok)
assert(not answer[5].ok and answer[6].ok)
""")

    def test_partial_joint_inputs_and_bare_payload_bindings_keep_scalar_semantics(self):
        self.lua_run(r"""
local rows={mock.sample(.2,nil,nil,nil,{A='brick'}),
 mock.sample(.5,nil,nil,nil,{A='brick'}),mock.sample(nil,.4),mock.sample(nil,.9)}
mock.compare(rows)
""")

    def test_explicit_payload_and_release_modes_keep_transform_and_baseline_restoration(self):
        self.lua_run(r"""
local held={A={name='brick',offset={0,0,.4},quaternion={0,0,0,1}}}
local rows={mock.sample(.1,.1,{other=mock.pose(3)},nil,held),
 mock.sample(.4,.1,{other=mock.pose(3)},nil,held),mock.sample(.4,.1),
 mock.sample(.7,.4,{other=mock.pose(3)},nil,held)}
mock.compare(rows)
""")

    def test_numeric_world_context_comparison_does_not_round_distinct_poses(self):
        self.lua_run(r"""
mock.sensitive_x=1
local x=1+2e-15;assert(x~=1)
local rows={mock.sample(.8,0,{brick=mock.pose(1),other=mock.pose(3)}),
 mock.sample(.8,0,{brick=mock.pose(x),other=mock.pose(3)})}
local answer=mock.compare(rows);assert(not answer[1].ok and answer[2].ok)
""")

    def test_mid_batch_exception_restores_both_arms_targets_objects_and_articulations(self):
        self.lua_run(r"""
mock.inject_error=true
local rows={mock.sample(.1,.1,{brick=mock.pose(5),other=mock.pose(6)}, {articulations={gate=.2}}),
 mock.sample(.777,.1,{brick=mock.pose(7),other=mock.pose(8)}, {articulations={gate=.3}})}
local answer=mpValidateSamples(rows)
assert(answer.ok==false and answer.error:find('injected_mesh_failure'))
mock.restored()
""")

    def test_public_statistics_report_every_geometry_call_and_exact_reuse(self):
        self.lua_run(r"""
mpSampleBatchStatistics({reset=true})
local held={A={name='brick',offset={0,0,.4},quaternion={0,0,0,1}}}
local rows={mock.sample(.1,0,{other=mock.pose(3)},nil,held),
 mock.sample(.2,0,{other=mock.pose(3)},nil,held),mock.sample(.3,0,{other=mock.pose(3)},nil,held)}
mock.compare(rows)
local stats=mpSampleBatchStatistics()
assert(stats.batches==1 and stats.samples_requested==3 and stats.samples_checked==3)
assert(stats.geometry_calls==3 and stats.shape_collision_queries==mock.collision_calls)
assert(stats.full_context_resets==1 and stats.world_reuse_samples==2)
for _,name in ipairs({'snapshot_seconds','restore_seconds','context_seconds',
 'configuration_payload_seconds','geometry_seconds','context_comparison_seconds','batch_seconds'}) do
 assert(type(stats[name])=='number' and stats[name]>=0,'missing phase clock: '..name)
end
local previous=mpSampleBatchStatistics({reset=true});assert(previous.samples_checked==3)
assert(mpSampleBatchStatistics().samples_checked==0)
""")

    def test_forced_legacy_path_changes_writes_not_samples_or_collision_results(self):
        self.lua_run(r"""
local held={A={name='brick',offset={0,0,.4},quaternion={0,0,0,1}}}
local rows={}
for i=1,5 do rows[#rows+1]=mock.sample(i*.01,0,{other=mock.pose(3)},nil,held) end
mpSampleBatchStatistics({reset=true});mock.resetCounts()
local optimized=mpValidateSamples(rows)
local fastWrites=mock.writes.joint+mock.writes.block
mpSampleBatchStatistics({reset=true});mock.resetCounts()
local legacy=mpValidateSamples(rows,{legacy_batch_reset=true})
assert(equal(optimized,legacy));mock.restored()
local stats=mpSampleBatchStatistics()
assert(stats.geometry_calls==#rows and stats.world_reuse_samples==0)
assert(stats.full_context_resets==#rows and stats.forced_legacy_batches==1)
assert(mock.writes.joint+mock.writes.block>fastWrites,'legacy control no longer exercises old resets')
""")

    def test_repeated_complete_context_avoids_redundant_joint_and_block_resets(self):
        self.lua_run(r"""
local held={A={name='brick',offset={0,0,.4},quaternion={0,0,0,1}}}
local rows={}
for i=1,8 do rows[#rows+1]=mock.sample(i*.01,0,{other=mock.pose(3)},{articulations={gate=.1}},held) end
mock.compare(rows)
-- A scalar-style reset on every sample performs >=24 joint and >=4 block
-- writes per sample. The complete, unchanged world should be initialized once.
assert(mock.writes.joint<24*#rows,'redundant per-sample joint reset remains: '..mock.writes.joint)
assert(mock.writes.block<4*#rows,'redundant per-sample object reset remains: '..mock.writes.block)
""")


if __name__ == "__main__":
    unittest.main()
