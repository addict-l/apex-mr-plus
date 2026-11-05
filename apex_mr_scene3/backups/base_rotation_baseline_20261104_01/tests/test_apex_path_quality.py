"""Exercise the actual Lua path-selection code without starting CoppeliaSim.

The installed Lua runtime executes production functions with synthetic geometry
and OMPL adapters. These are algorithm/contract regressions, not physical-scene
collision evidence; the complete scene is validated by the pipeline separately.
"""
import ctypes
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apex.motion import MotionPlanner


LUA_FIXTURE = r'''
local fakeTime=0
local solveCalls=0
local checkCalls=0
local candidates={}
local activeCandidate=0
local tickPerCheck=0
local geometryOracle=function(q,ctx) return true end
local simMock={getSystemTime=function()return fakeTime end,
    getJointInterval=function()return false,{-6.3,12.6} end}
local omplMock={Algorithm={RRTConnect=1},StateSpaceType={joint_position=1}}
omplMock.createTask=function()return {} end
omplMock.createStateSpace=function()return {} end
for _,name in ipairs({'setStateSpace','setAlgorithm','setStateValidityCheckingResolution',
    'setVerboseLevel','setStateValidationCallback','setStartState','setGoalState',
    'setup','destroyTask','destroyStateSpace'}) do omplMock[name]=function()end end
omplMock.getPlannerData=function()return {},nil,nil,{} end
omplMock.solve=function(_,budget)
    assert(budget>0)
    solveCalls=solveCalls+1;activeCandidate=solveCalls
    fakeTime=fakeTime+math.min(.04,budget)
    return candidates[activeCandidate]~=nil
end
omplMock.hasExactSolution=function()return candidates[activeCandidate]~=nil end
omplMock.getPath=function()
    local out={}
    for _,q in ipairs(candidates[activeCandidate]) do
        for _,x in ipairs(q) do out[#out+1]=x end
    end
    return out
end
package.preload.sim=function()return simMock end
package.preload.simIK=function()return {} end
package.preload.simOMPL=function()return omplMock end
'''

LUA_ADAPTER = r'''
A={A={joints={1,2,3,4,5,6}},B={joints={7,8,9,10,11,12}}}
transaction=function(f)return f()end
newContext=function(payloads,placed,options,side)
    return {payloads=payloads,placed=placed,options=options,side=side}
end
validJoint=function(side,q,ctx)
    checkCalls=checkCalls+1;checks=checks+1;fakeTime=fakeTime+tickPerCheck
    local ok=geometryOracle(q,ctx)
    if not ok then lastProblem={reason='synthetic_collision'} end
    return ok
end
local function q(x,y)return {x,y or 0,0,0,0,0} end
local function close(a,b)assert(math.abs(a-b)<1e-8,tostring(a)..' ~= '..tostring(b)) end
'''


class NativeLuaPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install = Path(os.environ.get('COPPELIASIM_ROOT', r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu'))
        dll = install / 'lua.dll'
        if os.name != 'nt' or not dll.exists():
            raise unittest.SkipTest('Installed native Lua runtime unavailable')
        cls.dll_directory = os.add_dll_directory(str(install))
        cls.lua = ctypes.CDLL(str(dll))
        cls.lua.luaL_newstate.restype = ctypes.c_void_p
        cls.lua.luaL_openlibs.argtypes = [ctypes.c_void_p]
        cls.lua.luaL_loadbufferx.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                           ctypes.c_char_p, ctypes.c_char_p]
        cls.lua.lua_pcallk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]
        cls.lua.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        cls.lua.lua_tolstring.restype = ctypes.c_char_p
        cls.lua.lua_close.argtypes = [ctypes.c_void_p]
        cls.backend = (ROOT / 'apex/motion_backend.lua').read_text(encoding='utf-8')

    def execute(self, scenario):
        code = (LUA_FIXTURE + '\n' + self.backend + '\n' + LUA_ADAPTER + '\n' + scenario).encode('utf-8')
        state = self.lua.luaL_newstate()
        self.assertTrue(state)
        try:
            self.lua.luaL_openlibs(state)
            status = self.lua.luaL_loadbufferx(state, code, len(code), b'path_quality_regression', None)
            if not status:
                status = self.lua.lua_pcallk(state, 0, 0, 0, 0, None)
            if status:
                self.fail(self.lua.lua_tolstring(state, -1, None).decode('utf-8', errors='replace'))
        finally:
            self.lua.lua_close(state)

    def test_direct_path_checks_interior_and_preserves_full_context(self):
        self.execute(r'''
local seenInterior=false
local payload={name='K1',offset={.03,0,0},quaternion={0,0,0,1}}
local placed={Bridge={0,0,1,0,0,0,1}}
local options={other_q=q(2),other_payload={name='K2'},
    articulations={SupportLift=.01},contact_exclusions={{'K1','B_Finger'}},contact_tolerance=.002}
geometryOracle=function(v,ctx)
    assert(ctx.payloads.B==payload and ctx.placed==placed and ctx.options==options)
    assert(ctx.options.articulations.SupportLift==.01)
    assert(ctx.options.other_payload.name=='K2')
    if v[1]>.1 and v[1]<.9 then seenInterior=true end
    return true
end
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',payload,placed,options)
assert(result.ok and seenInterior and checkCalls>200)
assert(solveCalls==0 and result.stats.ompl_solve_called==false)
assert(result.algorithm=='CollisionCheckedJointLine')
assert(result.path_source=='collision_checked_direct' and result.stats.direct_valid)
close(result.cost,1)
''')

    def test_midsegment_collision_falls_back_and_selects_best_independent_rrt(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)end
candidates={
    {q(0),q(0,.8),q(1,.8),q(1)},
    {q(0),q(0,.3),q(1,.3),q(1)},
    {q(0),q(0,.6),q(1,.6),q(1)}}
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {solve_time=1,simplify=false,max_rrt_candidates=3,quality_stretch_threshold=1.01})
assert(result.ok and result.algorithm=='RRTConnect' and solveCalls==3)
assert(result.stats.direct_attempted and not result.stats.direct_valid)
assert(result.stats.ompl_solve_called and result.stats.selected_candidate==2)
assert(#result.stats.candidates==3 and result.stats.validated_states>200)
close(result.cost,1.6)
for _,v in ipairs(result.path) do assert(geometryOracle(v))end
''')

    def test_invalid_direct_and_no_rrt_solution_do_not_return_a_line(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55)end
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},{solve_time=1})
assert(not result.ok and result.path==nil and solveCalls>0)
assert(result.stats.direct_rejection.reason=='synthetic_collision')
''')

    def test_candidates_share_one_solve_deadline_and_keep_best_found_path(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)end
candidates={{q(0),q(0,.8),q(1,.8),q(1)},
    {q(0),q(0,.3),q(1,.3),q(1)},
    {q(0),q(0,.21),q(1,.21),q(1)}}
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {solve_time=.05,simplify=false,max_rrt_candidates=3,quality_stretch_threshold=1.01})
assert(result.ok and solveCalls==2 and result.stats.selected_candidate==2)
assert(result.stats.search_wall_time_s<=.05+1e-10)
close(result.cost,1.6)
''')

    def test_direct_path_cannot_bypass_joint_limits(self):
        self.execute(r'''
local result=mpPlanJoint('B',q(0),q(7),'RRTConnect',{}, {},{})
assert(not result.ok and result.error=='Endpoint outside joint bounds' and solveCalls==0)
''')

    def test_shortcut_reduces_late_detour_and_checks_every_accepted_edge(self):
        self.execute(r'''
local path={q(0),q(.2,.4),q(.4,.4),q(.5,.6),q(.7,.8),q(.9,.6),q(1)}
local checked=0
local oracle=function(v)
    checked=checked+1
    return not(v[1]>.15 and v[1]<.35 and v[2]<.3)
end
local better,report=checkedShortcut(path,oracle,.005,96,10)
assert(report.accepted>0 and checked>50 and pathCost(better)<pathCost(path))
close(better[1][1],0);close(better[#better][1],1)
for i=2,#better do assert(checkedSegment(better[i-1],better[i],oracle,.005))end
''')

    def test_shortcut_budget_never_accepts_partially_checked_segment(self):
        self.execute(r'''
local path={q(0),q(0,.5),q(1,.5),q(1)}
local oracle=function(v)fakeTime=fakeTime+.01;return true end
local result,report=checkedShortcut(path,oracle,.005,96,.025)
assert(report.budget_exhausted and report.accepted==0)
close(pathCost(result),pathCost(path))
''')

    def test_final_certificate_rejects_invalid_ompl_result(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55)end
candidates={{q(0),q(1)}}
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {direct_first=false,simplify=false,max_rrt_candidates=1})
assert(not result.ok and result.error=='Dense path validation failed')
''')

    def test_new_rrt_cannot_regress_beyond_revalidated_reference_candidate(self):
        self.execute(r'''
local payload={name='K1'}
geometryOracle=function(v,ctx)
    assert(ctx.payloads.B==payload and ctx.options.articulations.SupportLift==.01)
    return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)
end
candidates={{q(0),q(0,.8),q(1,.8),q(1)},
    {q(0),q(0,.9),q(1,.9),q(1)},
    {q(0),q(0,.7),q(1,.7),q(1)}}
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',payload,{},
    {solve_time=1,simplify=false,articulations={SupportLift=.01},
     candidate_paths={{source='baseline_reference',path={q(0),q(0,.3),q(1,.3),q(1)}}}})
assert(result.ok and result.algorithm=='ValidatedReferencePath')
assert(result.path_source=='validated_reference_candidate' and result.stats.ompl_solve_called)
assert(result.stats.selected_reference_candidate==1 and result.stats.reference_candidates[1].valid)
assert(result.stats.reference_candidates[1].validated_states>200)
assert(result.stats.selected_path_backend=='simulator_geometry')
close(result.cost,1.6)
''')

    def test_reference_collision_is_rejected_against_current_context(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)end
candidates={{q(0),q(0,.3),q(1,.3),q(1)}}
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {solve_time=1,simplify=false,max_rrt_candidates=1,
     candidate_paths={{source='previously_clear',path={q(0),q(1)}}}})
assert(result.ok and result.algorithm=='RRTConnect')
assert(not result.stats.reference_candidates[1].valid)
assert(result.stats.reference_candidates[1].rejection=='current_context_collision')
close(result.cost,1.6)
''')

    def test_valid_reference_survives_no_rrt_solution_and_local_search_box(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)end
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {solve_time=1,simplify=false,joint_bound_margin=.01,
     candidate_paths={{source='known_valid',path={q(0),q(0,.3),q(1,.3),q(1)}}}})
assert(result.ok and result.algorithm=='ValidatedReferencePath' and solveCalls>0)
assert(result.stats.reference_candidates[1].valid)
close(result.cost,1.6)
''')

    def test_reference_bounds_and_endpoints_are_not_exempt(self):
        self.execute(r'''
geometryOracle=function(v)return not(v[1]>.45 and v[1]<.55 and math.abs(v[2])<.2)end
local result=mpPlanJoint('B',q(0),q(1),'RRTConnect',{}, {},
    {solve_time=1,simplify=false,candidate_paths={
      {source='out_of_bounds',path={q(0),q(0,7),q(1,7),q(1)}},
      {source='wrong_endpoint',path={q(0),q(0,.3),q(1,.3),q(.9)}}}})
assert(not result.ok and not result.stats.reference_candidates[1].valid)
assert(result.stats.reference_candidates[2].rejection=='endpoint_mismatch')
''')

    def test_refined_ik_uses_original_seed_winding_and_rechecks_geometry(self):
        self.execute(r'''
sim.getJointInterval=function()return true,{0,0}end
setq=function(arm,values)arm.q=copy(values)end
pose=function()return {0,0,0,0,0,0,1}end
ikFromSeed=function()return q(-math.pi/2)end
local checked=false
geometryOracle=function(v)
    close(v[1],3*math.pi/2);checked=true
    for _,x in ipairs(v)do assert(x>=-2*math.pi and x<=2*math.pi)end
    return true
end
local result=mpSolveIK('B',{0,0,0},{0,0,0,1},{},{},{seed=q(math.pi),multiple=false})
assert(result.ok and checked)
close(result.solutions[1][1]-math.pi,math.pi/2)
''')

    def test_equivalent_winding_must_fit_real_planning_limits(self):
        self.execute(r'''
sim.getJointInterval=function()return true,{0,0}end
local result=nearestEquivalent('B',q(.3),q(5.7))
close(result[1],.3) -- .3+2pi is closer but outside the allowed range.
local wrapped=nearestEquivalent('B',q(-math.pi+.02),q(math.pi-.02))
close(wrapped[1],math.pi+.02)
''')

    def test_valid_nearest_goal_can_be_disconnected_while_equivalent_goal_connects(self):
        self.execute(r'''
sim.getJointInterval=function()return true,{0,0}end
local payload={name='TemporarySupport'}
local permissions={{'A_SupportShoe','Bridge'}}
geometryOracle=function(v,ctx)
    assert(ctx.payloads.B==payload)
    assert(ctx.options.contact_exclusions==permissions)
    return not(v[1]>-1. and v[1]<-.5)
end
local options={solve_time=.05,contact_exclusions=permissions,simplify=false}
local nearest=mpPlanJoint('B',q(0),q(-math.pi/2),'RRTConnect',payload,{},options)
assert(not nearest.ok) -- Both endpoints are legal; every connection crosses the barrier.
local alternate=mpPlanJoint('B',q(0),q(3*math.pi/2),'RRTConnect',payload,{},options)
assert(alternate.ok and alternate.algorithm=='CollisionCheckedJointLine')
assert(alternate.stats.validated_states>500 and alternate.stats.direct_valid)
close(alternate.path[#alternate.path][1],3*math.pi/2)
close(math.sin(alternate.path[#alternate.path][1]),math.sin(-math.pi/2))
''')


class PlannerResultProvenanceTests(unittest.TestCase):
    def test_python_wrapper_does_not_relabel_direct_path_as_ompl(self):
        planner = MotionPlanner.__new__(MotionPlanner)
        planner.info = {'collision_geometry_policy': 'synthetic test oracle'}
        calls = []

        def call(*args):
            calls.append(args)
            return {'ok': True, 'algorithm': 'CollisionCheckedJointLine',
                    'path_source': 'collision_checked_direct',
                    'stats': {'ompl_solve_called': False}}

        planner._call = call
        result = planner.plan('B', [0.] * 6, [.1] * 6)
        self.assertEqual(result['algorithm'], 'CollisionCheckedJointLine')
        self.assertFalse(result['stats']['ompl_solve_called'])
        self.assertTrue(calls[0][-1]['direct_first'])
        self.assertEqual(calls[0][-1]['max_rrt_candidates'], 3)


if __name__ == '__main__':
    unittest.main()
