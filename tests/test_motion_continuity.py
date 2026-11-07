"""Synthetic motion regressions, never a substitute for dynamic execution.

The fixtures retain original TPG node identities, paths, and dependency edges.
No simulator is started. Lua parity uses the installed Lua interpreter library
and the real controller's isolated, side-effect-free plannedQ function.
"""
from copy import deepcopy
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normal.geometry import _configuration_at
from normal.program import parameterize, refresh_schedule
from normal.tpg import build_tpg, simulate_tpg


def q(value, second=0.):
    return [value, second, 0., 0., 0., 0.]


def task(name, arm='A', *, path=None, kind='transit', metadata=None, order=0):
    return {'id': name, 'arm': arm, 'kind': kind,
            'q_path': path if path is not None else [q(0.), q(.6)],
            'duration': 1., 'dependencies': [], 'reference_order': order,
            'metadata': deepcopy(metadata or {})}


def two_fragments(metadata=None):
    graph = build_tpg([task('travel', metadata=metadata)],
                      lambda _a, _b: False, max_joint_span=.3)
    assert len(graph['nodes']) == 2
    return graph


def at(node, elapsed):
    return np.asarray(_configuration_at(node, elapsed, 0., node['duration']))


def junction_derivatives(left, right, h=1e-4):
    t = left['duration']
    velocity_left = (at(left, t)-at(left, t-h))/h
    velocity_right = (at(right, h)-at(right, 0.))/h
    acceleration_left = (at(left, t)-2*at(left, t-h)+at(left, t-2*h))/h**2
    acceleration_right = (at(right, 2*h)-2*at(right, h)+at(right, 0.))/h**2
    return velocity_left, velocity_right, acceleration_left, acceleration_right


def lua_literal(value):
    if value is None:
        return 'nil'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        return '{'+','.join('['+lua_literal(k)+']='+lua_literal(v) for k,v in value.items())+'}'
    if isinstance(value, (list, tuple)):
        return '{'+','.join(lua_literal(v) for v in value)+'}'
    return repr(value)


class LuaControllerEvaluator:
    """Run actual controller mathematics without loading sim or a scene."""
    def __init__(self):
        installation = Path(os.environ.get('COPPELIASIM_ROOT',
                            r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu'))
        library = (Path('/Applications/coppeliaSim.app/Contents/Frameworks/liblua.dylib') if sys.platform == 'darwin' else installation/'lua.dll')
        if not library.is_file():
            raise unittest.SkipTest('Installed Lua library unavailable for real controller parity')
        self.lua = ctypes.CDLL(str(library))
        signatures = {
            'luaL_newstate': ([], ctypes.c_void_p),
            'luaL_openlibs': ([ctypes.c_void_p], None),
            'luaL_loadstring': ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
            'lua_pcallk': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.c_ssize_t, ctypes.c_void_p], ctypes.c_int),
            'lua_tolstring': ([ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)], ctypes.c_char_p),
            'lua_tonumberx': ([ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)], ctypes.c_double),
            'lua_close': ([ctypes.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lua, name)
            function.argtypes = arguments; function.restype = result
        self.source = (ROOT/'normal/controller.lua').read_text(encoding='utf-8')

    def evaluate(self, node, elapsed):
        # Load all real local helpers too (for example curveQ). Declaring the
        # controller callbacks has no simulator side effects; none are called.
        chunk = ("package.preload['simIK']=function() return {} end\npackage.preload['sim']=function() return {} end\n"+self.source+
                 '\nreturn table.unpack(plannedQ('+lua_literal(node)+','+repr(elapsed)+'))')
        return self.run(chunk, number_count=6)

    def run(self, chunk, *, number_count=None):
        state = self.lua.luaL_newstate()
        if not state:
            raise RuntimeError('Lua state allocation failed')
        try:
            self.lua.luaL_openlibs(state)
            status = self.lua.luaL_loadstring(state, chunk.encode('utf-8'))
            if not status:
                status = self.lua.lua_pcallk(state, 0, number_count or 1, 0, 0, None)
            if status:
                message = self.lua.lua_tolstring(state, -1, None)
                raise AssertionError('Controller Lua evaluation failed: '+message.decode('utf-8'))
            if number_count:
                return [self.lua.lua_tonumberx(state, i, None) for i in range(1, number_count+1)]
            return self.lua.lua_tolstring(state, -1, None).decode('utf-8')
        finally:
            self.lua.lua_close(state)


def run_controller_motion_fixture(graph, *, lag_one_step=False, wrap_measurements=None):
    """Actual dispatcher with synthetic servos; no physical validation claim.

    Lifecycle flags bypass unrelated scene setup only in this isolated test
    state. Event/dependency/continuation/target logic is the unmodified Lua.
    """
    mock = r'''
local fixtureTime=0
-- CoppeliaSim supplies the legacy atan2 alias; stock Lua 5.4 names it atan.
math.atan2=math.atan2 or math.atan
local fixtureActual,fixtureTarget={},{}
for i=1,12 do fixtureActual[i]=0;fixtureTarget[i]=0 end
local fixtureLag=LAG_FLAG
local fixtureWrap=WRAP_FLAG
local fixtureSim={handle_world=-1,verbosity_errors=1}
function fixtureSim.getSimulationTime() return fixtureTime end
function fixtureSim.getJointPosition(j)
 local value=fixtureActual[j]
 if fixtureWrap then
  value=math.atan(math.sin(value),math.cos(value))
  if fixtureWrap=='negative' and value>=math.pi-1e-12 then value=value-2*math.pi end
  if fixtureWrap=='positive' and value<=-math.pi+1e-12 then value=value+2*math.pi end
 end
 return value
end
function fixtureSim.setJointTargetPosition(j,value)
 fixtureTarget[j]=value
 if not fixtureLag then fixtureActual[j]=value end
end
function fixtureSim.readForceSensor(_) return 1,{0,0,0},{0,0,0} end
function fixtureSim.getObjectMatrix(_h,_r) return {1,0,0,0,0,1,0,0,0,0,1,0} end
function fixtureSim.addLog(_level,_message) end
function fixtureSim.getJointVelocity(j) return 0 end
package.preload['simIK']=function() return {} end
package.preload['sim']=function() return fixtureSim end
'''.replace('LAG_FLAG', lua_literal(lag_one_step)).replace('WRAP_FLAG', lua_literal(wrap_measurements))
    source = (ROOT/'normal/controller.lua').read_text(encoding='utf-8')
    setup = r'''
program=GRAPH_FIXTURE
cfg={runtime={}};runtime=cfg.runtime
arms={A={joints={1,2,3,4,5,6},hold={0,0,0,0,0,0}},
      B={joints={7,8,9,10,11,12},hold={0,0,0,0,0,0}}}
nodes={};pred={};states={};active={};held={};grips={};snaps={};support={};graspSensors={};snapSensors={}
events={};trace={};taskTimes={};enabled={};overloads={}
flags={buffer={},installed={},support_verified=true,arm_support_released=true,
       support_returned=true,gate_closed=true,permanent_verified=true}
done=false;failure=nil;settled=true;startTime=-1;stepCount=0
supportTare=0;supportSensor=99;supportLift=100;pressLoad=nil
peakTracking=0;peakVelocity=0;peakSupport=0
assemblyGuardNodes={};assemblyContactLimits={force_N=40,torque_Nm=8}
for _,node in ipairs(program.nodes) do
 nodes[node.id]=node;pred[node.id]={};states[node.id]='pending'
end
for _,edge in ipairs(program.edges) do
 pred[edge.target][#pred[edge.target]+1]=edge.source
end
local fixtureInitialized={}
for _,node in ipairs(program.nodes) do
 if not fixtureInitialized[node.arm] then
  fixtureInitialized[node.arm]=true
  for i,j in ipairs(arms[node.arm].joints) do
   fixtureActual[j]=node.q_start[i];fixtureTarget[j]=node.q_start[i]
  end
  arms[node.arm].hold={table.unpack(node.q_start)}
 end
end
local fixtureRows={}
for step=0,2000 do
 fixtureTime=step*.02
 if fixtureLag then for j=1,12 do fixtureActual[j]=fixtureTarget[j] end end
 sysCall_actuation()
 fixtureRows[#fixtureRows+1]=string.format('Q\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g',
  fixtureTime,fixtureTarget[1],fixtureSim.getJointPosition(1),fixtureTarget[7],
  fixtureTarget[6],fixtureSim.getJointPosition(6))
 if failure then error('Synthetic dispatcher fixture: '..failure) end
 if done then break end
end
assert(done,'Synthetic dispatcher fixture did not complete')
for _,event in ipairs(events) do
 fixtureRows[#fixtureRows+1]=string.format('E\t%s\t%s\t%.17g',
  event.kind,event.node or '',event.time)
end
return table.concat(fixtureRows,'\n')
'''.replace('GRAPH_FIXTURE', lua_literal(graph))
    output = LuaControllerEvaluator().run(mock+'\n'+source+'\n'+setup)
    samples, events = [], []
    for line in output.splitlines():
        row = line.split('\t')
        if row[0] == 'Q':
            samples.append(tuple(float(value) for value in row[1:]))
        else:
            events.append({'kind': row[1], 'node': row[2], 'time': float(row[3])})
    return np.asarray(samples), events


def curved_two_fragment_fixture(fixed_joint_6=0.):
    """Synthetic curve protocol fixture, with a milestone inside a blend."""
    from normal.curve import make_profile, eval_profile, eval_at_s
    knots = [q(0.), q(.08), q(.08+.08*math.cos(math.radians(5.)), .08*math.sin(math.radians(5.)))]
    for point in knots:
        point[5] = fixed_joint_6
    profile = make_profile(knots, .6, 1.2, 8.)
    graph = two_fragments()
    total = profile['duration']
    for index, node in enumerate(graph['nodes']):
        offset, duration = index*total/2., total/2.
        node['q_start'] = eval_profile(profile, offset)
        node['q_end'] = eval_profile(profile, offset+duration)
        node['q_samples'] = [eval_profile(profile, t) for t in np.linspace(offset, offset+duration, 41)]
        node['duration'] = node['nominal_duration'] = duration
        u = (offset+duration)/total
        tangent = np.asarray(eval_at_s(profile, profile['length']*(10*u**3-15*u**4+6*u**5), 1))
        tangent /= np.linalg.norm(tangent)
        node['motion_segments'] = [{'q0': knots[0], 'q1': knots[-1], 'duration': duration,
                                   'profile_duration': total, 'profile_offset': offset,
                                   'geometry_profile': profile, 'progress_direction': tangent.tolist()}]
        node['metadata'].update(motion_run_id=graph['nodes'][0]['id'],
                                motion_run_offset=offset, motion_run_duration=total)
    graph['nodes'][0]['metadata']['continuation_to'] = graph['nodes'][1]['id']
    graph['nodes'][1]['metadata']['continuation_from'] = graph['nodes'][0]['id']
    return graph, profile


class MotionContinuityTests(unittest.TestCase):
    def assertStopped(self, left, right):
        vl, vr, _al, _ar = junction_derivatives(left, right)
        self.assertLess(np.linalg.norm(vl), 1e-5)
        self.assertLess(np.linalg.norm(vr), 1e-5)
        self.assertFalse(left['metadata'].get('continuation_to'))
        self.assertFalse(right['metadata'].get('continuation_from'))

    def test_collinear_fragments_have_nonzero_c2_junction_and_unchanged_path(self):
        graph = two_fragments()
        original = deepcopy(graph)
        parameterize(graph)
        left, right = graph['nodes']
        vl, vr, al, ar = junction_derivatives(left, right)
        self.assertGreater(vl[0], .05, 'Artificial TPG waypoint still forces a stop')
        self.assertGreater(vr[0], .05)
        np.testing.assert_allclose(vl, vr, atol=1e-4, rtol=0.)
        np.testing.assert_allclose(al, ar, atol=.003, rtol=0.)
        self.assertEqual(left['metadata']['continuation_to'], right['id'])
        self.assertEqual(right['metadata']['continuation_from'], left['id'])
        for before, node in zip(original['nodes'], graph['nodes']):
            for key in ('id', 'q_start', 'q_end', 'q_samples'):
                self.assertEqual(node[key], before[key])
            samples = np.asarray([at(node, t) for t in np.linspace(0., node['duration'], 51)])
            np.testing.assert_allclose(samples[:, 1:], 0., atol=1e-12)
            self.assertTrue(np.all(np.diff(samples[:, 0]) >= -1e-12))
            np.testing.assert_allclose(samples[0], before['q_start'], atol=1e-10)
            np.testing.assert_allclose(samples[-1], before['q_end'], atol=1e-10)
        for edge in original['edges']:
            self.assertIn(edge, graph['edges'])

    def test_interior_cross_arm_dependency_gates_whole_run_even_when_delayed(self):
        graph = build_tpg([task('other', 'B', path=[q(0.)], kind='skill'),
                           task('travel', order=1)],
                          lambda a,b: a['task_id']=='other' and b.get('fragment_index')==1,
                          max_joint_span=.3)
        original_edges = deepcopy(graph['edges'])
        parameterize(graph)
        other, first, second = graph['nodes']
        delayed = simulate_tpg(graph, {'other': 7.})['timestamps']
        self.assertGreaterEqual(delayed[first['id']]['start'], delayed[other['id']]['end'])
        self.assertAlmostEqual(delayed[second['id']]['start'], delayed[first['id']]['end'])
        self.assertEqual(first['metadata']['continuation_to'], second['id'])
        for edge in original_edges:
            self.assertIn(edge, graph['edges'])

    def test_external_dependency_return_path_rejects_continuation_cycle(self):
        graph = build_tpg([task('travel'), task('other', 'B', path=[q(0.)], kind='skill', order=1)],
                          lambda _a,_b: False, max_joint_span=.3)
        first, second, other = graph['nodes']
        # A legal DAG whose extra dependency cannot be moved to run entry:
        # A0 -> B -> A1. Reference indices do not control event execution.
        graph['edges'].extend([
            {'source': first['id'], 'target': other['id'], 'type': 'type2', 'reason': 'test_dependency'},
            {'source': other['id'], 'target': second['id'], 'type': 'type2', 'reason': 'test_dependency'},
        ])
        parameterize(graph)
        predicted = simulate_tpg(graph)['timestamps']
        self.assertGreaterEqual(predicted[second['id']]['start'], predicted[other['id']]['end'])
        self.assertStopped(first, second)

    def test_middle_world_effect_and_payload_change_preserve_stop(self):
        for mutation in ('effects', 'payload', 'contact_exclusions', 'articulation'):
            with self.subTest(mutation=mutation):
                graph = two_fragments()
                first, second = graph['nodes']
                if mutation == 'effects':
                    first['metadata']['effects'] = {'support_contacts': {}}
                elif mutation == 'payload':
                    second['payload'] = {'part': 'K1'}
                elif mutation == 'contact_exclusions':
                    second['metadata']['contact_exclusions'] = [['tool', 'Bridge']]
                else:
                    second['metadata']['options'] = {'articulations': {'SupportLift': .01}}
                parameterize(graph)
                self.assertStopped(first, second)

    def test_completion_effect_remains_only_on_final_continuous_node(self):
        effects = {'support_contacts': {}}
        graph = two_fragments({'effects': effects})
        parameterize(graph)
        first, last = graph['nodes']
        self.assertNotIn('effects', first['metadata'])
        self.assertEqual(last['metadata']['effects'], effects)
        self.assertGreater(junction_derivatives(first, last)[0][0], .05)

    def test_noncollinear_corner_and_direction_reversal_preserve_stop(self):
        for endpoint in (q(.3, .3), q(0.)):
            with self.subTest(endpoint=endpoint):
                graph = two_fragments()
                first, second = graph['nodes']
                second['q_end'] = endpoint
                second['q_samples'] = [second['q_start'][:], endpoint[:]]
                parameterize(graph)
                self.assertStopped(first, second)

    def test_skill_boundary_is_atomic_and_stopped(self):
        graph = build_tpg([task('approach', path=[q(0.), q(.3)]),
                           task('grasp', path=[q(.3)], kind='skill', order=1,
                                metadata={'action': 'grasp', 'dwell_s': .7}),
                           task('carry', path=[q(.3), q(.6)], order=2)],
                          lambda _a,_b: False, max_joint_span=.3)
        parameterize(graph)
        approach, skill, carry = graph['nodes']
        self.assertEqual(skill['node_type'], 'skill')
        self.assertGreaterEqual(skill['duration'], .7)
        self.assertStopped(approach, skill)
        self.assertStopped(skill, carry)

    def test_shared_quintic_respects_velocity_acceleration_and_jerk_bounds(self):
        limits = {'max_velocity_rad_s': .17, 'max_acceleration_rad_s2': .3, 'max_jerk_rad_s3': .7}
        graph = two_fragments(limits)
        parameterize(graph, vmax=.21, amax=.4, jmax=1.3)
        for node in graph['nodes']:
            for segment in node['motion_segments']:
                duration = segment.get('profile_duration', segment['duration'])
                delta = max(abs(a-b) for a,b in zip(segment['q0'], segment['q1']))
                # Exact derivative extrema of 10u^3-15u^4+6u^5 on [0,1].
                self.assertLessEqual(1.875*delta/duration, limits['max_velocity_rad_s']+1e-10)
                self.assertLessEqual((10/math.sqrt(3))*delta/duration**2,
                                     limits['max_acceleration_rad_s2']+1e-10)
                self.assertLessEqual(60*delta/duration**3, limits['max_jerk_rad_s3']+1e-10)

    def test_geometry_and_real_lua_use_profile_offset_and_duration(self):
        node = {'q_start': q(.3), 'q_end': q(.6), 'q_samples': [q(.3), q(.6)], 'duration': 1.,
                'motion_segments': [{'q0': q(0.), 'q1': q(.6), 'duration': 1.,
                                     'profile_offset': 1., 'profile_duration': 2.}]}
        lua = LuaControllerEvaluator()
        for elapsed in (0., .013, .27, .79, 1.):
            with self.subTest(elapsed=elapsed):
                u = (1.+elapsed)/2.
                expected = q(.6*(10*u**3-15*u**4+6*u**5))
                np.testing.assert_allclose(at(node, elapsed), expected, atol=2e-12, rtol=0.)
                np.testing.assert_allclose(lua.evaluate(node, elapsed), expected, atol=2e-12, rtol=0.)

    def test_real_dispatcher_continues_same_actuation_without_target_reset(self):
        for lag in (False, True):
            with self.subTest(one_step_servo_lag=lag):
                graph = two_fragments()
                parameterize(graph)
                first, last = graph['nodes']
                samples, events = run_controller_motion_fixture(graph, lag_one_step=lag)
                times = {(event['kind'], event['node']): event['time'] for event in events}
                beginning = times['node_started', first['id']]
                boundary = times['node_completed', first['id']]
                self.assertEqual(boundary, times['node_started', last['id']])
                total = first['motion_segments'][0]['profile_duration']
                u = np.clip((samples[:, 0]-beginning)/total, 0., 1.)
                expected = .6*(10*u**3-15*u**4+6*u**5)
                np.testing.assert_allclose(samples[:, 1], expected, atol=2e-12, rtol=0.)
                # Test the observed target increments around the handover, not
                # merely the continuity metadata emitted by the parameterizer.
                index = int(np.argmin(abs(samples[:, 0]-boundary)))
                speeds = np.diff(samples[index-1:index+2, 1])/.02
                self.assertTrue(np.all(speeds > .05), speeds)
                self.assertGreaterEqual(samples[index, 2], .3-1e-8,
                                        'Node acknowledgment preceded actual waypoint crossing')

    def test_real_dispatcher_waits_for_cross_arm_gate_then_keeps_shared_clock(self):
        graph = build_tpg([task('other', 'B', path=[q(0.)], kind='skill',
                                metadata={'action': 'wait', 'dwell_s': .7}),
                           task('travel', order=1)],
                          lambda a,b: a['task_id']=='other' and b.get('fragment_index')==1,
                          max_joint_span=.3)
        parameterize(graph)
        other, first, last = graph['nodes']
        samples, events = run_controller_motion_fixture(graph)
        times = {(event['kind'], event['node']): event['time'] for event in events}
        beginning = times['node_started', first['id']]
        self.assertGreaterEqual(beginning, times['node_completed', other['id']])
        self.assertEqual(times['node_completed', first['id']], times['node_started', last['id']])
        self.assertTrue(np.all(samples[samples[:, 0] < beginning, 1] == 0.))
        total = first['motion_segments'][0]['profile_duration']
        u = np.clip((samples[:, 0]-beginning)/total, 0., 1.)
        expected = .6*(10*u**3-15*u**4+6*u**5)
        np.testing.assert_allclose(samples[:, 1], expected, atol=2e-12, rtol=0.)

    def test_real_dispatcher_accepts_equivalent_positive_and_negative_pi_measurements(self):
        for fixed, branch in ((math.pi, 'negative'), (-math.pi, 'positive')):
            with self.subTest(fixed_joint_6=fixed, measured_branch=branch):
                graph = two_fragments()
                for node in graph['nodes']:
                    node['q_start'][5] = fixed; node['q_end'][5] = fixed
                    for point in node['q_samples']:
                        point[5] = fixed
                parameterize(graph)
                first, last = graph['nodes']
                samples, events = run_controller_motion_fixture(graph, wrap_measurements=branch)
                times = {(event['kind'], event['node']): event['time'] for event in events}
                self.assertEqual(times['node_completed', first['id']], times['node_started', last['id']])
                np.testing.assert_allclose(samples[:, 4], fixed, atol=1e-12)
                np.testing.assert_allclose(samples[:, 5], -fixed, atol=1e-12)

    def test_real_dispatcher_crosses_cyclic_branch_without_premature_ack(self):
        for direction in (1., -1.):
            with self.subTest(direction=direction):
                start = [0., 0., 0., 0., 0., direction*(math.pi-.3)]
                end = [0., 0., 0., 0., 0., direction*(math.pi+.3)]
                graph = build_tpg([task('travel', path=[start, end])],
                                  lambda _a,_b: False, max_joint_span=.31)
                parameterize(graph)
                first, last = graph['nodes']
                samples, events = run_controller_motion_fixture(
                    graph, lag_one_step=True, wrap_measurements='negative')
                times = {(event['kind'], event['node']): event['time'] for event in events}
                boundary = times['node_completed', first['id']]
                self.assertEqual(boundary, times['node_started', last['id']])
                index = int(np.argmin(abs(samples[:, 0]-boundary)))
                actual_unwrapped = np.unwrap(samples[:, 5])
                self.assertGreaterEqual(direction*(actual_unwrapped[index]-first['q_end'][5]), -1e-8)
                self.assertLess(direction*(actual_unwrapped[index-1]-first['q_end'][5]), 0.)
                # Commands retain the unwrapped path; only measurements wrap.
                self.assertGreater(np.max(abs(samples[:, 4])), math.pi)
                self.assertTrue(np.all(abs(samples[:, 5]) <= math.pi+1e-12))
                self.assertTrue(np.all(direction*np.diff(samples[:, 4]) >= -1e-12))

    def test_curved_profile_geometry_and_real_lua_match_with_nonzero_offset(self):
        from normal.curve import eval_profile
        graph, profile = curved_two_fragment_fixture()
        lua = LuaControllerEvaluator()
        for node in graph['nodes']:
            offset = node['metadata']['motion_run_offset']
            for elapsed in np.linspace(0., node['duration'], 17):
                expected = eval_profile(profile, offset+float(elapsed))
                np.testing.assert_allclose(at(node, elapsed), expected, atol=3e-12, rtol=0.)
                np.testing.assert_allclose(lua.evaluate(node, float(elapsed)), expected, atol=3e-12, rtol=0.)

    def test_real_dispatcher_curved_milestone_keeps_clock_and_uses_actual_tangent_progress(self):
        from normal.curve import eval_profile
        graph, profile = curved_two_fragment_fixture(fixed_joint_6=math.pi)
        first, last = graph['nodes']
        samples, events = run_controller_motion_fixture(graph, lag_one_step=True, wrap_measurements='negative')
        times = {(event['kind'], event['node']): event['time'] for event in events}
        boundary = times['node_completed', first['id']]
        self.assertEqual(boundary, times['node_started', last['id']])
        beginning = times['node_started', first['id']]
        expected = np.asarray([eval_profile(profile, t-beginning) for t in samples[:, 0]])
        np.testing.assert_allclose(samples[:, 1], expected[:, 0], atol=3e-12, rtol=0.)
        np.testing.assert_allclose(samples[:, 4], math.pi, atol=3e-12, rtol=0.)
        np.testing.assert_allclose(samples[:, 5], -math.pi, atol=3e-12, rtol=0.)
        index = int(np.argmin(abs(samples[:, 0]-boundary)))
        # The actual pose is one command sample behind in this fixture. Test
        # full 6D tangent progress independently of the controller's own flag.
        actual = expected[index-1]
        previous_actual = expected[index-2]
        tangent = np.asarray(first['motion_segments'][0]['progress_direction'])
        end = np.asarray(first['q_end'])
        self.assertGreaterEqual(float((actual-end)@tangent), -1e-8)
        self.assertLess(float((previous_actual-end)@tangent), 0.)
        self.assertTrue(np.all(np.diff(samples[index-1:index+2, 1])/.02 > .01))

    def test_parameterizer_rejects_unchecked_curve_and_tighter_requested_limits(self):
        def fixture(checked):
            graph, profile = curved_two_fragment_fixture()
            # A synthetic certificate flag exercises the consumer contract;
            # this test does not certify real robot geometry.
            profile['collision_validated'] = checked
            for index,node in enumerate(graph['nodes']):
                node['curve_profile'] = profile
                node['curve_run_id'] = graph['nodes'][0]['id']
                node['curve_range_s'] = [index*profile['length']/2., (index+1)*profile['length']/2.]
            return graph, profile
        graph, _profile = fixture(False)
        with self.assertRaisesRegex(ValueError, 'unchecked curve'):
            parameterize(graph)
        for key,argument in (('velocity','vmax'), ('acceleration','amax'), ('jerk','jmax')):
            with self.subTest(bound=key):
                graph, profile = fixture(True)
                original_bounds = deepcopy(profile['derivative_bounds'])
                requested = {argument: .99*max(profile['derivative_bounds'][key])}
                with self.assertRaisesRegex(ValueError, 'exceeds '+key):
                    parameterize(graph, **requested)
                self.assertEqual(profile['derivative_bounds'], original_bounds)



class RetryObservationTests(unittest.TestCase):
    def observation(self, kind, position):
        evaluator=LuaControllerEvaluator()
        mock="""
math.atan2=math.atan2 or math.atan
local mock={}
function mock.getJointPosition(_) return 0 end
function mock.getObjectPosition(_) return POSITION end
function mock.getObjectPose(_) return POSITION end
package.preload['sim']=function() return mock end
package.preload['simIK']=function() return {} end
""".replace('POSITION',lua_literal(position))
        setup="""
cfg={goal={cap={0,0,0}}};blocks={cap=1}
arms={B={joints={1,2,3,4,5,6},tip=7}}
held={B='cap'};retry={attempt=3,blocked=false}
demo={retry_skill={part='cap',arm='B',kind=KIND,q_end={0,0,0,0,0,0},
                  metadata={payload_after={offset={0,0,0}}}},
      retry_config={position_tolerance_m=.009}}
local answer=skillObservation()
return answer.aligned and 'aligned' or 'failed'
""".replace('KIND',lua_literal(kind))
        return evaluator.run(mock+'\n'+evaluator.source+'\n'+
            (ROOT/'recovery/demo_controller.lua').read_text()+'\n'+setup)

    def test_final_attempt_does_not_override_measured_installation_failure(self):
        self.assertEqual(self.observation('place',[.025,0,0]),'failed')
        self.assertEqual(self.observation('place',[.001,0,0]),'aligned')

    def test_grasp_alignment_uses_observed_object_in_tool_frame(self):
        self.assertEqual(self.observation('pick',[0,.03,0]),'failed')
        self.assertEqual(self.observation('pick',[0,.001,0]),'aligned')


if __name__ == '__main__':
    unittest.main()
