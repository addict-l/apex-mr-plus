"""Behavioral tests; synthetic traces below are test fixtures, never evidence."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.tpg import build_tpg, simulate_tpg
from apex.program import parameterize
from apex.validation import validate_execution
from apex.compile import Compiler, grasp


def q(value):
    return [value, 0., 0., 0., 0., 0.]


def task(name, arm, kind='transit', path=None, dependencies=(), order=0, metadata=None):
    return {'id': name, 'arm': arm, 'kind': kind, 'q_path': path or [q(0)],
            'duration': 1., 'dependencies': list(dependencies), 'reference_order': order,
            'metadata': metadata or {}}


def audit_fixture():
    """A complete synthetic actual-event log for validator mutation tests."""
    specs = [
        ('verify', 'B', 'support_load_verified', {'valid': True, 'force_N': 3.0}),
        ('release', 'A', 'arm_support_released', {}),
        ('k1', 'B', 'part_installed', {'part': 'K1'}),
        ('k2', 'B', 'part_installed', {'part': 'K2'}),
        ('permanent', 'B', 'permanent_structure_verified', {'valid': True}),
        ('unload', 'B', 'support_unloaded', {'retraction_m': .008}),
        ('extract', 'B', 'support_extracted', {}),
        ('return', 'B', 'support_returned', {}),
        ('gate', 'B', 'gate_closed', {}),
    ]
    tasks = [task(name, side, 'skill', dependencies=[] if index == 0 else [specs[index-1][0]], order=index)
             for index, (name, side, _, _) in enumerate(specs)]
    graph = build_tpg(tasks, lambda _a, _b: False)
    events = []
    for index, (name, side, semantic, details) in enumerate(specs):
        node = graph['task_map'][name]['first_node']
        events.extend([
            {'kind': 'node_started', 'node': node, 'time': float(index)},
            {'kind': semantic, 'node': node, 'time': float(index+1), 'details': details},
            {'kind': 'node_completed', 'node': node, 'time': float(index+1)},
        ])
    events.sort(key=lambda e: e['time'])
    goals = {part: [0., 0., 1., 0., 0., 0., 1.] for part in ['K1', 'K2', 'Gate', 'Support']}
    execution = {
        'success': True, 'failure': None, 'events': events,
        'trace': [{'time': i/10, 'support_valid': True, 'support_force_N': 0.0 if i >= 42 else 3.0,
                   'permanent_support_valid': i >= 42, 'permanent_sensor_valid': i >= 42,
                   'permanent_force_N': 8.0 if i >= 42 else 0.,
                   'bearing_sensor': 'PermanentBridgeFT' if i >= 42 else 'none',
                   'permanent_verification_active': 42 <= i < 50,
                   'support_lift_position_m': 0.0 if i >= 42 else .008,
                   'bridge_position_error_m': 0., 'bridge_orientation_error_rad': 0.,
                   'insertion_load_N': 8.0 if i >= 42 else 0.} for i in range(10, 51)],
        'part_poses': deepcopy(goals),
        'dynamic_enablement': {side: {'joints': [True]*6} for side in ['A', 'B']},
        'final_state': {'held': {'A': None, 'B': None}},
    }
    execution['dynamic_enablement'].update(support_sensor=True, support_lift=True)
    return graph, execution, {'goal_poses': goals}


def required_physics_audit_fixture():
    """Synthetic complete evidence with the formal scene-3 acceptance flags."""
    graph, execution, req = audit_fixture()
    req.update(require_actual_applied_load=True, required_dynamic_connections=['K1', 'K2', 'Gate', 'Bridge'])
    execution['snap_dynamic'] = {part: True for part in req['required_dynamic_connections']}
    for sample in execution['trace']:
        sample['applied_load_N'] = sample['insertion_load_N']
        sample['applied_load_substeps'] = 4 if sample['insertion_load_N'] > 0 else 0
        if sample['time'] >= 4.2:
            sample['snap'] = {'Bridge': {'valid': True, 'force_N': [0., 0., 13.788], 'torque_Nm': [0., 0., 0.]}}
    # A final sensor read is distinct from samples in the earlier proof phase.
    final_sample = deepcopy(execution['trace'][-1])
    final_sample.update(time=9., insertion_load_N=0., applied_load_N=0., applied_load_substeps=0,
                        permanent_force_N=5.788, permanent_verification_active=False)
    final_sample['snap']['Bridge']['force_N'] = [0., 0., 5.788]
    execution['trace'].append(final_sample)
    return graph, execution, req


def assembly_audit_fixture():
    """Synthetic contact observations, independently located by node events."""
    graph, execution, req = audit_fixture()
    req.update(assembly_tool_force_limit_N=40., assembly_tool_torque_limit_Nm=8.)
    for name in ('k1', 'k2', 'gate'):
        node = next(node for node in graph['nodes'] if node['id'] == graph['task_map'][name]['first_node'])
        node['metadata']['assembly_force_guard'] = True
    last_sample = deepcopy(execution['trace'][-1])
    for index in range(51, 91):
        execution['trace'].append({**deepcopy(last_sample), 'time': index/10})
    for sample in execution['trace']:
        for side in ('A', 'B'):
            # False flags deliberately cannot suppress a real guarded interval.
            sample[side+'_assembly_contact'] = False
            sample[side+'_tool_sensor_valid'] = True
            sample[side+'_tool_force_N'] = [24., 32., 0.]
            sample[side+'_tool_torque_Nm'] = [0., 0., 8.]
    return graph, execution, req


class TPGBehaviorTests(unittest.TestCase):
    def test_assembly_force_guard_covers_every_motion_fragment(self):
        graph = build_tpg([task('install_K1_approach', 'B', path=[q(0.), q(.8)],
                                metadata={'assembly_force_guard': True})],
                          lambda _a, _b: False, max_joint_span=.1)
        self.assertGreater(len(graph['nodes']), 1)
        self.assertTrue(all(node['metadata'].get('assembly_force_guard') is True for node in graph['nodes']))

    def test_buffer_receiver_approach_waits_for_supplier_clear_but_not_its_next_motion(self):
        # Use the real compiler's lifecycle/pick/place expansion. Only IK and
        # RRT numerical answers are synthetic: this test proves causal edges,
        # not a geometric or dynamic execution result.
        class MotionStub:
            def fk(self, arm, configuration):
                return {'pose': [-.08, .14, 1.002, 0., 0., 0., 1.]}

            def plan(self, arm, start, end, *args, **kwargs):
                return {'ok': True, 'path': [start[:], end[:]], 'stats': {'test_fixture': True}}

        with TemporaryDirectory(prefix='scene3_buffer_causality_test_') as directory:
            compiler = Compiler.__new__(Compiler)
            compiler.out = Path(directory)
            compiler.cache = compiler.out/'rrt_paths'; compiler.cache.mkdir()
            compiler.p = MotionStub(); compiler.sig = 'synthetic_causality_test_only'
            compiler.calls = []; compiler.qual = []
            compiler.home = {'A': q(0), 'B': q(0)}
            compiler.cfg = {'free_q': {'A': q(.1), 'B': q(.1)},
                            'runtime': {'support_extension_m': .010}}
            # This fixture intentionally bypasses __init__; retain the same
            # configured actuator travel consumed by real lifecycle expansion.
            compiler.extension = compiler.cfg['runtime']['support_extension_m']
            compiler.source = {
                'K1': [-.53,-.36,.806,0,0,0,1], 'K2': [-.38,-.36,.806,0,0,0,1],
                'Gate': [.48,-.35,.851,0,0,0,1], 'Bridge': [.04,.20,1.032,0,0,0,1],
                'TemporarySupport': [.48,.43,.760,0,0,0,1]}
            compiler.goals = {**compiler.source, 'K1': [.24,.20,.972,0,0,0,1],
                              'K2': [.24,.20,1.096,0,0,0,1], 'Gate': [.06,.105,.857,0,0,0,1]}
            compiler.buffer = {part: [0,-.36,.808,0,0,0,1] for part in ['K1','K2']}
            compiler.tool_contacts = lambda arm, part: [[arm+'_Palm', part]]
            compiler.resting_contacts = lambda part: [[part, 'Table']]
            compiler.ik = lambda arm, tcp, *args, **kwargs: list(tcp[:3])+[0.,0.,0.]
            assignments = {'supply_K1':'A', 'supply_K2':'A', 'install_K1':'B', 'install_K2':'B', 'close_gate':'B'}
            choices = []
            for name, arm in assignments.items():
                part = 'Gate' if name == 'close_gate' else name.split('_')[1]
                offset, quaternion = grasp(part, arm, 'install' if name.startswith('install_') else None)
                choices.append({'task':name,'grasp_offset':offset,'grasp_quaternion':quaternion})
            allocation = {'selected_support': {'candidate':'outer_front', 'placer_arm':'B', 'remover_arm':'B',
                          'removal_order':'after_gate', 'goal_pose':[.08,-.05,.772,0,0,0,1]},
                          'task_assignments':assignments, 'choices':choices}
            with redirect_stdout(io.StringIO()):
                tasks = compiler.compile(allocation)
            graph = build_tpg(tasks, lambda a,b: False)
            extension = graph['task_map']['verify_support']['metadata']
            retraction = graph['task_map']['verify_permanent']['metadata']
            self.assertEqual(extension['articulation_motion']['SupportLift'], {'start': 0., 'end': .010})
            self.assertEqual(extension['effects']['articulations']['SupportLift'], .010)
            self.assertEqual(graph['task_map']['A_lower_from_support']['metadata']['options']['articulations']['SupportLift'], .010)
            self.assertEqual(retraction['articulation_motion']['SupportLift'], {'start': .010, 'end': 0.})
            self.assertEqual(retraction['effects']['articulations']['SupportLift'], 0.)
            clear = 'deposit_K1_retreat_contact_clear'
            following_motion = 'supplier_clear_K1'
            delayed = simulate_tpg(graph, {clear: 4., following_motion: 20.})
            intervals = {row['id']:row for row in delayed['task_schedule']}
            self.assertGreaterEqual(intervals['pick_K1_approach']['start'], intervals[clear]['end'])
            self.assertLess(intervals['pick_K1_approach']['start'], intervals[following_motion]['end'])
            # Reintroducing only the old approach edge reproduces the unsafe
            # overlap even though the final grasp still waits for the clear.
            broken_tasks = deepcopy(tasks)
            approach = next(row for row in broken_tasks if row['id']=='pick_K1_approach')
            approach['dependencies'] = ['deposit_K1' if dep==clear else dep for dep in approach['dependencies']]
            broken = simulate_tpg(build_tpg(broken_tasks, lambda a,b: False), {clear: 4., following_motion: 20.})
            broken_intervals = {row['id']:row for row in broken['task_schedule']}
            self.assertLess(broken_intervals['pick_K1_approach']['start'], broken_intervals[clear]['end'])

    def test_partition_retains_turning_points_and_swept_fragments(self):
        original = [q(0), q(.03), q(.22), q(.12), q(.38)]
        graph = build_tpg([task('move', 'A', path=original)])
        self.assertGreater(len(graph['nodes']), 3)
        samples = [point for node in graph['nodes'] for point in node['q_samples']]
        for point in original:
            self.assertTrue(any(max(abs(a-b) for a,b in zip(point, candidate)) < 1e-12 for candidate in samples))
        for node in graph['nodes']:
            self.assertLessEqual(max(p[0] for p in node['q_samples'])-min(p[0] for p in node['q_samples']), .10+1e-9)
        self.assertFalse(graph['certificate']['collision_oracle_supplied'])

    def test_cross_arm_conflicts_and_load_handoff_use_completion_edges(self):
        tasks = [task('move_B', 'B', path=[q(0), q(.25)]),
                 task('verify_load', 'B', 'verify_support', order=1),
                 task('release_A', 'A', 'release_support', dependencies=['verify_load'], order=2),
                 task('fetch_A', 'A', path=[q(0), q(.22)], dependencies=['release_A'], order=3),
                 task('move_B_later', 'B', path=[q(.25), q(.4)], order=4)]
        graph = build_tpg(tasks, lambda a,b: {'conflict': True, 'evidence': {'test_oracle': True}})
        self.assertTrue(any(e['type'] == 'type2' and e['reason'] == 'task_dependency' for e in graph['edges']))
        self.assertTrue(any(e['reason'] == 'motion_conflict' for e in graph['edges']))
        delayed = simulate_tpg(graph, {'verify_load': 4.})
        for edge in graph['edges']:
            self.assertLessEqual(delayed['timestamps'][edge['source']]['end'], delayed['timestamps'][edge['target']]['start']+1e-8)
        timing = {row['id']: row for row in delayed['task_schedule']}
        self.assertGreaterEqual(timing['release_A']['start'], timing['verify_load']['end'])
        self.assertGreaterEqual(timing['fetch_A']['start'], timing['release_A']['end'])

    def test_oracle_rejecting_reference_serialization_stops_build(self):
        with self.assertRaisesRegex(ValueError, 'serialization rejected'):
            build_tpg([task('a', 'A'), task('b', 'B', order=1)],
                      lambda a,b: {'conflict': True, 'allowed_orders': [[b['id'],a['id']]]})

    def test_missing_geometry_and_reverse_dependencies_are_not_certificates(self):
        tasks = [task('a', 'A'), task('b', 'B', order=1)]
        del tasks[0]['q_path']; tasks[0]['kind'] = 'skill'
        graph = build_tpg(tasks, lambda a,b: False)
        self.assertFalse(graph['certificate']['all_nodes_have_geometry'])
        self.assertFalse(graph['certificate']['motion_overlap_pairs_checked'])
        with self.assertRaisesRegex(ValueError, 'requires joint geometry'):
            parameterize(graph)
        with self.assertRaisesRegex(ValueError, 'Reference order'):
            build_tpg([task('a', 'A', dependencies=['b']), task('b', 'B', order=1)])

    def test_contact_parameterization_respects_custom_bounds(self):
        graph = build_tpg([task('extract', 'B', 'extract_support', [q(0), q(.2)],
                               metadata={'max_velocity_rad_s': .025, 'max_acceleration_rad_s2': .12})])
        parameterize(graph)
        segment = graph['nodes'][0]['motion_segments'][0]
        duration = segment['duration']
        self.assertLessEqual(1.875*.2/duration, .025+1e-10)
        self.assertLessEqual((10/math.sqrt(3))*.2/duration**2, .12+1e-10)
        self.assertEqual(segment['q0'], q(0)); self.assertEqual(segment['q1'], q(.2))
        with self.assertRaises(ValueError):
            parameterize(graph, vmax=0)

    def test_support_contact_survives_partial_lowering_motion(self):
        # The support-contact mode must survive completion of early path
        # fragments, including when feedback delays the final fragment.
        lowering = task('lower_A', 'A', path=[q(0), q(.45)],
                        metadata={'effects': {'support_contacts': {'A': None}}})
        graph = build_tpg([lowering], lambda a,b: False)
        final = graph['task_map']['lower_A']['last_node']
        prediction = simulate_tpg(graph, {final: 2.0})
        by_id = {node['id']: node for node in graph['nodes']}
        contacts = {'A': [['A_SupportShoe', 'Bridge']]}
        released_at = None
        observed_intermediate_completion = False
        for event in prediction['events']:
            if event['event'] != 'finish':
                continue
            node = by_id[event['node_id']]
            for side, value in node['metadata'].get('effects', {}).get('support_contacts', {}).items():
                if value is None:
                    contacts.pop(side, None); released_at = event['time']
                else:
                    contacts[side] = value
            if node['id'] != final:
                observed_intermediate_completion = True
                self.assertIn('A', contacts)
        self.assertTrue(observed_intermediate_completion)
        self.assertNotIn('A', contacts)
        self.assertEqual(released_at, prediction['timestamps'][final]['end'])
        self.assertIn('effects', lowering['metadata'])


class ActualExecutionAuditTests(unittest.TestCase):
    def report(self, graph, execution, req):
        return validate_execution(graph, execution, req)

    def codes(self, report):
        return {issue['code'] for issue in report['violations']}

    def test_complete_test_fixture_passes_without_mutating_input(self):
        graph, execution, req = audit_fixture()
        previous = deepcopy(execution)
        report = self.report(graph, execution, req)
        self.assertTrue(report['success'], report['violations'])
        self.assertEqual(execution, previous)

    def test_nominal_prediction_cannot_substitute_for_actual_evidence(self):
        graph, execution, req = audit_fixture()
        execution['events'] = simulate_tpg(graph)['events']
        report = self.report(graph, execution, req)
        self.assertFalse(report['success'])
        self.assertIn('missing_or_reversed_actual_interval', self.codes(report))

    def test_missing_node_and_duplicate_completion_fail(self):
        graph, execution, req = audit_fixture()
        execution['events'] = [e for e in execution['events'] if e['node'] != graph['nodes'][-1]['id']]
        self.assertFalse(self.report(graph, execution, req)['success'])
        graph, execution, req = audit_fixture()
        execution['events'].append(deepcopy(execution['events'][-1]))
        self.assertIn('duplicate_node_event', self.codes(self.report(graph, execution, req)))

    def test_actual_delay_requires_successors_to_wait(self):
        graph, execution, req = audit_fixture()
        for event in execution['events']:
            if event['node'] == graph['nodes'][0]['id'] and event['kind'] == 'node_completed':
                event['time'] = 1.5
        execution['events'].sort(key=lambda e:e['time'])
        self.assertIn('predecessor_not_completed', self.codes(self.report(graph, execution, req)))

    def test_force_evidence_loss_and_unobserved_gaps_fail(self):
        graph, execution, req = audit_fixture()
        execution['trace'][10]['support_force_N'] = 0.0
        self.assertIn('invalid_or_lost_support_force_sample', self.codes(self.report(graph, execution, req)))
        graph, execution, req = audit_fixture()
        execution['trace'] = [execution['trace'][0], execution['trace'][-1]]
        self.assertIn('support_trace_has_unobserved_gap', self.codes(self.report(graph, execution, req)))

    def test_contact_and_unload_require_measured_evidence(self):
        graph, execution, req = audit_fixture()
        for event in execution['events']:
            if event['kind'] == 'support_load_verified': event['details']['valid'] = False
            if event['kind'] == 'support_unloaded': event['details']['retraction_m'] = .001
        codes = self.codes(self.report(graph, execution, req))
        self.assertIn('load_transfer_missing_measured_force', codes)
        self.assertIn('insufficient_measured_shoe_retraction', codes)

    def test_measured_permanent_bearing_allows_temporary_support_to_unload(self):
        graph, execution, req = audit_fixture()
        for sample in execution['trace']:
            if sample['time'] >= 4.5: sample['support_force_N'] = 0.
        self.assertTrue(self.report(graph, execution, req)['success'])
        for sample in execution['trace']:
            sample['permanent_force_N'] = 0.
        codes = self.codes(self.report(graph, execution, req))
        self.assertIn('missing_measured_permanent_load_test', codes)
        self.assertIn('invalid_or_lost_support_force_sample', codes)

    def test_handoff_and_continuity_have_distinct_measured_thresholds(self):
        graph, execution, req = audit_fixture()
        for sample in execution['trace']:
            if 2. <= sample['time'] < 4.2: sample['support_force_N'] = .20
        self.assertTrue(self.report(graph, execution, req)['success'])
        next(e for e in execution['events'] if e['kind']=='support_load_verified')['details']['force_N'] = .20
        self.assertIn('load_transfer_missing_measured_force', self.codes(self.report(graph, execution, req)))

    def test_K1_contact_cannot_replace_dedicated_permanent_proof(self):
        graph, execution, req = audit_fixture()
        for sample in execution['trace']:
            if sample['time'] >= 4.2: sample['bearing_sensor'] = 'K1Snap'
        codes = self.codes(self.report(graph, execution, req))
        self.assertIn('missing_measured_permanent_load_test', codes)
        self.assertNotIn('invalid_or_lost_support_force_sample', codes)

    def test_permanent_proof_requires_sustained_independent_bearing(self):
        for field, bad_value in [('support_lift_position_m', .008),
                                 ('support_force_N', 4.), ('permanent_sensor_valid', False)]:
            with self.subTest(field=field):
                graph, execution, req = audit_fixture()
                # A failed physical check halfway through resets the proof
                # window; older qualifying samples cannot be accumulated.
                next(r for r in execution['trace'] if r['time']==4.6)[field] = bad_value
                codes = self.codes(self.report(graph, execution, req))
                self.assertIn('permanent_proof_duration_or_continuity_not_met', codes)

    def test_actual_load_requirement_rejects_command_only_and_callback_gaps(self):
        graph, execution, req = required_physics_audit_fixture()
        self.assertTrue(self.report(graph, execution, req)['success'])
        for sample in execution['trace']:
            sample.pop('applied_load_N'); sample.pop('applied_load_substeps')
        self.assertIn('missing_measured_permanent_load_test', self.codes(self.report(graph, execution, req)))
        for field, value in [('applied_load_N', 7.9), ('applied_load_N', float('nan')),
                             ('applied_load_substeps', 0), ('applied_load_substeps', .5),
                             ('applied_load_substeps', True)]:
            with self.subTest(field=field, value=value):
                graph, execution, req = required_physics_audit_fixture()
                next(sample for sample in execution['trace'] if sample['time'] == 4.6)[field] = value
                self.assertIn('permanent_proof_duration_or_continuity_not_met',
                              self.codes(self.report(graph, execution, req)))

    def test_formal_final_connections_and_dedicated_sensor_evidence_are_required(self):
        for connection in ['K1', 'K2', 'Gate', 'Bridge']:
            with self.subTest(connection=connection):
                graph, execution, req = required_physics_audit_fixture()
                execution['snap_dynamic'][connection] = False
                self.assertIn('required_connection_not_dynamic', self.codes(self.report(graph, execution, req)))
        graph, execution, req = required_physics_audit_fixture()
        execution.pop('snap_dynamic')
        self.assertIn('required_connection_not_dynamic', self.codes(self.report(graph, execution, req)))
        for field, value in [('valid', False), ('force_N', [0., 0.]), ('torque_Nm', [0., float('nan'), 0.])]:
            with self.subTest(field=field):
                graph, execution, req = required_physics_audit_fixture()
                execution['trace'][-1]['snap']['Bridge'][field] = value
                self.assertIn('invalid_final_dedicated_sensor_telemetry', self.codes(self.report(graph, execution, req)))

    def test_dedicated_sensor_vector_gap_resets_the_actual_load_proof(self):
        graph, execution, req = required_physics_audit_fixture()
        # Summary force/validity fields and API calls remain valid. Missing raw
        # dedicated sensor evidence still interrupts the accepted interval.
        next(sample for sample in execution['trace'] if sample['time'] == 4.6)['snap']['Bridge']['force_N'] = []
        codes = self.codes(self.report(graph, execution, req))
        self.assertIn('permanent_proof_duration_or_continuity_not_met', codes)
        self.assertNotIn('invalid_final_dedicated_sensor_telemetry', codes)

    def test_load_plot_never_labels_a_command_as_actual_applied_evidence(self):
        from apex.plots import _validation_load_trace
        series, label, title = _validation_load_trace([{'insertion_load_N': 8.}])
        self.assertEqual(series, [8.]); self.assertIn('命令', label); self.assertIn('缺失', title)
        series, label, title = _validation_load_trace([
            {'insertion_load_N': 8., 'applied_load_N': 8., 'applied_load_substeps': 4},
            {'insertion_load_N': 8., 'applied_load_N': 8., 'applied_load_substeps': 0},
            {'insertion_load_N': 8., 'applied_load_N': 0., 'applied_load_substeps': 0}])
        self.assertEqual(series[0], 8.); self.assertTrue(math.isnan(series[1])); self.assertEqual(series[2], 0.)
        self.assertIn('子步', label); self.assertIn('调用', title)

    def test_assembly_contact_limits_audit_norms_despite_false_trace_flags(self):
        graph, execution, req = assembly_audit_fixture()
        report = self.report(graph, execution, req)
        self.assertTrue(report['success'], report['violations'])
        self.assertEqual(report['assembly_contact_audit']['max_force_N']['B'], 40.)
        self.assertEqual(report['assembly_contact_audit']['max_torque_Nm']['B'], 8.)
        for field, value, code in [
                ('B_tool_force_N', [269., 0., 0.], 'assembly_tool_force_limit_exceeded'),
                ('B_tool_force_N', [30., 30., 0.], 'assembly_tool_force_limit_exceeded'),
                ('B_tool_torque_Nm', [0., 26., 0.], 'assembly_tool_torque_limit_exceeded')]:
            with self.subTest(field=field, value=value):
                graph, execution, req = assembly_audit_fixture()
                next(sample for sample in execution['trace'] if sample['time'] == 2.5)[field] = value
                self.assertIn(code, self.codes(self.report(graph, execution, req)))

    def test_assembly_contact_requires_valid_complete_sensor_vectors(self):
        for field, value in [('B_tool_sensor_valid', False), ('B_tool_force_N', None),
                             ('B_tool_force_N', [0., 0.]), ('B_tool_torque_Nm', [0., float('nan'), 0.])]:
            with self.subTest(field=field):
                graph, execution, req = assembly_audit_fixture()
                next(sample for sample in execution['trace'] if sample['time'] == 2.5)[field] = value
                self.assertIn('invalid_assembly_tool_sensor_sample', self.codes(self.report(graph, execution, req)))

    def test_assembly_contact_cannot_pass_with_unobserved_event_intervals(self):
        graph, execution, req = assembly_audit_fixture()
        execution['trace'] = [sample for sample in execution['trace'] if not 8. <= sample['time'] <= 9.]
        self.assertIn('missing_assembly_force_samples', self.codes(self.report(graph, execution, req)))
        graph, execution, req = assembly_audit_fixture()
        execution['trace'] = [sample for sample in execution['trace'] if not 8.2 <= sample['time'] <= 8.7]
        self.assertIn('assembly_force_trace_has_unobserved_gap', self.codes(self.report(graph, execution, req)))
        graph, execution, req = assembly_audit_fixture()
        for node in graph['nodes']: node['metadata'].pop('assembly_force_guard', None)
        self.assertIn('assembly_event_node_missing_force_guard', self.codes(self.report(graph, execution, req)))

    def test_flagged_final_physics_step_is_checked_after_node_completion(self):
        graph, execution, req = assembly_audit_fixture()
        sample = deepcopy(execution['trace'][-1])
        sample.update(time=9.02, B_assembly_contact=True, B_tool_force_N=[41., 0., 0.])
        execution['trace'].append(sample)
        self.assertIn('assembly_tool_force_limit_exceeded', self.codes(self.report(graph, execution, req)))

    def test_final_pose_dynamic_state_and_empty_hands_are_required(self):
        graph, execution, req = audit_fixture()
        execution['part_poses']['Support'][0] += .1
        execution['dynamic_enablement']['A']['joints'][1] = False
        execution['final_state']['held']['B'] = 'Support'
        codes = self.codes(self.report(graph, execution, req))
        self.assertTrue({'goal_position_error', 'dynamic_joints_not_enabled', 'final_hands_not_known_empty'} <= codes)

    def test_gate_guard_is_independent_of_graph_edges(self):
        graph, execution, req = audit_fixture()
        for event in execution['events']:
            if event['kind'] == 'gate_closed': event['time'] = 5.5
        execution['events'].sort(key=lambda e:e['time'])
        req['gate_blocks_extraction'] = True
        self.assertIn('inner_support_trapped_by_gate', self.codes(self.report(graph, execution, req)))


if __name__ == '__main__':
    unittest.main()
