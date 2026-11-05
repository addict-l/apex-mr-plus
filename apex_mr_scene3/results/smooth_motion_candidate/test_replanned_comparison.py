"""Synthetic tests for postprocessing policy, never simulator evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replanned_comparison import geometry_evidence, task_guard_report


def fixture(order='before_gate', pose_suffix='0000', candidate='outer_front'):
    prefix = ['place_support', 'verify_support', 'release_A']
    material = []
    for part in ('K1', 'K2'):
        material += ['supply_'+part+'_approach', 'supply_'+part, 'deposit_'+part+'_approach', 'deposit_'+part,
                     'deposit_'+part+'_retreat_contact_clear', 'pick_'+part+'_approach', 'pick_'+part,
                     'install_'+part+'_approach', 'install_'+part]
    remove = ['unload_support', 'remove_support_approach', 'regrasp_support', 'extract_support', 'return_support']
    gate = ['pick_gate_approach', 'pick_gate', 'close_gate']
    sequence = prefix+material+['verify_permanent']+(remove+gate if order == 'before_gate' else gate+remove)
    nodes = []; events = []; task_map = {}; edges = []
    for index, task in enumerate(sequence):
        pose = task.endswith('_approach')
        part = next((p for p in ('K1', 'K2') if p in task), None)
        arm = 'A' if task.startswith(('supply_', 'deposit_')) or task == 'release_A' else 'B'
        action = 'wait'
        if task.startswith(('supply_', 'pick_')) and not pose: action = 'grasp'
        if task.startswith('deposit_') and not task.endswith('clear') and not pose: action = 'release_buffer'
        if task.startswith('install_') and not pose: action = 'install'
        if task == 'place_support': action = 'release_support'
        if task in ['verify_support', 'verify_permanent', 'unload_support', 'extract_support', 'return_support', 'close_gate']: action = task
        if task == 'release_A': action = 'release_arm_support'
        node_id = task+('::p'+pose_suffix if pose else '::skill')
        node = {'id': node_id, 'task_id': task, 'arm': arm, 'part': part,
                'node_type': 'pose' if pose else 'skill', 'metadata': {'action': action}, 'duration': 1.}
        if task.startswith(('install_', 'close_gate')): node['metadata']['assembly_force_guard'] = True
        nodes.append(node); task_map[task] = {'node_ids': [node_id], 'first_node': node_id, 'last_node': node_id}
        if index:
            edges.append({'source': nodes[-2]['id'], 'target': node_id, 'source_task': sequence[index-1],
                          'target_task': task, 'reason': 'task_dependency', 'type': 'type2'})
        events.append({'kind': 'node_started', 'node': node_id, 'time': index*1.1})
        if action in ('grasp', 'release_buffer') and part:
            events.append({'kind': 'grasp' if action == 'grasp' else 'buffer_deposit', 'node': node_id,
                           'arm': arm, 'time': index*1.1+1., 'details': {'part': part}})
        semantic = {'unload_support': 'support_unloaded', 'return_support': 'support_returned', 'close_gate': 'gate_closed'}
        if task in semantic: events.append({'kind': semantic[task], 'node': node_id, 'time': index*1.1+1.})
        events.append({'kind': 'node_completed', 'node': node_id, 'time': index*1.1+1.})
    graph = {'nodes': nodes, 'edges': edges, 'task_map': task_map,
             'validation_requirements': {'gate_blocks_extraction': candidate == 'inner_pocket'}}
    allocation = {'selected_support': {'candidate': candidate, 'removal_order': order},
                  'handoffs': [{'part': p, 'from_arm': 'A', 'to_arm': 'B'} for p in ('K1', 'K2')]}
    return graph, {'events': events}, allocation


class ReplannedComparisonTests(unittest.TestCase):
    def test_pose_renumbering_does_not_drop_task_semantic_checks(self):
        before, _, _ = fixture()
        after, execution, allocation = fixture(pose_suffix='0042')
        report = task_guard_report(before, after, execution, allocation)
        self.assertTrue(report['passed'], report['violations'])

    def test_after_gate_can_replace_before_gate_for_outer_support(self):
        before, _, _ = fixture()
        after, execution, allocation = fixture(order='after_gate', pose_suffix='0042')
        report = task_guard_report(before, after, execution, allocation)
        self.assertTrue(report['passed'], report['violations'])
        self.assertTrue(report['order_specific_old_edges_replaced'])
        selected = [x for x in report['checks'] if x['reason'] == 'selected_after_gate_order']
        self.assertEqual(selected[0]['source_task'], 'close_gate')
        self.assertEqual(selected[0]['target_task'], 'unload_support')

    def test_old_collision_edge_can_disappear_under_replanned_geometry(self):
        before, _, _ = fixture()
        before['edges'].append({'source': 'return_support::skill', 'target': 'close_gate::skill',
                                'reason': 'motion_conflict', 'type': 'type2'})
        after, execution, allocation = fixture(order='after_gate')
        report = task_guard_report(before, after, execution, allocation)
        self.assertTrue(report['passed'], report['violations'])
        self.assertEqual(report['old_motion_conflict_edges_not_preserved_by_ID'], 1)

    def test_deleted_load_handoff_dependency_fails_despite_ordered_events(self):
        before, _, _ = fixture(); after, execution, allocation = fixture()
        after['edges'] = [x for x in after['edges'] if x['source_task'] != 'verify_support']
        report = task_guard_report(before, after, execution, allocation)
        self.assertFalse(report['passed'])
        self.assertTrue(any('verify_support -> release_A' in issue for issue in report['violations']))

    def test_inner_after_gate_is_rejected(self):
        before, _, _ = fixture()
        after, execution, allocation = fixture(order='after_gate', candidate='inner_pocket')
        report = task_guard_report(before, after, execution, allocation)
        self.assertFalse(report['passed'])
        self.assertIn('inner support cannot be removed after gate closure', report['violations'])

    def test_falsely_declared_order_does_not_override_actual_events(self):
        before, _, _ = fixture(); after, execution, allocation = fixture()
        allocation['selected_support']['removal_order'] = 'after_gate'
        report = task_guard_report(before, after, execution, allocation)
        self.assertFalse(report['passed'])
        self.assertIn('actual semantic events violate selected support order', report['violations'])

    def test_actual_buffer_ownership_and_A_source_access_are_checked(self):
        before, _, _ = fixture(); after, execution, allocation = fixture()
        for event in execution['events']:
            if event['kind'] == 'grasp' and event['node'] == 'supply_K1::skill': event['arm'] = 'B'
        report = task_guard_report(before, after, execution, allocation)
        self.assertFalse(report['passed'])
        self.assertIn('A-exclusive actual source grasp violated: K1', report['violations'])
        _, execution, _ = fixture()
        execution['events'] = [x for x in execution['events'] if x['kind'] != 'buffer_deposit']
        self.assertFalse(task_guard_report(before, after, execution, allocation)['passed'])

    def test_changed_skill_action_and_missing_force_guard_are_rejected(self):
        before, _, _ = fixture(); after, execution, allocation = fixture()
        for node in after['nodes']:
            if node['task_id'] == 'install_K1': node['metadata']['action'] = 'wait'
            if node['task_id'] == 'install_K2_approach': node['metadata']['assembly_force_guard'] = False
        report = task_guard_report(before, after, execution, allocation)
        self.assertFalse(report['passed'])
        self.assertIn('changed skill semantics: install_K1', report['violations'])
        self.assertIn('assembly force guard lost: install_K2_approach', report['violations'])

    def test_new_oracle_flag_alone_cannot_replace_full_rollout_coverage(self):
        graph = {'nominal_makespan': 3., 'certificate': {k: True for k in
                 ['dag', 'motion_overlap_pairs_checked', 'collision_oracle_supplied', 'all_nodes_have_geometry',
                  'waiting_endpoints_verified', 'nominal_world_state_verified']}}
        rollout = dict(valid=True, coverage_complete=True, no_sampled_collision=True,
                       nominal_world_state_reconstructed=True, stationary_waiting_poses_included=True,
                       planned_samples=20, tested_samples=20, first_violation=None, world_state_gaps=0,
                       missing_source_poses=0, nominal_makespan_s=3., step_s=.06)
        self.assertTrue(geometry_evidence(graph, rollout)['passed'])
        list_schema = deepcopy(rollout)
        list_schema.update(world_state_gaps=[], missing_source_poses=[])
        self.assertTrue(geometry_evidence(graph, list_schema)['passed'])
        list_schema['world_state_gaps'] = ['missing payload state']
        self.assertFalse(geometry_evidence(graph, list_schema)['passed'])
        bad = deepcopy(rollout); bad['tested_samples'] = 19
        self.assertFalse(geometry_evidence(graph, bad)['passed'])
        bad = deepcopy(rollout); bad['stationary_waiting_poses_included'] = False
        self.assertFalse(geometry_evidence(graph, bad)['passed'])


if __name__ == '__main__': unittest.main()
