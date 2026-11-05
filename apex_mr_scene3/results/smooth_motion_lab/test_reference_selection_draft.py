"""Synthetic reference selection tests; no simulator or accepted-run writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_selection_draft import choose_reference, value_hash


def q(x, y=0.):
    return [x, y, 0., 0., 0., 0.]


def fixture(detour=False):
    tasks = []
    def add(tid, arm, kind, path, dep=(), action=None):
        tasks.append({'id': tid, 'arm': arm, 'kind': kind, 'part': 'K1' if action else None,
                      'q_path': path, 'duration': .2, 'dependencies': list(dep), 'reference_order': len(tasks),
                      'payload': None, 'metadata': {'action': action, 'dwell_s': .1} if action else {'planner': 'planner'}})
    add('move_A', 'A', 'transit', [q(0.), q(.5, 2.), q(1.)] if detour else [q(0.), q(1.)])
    add('install_K1', 'A', 'install', [q(1.)], ['move_A'], 'install')
    add('move_B', 'B', 'transit', [q(0.), q(1.)])
    return {'tasks': tasks,
            'allocation': {'task_assignments': {'install_K1': 'A'},
                'choices': [{'task': 'install_K1', 'arm': 'A', 'grasp_offset': [0, 0, .03], 'grasp_quaternion': [1, 0, 0, 0], 'cost': 99}],
                'selected_support': {'candidate': 'outer_front', 'removal_order': 'before_gate', 'placer_arm': 'B',
                    'remover_arm': 'B', 'goal_pose': [0, 0, 0, 0, 0, 0, 1], 'endpoint_evidence': {'grasp_offset': [0, 0, .1]}},
                'handoffs': [], 'required_buffer_edges': [], 'required_lifecycle_edges': [], 'assembly_order': ['K1'], 'buffer_capacity': 1},
            'problem': {'qualification_signature': 'fresh-qualification'}, 'validation_requirements': {'force_min': 1.},
            'planner_info': {'initialhome_q': {'A': q(0.), 'B': q(0.)}},
            'motion_stats': {'path_count': 2, 'calls': [{'from': 'detour' if detour else 'straight'}]}}


class ReferenceSelectionTests(unittest.TestCase):
    args = {'expected_task_count': 3, 'physical_scene_sha256': 'same-scene-hash', 'span': .24, 'vmax': .6, 'amax': 1.2, 'jmax': 8.}

    def test_whole_better_baseline_selected_preserving_fresh_wrapper_and_all_inputs(self):
        fresh, baseline = fixture(True), fixture(False)
        fresh['allocation']['choices'][0]['cost'] = 2
        baseline['problem']['qualification_signature'] = 'historical-qualification'
        before_f, before_b = deepcopy(fresh), deepcopy(baseline)
        selected, report = choose_reference(fresh, baseline, self.args)
        self.assertTrue(report['baseline_eligible'], report['rejection_reasons'])
        self.assertEqual(report['selected'], 'baseline')
        self.assertEqual(selected['tasks'], baseline['tasks'])
        self.assertEqual(selected['allocation'], fresh['allocation'])
        self.assertEqual(selected['problem'], fresh['problem'])
        self.assertEqual(selected['validation_requirements'], fresh['validation_requirements'])
        self.assertEqual(selected['motion_stats'], baseline['motion_stats'])
        self.assertEqual(selected['reference_signature'], value_hash(baseline['tasks']))
        self.assertFalse(report['collision_validated'])
        self.assertFalse(report['actual_execution_claimed'])
        self.assertEqual(fresh, before_f); self.assertEqual(baseline, before_b)
        for row in report['candidate_ranking'].values():
            self.assertEqual(row['collision_edge_count'], 0)
            self.assertGreater(row['serial_parameterized_service_proxy_s'], 0)

    def test_fresh_shorter_or_exact_tie_retains_fresh(self):
        for baseline in (fixture(True), fixture(False)):
            _, report = choose_reference(fixture(False), baseline, self.args)
            self.assertEqual(report['selected'], 'fresh')
            self.assertTrue(report['baseline_eligible'])

    def test_changed_support_order_disqualifies_better_historical_path(self):
        baseline = fixture(False)
        baseline['allocation']['selected_support']['removal_order'] = 'after_gate'
        _, report = choose_reference(fixture(True), baseline, self.args)
        self.assertFalse(report['baseline_eligible'])
        self.assertEqual(report['selected'], 'fresh')
        self.assertIn('same_support_candidate_order_arms_goal_and_grasp', report['rejection_reasons'])
        self.assertNotIn('baseline', report['candidate_ranking'])

    def test_skill_payload_dependency_assignment_and_grasp_changes_rejected(self):
        changes = [lambda b: b['tasks'][1]['metadata'].update(action='release_buffer'),
                   lambda b: b['tasks'][1].update(part='K2'),
                   lambda b: b['tasks'][1].update(dependencies=[]),
                   lambda b: b['tasks'][0].update(payload={'part': 'K2'}),
                   lambda b: b['allocation']['task_assignments'].update(install_K1='B'),
                   lambda b: b['allocation']['choices'][0].update(grasp_offset=[0, 0, .04]),
                   lambda b: b['tasks'][0]['metadata'].update(assembly_force_guard=True)]
        for change in changes:
            baseline = fixture(False); change(baseline)
            with self.subTest(change=change):
                _, report = choose_reference(fixture(True), baseline, self.args)
                self.assertFalse(report['baseline_eligible'])
                self.assertEqual(report['selected'], 'fresh')

    def test_initial_winding_and_internal_joint_discontinuity_rejected(self):
        baseline = fixture(False)
        baseline['planner_info']['initialhome_q']['A'][0] += 2*3.141592653589793
        _, report = choose_reference(fixture(True), baseline, self.args)
        self.assertFalse(report['eligibility_checks']['same_declared_initial_joints'])
        self.assertFalse(report['baseline_eligible'])
        baseline = fixture(False); baseline['tasks'][1]['q_path'][0][0] += .01
        _, report = choose_reference(fixture(True), baseline, self.args)
        self.assertFalse(report['eligibility_checks']['baseline_joint_paths_close'])

    def test_partial_baseline_and_changed_final_joint_semantics_rejected(self):
        baseline = fixture(False); baseline['tasks'].pop()
        _, report = choose_reference(fixture(True), baseline, self.args)
        self.assertFalse(report['baseline_eligible'])
        baseline = fixture(False); baseline['tasks'][-1]['q_path'][-1] = q(2.)
        _, report = choose_reference(fixture(True), baseline, self.args)
        self.assertFalse(report['eligibility_checks']['same_final_joint_semantics'])

    def test_partial_fresh_is_rejected_not_compared_as_faster(self):
        fresh = fixture(True); fresh['tasks'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            choose_reference(fresh, fixture(False), self.args)

    def test_physical_scene_hash_must_be_known_and_equal(self):
        for options in ({'expected_task_count': 3}, {'expected_task_count': 3, 'fresh_scene_sha256': 'a', 'baseline_scene_sha256': 'b'}):
            _, report = choose_reference(fixture(True), fixture(False), options)
            self.assertFalse(report['baseline_eligible'])
            self.assertFalse(report['eligibility_checks']['same_known_physical_source_scene'])

    def test_file_hash_provenance_and_manifest_physics_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, detour in [('fresh', True), ('baseline', False)]:
                d = root/label; d.mkdir()
                (d/'alloc_and_reference.json').write_text(json.dumps(fixture(detour)), encoding='utf-8')
                (d/'pipeline_manifest.json').write_text(json.dumps({'input_fingerprint': {'files': {'scenes/10_Scene3_Dynamic.ttt': 'same'}}}), encoding='utf-8')
            paths = [root/label/'alloc_and_reference.json' for label in ('fresh', 'baseline')]
            raw = [p.read_bytes() for p in paths]
            _, report = choose_reference(*paths, {'expected_task_count': 3})
            self.assertEqual(report['selected'], 'baseline')
            for label, path, content in zip(('fresh', 'baseline'), paths, raw):
                self.assertEqual(report['sources'][label]['path'], str(path.resolve()))
                self.assertEqual(report['sources'][label]['file_sha256'], hashlib.sha256(content).hexdigest())
                self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__': unittest.main()
