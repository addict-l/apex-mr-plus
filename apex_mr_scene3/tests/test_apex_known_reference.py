"""Complete-reference eligibility, optimum ties, and fixed-end refinement."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.compile import (Compiler, CompileFailure, KNOWN_REFERENCE_ROOT, _value_hash,
                          load_known_complete_reference, known_reference_allocation,
                          refine_known_complete_reference)


def q(x, y=0.):
    return [x, y, 0., 0., 0., 0.]


class CompleteReferenceEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='scene3_complete_reference_')
        self.root = Path(self.temp.name)
        self.cfg = {'initial': {}, 'goal': {}, 'sizes': {}, 'runtime': {},
                    'free_q': {'A': q(0), 'B': q(0)}, 'initial_support_contacts': {}, 'dt': .02,
                    'arms': {'A': {'home_q': q(0)}, 'B': {'home_q': q(0)}}}
        files = ['scenes/10_Scene3_Dynamic.ttt', 'config/apex_runtime.json',
                 'config/dynamic_homes.json', 'config/experiment_definition.json', 'config/scene_manifest.json']
        hashes = {}
        for name in files:
            path = self.root/name;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic physical input')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tasks = [{'id': f'move_{i:03}', 'arm': 'B', 'kind': 'transit',
                       'q_path': [q(i*.001), q((i+1)*.001)],
                       'dependencies': [] if i == 0 else [f'move_{i-1:03}']} for i in range(134)]
        self.manifest = {'status': 'complete', 'execution_performed': True, 'run_id': 'synthetic',
                         'input_fingerprint': {'files': hashes}, 'scene_signature': 'physical_inputs',
                         'reference_signature': _value_hash(self.tasks)}
        self.records = {
            'alloc_and_reference.json': {'tasks': self.tasks, 'scene_signature': 'physical_inputs'},
            'reference_tasks.json': self.tasks,
            'source_signature.json': {'inputs': {'files': hashes}, 'resolved_scene': self.cfg},
            'pipeline_manifest.json': self.manifest,
            'summary.json': {'success': True, 'run_id': 'synthetic', 'reference_task_count': 134},
            'execution_validation.json': {'success': True, 'run_id': 'synthetic'},
            'independent_final_audit.json': {'success': True, 'run_id': 'synthetic',
                                           'source_scene_sha256': hashes[files[0]]},
        }
        self.write_records()
        self.patch = patch('apex.compile.ROOT', self.root);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def write_records(self):
        target = self.root/KNOWN_REFERENCE_ROOT;target.mkdir(parents=True, exist_ok=True)
        for name, value in self.records.items():
            (target/name).write_text(json.dumps(value), encoding='utf-8')

    def test_intact_reference_is_eligible_but_changed_runtime_or_scene_is_not(self):
        baseline, evidence = load_known_complete_reference(self.cfg)
        self.assertTrue(evidence['eligible'])
        self.assertEqual(len(baseline['tasks']), 134)
        (self.root/'config/apex_runtime.json').write_bytes(b'changed force or travel setting')
        self.assertFalse(load_known_complete_reference(self.cfg)[1]['eligible'])

    def test_recomputed_hash_does_not_hide_a_broken_joint_chain(self):
        self.tasks[10]['q_path'][0] = q(.02)
        self.manifest['reference_signature'] = _value_hash(self.tasks)
        self.write_records()
        result, evidence = load_known_complete_reference(self.cfg)
        self.assertIsNone(result)
        self.assertIn('reference_joint_chain_not_closed', evidence['reason'])

    def test_nested_v2_audit_source_identity_is_checked_without_editing_the_backup(self):
        audit = self.records['independent_final_audit.json']
        source = audit.pop('source_scene_sha256')
        audit['delivery'] = {'original_physical_scene_sha256': source}
        self.write_records()
        self.assertTrue(load_known_complete_reference(self.cfg)[1]['eligible'])
        audit['delivery']['original_physical_scene_sha256'] = 'wrong'
        self.write_records()
        self.assertFalse(load_known_complete_reference(self.cfg)[1]['eligible'])

    def test_missing_physics_or_changed_resolved_home_cannot_reuse_reference(self):
        changed = deepcopy(self.cfg);changed['arms']['B']['home_q'] = q(.01)
        self.assertFalse(load_known_complete_reference(changed)[1]['eligible'])
        changed = deepcopy(self.cfg);changed['runtime'] = {'support_extension_m': .012}
        self.assertFalse(load_known_complete_reference(changed)[1]['eligible'])


class CompleteReferenceChoiceTests(unittest.TestCase):
    def allocation(self, objective=10.):
        support = {'id': 'before', 'candidate': 'outer', 'removal_order': 'before_gate',
                   'placer_arm': 'B', 'remover_arm': 'B', 'goal_pose': [0., 0., 1., 0., 0., 0., 1.],
                   'endpoint_evidence': {'grasp_offset': [0., 0., .1], 'grasp_quaternion': [0., 0., 0., 1.]}}
        return {'solved': True, 'optimal': True, 'objective': objective,
                'selected_support': support, 'selected_mode_ids': ['before', 'install:B'],
                'task_assignments': {'install': 'B'},
                'choices': [{'id': 'install:B', 'task': 'install', 'arm': 'B',
                             'grasp_offset': [0., 0., .1], 'grasp_quaternion': [0., 0., 0., 1.]}]}

    def test_known_support_is_chosen_only_for_equal_unrestricted_cost(self):
        problem = {'support_modes': [{'id': 'before'}, {'id': 'after'}]}
        baseline = self.allocation()
        with patch('apex.compile.solve_allocation', return_value=self.allocation()) as solve:
            chosen, evidence = known_reference_allocation(problem, self.allocation(), baseline)
            self.assertTrue(evidence['accepted'])
            self.assertIn('independently checked', chosen['optimal_scope'])
            solve.assert_called_once_with(problem, banned_modes=['after'])
        with patch('apex.compile.solve_allocation', return_value=self.allocation(10.01)):
            self.assertIsNone(known_reference_allocation(problem, self.allocation(), baseline)[0])

    def test_equal_cost_does_not_hide_changed_assignment_or_grasp(self):
        problem = {'support_modes': [{'id': 'before'}, {'id': 'after'}]}
        changed = self.allocation();changed['choices'][0]['grasp_offset'] = [.1, 0., .1]
        with patch('apex.compile.solve_allocation', return_value=changed):
            self.assertIsNone(known_reference_allocation(problem, self.allocation(), self.allocation())[0])


class FixedReferenceRefinementTests(unittest.TestCase):
    def fixture(self, fail_optimization=False, geometry_ok=True, worsen=False):
        c = Compiler.__new__(Compiler)
        c.home = {'A': q(0), 'B': q(0)};c.source = {'K1': [0., 0., 0., 0., 0., 0., 1.]};c.calls = []
        payload = {'name': 'K1', 'offset': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
        md = {'placed': deepcopy(c.source), 'contact_exclusions': [['A_Pad', 'Bridge']],
              'options': {'articulations': {'SupportLift': 0.}}}
        tasks = [
            {'id': 'bent_B', 'arm': 'B', 'kind': 'transit', 'q_path': [q(0), q(.1, .1), q(.2)],
             'metadata': deepcopy(md), 'payload': None, 'dependencies': []},
            {'id': 'grasp_B', 'arm': 'B', 'kind': 'grasp', 'q_path': [q(.2), q(.2)],
             'metadata': {**deepcopy(md), 'effects': {'payloads': {'B': payload}}},
             'payload': None, 'dependencies': ['bent_B']},
            {'id': 'bent_A', 'arm': 'A', 'kind': 'transit', 'q_path': [q(0), q(.1, .1), q(.2)],
             'metadata': deepcopy(md), 'payload': None, 'dependencies': ['grasp_B']},
            {'id': 'straight_B', 'arm': 'B', 'kind': 'transit', 'q_path': [q(.2), q(.3)],
             'metadata': deepcopy(md), 'payload': payload, 'dependencies': ['bent_A']},
        ]

        class SyntheticPlanner:
            def __init__(self):
                self.calls = [];self.samples = []

            def plan(self, arm, start, end, held, placed, solve_time, options):
                self.calls.append(deepcopy(dict(arm=arm, start=start, end=end, payload=held,
                                               placed=placed, solve_time=solve_time, options=options)))
                if fail_optimization:
                    return {'ok': False, 'error': 'Synthetic optimization budget exhausted', 'stats': {}}
                return {'ok': True, 'path': [start, q(.1, 2.), end] if worsen else [start, end],
                        'algorithm': 'ValidatedReferencePath', 'path_source': 'synthetic_test_only',
                        'stats': {'reference_candidates': [{'valid': True}], 'candidates': []}}

            def validate_samples(self, samples):
                self.samples.extend(deepcopy(samples))
                return [{'ok': geometry_ok} for sample in samples]

        c.p = SyntheticPlanner()
        return c, {'tasks': tasks}, payload

    def test_fixed_end_refinement_preserves_skills_closure_and_other_arm_payload(self):
        compiler, reference, payload = self.fixture()
        before = deepcopy(reference['tasks'])
        with redirect_stdout(io.StringIO()):
            tasks, stats = refine_known_complete_reference(compiler, reference)
        self.assertEqual(stats['planner_query_count'], 2)
        self.assertEqual(stats['unchanged_reference_path_count'], 1)
        self.assertEqual(tasks[1], before[1])
        for old, new in zip(before, tasks):
            self.assertEqual(old['q_path'][0], new['q_path'][0])
            self.assertEqual(old['q_path'][-1], new['q_path'][-1])
            self.assertEqual(old['dependencies'], new['dependencies'])
        self.assertEqual(compiler.p.calls[1]['options']['other_q'], q(.2))
        self.assertEqual(compiler.p.calls[1]['options']['other_payload'], payload)
        self.assertTrue(all(call['solve_time'] == 3. and call['options']['direct_first'] is False for call in compiler.p.calls))
        self.assertEqual(reference['tasks'], before)

    def test_failed_optimizer_retains_original_only_after_fresh_dense_validation(self):
        compiler, reference, _ = self.fixture(fail_optimization=True)
        with redirect_stdout(io.StringIO()):
            tasks, stats = refine_known_complete_reference(compiler, reference)
        self.assertGreater(len(compiler.p.samples), 100)
        self.assertEqual(tasks[0]['q_path'], reference['tasks'][0]['q_path'])
        self.assertEqual(stats['refinements'][0]['path_source'], 'revalidated_original_reference')

    def test_colliding_original_cannot_be_retained(self):
        compiler, reference, _ = self.fixture(fail_optimization=True, geometry_ok=False)
        with redirect_stdout(io.StringIO()), self.assertRaises(CompileFailure):
            refine_known_complete_reference(compiler, reference)

    def test_worse_candidate_never_replaces_revalidated_original(self):
        compiler, reference, _ = self.fixture(worsen=True)
        with redirect_stdout(io.StringIO()):
            tasks, stats = refine_known_complete_reference(compiler, reference)
        self.assertEqual(tasks[0]['q_path'], reference['tasks'][0]['q_path'])
        self.assertEqual(stats['refinements'][0]['selected_cost'], stats['refinements'][0]['baseline_cost'])


if __name__ == '__main__':
    unittest.main()
