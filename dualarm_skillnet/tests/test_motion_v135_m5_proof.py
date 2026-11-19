"""Formal M5 requires a current complete graph/rollout, not an ABBA prefix."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import motion_v135_preflight as p
from tests.test_motion_v135_preflight import put


class CompleteProfileProofTests(unittest.TestCase):
    def fixture(self, root):
        required = ('geometry.py', 'tpg.py', 'program.py', 'motion.py', 'motion_backend.lua',
                    'sim_connection.py', 'PROVENANCE.json')
        version = {'core': {'apexmr/'+name: 'current' for name in required},
                   'tools': {'tools/motion_v135_tpg_profile.py': 'current-tool'}}
        snapshot = {'idle_fps': 8, 'fast_idle_tag_hex': '', 'stack': [0, None],
                    'simulation_state': 0, 'simulation_time_s': 0., 'simulation_timestep_s': .02,
                    'dynamics_step_s': .002, 'dynamics_engine': 0, 'gravity': [0., 0., -9.81]}
        paths = []
        for case in ('00', '02'):
            source = {'task_sha256': 'input-'+case, 'source_tasks_sha256': 'tasks-'+case,
                      'source_workcell_sha256': 'scene-'+case,
                      'joint_sample_step_rad': .07, 'rollout_step_s': .05}
            put(root/'m1'/('tpg'+case+'_a1')/'profile.json', source)
            code = put(root/'jobs'/case/'code_version.json', version['core'])
            destination = root/'profiles'/case
            job = put(root/'jobs'/case/'result.json', {'status': 'passed', 'live': True, 'seconds': 1200.,
                'cleanup': {'passed': True}, 'code_unchanged_during_job': True, 'code_version': str(code),
                'command': ['python', 'tools/motion_v135_tpg_profile.py', '--out', str(destination),
                            '--case', case, '--fast-idle', '--until', 'rollout']})
            data = {**source, 'case': case, 'status': 'completed', 'cleanup_passed': True,
                'diagnostic_only': True, 'formal_run': False, 'until': 'rollout',
                'code_sha256': version['core'], 'project_code_sha256': version['core'],
                'profile_tool_sha256': 'current-tool', 'controlled_job_result': str(job),
                'budget_seconds': 1180., 'graph_counts': {'nodes': 12}, 'fast_idle_enabled': True,
                'fast_idle_scope': {'enabled': True, 'restored': True, 'before': snapshot, 'after': deepcopy(snapshot)},
                'rollout': {'valid': True, 'coverage_complete': True, 'no_sampled_collision': True,
                    'stationary_waiting_poses_included': True, 'planned_samples': 4, 'tested_samples': 4}}
            paths.append(put(destination/'profile.json', data))
        return paths, version

    def test_complete_current_controlled_profiles_pass(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'out') as folder:
            root = Path(folder)
            paths, version = self.fixture(root)
            with patch.object(p.control, 'BASE', root):
                proof = p.verify_profiles(paths, version, require_complete=True)
        self.assertTrue(proof['passed'], proof.get('reason'))

    def test_missing_graph_coverage_restore_or_control_proof_is_rejected(self):
        for field in ('graph_counts', 'controlled_job_result', 'project_code_sha256', 'fast_idle_scope'):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=p.ROOT/'out') as folder:
                root = Path(folder)
                paths, version = self.fixture(root)
                data = p.old.read_json(paths[0]); data.pop(field); put(paths[0], data)
                with patch.object(p.control, 'BASE', root):
                    proof = p.verify_profiles(paths, version, require_complete=True)
                self.assertFalse(proof['passed'])

    def test_partial_rollout_or_changed_physics_is_rejected(self):
        for kind in ('coverage', 'physics', 'app'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=p.ROOT/'out') as folder:
                root = Path(folder); paths, version = self.fixture(root)
                data = p.old.read_json(paths[0])
                if kind == 'coverage': data['rollout']['tested_samples'] = 3
                elif kind == 'physics': data['fast_idle_scope']['after']['simulation_timestep_s'] = .05
                else: data['fast_idle_scope']['after']['idle_fps'] = 0
                put(paths[0], data)
                with patch.object(p.control, 'BASE', root):
                    proof = p.verify_profiles(paths, version, require_complete=True)
                self.assertFalse(proof['passed'])


if __name__ == '__main__':
    unittest.main()
