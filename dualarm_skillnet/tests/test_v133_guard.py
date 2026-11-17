"""Revision authorization never weakens immutable task/archive checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import v133_guard


class RevisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=v133_guard.ROOT / 'out')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = {'protected_files': {}, 'code_hashes': {}}

    def file(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_authorized_search_change_is_reported_without_rebaselining(self):
        name = 'skillnet/planning/search.py'
        self.baseline['code_hashes'][name] = self.file(name, 'old')
        self.file(name, 'new')
        answer = v133_guard.verify_local(self.baseline, self.root)
        self.assertTrue(answer['passed'])
        self.assertEqual(answer['authorized_changes'], [name])
        self.assertEqual(self.baseline['code_hashes'][name], hashlib.sha256(b'old').hexdigest())

    def test_old_task_and_spec_cannot_use_authorized_core_exception(self):
        for name in ('tasks/old.toml', 'FIX_v1.3.3.md'):
            self.baseline['protected_files'][name] = self.file(name, 'old')
            self.file(name, 'changed')
        answer = v133_guard.verify_local(self.baseline, self.root)
        self.assertFalse(answer['passed'])
        self.assertEqual(set(answer['protected_changes']), {'tasks/old.toml', 'FIX_v1.3.3.md'})

    def test_unapproved_core_change_or_new_core_file_is_rejected(self):
        name = 'skillnet/core/engine.py'
        self.baseline['code_hashes'][name] = self.file(name, 'old')
        self.file(name, 'changed')
        self.file('skillnet/planning/shortcut.py', 'new core')
        answer = v133_guard.verify_local(self.baseline, self.root)
        self.assertFalse(answer['passed'])
        self.assertIn(name, answer['code_changes'])
        self.assertIn('skillnet/planning/shortcut.py', answer['new_core_files'])

    def test_d_measured_force_repair_does_not_authorize_archive_validation_edits(self):
        expected = ('simbridge/compiler.py', 'simbridge/controller_ext.lua')
        for name in (*expected, 'apexmr/validation.py'):
            self.baseline['code_hashes'][name] = self.file(name, 'before')
            self.file(name, 'after')
        answer = v133_guard.verify_local(self.baseline, self.root)
        self.assertEqual(set(answer['authorized_changes']), set(expected))
        self.assertEqual(answer['code_changes'], ['apexmr/validation.py'])
        self.assertFalse(answer['passed'])

    def test_explicit_user_document_update_preserves_original_hash_chain(self):
        name='FIX_v1.3.3.md';before=self.file(name,'old instruction')
        self.baseline['protected_files'][name]=before
        after=self.file(name,'new user instruction')
        self.baseline['user_document_updates']=[{'path':name,'previous_sha256':before,'current_sha256':after}]
        self.assertTrue(v133_guard.verify_local(self.baseline,self.root)['passed'])
        self.file(name,'unrequested edit')
        self.assertFalse(v133_guard.verify_local(self.baseline,self.root)['passed'])

    def test_sim_migration_only_allows_reach_removal_not_layout_changes(self):
        name='tasks/sim/s2_tower_sim.toml'
        self.baseline['protected_files'][name]=self.file(name,'name="x"\n[capabilities.reach]\ndefault="VALID"\n')
        self.baseline['reach_table_migrations']={name:{'before_data':{'name':'x','capabilities':{'reach':{'default':'VALID'}}}}}
        self.file(name,'name="x"\n')
        self.assertTrue(v133_guard.verify_local(self.baseline,self.root)['passed'])
        self.file(name,'name="different"\n')
        self.assertFalse(v133_guard.verify_local(self.baseline,self.root)['passed'])
