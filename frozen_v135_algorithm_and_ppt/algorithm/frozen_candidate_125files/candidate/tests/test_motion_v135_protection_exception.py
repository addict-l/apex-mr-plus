"""The user's one exact historical-artifact exception never widens the guard."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import motion_v135_control as control

NAME = 'results/apex/geometry/rollout_verification.json'
ORIGINAL = 'b1860c844c677aca51185a371f9cd984ec14516d9e2ab97b984a694e950a56ad'
ACCEPTED = '8a8a1a66e4d98ecef2caefbf7a01d359a5cb737102760138ab86f0cd2335c47c'


class ExactProtectionExceptionTests(unittest.TestCase):
    def fixture(self, root, *, authorized=True, actual=ACCEPTED, extra=False):
        base = root/'results/sim_v1_3/motion_v135'
        base.mkdir(parents=True)
        target = root/NAME
        target.parent.mkdir(parents=True)
        target.write_bytes(b'overwritten historical artifact; never acceptance evidence')
        baseline = {'source_files': [], 'history_files': [
            {'path': NAME, 'sha256': ORIGINAL, 'size': 949, 'mtime_ns': 1}]}
        (base/'baseline.json').write_text(json.dumps(baseline), encoding='utf-8')
        register = {'authorization_id': 'user-motion-v135-resume-20261117',
            'baseline_sha256': hashlib.sha256((base/'baseline.json').read_bytes()).hexdigest(),
            'exceptions': [{'path': NAME, 'original_sha256': ORIGINAL,
                            'accepted_sha256': ACCEPTED, 'not_acceptance_evidence': True}]}
        if authorized:
            (base/'protection_exceptions.json').write_text(json.dumps(register), encoding='utf-8')
        if extra:
            (root/'results/another_unapproved_file.json').write_bytes(b'not authorized')
        original_hash = control.old.sha256
        def hashed(path):
            return actual if Path(path) == target else original_hash(path)
        return base, hashed, register

    def audit(self, root, base, hashed):
        guard = type('Guard', (), {'check': lambda self: {'passed': True, 'changes': []}})
        with (patch.object(control, 'ROOT', root), patch.object(control, 'BASE', base),
              patch.object(control, '_archive_guard', None),
              patch.object(control.old, 'ArchiveGuard', guard),
              patch.object(control.old, 'sha256', side_effect=hashed)):
            return control.audit(full=True)

    def test_exact_authorized_change_passes_with_explicit_non_evidence_receipt(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
            root = Path(folder)
            base, hashed, _ = self.fixture(root)
            original_baseline = (base/'baseline.json').read_bytes()
            result = self.audit(root, base, hashed)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['changed'], [])
            self.assertEqual(result['authorized_exceptions'][0]['path'], NAME)
            self.assertTrue(result['authorized_exceptions'][0]['not_acceptance_evidence'])
            self.assertEqual((base/'baseline.json').read_bytes(), original_baseline)
            self.assertFalse((base/'global_stop.json').exists())

    def test_register_missing_never_accepts_the_overwrite(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
            root = Path(folder)
            base, hashed, _ = self.fixture(root, authorized=False)
            self.assertFalse(self.audit(root, base, hashed)['passed'])
            self.assertTrue((base/'global_stop.json').exists())

    def test_further_artifact_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
            root = Path(folder)
            base, hashed, _ = self.fixture(root, actual='f'*64)
            self.assertFalse(self.audit(root, base, hashed)['passed'])

    def test_another_history_difference_still_stops(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
            root = Path(folder)
            base, hashed, _ = self.fixture(root, extra=True)
            result = self.audit(root, base, hashed)
            self.assertFalse(result['passed'])
            self.assertIn('results/another_unapproved_file.json', result['changed'])

    def test_register_cannot_change_path_hash_or_evidence_scope(self):
        for field, replacement in (('path', 'results/other.json'),
                                   ('accepted_sha256', 'e'*64),
                                   ('original_sha256', 'd'*64),
                                   ('not_acceptance_evidence', False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
                root = Path(folder)
                base, hashed, register = self.fixture(root)
                register['exceptions'][0][field] = replacement
                (base/'protection_exceptions.json').write_text(json.dumps(register), encoding='utf-8')
                self.assertFalse(self.audit(root, base, hashed)['passed'])

    def test_original_bytes_need_no_exception(self):
        with tempfile.TemporaryDirectory(dir=control.ROOT/'out') as folder:
            root = Path(folder)
            base, hashed, _ = self.fixture(root, authorized=False, actual=ORIGINAL)
            self.assertTrue(self.audit(root, base, hashed)['passed'])


if __name__ == '__main__':
    unittest.main()
