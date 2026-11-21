"""Timeout-safe motion evidence uses the current task output only."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from simbridge import runner


class MotionJournalTests(unittest.TestCase):
    def test_completed_records_are_append_only_and_progress_is_compact(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'out') as folder:
            root = Path(folder)
            sink = runner.MotionEvidenceJournal(root, time.monotonic(), lambda: 'real_planning')
            event = {'query': 'validate', 'phase': 'before', 'action': 'pick(A, red, stock)',
                     'q': {'A': [0.]*6, 'B': [1.]*6}, 'held': {}, 'placed': {'red': [0.]*7}}
            sink(event)
            event.update(phase='after', status='UNKNOWN', record={'failure_evidence': 'retained'})
            sink(event)
            rows = [json.loads(line) for line in (root/'motion_live.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['event']['phase'], 'before')
            self.assertEqual(rows[1]['event']['record']['failure_evidence'], 'retained')
            progress = json.loads((root/'motion_progress.json').read_text(encoding='utf-8'))
            self.assertEqual(progress['phase'], 'real_planning/validate:after')
            self.assertNotIn('record', progress['event'])
            self.assertEqual(progress['event']['q'], event['q'])
            self.assertEqual(sink.events, 2)

    def test_persistence_error_propagates_and_never_becomes_motion_success(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'out') as folder:
            sink = runner.MotionEvidenceJournal(Path(folder), time.monotonic(), lambda: 'real_planning')
            with patch('simbridge.runner.atomic_progress_json', side_effect=OSError('disk error')):
                with self.assertRaisesRegex(OSError, 'disk error'):
                    sink({'query': 'qualify', 'phase': 'before'})
            self.assertEqual(len((Path(folder)/'motion_live.jsonl').read_text(encoding='utf-8').splitlines()), 1)

    def test_nonfinite_event_is_rejected_before_a_partial_line_is_written(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'out') as folder:
            sink = runner.MotionEvidenceJournal(Path(folder), time.monotonic(), lambda: 'real_planning')
            with self.assertRaises(ValueError):
                sink({'query': 'validate', 'q': [float('nan')]})
            self.assertFalse((Path(folder)/'motion_live.jsonl').exists())

    def test_real_backend_mock_geometry_records_are_json_serializable(self):
        from tests.test_coppelia_backend import backend_fixture
        engine, _, backend = backend_fixture()
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'out') as folder:
            sink = runner.MotionEvidenceJournal(Path(folder), time.monotonic(), lambda: 'real_planning')
            backend.progress_callback = sink
            backend.begin_reference(1)
            backend.validate(engine.parse_action('pick(A, red, stock_l1)'), engine.initial)
            rows = [json.loads(line) for line in (Path(folder)/'motion_live.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[-1]['event']['status'], 'VALID')
            self.assertEqual(rows[-1]['event']['record']['action'], 'pick(A, red, stock_l1)')


if __name__ == '__main__':
    unittest.main()
