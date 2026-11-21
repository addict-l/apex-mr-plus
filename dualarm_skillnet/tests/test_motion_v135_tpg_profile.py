"""Thin statistics routing and reversible codec profiling without a simulator."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from apexmr.motion import MotionPlanner
from tools import motion_v135_tpg_profile as profiling


class StatisticsInterfaceTests(unittest.TestCase):
    def test_sample_statistics_is_one_explicit_read_or_reset_rpc(self):
        planner = MotionPlanner.__new__(MotionPlanner)
        planner._call = Mock(return_value={"samples_checked": 9})
        self.assertEqual(planner.sample_batch_statistics(), {"samples_checked": 9})
        planner._call.assert_called_once_with("mpSampleBatchStatistics", {"reset": False})
        planner._call.reset_mock()
        planner.sample_batch_statistics(reset=True)
        planner._call.assert_called_once_with("mpSampleBatchStatistics", {"reset": True})


class CodecProfilingTests(unittest.TestCase):
    def codec(self):
        return SimpleNamespace(dumps=lambda value, **kw: b"encoded", loads=lambda raw, **kw: {"ok": True})

    def test_observations_preserve_arguments_results_and_restore_original_functions(self):
        codec = self.codec()
        original_dumps, original_loads = codec.dumps, codec.loads
        with profiling.CodecMetrics(codec) as measure:
            self.assertEqual(codec.dumps({"sample": 1}), b"encoded")
            self.assertEqual(codec.loads(b"encoded"), {"ok": True})
            self.assertEqual(measure.stats["encode_calls"], 1)
            self.assertEqual(measure.stats["decode_calls"], 1)
            self.assertEqual(measure.stats["encoded_bytes"], 7)
            self.assertEqual(measure.stats["decoded_bytes"], 7)
            self.assertGreaterEqual(measure.stats["encode_seconds"], 0.)
            self.assertGreaterEqual(measure.stats["decode_seconds"], 0.)
        self.assertIs(codec.dumps, original_dumps)
        self.assertIs(codec.loads, original_loads)

    def test_codec_exception_is_preserved_and_patch_is_always_restored(self):
        codec = self.codec()
        def fail(raw, **kwargs):
            raise ValueError("decoder sentinel")
        codec.loads = fail
        original = codec.dumps
        with self.assertRaisesRegex(ValueError, "decoder sentinel"):
            with profiling.CodecMetrics(codec) as measure:
                codec.loads(b"bad")
        self.assertIs(codec.dumps, original)
        self.assertIs(codec.loads, fail)
        self.assertEqual(measure.stats["decode_calls"], 1)
        self.assertEqual(measure.stats["decoded_bytes"], 3)

    def test_absent_codec_is_explicit_and_never_changes_the_client(self):
        with profiling.CodecMetrics(None) as measure:
            self.assertFalse(measure.stats["available"])
            self.assertEqual(measure.stats["encode_calls"], 0)


class LegacyProfileControlTests(unittest.TestCase):
    def test_force_legacy_flag_only_changes_outer_diagnostic_options(self):
        native = SimpleNamespace(_call=Mock(), sample_state_pairs=Mock(return_value=[{"ok": True}]))
        wrapper = profiling.ProfiledPlanner(native, lambda: None, lambda: None, legacy_reset=True)
        samples = [{"A": [0]*6, "B": [0]*6, "options": {"contact_tolerance": .003}}]
        options = {"other": 1}
        wrapper.sample_state_pairs(samples, options)
        native.sample_state_pairs.assert_called_once_with(samples, {"other": 1, "legacy_batch_reset": True})
        self.assertEqual(options, {"other": 1})
        self.assertEqual(samples[0]["options"], {"contact_tolerance": .003})


if __name__ == "__main__":
    unittest.main()
