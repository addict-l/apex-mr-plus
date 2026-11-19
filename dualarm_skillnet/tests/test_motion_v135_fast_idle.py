"""Owned stopped-scene rate changes preserve physics and restore app state."""
from contextlib import contextmanager
import struct
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

from apexmr.sim_connection import Simulator


class FakeSim:
    simulation_stopped = 0
    intparam_idle_fps = 7
    intparam_dynamic_engine = 8
    handle_app = -13
    handle_scene = -12

    def __init__(self, fps=8, tag=None):
        self.fps, self.tag = fps, tag
        self.state, self.time = 0, 0.
        self.writes = []
        self.enable_error = None
        self.restore_error = False
        self.nonce = "owned-test"

    def getNamedStringParam(self, key): return self.nonce
    def getSimulationState(self): return self.state
    def getSimulationTime(self): return self.time
    def getSimulationTimeStep(self): return .02
    def getFloatProperty(self, handle, key): return .002
    def getVector3Property(self, handle, key): return [0., 0., -9.81]
    def getInt32Param(self, key): return self.fps if key == self.intparam_idle_fps else 0
    def readCustomStringData(self, handle, key): return self.tag
    def readCustomBufferData(self, handle, key): return self.tag
    def unpackInt32Table(self, raw): return list(struct.unpack("<ii", raw))

    def setInt32Param(self, key, value):
        if key != self.intparam_idle_fps: raise AssertionError("physical write")
        self.writes.append(("fps", value)); self.fps = value

    def writeCustomStringData(self, handle, key, value):
        if not key: raise AssertionError("delete all tags")
        if self.restore_error: raise ConnectionError("restore channel failed")
        self.writes.append(("tag", value)); self.tag = value or None

    def writeCustomBufferData(self, handle, key, value):
        return self.writeCustomStringData(handle, key, value)

    def fastIdleLoop(self, enable):
        stage, saved = self.unpackInt32Table(self.tag) if self.tag else (0, self.fps)
        stage = stage + 1 if enable else max(0, stage - 1)
        self.setInt32Param(self.intparam_idle_fps, 0 if stage else saved)
        if enable and self.enable_error: raise self.enable_error
        self.writeCustomStringData(self.handle_app, "__IDLEFPSSTACKSIZE__", struct.pack("<ii", stage, saved))


def owner(sim):
    result = Simulator(".", port=23210)
    result.process = SimpleNamespace(pid=123, poll=lambda: None)
    result._instance, result.sim, result.client = "owned-test", sim, object()
    return result


class StoppedFastIdleTests(unittest.TestCase):
    def test_packed_custom_data_uses_binary_wire_api_before_and_after_enabling(self):
        class BinaryWireSim(FakeSim):
            def readCustomStringData(self, handle, key):
                if self.tag:
                    raise UnicodeError('CBOR TEXT:not UTF-8')
                return self.tag
            def readCustomBufferData(self, handle, key): return self.tag
            def writeCustomBufferData(self, handle, key, value):
                self.buffer_writes = getattr(self, 'buffer_writes', 0)+1
                return super().writeCustomBufferData(handle, key, value)
        for fps, raw in [(8,None),(8,struct.pack('<ii',0,255)),(0,struct.pack('<ii',2,255))]:
            sim = BinaryWireSim(fps, raw)
            with owner(sim).fast_idle_stopped() as receipt:
                self.assertEqual(sim.fps, 0)
            self.assertEqual((sim.fps,sim.tag),(fps,raw))
            self.assertTrue(receipt['restored'])
            self.assertEqual(sim.buffer_writes, 1)

    def test_owned_scope_enables_fast_idle_then_restores_exact_original_state(self):
        sim = FakeSim()
        with owner(sim).fast_idle_stopped() as receipt:
            self.assertEqual(sim.fps, 0)
            self.assertEqual(receipt["before"]["simulation_timestep_s"], .02)
        self.assertEqual((sim.fps, sim.tag), (8, None))
        self.assertTrue(receipt["restored"])
        self.assertEqual(receipt["before"], receipt["after"])

    def test_preexisting_nested_scope_and_stale_zero_saved_fps_restore_exactly(self):
        for fps, raw in [(0, struct.pack('<ii', 2, 8)), (12, struct.pack('<ii', 0, 8))]:
            sim = FakeSim(fps, raw)
            with owner(sim).fast_idle_stopped() as receipt:
                self.assertEqual(sim.fps, 0)
            self.assertEqual((sim.fps, sim.tag), (fps, raw))
            self.assertTrue(receipt['restored'])

    def test_body_error_and_partial_enable_error_restore_before_state(self):
        sim = FakeSim()
        with self.assertRaisesRegex(ValueError, 'body'):
            with owner(sim).fast_idle_stopped(): raise ValueError('body')
        self.assertEqual((sim.fps, sim.tag), (8, None))
        sim = FakeSim(0, struct.pack('<ii', 2, 8))
        sim.enable_error = ValueError('partial enable')
        with self.assertRaisesRegex(ValueError, 'partial enable'):
            with owner(sim).fast_idle_stopped(): self.fail('must not enter')
        self.assertEqual((sim.fps, sim.tag), (0, struct.pack('<ii', 2, 8)))

    def test_enable_reply_lost_after_stack_write_balances_only_its_own_increment(self):
        sim = FakeSim(0, struct.pack('<ii', 2, 8))
        original = sim.fastIdleLoop
        def lost_reply(enable):
            original(enable)
            if enable: raise ConnectionError('reply lost after write')
        sim.fastIdleLoop = lost_reply
        with self.assertRaisesRegex(ConnectionError, 'reply lost'):
            with owner(sim).fast_idle_stopped(): self.fail('must not enter')
        self.assertEqual((sim.fps, sim.tag), (0, struct.pack('<ii', 2, 8)))

    def test_unowned_running_or_expired_scopes_never_write_app_state(self):
        for change in ('unowned', 'running', 'expired', 'other-nonce'):
            sim = FakeSim(); instance = owner(sim)
            if change == 'unowned': instance.process = None
            if change == 'running': sim.state = 1
            if change == 'other-nonce': sim.nonce = 'unrelated'
            with self.assertRaises((RuntimeError, TimeoutError)):
                with instance.fast_idle_stopped(deadline=0 if change == 'expired' else None):
                    self.fail('must not enter')
            self.assertEqual(sim.writes, [])

    def test_restore_failure_or_external_stack_change_is_not_reported_restored(self):
        from apexmr.sim_connection import FastIdleRestorationError
        for change in ('restore-error', 'other-scope'):
            sim = FakeSim()
            with self.assertRaises(FastIdleRestorationError) as caught:
                with owner(sim).fast_idle_stopped():
                    if change == 'restore-error': sim.restore_error = True
                    else: sim.fastIdleLoop(True)
            self.assertFalse(caught.exception.receipt['restored'])
            if change == 'other-scope':
                self.assertEqual(sim.unpackInt32Table(sim.tag)[0], 2)

    def test_stopped_snapshot_is_json_safe_and_refuses_attached_instances(self):
        sim = FakeSim()
        self.assertEqual(owner(sim).stopped_scene_snapshot()['gravity'], [0., 0., -9.81])
        instance = owner(sim); instance.process = None
        with self.assertRaises(RuntimeError): instance.stopped_scene_snapshot()

    def test_physical_state_change_is_detected_without_attempting_physics_restore(self):
        from apexmr.sim_connection import FastIdleRestorationError
        sim = FakeSim()
        with self.assertRaises(FastIdleRestorationError):
            with owner(sim).fast_idle_stopped(): sim.time = 1.
        self.assertEqual((sim.fps, sim.tag), (8, None))
        self.assertEqual(sim.time, 1.)


class FixedTapeAbbaTests(unittest.TestCase):
    def test_fixed_work_abba_preserves_answers_and_resumes_committed_phases(self):
        from tools.motion_v135_fast_idle import replay_abba
        sim = FakeSim(); instance = owner(sim)
        seen = []
        class Planner:
            def sample_state_pairs(self, samples, options=None):
                seen.append((sim.fps, samples, options))
                return [{'ok': row['ok'], 'problem': None} for row in samples]
            def sample_batch_statistics(self): return {'batches': len(seen)}
        tape = [{'samples': [{'ok': True}], 'options': {}, 'expected': [{'ok': True, 'problem': None}]},
                {'samples': [{'ok': False}], 'options': {'contact_tolerance': .003},
                 'expected': [{'ok': False, 'problem': None}]}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'out') as tmp:
            result = replay_abba(instance, Planner(), tape, Path(tmp), time.monotonic()+60, warmup=1)
            self.assertEqual([row['phase'] for row in result['phases']], ['A1','B1','B2','A2'])
            self.assertEqual([row[0] for row in seen], [8]*3+[0]*6+[8]*3)
            self.assertTrue(result['passed'])
            self.assertEqual((sim.fps, sim.tag), (8, None))
            replay_abba(instance, Planner(), tape, Path(tmp), time.monotonic()+60, warmup=1)
            self.assertEqual(len(seen), 12)

    def test_candidate_answer_disagreement_restores_idle_and_does_not_commit_pass(self):
        from tools.motion_v135_fast_idle import replay_abba
        sim = FakeSim()
        class Planner:
            def __init__(self): self.calls = 0
            def sample_state_pairs(self, samples, options=None):
                self.calls += 1
                return [{'ok': sim.fps != 0}]
            def sample_batch_statistics(self): return {'batches': self.calls}
        tape = [{'samples': [{'A': [0.]*6}], 'expected': [{'ok': True}]}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'out') as tmp:
            with self.assertRaisesRegex(ValueError, 'differs'):
                replay_abba(owner(sim), Planner(), tape, Path(tmp), time.monotonic()+60, warmup=0)
            self.assertTrue((Path(tmp)/'A1.json').is_file())
            self.assertFalse((Path(tmp)/'B1.json').exists())
            self.assertEqual((sim.fps, sim.tag), (8, None))

    def test_resume_rejects_changed_tape_and_expired_deadline_runs_no_queries(self):
        from tools.motion_v135_fast_idle import replay_abba
        sim = FakeSim()
        class Planner:
            def __init__(self): self.calls = 0
            def sample_state_pairs(self, samples, options=None): self.calls += 1; return [{'ok': True}]
            def sample_batch_statistics(self): return {'batches': self.calls}
        tape = [{'samples': [{'A': [0.]*6}], 'expected': [{'ok': True}]}]
        planner = Planner()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'out') as tmp:
            with self.assertRaises(TimeoutError): replay_abba(owner(sim), planner, tape, Path(tmp), 0)
            self.assertEqual(planner.calls, 0)
            tape[0]['samples'][0]['A'][0] = .1
            with self.assertRaisesRegex(ValueError, 'changed'):
                replay_abba(owner(sim), planner, tape, Path(tmp), time.monotonic()+60)

    def test_capture_preserves_request_and_stops_only_after_fixed_completed_calls(self):
        from tools.motion_v135_fast_idle import CapturePlanner, TapeCaptured
        import io
        class Planner:
            def sample_state_pairs(self, samples, options=None): return [{'ok': False, 'problem': {'reason':'table'}}]
        journal = io.StringIO()
        capture = CapturePlanner(Planner(), 1, journal, time.monotonic()+60)
        samples = [{'A': [0.]*6, 'payloads': {'B': {'name': 'beam', 'offset': [.0,.0,.026]}}}]
        with self.assertRaises(TapeCaptured): capture.sample_state_pairs(samples, {'contact_tolerance':.003})
        samples[0]['A'][0] = 1.
        self.assertEqual(capture.tape[0]['samples'][0]['A'][0], 0.)
        self.assertEqual(capture.tape[0]['expected'][0]['problem']['reason'], 'table')


if __name__ == "__main__": unittest.main()
