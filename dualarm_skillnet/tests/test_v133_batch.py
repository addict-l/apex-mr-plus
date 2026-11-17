"""Durable v133 batching: identity, resume and raw/qualified pairing."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import overnight_batch as old
from tools import v133_batch as batch


class RevisionBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=old.ROOT/'out')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_worker_phase_does_not_override_manifest_stage(self):
        answer = batch.worker_result({'id':'q1','phase':'sim','kind':'qualify','family':'F1'},
                                     {'phase':'complete','status':'completed'})
        self.assertNotIn('phase', answer)
        self.assertEqual(answer['execution_phase'], 'complete')
        self.assertEqual(answer['kind'], 'qualify')

    def test_controller_and_executor_changes_require_fresh_motion_gate(self):
        fingerprint = batch.motion_fingerprint()
        self.assertIn('controller_ext.lua', fingerprint)
        self.assertIn('executor.py', fingerprint)
        for name in ('controller_ext.lua', 'executor.py'):
            changed = {**fingerprint, name: 'changed'}
            with patch.object(batch, 'BASE', self.base), patch.object(batch, 'motion_fingerprint', return_value=changed):
                old.atomic_json(self.base/'last_motion_gate.json', {
                    'passed': True, 'motion_code_sha256': fingerprint})
                with self.assertRaisesRegex(RuntimeError, '变化'):
                    batch.require_motion_gate({'kind': 'sim'})

    def test_prepare_preserves_completed_definitions_and_input_hashes(self):
        task = self.base/'task.toml'
        task.write_text('name="mock"\n', encoding='utf-8')
        jobs=[{'id':'p1','phase':'plan','kind':'plan','task':str(task),
               'output':str(self.base/'taskout'),'timeout_seconds':60}]
        first=batch.prepare('mock',jobs,base=self.base)
        old_bytes=first.read_bytes()
        batch.prepare('mock',jobs,base=self.base)
        self.assertEqual(first.read_bytes(),old_bytes)
        task.write_text('name="changed"\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'SHA|输入|改变'):
            batch.prepare('mock',jobs,base=self.base)

    def test_wide_report_includes_raw_and_qualified_and_missing_as_pending(self):
        manifest={'jobs':[
            {'id':'r','phase':'plan','kind':'plan','family':'F1','index':0,'table':'raw'},
            {'id':'q','phase':'plan','kind':'plan','family':'F1','index':0,'table':'qualified'}]}
        answer=batch.paired_planning(manifest,{'r':{'status':'unknown','elapsed_seconds':.01,
            'unknown':['reach(B, board_2)']},'q':{'phase':'complete','status':'solved','bound':1.4,'makespan':30}})
        self.assertEqual(len(answer),1)
        self.assertEqual(answer[0]['raw_status'],'unknown')
        self.assertEqual(answer[0]['qualified_status'],'solved')
        self.assertEqual(answer[0]['qualified_bound'],1.4)
        self.assertEqual(answer[0]['distance_prior_status'],'unknown')
        self.assertEqual(answer[0]['ik_status'],'solved')
        answer=batch.paired_planning(manifest,{'r':{'status':'unknown'}})
        self.assertEqual(answer[0]['qualified_status'],'pending')

    def test_changed_manifest_definition_is_not_silently_reprepared(self):
        task=self.base/'task.toml';task.write_text('name="x"\n',encoding='utf-8')
        job={'id':'a','phase':'sim','kind':'sim','task':str(task),'output':str(self.base/'out')}
        batch.prepare('mock',[job],base=self.base)
        with self.assertRaisesRegex(ValueError,'定义|改变'):
            batch.prepare('mock',[{**job,'timeout_seconds':55}],base=self.base)

    def test_real_jobs_require_current_motion_gate_but_gate_can_run_first(self):
        with patch.object(batch,'BASE',self.base),patch.object(batch,'motion_fingerprint',return_value={'frames.py':'new'}):
            batch.require_motion_gate({'kind':'gate','test_name':old.LIVE_TESTS[0]})
            with self.assertRaisesRegex(RuntimeError,'先通过'):
                batch.require_motion_gate({'kind':'qualify'})
            old.atomic_json(self.base/'last_motion_gate.json',{'passed':True,'motion_code_sha256':{'frames.py':'old'}})
            with self.assertRaisesRegex(RuntimeError,'变化'):
                batch.require_motion_gate({'kind':'sim'})
            old.atomic_json(self.base/'last_motion_gate.json',{'passed':True,'motion_code_sha256':{'frames.py':'new'}})
            batch.require_motion_gate({'kind':'sim'})

    def test_missing_qualification_does_not_prevent_independent_distance_prior_jobs(self):
        from simbridge.taskio import write_task_data
        with patch.object(batch,'BASE',self.base):
            for family in batch.FAMILIES:
                for index in range(40):
                    name=f'gen_{family.lower()}_s20261115_{index:03d}'
                    write_task_data(self.base/'generated'/f'{name}.toml',{'name':name,'arms':['A','B'],
                        'location':[{'id':'board','kind':'buffer','pos':[0.,0.,0.]}],
                        'slot':[],'capabilities':{'liftable':{'default':'VALID'}}})
            path=batch.prepare_planning()
            self.assertTrue(path.is_relative_to(self.base))
            manifest=old.load_manifest(path)
            prior=[j for j in manifest['jobs'] if j['table']=='raw']
            missing=[j for j in manifest['jobs'] if j['table']=='qualified']
            self.assertEqual(len(prior),200)
            self.assertEqual(len(missing),200)
            self.assertTrue(all('skip_reason' not in j for j in prior))
            self.assertTrue(all('skip_reason' in j and not j.get('task') for j in missing))

    def test_motion_gate_revisions_get_new_immutable_attempt_ids(self):
        with patch.object(batch,'BASE',self.base),patch.object(batch,'motion_fingerprint',return_value={'frames.py':'first'}):
            first=batch.prepare_native('d_gate')
        with patch.object(batch,'BASE',self.base),patch.object(batch,'motion_fingerprint',return_value={'frames.py':'second'}):
            with self.assertRaises(ValueError):batch.prepare_native('d_gate')
            second=batch.prepare_native('d_gate',attempt=2)
        self.assertNotEqual(first,second)
        self.assertEqual(old.read_json(second)['jobs'][0]['motion_code_sha256'],{'frames.py':'second'})

    def test_connection_failures_accumulate_across_stage_attempts(self):
        for index in range(3):
            old.Journal(self.base/'states'/f'd_gate_attempt{index}').append({
                'id':str(index),'phase':'gates','kind':'gate','status':'failed','executed':True,
                'failure_phase':'simulator_connect','completed_at':f'2026-11-16T00:00:0{index}+00:00'})
        self.assertEqual(batch.connection_failures(self.base),3)
        old.Journal(self.base/'states'/'later').append({'id':'ok','phase':'sim','kind':'qualify',
            'status':'completed','executed':True,'simulator_pid':123,'completed_at':'2026-11-16T00:00:04+00:00'})
        self.assertEqual(batch.connection_failures(self.base),0)

    def test_hard_timeout_keeps_checkpointed_connection_failure_stage(self):
        output=self.base/'qualification'
        old.atomic_json(output/'task'/'qualify_log.json',{'phase':'simulator_connect','simulator_pid':22})
        result=batch.adapt_result({'kind':'qualify','output':str(output)},
            {'status':'timeout','failure_phase':'task_wall_timeout','reason':'hard limit'})
        self.assertEqual(result['failure_phase'],'simulator_connect')
        self.assertTrue(result['hard_timeout'])
        self.assertEqual(result['timeout_phase'],'task_wall_timeout')

    def test_progress_checkpoint_and_original_phase_survive_close_failure(self):
        output=self.base/'case'
        old.atomic_json(output/'simulation'/'task'/'run_progress.json',{'phase':'simulator_connect','pid':33})
        result=batch.adapt_result({'kind':'sim','output':str(output)},
            {'status':'timeout','failure_phase':'task_wall_timeout'})
        self.assertEqual(result['failure_phase'],'simulator_connect')
        old.Journal(self.base/'states'/'a').append({'id':'a','phase':'sim','status':'failed',
            'failure_phase':'simulator_close','startup_failure_phase':'simulator_connect',
            'simulator_pid':44,'completed_at':'2026-11-16T01:00:00+00:00'})
        self.assertEqual(batch.connection_failures(self.base),1)
