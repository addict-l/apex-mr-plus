"""New planning trials cannot inherit old successes or change their inputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import overnight_batch as old
from tools import v133_batch as batch


class ExperimentBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=old.ROOT/'out')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        inputs={}
        for table,source in (('raw','distance_prior'),('qualified','ik_qualified')):
            task=self.base/'inputs'/f'{table}.toml'
            task.parent.mkdir(parents=True,exist_ok=True)
            task.write_text(f'name="mock_{table}"\ncapability_source="{source}"\n',encoding='utf-8')
            inputs[table]=task
        jobs = []
        for family in batch.FAMILIES:
            for index in range(40):
                for table in ('raw', 'qualified'):
                    task = inputs[table]
                    jobs.append({'id':f'p-{table}-{family}-{index}', 'phase':'plan', 'kind':'plan',
                        'family':family, 'index':index, 'table':table, 'task':str(task),
                        'output':str(self.base/'old'/family/str(index)/table), 'timeout_seconds':60.})
        with patch.object(batch, 'planning_fingerprint', return_value={'core':'before'}):
            self.original = batch.prepare('planning200', jobs, base=self.base)

    def retry(self):
        with patch.object(batch, 'BASE', self.base), patch.object(batch, 'planning_fingerprint', return_value={'core':'after'}):
            return batch.prepare_planning(stage='planning200_retry1', source_manifest=self.original)

    def test_retry_preserves_all_four_hundred_input_cells_but_uses_new_state_and_outputs(self):
        before = self.original.read_bytes()
        target = self.retry()
        original, retry = old.load_manifest(self.original), old.load_manifest(target)
        key = lambda job: (job['family'], job['index'], job['table'], job['task'], job['task_sha256'], job['timeout_seconds'])
        self.assertEqual(sorted(map(key, original['jobs'])), sorted(map(key, retry['jobs'])))
        self.assertNotEqual(original['state_dir'], retry['state_dir'])
        self.assertTrue(set(j['output'] for j in original['jobs']).isdisjoint(j['output'] for j in retry['jobs']))
        self.assertTrue(all(j['planning_code_sha256']=={'core':'after'} for j in retry['jobs']))
        self.assertEqual(before, self.original.read_bytes())

    def test_new_pending_trial_cannot_select_old_qualified_successes_for_simulation(self):
        self.retry()
        manifest = old.load_manifest(self.original)
        journal = old.Journal(manifest['state_dir'])
        for job in manifest['jobs']:
            if job['table'] == 'qualified' and job['index'] < 4:
                old.atomic_json(Path(job['output'])/'plan.json', {'status':'solved', 'proposal_enabled':False, 'witness_loaded':False})
                journal.append({'id':job['id'], 'status':'solved', 'passed':True, 'definition_sha256':job['definition_sha256']})
        with patch.object(batch, 'BASE', self.base):
            path = batch.prepare_sim20(planning_stage='planning200_retry1')
        jobs = old.load_manifest(path)['jobs']
        self.assertEqual(len(jobs), 20)
        self.assertTrue(all(j.get('skip_reason') and not j.get('task') for j in jobs))

    def test_rebuilding_diagnostic_trial_cannot_replace_selected_official_csv(self):
        self.retry()
        with patch.object(batch, 'BASE', self.base), patch.object(batch, 'selected_stage', return_value='planning200_retry1'):
            batch.rebuild('planning200_retry1', base=self.base)
            current = (self.base/'planning_200.csv').read_bytes()
            first = old.load_manifest(self.original)['jobs'][0]
            old.Journal(self.base/'states/planning200').append({'id':first['id'], 'phase':'plan', 'status':'solved', 'passed':True})
            batch.rebuild('planning200', base=self.base)
        self.assertEqual(current, (self.base/'planning_200.csv').read_bytes())
        self.assertTrue((self.base/'stage_reports/planning200/paired_planning_200.csv').is_file())


if __name__ == '__main__':
    unittest.main()
