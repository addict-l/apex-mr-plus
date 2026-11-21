"""Late stages resume in order without replaying completed experiments."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import overnight_batch as old
from tools import v133_workflow as flow


class LateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=old.ROOT/'out')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)

    def test_all_stages_keep_required_order_and_no_implicit_retries(self):
        calls=[]
        def prepare(stage):calls.append(('prepare',stage))
        def run(stage):calls.append(('run',stage));return {'status':'phase_complete'}
        answer=flow.run_pipeline(base=self.base,preparer=prepare,runner=run,
            checker=lambda:{'passed':True},reporter=lambda:None,ready=lambda:None)
        self.assertEqual(answer['status'],'finished')
        self.assertEqual([s for k,s in calls if k=='run'],list(flow.STAGES))
        self.assertEqual(len(answer['stages']),4)

    def test_global_stop_prevents_later_independent_stages(self):
        calls=[]
        def run(stage):
            calls.append(stage)
            return {'status':'global_stop','global_stop':'C盘不足5GiB'}
        answer=flow.run_pipeline(base=self.base,preparer=lambda stage:None,runner=run,
            checker=lambda:{'passed':True},reporter=lambda:None,ready=lambda:None)
        self.assertEqual(calls,['qualify200'])
        self.assertEqual(answer['status'],'global_stop')

    def test_ordinary_partial_stage_does_not_claim_success_or_stop_independent_work(self):
        calls=[]
        def run(stage):calls.append(stage);return {'status':'phase_budget_exhausted' if stage=='qualify200' else 'phase_complete'}
        answer=flow.run_pipeline(base=self.base,preparer=lambda stage:None,runner=run,
            checker=lambda:{'passed':True},reporter=lambda:None,ready=lambda:None)
        self.assertEqual(calls,list(flow.STAGES))
        self.assertEqual(answer['stages']['qualify200']['status'],'phase_budget_exhausted')
        self.assertEqual(answer['status'],'finished')

    def test_physical_or_stage_global_stop_is_persistent_across_supervisors(self):
        old.atomic_json(self.base/'physical_stage_outcome.json',{'finished':True,'global_stop':'three startup failures'})
        self.assertIn('three',flow.global_stop_reason(self.base))
        old.atomic_json(self.base/'physical_stage_outcome.json',{'finished':True})
        old.atomic_json(self.base/'states/d_gate/state.json',{'global_stop':'archive mismatch'})
        self.assertIn('archive mismatch',flow.global_stop_reason(self.base))

    def test_selected_trial_is_used_for_execution_and_has_its_own_cursor(self):
        selection={'selection_present':True,'selection_id':'retry1','selection_sha256':'frozen',
                   'stages':{'planning200':'planning200_retry1'}}
        calls=[]
        with patch.object(flow.experiments,'load_selection',return_value=selection):
            answer=flow.run_pipeline(base=self.base,preparer=lambda stage:None,
                runner=lambda stage:calls.append(stage) or {'status':'phase_complete'},
                checker=lambda:{'passed':True},reporter=lambda:None,ready=lambda:None)
        self.assertEqual(answer['status'],'finished')
        self.assertEqual(calls,['qualify200','planning200_retry1','sim20','gates'])
        self.assertTrue((self.base/'experiments/retry1/workflow_state.json').is_file())
        self.assertFalse((self.base/'workflow_state.json').exists())

    def test_selection_change_during_execution_stops_instead_of_mixing_trials(self):
        first={'selection_present':True,'selection_id':'retry1','selection_sha256':'one','stages':{}}
        changed={**first,'selection_sha256':'two'}
        calls=[]
        with patch.object(flow.experiments,'load_selection',side_effect=[first,changed]):
            result=flow.run_pipeline(base=self.base,preparer=lambda stage:None,
                runner=lambda stage:calls.append(stage) or {'status':'phase_complete'},
                checker=lambda:{'passed':True},reporter=lambda:None,ready=lambda:None)
        self.assertEqual(result['status'],'global_stop')
        self.assertEqual(calls,[])

    def test_new_core_cannot_reuse_old_passed_proofs_without_current_fingerprint(self):
        selection={'selection_present':True,'planning_code_sha256':{'core':'new'},
            'regressions':{'full':'full.json','original11':'originals.json'}}
        old.atomic_json(self.base/'full.json',{'status':'passed','planning_code_sha256':{'core':'old'}})
        old.atomic_json(self.base/'originals.json',{'passed':True,'rows':[{'passed':True}]*11,
                                                  'planning_code_sha256':{'core':'new'}})
        with patch.object(flow.batch,'planning_fingerprint',return_value={'core':'new'}):
            with self.assertRaisesRegex(RuntimeError,'指纹|fingerprint'):
                flow.verify_selection_regressions(selection,project=self.base)
