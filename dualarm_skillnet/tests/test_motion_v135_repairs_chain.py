"""Offline serial repair orchestration; external phase boundaries are fake."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]


class RepairsChainTests(unittest.TestCase):
    def execute(self,base,calls,*,gate_pass=True,identity=None,diag_status='finished'):
        from tools import motion_v135_repairs_chain as chain
        from tools import overnight_batch as old
        prepared=base/'prepared.json'
        if not prepared.exists():old.atomic_json(prepared,[{'command':['fixture',str(i)]} for i in range(8)])
        version=identity or {'core':{'fixed':'one'},'tools':{},'inputs':{}}
        def gate(*args,**kwargs):calls.append('gate');return {'passed':gate_pass,'reason':None if gate_pass else 'native failed'}
        def probe(row,index,*args,**kwargs):calls.append('probe'+str(index));return {'status':'completed','diagnostic_status':'blocked' if index==0 else 'valid','passed':True,'formal_assembly_pass':False}
        def prepare(queue_id,gate_job,selected_ids=None):
            calls.append(('prepare',queue_id,tuple(selected_ids)));return {'status':'prepared'}
        def run_queue(queue_id,**kwargs):calls.append(('queue',queue_id));return {'status':diag_status,'counts':{'passed':3,'failed':4}}
        def profiles(run_id,gate_job,**kwargs):calls.append(('m5',run_id,kwargs));return {'status':'finished'}
        with patch.object(chain,'BASE',base),patch.object(chain.control,'record'):
            return chain.run('mock',prepared=prepared,queue_id='repaired7_c3',profiles_id='full_profiles_c3',
                snapshotter=lambda:version,environment_check=lambda root:None,gate_checker=gate,
                probe_performer=probe,diagnostics_module=SimpleNamespace(prepare=prepare,run=run_queue),
                m5_module=SimpleNamespace(run=profiles),heartbeat_seconds=.05)

    def test_gate_failure_stops_probes_queue_and_profiles(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            calls=[];result=self.execute(Path(tmp),calls,gate_pass=False)
        self.assertEqual(calls,['gate']);self.assertEqual(result['status'],'gate_failed')
        self.assertTrue(result['requires_rollback'])

    def test_fixed_order_and_completed_resume_never_launches_duplicate_work(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            calls=[];base=Path(tmp);result=self.execute(base,calls)
            self.assertEqual(result['status'],'finished')
            self.assertEqual(calls[:9],['gate']+['probe'+str(i) for i in range(8)])
            selected=('v134-fixed20-00','v134-fixed20-12','v134-fixed20-08','v134-fixed20-16','v134-fixed20-02','v134-fixed20-06','v134-fixed20-07')
            self.assertEqual(calls[9],('prepare','repaired7_c3',selected))
            self.assertEqual(calls[10],('queue','repaired7_c3'))
            self.assertEqual(calls[11],('m5','full_profiles_c3',{'complete_optimized':True}))
            before=list(calls);self.execute(base,calls)
            self.assertEqual(calls,before+['gate'])

    def test_changed_identity_cannot_reuse_existing_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            base=Path(tmp);calls=[];self.execute(base,calls)
            with self.assertRaisesRegex(ValueError,'version|fingerprint'):
                self.execute(base,calls,identity={'core':{'fixed':'changed'},'tools':{},'inputs':{}})

    def test_interrupted_queue_resumes_without_repeating_completed_probes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            calls=[];base=Path(tmp);first=self.execute(base,calls,diag_status='interrupted')
            self.assertEqual(first['status'],'interrupted')
            second=self.execute(base,calls)
            self.assertEqual(second['status'],'finished')
            self.assertEqual(sum(isinstance(c,str) and c.startswith('probe') for c in calls),8)


if __name__=='__main__':unittest.main()
