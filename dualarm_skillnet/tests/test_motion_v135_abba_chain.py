"""Serial supervisor behavior with native evidence and external-process mocks."""
import tempfile
from pathlib import Path
from copy import deepcopy
import sys
import unittest


class AbbaChainTests(unittest.TestCase):
    def native_fixture(self, base):
        from tests.test_motion_v135_preflight import NativeEvidenceTests
        from tools import overnight_batch as old
        output, evidence, job, version = NativeEvidenceTests().fixture(base/'seed')
        command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve().parents[1]/'tools/motion_v135_single.py'),
                 '--gate',old.LIVE_TESTS[0],'--out',str(output),'--seconds','1200']
        job.update(command=command,seconds=1200.)
        return output,evidence,job,version

    def test_normal_gate_failure_stops_before_any_abba_job(self):
        from tools import motion_v135_abba_chain as chain
        from unittest.mock import patch
        from tools import overnight_batch as old
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'out') as tmp:
            base = Path(tmp); calls=[]
            version={'core': {'fixture': 'fixed'}, 'tools': {}, 'inputs': {}}
            code=base/'code.json';old.atomic_json(code,version['core'])
            def launch(name, command, seconds, live=False):
                calls.append(command)
                return {'status':'failed','code_unchanged_during_job':True,
                        'code_version':str(code),'cleanup':{'passed':True}, 'command':command}
            with patch.object(chain,'BASE',base),patch.object(chain.control,'record'):
                result=chain.run('mock',snapshotter=lambda:version,environment_check=lambda root:None,job_runner=launch)
            self.assertEqual(result['status'],'gate_failed')
            self.assertEqual(len(calls),1)

    def test_completed_gate_is_strictly_adopted_profiles_continue_and_resume_skips_both(self):
        from tools import motion_v135_abba_chain as chain
        from tools import overnight_batch as old
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'out') as tmp:
            base=Path(tmp);output,evidence,job,version=self.native_fixture(base)
            old.atomic_json(base/'jobs/external_gate/result.json',job)
            calls=[]
            def launch(name,command,seconds,live=False):
                calls.append((name,command,seconds,live))
                return {**job,'status':'failed','command':command}
            with patch.object(chain,'BASE',base),patch.object(chain.control,'record'):
                result=chain.run('mock','external_gate',output,snapshotter=lambda:version,
                                 environment_check=lambda root:None,job_runner=launch)
                self.assertEqual(result['status'],'finished_with_failures')
                self.assertTrue(result['steps']['gate_parallel']['passed'])
                self.assertEqual(len(calls),2)
                self.assertTrue(all(row[2:]==(1200.,True) for row in calls))
                chain.run('mock','external_gate',output,snapshotter=lambda:version,
                          environment_check=lambda root:None,job_runner=launch)
                self.assertEqual(len(calls),2)
                changed=deepcopy(version);changed['core']['simbridge/compiler.py']='changed'
                with self.assertRaisesRegex(ValueError,'version'):
                    chain.run('mock','external_gate',output,snapshotter=lambda:changed,
                              environment_check=lambda root:None,job_runner=launch)

    def test_live_existing_gate_waits_then_adopts_without_launching_duplicate(self):
        from tools import motion_v135_abba_chain as chain
        from tools import overnight_batch as old
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'out') as tmp:
            base=Path(tmp);output,evidence,job,version=self.native_fixture(base)
            old.atomic_json(base/'jobs/external_gate/state.json',{**job,'pid':999999,'birth':'identity','status':'running'})
            calls=[]
            def wake(_):old.atomic_json(base/'jobs/external_gate/result.json',job)
            def launch(name,command,seconds,live=False):calls.append(name);return {**job,'status':'failed','command':command}
            with (patch.object(chain,'BASE',base),patch.object(chain.control,'record'),
                  patch.object(chain.old,'process_birth',return_value='identity'),patch.object(chain.time,'sleep',side_effect=wake)):
                result=chain.run('mock','external_gate',output,snapshotter=lambda:version,
                                 environment_check=lambda root:None,job_runner=launch)
            self.assertEqual(result['status'],'finished_with_failures')
            self.assertEqual(calls,['abba_mock_00','abba_mock_02'])

    def test_missing_native_execution_blocks_profiles_and_partial_profile_uses_new_attempt(self):
        from tools import motion_v135_abba_chain as chain
        from tools import overnight_batch as old
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'out') as tmp:
            base=Path(tmp);output,evidence,job,version=self.native_fixture(base)
            old.atomic_json(base/'jobs/external_gate/result.json',job)
            missing=(evidence/'execution.json');original=missing.read_bytes();missing.unlink()
            calls=[]
            def launch(name,command,seconds,live=False):calls.append(command);return {**job,'status':'failed','command':command}
            with patch.object(chain,'BASE',base),patch.object(chain.control,'record'):
                result=chain.run('blocked','external_gate',output,snapshotter=lambda:version,
                                 environment_check=lambda root:None,job_runner=launch)
                self.assertEqual(result['status'],'gate_failed');self.assertEqual(calls,[])
                missing.write_bytes(original)
                partial=base/'m5/resume/F1_014/progress.json';old.atomic_json(partial,{'interrupted':True})
                result=chain.run('resume','external_gate',output,snapshotter=lambda:version,
                                 environment_check=lambda root:None,job_runner=launch)
                self.assertEqual(result['status'],'finished_with_failures')
                self.assertEqual(Path(calls[0][calls[0].index('--out')+1]).name,'abba_00_resume1')
                self.assertEqual(old.read_json(partial),{'interrupted':True})


if __name__=='__main__':unittest.main()
