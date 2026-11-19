"""Stopped-scene context lifetime ends before any physical execution."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simbridge import executor
from tests.test_v13_executor import execution_fixture, synthetic_execution

ROOT = Path(__file__).resolve().parents[1]


class GeometryScopeTests(unittest.TestCase):
    def attempt(self, mode='normal', fail_stage=None, invalid_rollout=False):
        engine, geometry, _, steps, tasks, _, cfg=execution_fixture()
        state={'active':False,'closed':0};receipt={'restored':False}
        @contextmanager
        def factory():
            if mode=='enter_failure':raise RuntimeError('scope enter failed')
            state['active']=True
            try:yield receipt
            finally:
                state['active']=False;state['closed']+=1
                if mode=='close_failure':raise RuntimeError('scope restoration failed')
                if mode!='unconfirmed':receipt['restored']=True
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor,'install_controller') as install,
                  patch.object(executor,'execute') as execute):
                if invalid_rollout:
                    extra=patch.object(executor,'verify_rollout',return_value={'valid':False,'first_violation':{'reason':'sample collision'}})
                elif fail_stage:
                    target={'tpg_geometry':'build_tpg','parameterize':'parameterize','rollout':'verify_rollout','deadline':'verify_rollout'}[fail_stage]
                    error=TimeoutError('shared task deadline') if fail_stage=='deadline' else RuntimeError('stage failed')
                    extra=patch.object(executor,target,side_effect=error)
                else:
                    extra=patch.object(executor,'verify_rollout',return_value={'valid':True})
                with extra:
                    result=executor.execute_plan(None,None,cfg,tasks,engine.problem,folder,
                        planner=geometry,plan_steps=steps,stopped_geometry_context=factory)
                saved=json.loads((Path(folder)/'summary.json').read_text(encoding='utf-8'))
                install.assert_not_called();execute.assert_not_called()
        self.assertEqual(saved['status'],result['status'])
        self.assertEqual(saved['failure_phase'],result['failure_phase'])
        return result,state

    def test_success_scope_covers_queries_and_closes_before_install_and_execution(self):
        engine, geometry, _, steps, tasks, _, cfg = execution_fixture()
        state = {'active': False}
        receipt = {'restored': False}
        original = geometry.sample_state_pairs
        def sample(*args, **kwargs):
            self.assertTrue(state['active'])
            return original(*args, **kwargs)
        geometry.sample_state_pairs = sample
        @contextmanager
        def factory():
            state['active'] = True
            try: yield receipt
            finally: state['active'] = False; receipt['restored'] = True
        def install(sim, current, graph, filename):
            self.assertFalse(state['active']);self.assertTrue(receipt['restored'])
            current['program'] = graph
        def measured(sim, client, current, output, **kwargs):
            self.assertFalse(state['active'])
            return synthetic_execution(current['program'], current)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor,'install_controller',side_effect=install),
                  patch.object(executor,'execute',side_effect=measured)):
                result=executor.execute_plan(None,None,cfg,tasks,engine.problem,folder,
                    planner=geometry,plan_steps=steps,stopped_geometry_context=factory)
            saved=json.loads((Path(folder)/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(result['passed'],result.get('reason'))
        self.assertTrue(saved['stopped_geometry_context']['receipt']['restored'])
        self.assertIn('geometry_context_enter',result['stage_seconds'])
        self.assertIn('geometry_context_restore',result['stage_seconds'])

    def test_rollout_early_return_restores_scope_and_preserves_collision_phase(self):
        result,state=self.attempt(invalid_rollout=True)
        self.assertEqual(state['closed'],1)
        self.assertEqual(result['failure_phase'],'rollout')
        self.assertTrue(result['stopped_geometry_context']['receipt']['restored'])

    def test_geometry_parameterization_rollout_and_deadline_exceptions_close_scope(self):
        for stage in ['tpg_geometry','parameterize','rollout','deadline']:
            with self.subTest(stage=stage):
                result,state=self.attempt(fail_stage=stage)
                self.assertEqual(state['closed'],1)
                self.assertEqual(result['failure_phase'],'rollout' if stage=='deadline' else stage)
                self.assertEqual(result['status'],'timeout' if stage=='deadline' else 'failed')
                self.assertTrue(result['stopped_geometry_context']['receipt']['restored'])

    def test_scope_entry_failure_never_starts_geometry_or_execution(self):
        result,state=self.attempt(mode='enter_failure')
        self.assertEqual(state['closed'],0)
        self.assertEqual(result['failure_phase'],'geometry_context_enter')

    def test_restore_failure_or_missing_confirmation_blocks_install_and_execution(self):
        for mode in ['close_failure','unconfirmed']:
            with self.subTest(mode=mode):
                result,state=self.attempt(mode=mode)
                self.assertEqual(state['closed'],1)
                self.assertTrue(result['geometry_context_failure'])
                self.assertEqual(result['failure_phase'],'geometry_context_restore')
                self.assertFalse(result['remote_interface_failure'])

    def test_scope_setup_and_restore_time_consume_existing_execution_budget(self):
        engine,geometry,_,steps,tasks,_,cfg=execution_fixture()
        clock={'now':100.}
        @contextmanager
        def factory():
            receipt={'restored':False};clock['now']+=2.
            try:yield receipt
            finally:clock['now']+=3.;receipt['restored']=True
        def install(sim,current,graph,filename):current['program']=graph
        def measured(sim,client,current,output,**kwargs):
            self.assertEqual(kwargs['timeout_wall_s'],5.)
            return synthetic_execution(current['program'],current)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor.time,'monotonic',side_effect=lambda:clock['now']),
                  patch.object(executor,'install_controller',side_effect=install),
                  patch.object(executor,'execute',side_effect=measured)):
                result=executor.execute_plan(None,None,cfg,tasks,engine.problem,folder,
                    planner=geometry,plan_steps=steps,timeout_wall_s=10.,stopped_geometry_context=factory)
        self.assertTrue(result['passed'],result.get('reason'))
        self.assertEqual(result['stopped_geometry_context']['enter_seconds'],2.)
        self.assertEqual(result['stopped_geometry_context']['restore_seconds'],3.)
        self.assertEqual(result['executor_wall_seconds'],5.)

    def test_runner_binds_original_task_deadline_in_lazy_zero_argument_factory(self):
        from tests.test_motion_v135_runner_reference import RunnerReferenceTests
        from simbridge.backend import CoppeliaBackend
        from simbridge import runner
        helper=RunnerReferenceTests()
        with patch.object(runner,'run',wraps=runner.run) as planning:
            def check(result,output,execute,saved):
                self.assertTrue(result['passed'],result.get('reason'))
                factory=execute.call_args.kwargs['stopped_geometry_context']
                original_deadline=planning.call_args_list[0].kwargs['deadline']
                self.assertEqual(factory.keywords,{'deadline':original_deadline})
                factory.func.assert_not_called()
                factory()
                factory.func.assert_called_once_with(deadline=original_deadline)
            helper.invoke(CoppeliaBackend,check)

    def test_partial_entry_restoration_failure_keeps_receipt_and_never_builds_or_executes(self):
        from apexmr.sim_connection import FastIdleRestorationError
        engine,geometry,_,steps,tasks,_,cfg=execution_fixture()
        receipt={'restored':False,'restoration_error':'app state not restored'}
        @contextmanager
        def factory():
            raise FastIdleRestorationError('partial entry restore failed',receipt)
            yield
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor,'build_tpg') as build,
                  patch.object(executor,'install_controller') as install,
                  patch.object(executor,'execute') as execute):
                result=executor.execute_plan(None,None,cfg,tasks,engine.problem,folder,
                    planner=geometry,plan_steps=steps,stopped_geometry_context=factory)
                build.assert_not_called();install.assert_not_called();execute.assert_not_called()
        self.assertEqual(result['failure_phase'],'geometry_context_restore')
        self.assertEqual(result['stopped_geometry_context']['receipt'],receipt)
        self.assertFalse(result['passed'])

    def test_none_factory_retains_existing_entry_without_app_context_stages(self):
        engine,geometry,_,steps,tasks,_,cfg=execution_fixture()
        def install(sim,current,graph,filename):current['program']=graph
        def measured(sim,client,current,output,**kwargs):return synthetic_execution(current['program'],current)
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            with (patch.object(executor,'install_controller',side_effect=install),
                  patch.object(executor,'execute',side_effect=measured)):
                result=executor.execute_plan(None,None,cfg,tasks,engine.problem,folder,
                    planner=geometry,plan_steps=steps,stopped_geometry_context=None)
        self.assertTrue(result['passed'],result.get('reason'))
        self.assertFalse(result['stopped_geometry_context']['enabled'])
        self.assertNotIn('geometry_context_enter',result['stage_seconds'])
        self.assertNotIn('geometry_context_restore',result['stage_seconds'])


if __name__=='__main__':unittest.main()
