"""Pure audit-gate/math tests; no new-run acceptance or simulator execution."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_and_compare import require_new_completed, command_bounds, curve_geometry, OLD_RUN
from apex.curve import make_profile


class NewRunGateTests(unittest.TestCase):
    def test_baseline_run_is_never_accepted_as_new(self):
        with self.assertRaisesRegex(ValueError,'new run'):
            require_new_completed({'run_id':OLD_RUN},{'run_id':OLD_RUN},OLD_RUN)

    def test_unfinished_or_mismatched_run_rejected(self):
        manifest={'run_id':'new','status':'complete','main_scene_promoted':True}
        execution={'run_id':'new','success':True,'done':True}
        require_new_completed(manifest,execution,'new')
        for changed in [dict(manifest,status='running'),dict(manifest,main_scene_promoted=False),dict(manifest,run_id='stale')]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                require_new_completed(changed,execution,'new')
        with self.assertRaises(ValueError):require_new_completed(manifest,dict(execution,success=False),'new')


class CommandBoundTests(unittest.TestCase):
    def graph(self,duration=5.):
        return {'time_parameterization':{'max_velocity_rad_s':.6,'max_acceleration_rad_s2':1.2,'max_jerk_rad_s3':8.},
            'nodes':[{'id':'motion','metadata':{},'motion_segments':[{'q0':[0]*6,'q1':[.1]*6,'duration':duration}]}]}

    def test_quintic_bounds_recomputed_from_actual_profile_duration(self):
        graph=self.graph();out=command_bounds(graph)
        self.assertTrue(out['passed']);self.assertAlmostEqual(out['recomputed_command_maxima']['velocity'],.0375)
        self.assertFalse(command_bounds(self.graph(.1))['passed'])

    def test_continuation_fragment_uses_shared_full_duration(self):
        graph=self.graph(.01);graph['nodes'][0]['motion_segments'][0]['profile_duration']=5.
        self.assertTrue(command_bounds(graph)['passed'])

    def test_curved_bounds_recomputed_not_merely_trusted(self):
        profile=make_profile([[0]*6,[.1]*6,[.2]*6],.6,1.2,8.)
        profile['collision_validated']=True
        graph=self.graph();graph['nodes']=[{'id':'one','metadata':{},'curve_run_id':'curve','curve_profile':profile}]
        self.assertTrue(command_bounds(graph)['passed'])
        self.assertEqual(command_bounds(graph)['curve_runs_checked'],1)
        changed=deepcopy(graph);changed['nodes'][0]['curve_profile']['duration']/=100
        self.assertFalse(command_bounds(changed)['passed'])

    def test_zero_or_variable_curve_counts_not_fixed_to_old_six(self):
        self.assertTrue(curve_geometry({'nodes':[]},{'candidate_macros':[],'accepted_macros':0})['passed'])
        graph={'nodes':[{'id':'one','task_id':'move','curve_run_id':'curve','curve_profile':{'x':1}}]}
        validation={k:True for k in ['accepted','valid','geometry_valid','tcp_corridor_valid','validation_performed','context_unchanged']}
        validation.update(geometry_samples_checked=12,geometry_samples_submitted=12,max_tcp_deviation_m=.0001,
            max_orientation_error_rad=.001,thresholds={'max_tcp_deviation_m':.0005,'max_orientation_error_rad':.005})
        evidence={'candidate_macros':[{'macro':'move','task_ids':['move'],'accepted':True,'attempts':[{'validation':validation}]}],'accepted_macros':1}
        self.assertTrue(curve_geometry(graph,evidence)['passed'])
        evidence['candidate_macros'][0]['attempts'][0]['validation']['geometry_samples_checked']=11
        self.assertFalse(curve_geometry(graph,evidence)['passed'])


if __name__=='__main__':unittest.main()
