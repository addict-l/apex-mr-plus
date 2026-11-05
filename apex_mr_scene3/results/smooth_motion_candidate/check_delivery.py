"""Cold-load the audited initial scene without starting its simulation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from sim_connection import Simulator
from apex.pipeline import source_fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((ROOT/'results/apex/pipeline_manifest.json').read_text(encoding='utf-8'))
delivery = json.loads((ROOT/'results/apex/delivery.json').read_text(encoding='utf-8'))
main = ROOT/'apex_mr_scene3.ttt'
planned = ROOT/'scenes/11_Scene3_APEX_Planned.ttt'
checks = {'main_matches_planned': digest(main) == digest(planned) == delivery['main_scene_sha256'],
          'frozen_inputs_match': source_fingerprint()['sha256'] == manifest['input_fingerprint']['sha256']}
with Simulator(ROOT, port=23067, timeout=45) as owner:
    client, sim = owner.connect()
    sim.loadScene(str(main))
    root = sim.getObject('/APEX_MR_Scene3')
    cfg = sim.readCustomTableData(root, 'Scene3APEX')
    graph = cfg['program']
    checks['simulation_time_zero'] = sim.getSimulationTime() == 0
    checks['scheduler_present'] = sim.getObject('/APEX_MR_Scene3/CentralTPGScheduler', {'noError': True}) != -1
    checks['complete_program'] = len(graph['nodes']) == 390 and len(graph['edges']) == 753
    checks['shared_profiles_present'] = graph['time_parameterization']['pass_through_boundaries'] == 285
    checks['six_validated_C2_macros'] = graph['time_parameterization']['C2_blended_macros'] == 6
    checks['unchanged_contact_limits'] = cfg['runtime']['assembly_tool_force_limit_N'] == 40 and cfg['runtime']['assembly_tool_torque_limit_Nm'] == 8
    checks['actual_load_proof_required'] = cfg['validation_requirements']['require_actual_applied_load'] is True
    errors = {}
    for part, source in cfg['initial'].items():
        handle = sim.getObject('/APEX_MR_Scene3/'+part)
        actual = sim.getObjectPose(handle, sim.handle_world)
        errors[part] = sum((actual[i]-source[i])**2 for i in range(3))**.5
    checks['initial_object_positions'] = max(errors.values(), default=0) < 1e-6
report = {'run_id': delivery['run_id'], 'success': all(checks.values()), 'checks': checks,
          'main_scene_sha256': digest(main), 'source_scene_sha256': delivery['source_scene_sha256'],
          'initial_position_errors_m': errors, 'scope': 'Cold initial-state load; no additional execution claimed'}
(ROOT/'results/smooth_motion_candidate/delivery_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
assert report['success']
