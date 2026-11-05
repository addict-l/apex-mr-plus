"""Cold-load the current audited delivery and check its exact manifest graph."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from sim_connection import Simulator
from apex.pipeline import source_fingerprint


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lua_table_value(value):
    # Lua tables omit nil-valued fields and have one empty-table type.
    # Keep every nonempty value and every numeric digit for this comparison.
    if isinstance(value, dict):
        result = {k: lua_table_value(v) for k, v in value.items() if v is not None}
        return result if result else []
    if isinstance(value, list):
        return [lua_table_value(v) for v in value]
    return value


def main():
    out = ROOT/'results/apex'
    delivery = read(out/'delivery.json')
    manifest = read(out/'pipeline_manifest.json')
    graph = read(out/'tpg.json')
    audit = read(out/'execution_validation.json')
    checks = {
        'same_successful_run': delivery['run_id'] == manifest['run_id'] == audit['run_id'] and
            manifest['status'] == 'complete' and audit['success'] is True,
        'delivery_matches_audited_planned_scene': sha(ROOT/'apex_mr_scene3.ttt') ==
            sha(ROOT/'scenes/11_Scene3_APEX_Planned.ttt') == delivery['main_scene_sha256'] ==
            manifest['planned_scene_sha256'],
        'frozen_inputs_unchanged': source_fingerprint() == manifest['input_fingerprint'],
    }
    with Simulator(ROOT, port=23067, timeout=45) as owner:
        client, sim = owner.connect()
        sim.loadScene(str(ROOT/'apex_mr_scene3.ttt'))
        root = sim.getObject('/APEX_MR_Scene3')
        cfg = sim.readCustomTableData(root, 'Scene3APEX')
        embedded = cfg['program']
        checks['simulation_time_zero'] = sim.getSimulationTime() == 0
        checks['scheduler_present'] = sim.getObject('/APEX_MR_Scene3/CentralTPGScheduler', {'noError': True}) != -1
        checks['embedded_nodes_match_audited_graph'] = lua_table_value(embedded['nodes']) == lua_table_value(graph['nodes'])
        checks['embedded_edges_match_audited_graph'] = embedded['edges'] == graph['edges']
        checks['continuous_profiles_match'] = embedded['time_parameterization'] == graph['time_parameterization']
        checks['force_limits_unchanged'] = cfg['runtime']['assembly_tool_force_limit_N'] == 40 and cfg['runtime']['assembly_tool_torque_limit_Nm'] == 8
        checks['actual_applied_proof_required'] = cfg['validation_requirements']['require_actual_applied_load'] is True
        errors = {}
        for part, initial in cfg['initial'].items():
            actual = sim.getObjectPose(sim.getObject('/APEX_MR_Scene3/'+part), sim.handle_world)
            errors[part] = sum((actual[i]-initial[i])**2 for i in range(3))**.5
        checks['initial_objects_in_place'] = max(errors.values(), default=0) < 1e-6
    report = {'run_id': delivery['run_id'], 'success': all(checks.values()), 'checks': checks,
              'node_count': len(graph['nodes']), 'edge_count': len(graph['edges']),
              'main_scene_sha256': delivery['main_scene_sha256'], 'initial_position_errors_m': errors,
              'scope': 'Read-only cold initial-state load. No additional dynamic execution.'}
    target = Path(__file__).with_name('delivery_validation.json')
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert report['success'], 'Cold delivery checks failed'


if __name__ == '__main__':
    main()
