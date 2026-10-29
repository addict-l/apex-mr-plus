"""Independent audit of actual TPG events, grasp/snap events and goal poses."""
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'normal'

def main():
    graph=json.loads((OUT/'tpg.json').read_text(encoding='utf-8'))
    execution=json.loads((OUT/'execution.json').read_text(encoding='utf-8'))
    times={};violations=[]
    for event in execution['events']:
        if event['kind'] in ['node_started','node_completed']:
            key='start' if event['kind']=='node_started' else 'end'
            row=times.setdefault(event['node'],{})
            if key in row:violations.append({'node':event['node'],'error':'duplicate '+key})
            row[key]=event['time']
    for node in graph['nodes']:
        row=times.get(node['id'],{})
        if 'start' not in row or 'end' not in row or row.get('end',0)<row.get('start',1):
            violations.append({'node':node['id'],'error':'missing or reversed actual interval'})
    for edge in graph['edges']:
        before=times.get(edge['source'],{}).get('end');after=times.get(edge['target'],{}).get('start')
        if before is None or after is None or before>after+1e-8:
            violations.append({'edge':edge,'error':'actual predecessor not completed before successor started'})
    for arm in ['A','B']:
        rows=sorted([(times[n['id']]['start'],times[n['id']]['end'],n['id']) for n in graph['nodes']
                     if n['arm']==arm and 'end' in times.get(n['id'],{})])
        for first,second in zip(rows,rows[1:]):
            if first[1]>second[0]+1e-8:violations.append({'arm':arm,'error':'overlapping node service'})
    quaternions={name:2*math.acos(min(1.,abs(pose[6]))) for name,pose in execution['part_poses'].items()}
    execution['goal_orientation_errors_rad']=quaternions
    execution['actual_node_timestamps']=times
    corrections=[e['details'] for e in execution['events'] if e['kind']=='support_height_compensation']
    ready_events=[e for e in execution['events'] if e['kind']=='support_ready']
    loaded=[r for r in execution['trace'] if r.get('applied_insertion_load_N',0)>0]
    beam_torques=[r['snap']['beam']['torque_norm_Nm'] for r in loaded
                  if r.get('snap',{}).get('beam',{}).get('valid')]
    actual_forces=[r.get('support',{}).get('A',0) for r in loaded]
    connection_peaks={}
    for row in execution['trace']:
        for part,sensor in row.get('snap',{}).items():
            if sensor.get('valid'):
                connection_peaks[part]=max(connection_peaks.get(part,0),sensor['torque_norm_Nm'])
    force_barrier=bool(ready_events and all(e['details'].get('force_N',0)>.15 for e in ready_events))
    dynamic=all(all(item['joints']) and item['tool_sensor'] and item['pad_sensor'] and item['pad']
                for item in execution['dynamic_enablement'].values())
    report={'success':bool(execution.get('success') and not violations and dynamic and max(quaternions.values())<.04
                           and force_barrier and beam_torques and actual_forces and max(actual_forces)>.4),
            'all_graph_predecessors_honored':not violations,'violations':violations,
            'actual_nodes':len(times),'expected_nodes':len(graph['nodes']),
            'all_arms_and_sensors_dynamically_enabled':dynamic,
            'grasp_events':sum(e['kind']=='grasp' for e in execution['events']),
            'snap_events':sum(e['kind']=='snap_connected' for e in execution['events']),
            'max_goal_position_error_m':max(execution['goal_position_errors_m'].values()),
            'max_goal_orientation_error_rad':max(quaternions.values()),
            'support_height_corrections':corrections,
            'support_ready_measured_force_barrier_verified':force_barrier,
            'insertion_loaded_trace_samples':len(loaded),
            'applied_insertion_load_N':max((r.get('applied_insertion_load_N',0) for r in loaded),default=0),
            'peak_beam_snap_torque_during_insertion_Nm':max(beam_torques,default=None),
            'peak_support_normal_during_insertion_N':max(actual_forces,default=None),
            'per_connection_peak_torque_Nm':connection_peaks,
            'peak_support_normal_N':execution['peak_support_normal_N'],
            'controller_sha256':hashlib.sha256((ROOT/'normal'/'controller.lua').read_bytes()).hexdigest(),
            'scope':'actual event and measured pose audit; ideal rigid grasp/snap and sampled geometry remain model assumptions'}
    execution['success']=report['success']
    (OUT/'execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'execution_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_path=OUT/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'));summary['success']=report['success']
        summary['max_goal_orientation_error_rad']=report['max_goal_orientation_error_rad']
        summary['actual_TPG_edges_verified']=not violations
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['success']:raise RuntimeError('Actual normal execution audit failed')

if __name__=='__main__':main()
