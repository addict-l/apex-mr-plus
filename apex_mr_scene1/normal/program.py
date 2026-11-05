"""Time parameterize the unchanged OMPL polylines for dynamic servos."""
from copy import deepcopy
import math
import numpy as np
from .tpg import simulate_tpg


def compress_collinear(path):
    if len(path)<3:return path
    out=[list(path[0])]
    for i in range(1,len(path)-1):
        a=np.asarray(path[i])-np.asarray(out[-1]);b=np.asarray(path[i+1])-np.asarray(path[i])
        if np.linalg.norm(a)<1e-10:continue
        if np.linalg.norm(b)>1e-10 and np.dot(a,b)>0 and np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b))<1e-7:
            continue
        out.append(list(path[i]))
    out.append(list(path[-1]));return out


def parameterize(tpg,vmax=.60,amax=1.20):
    for node in tpg['nodes']:
        local_vmax=vmax;local_amax=amax
        if node.get('kind') in ('support_start','support_end'):
            local_vmax=min(vmax,float(node['metadata'].get('max_velocity_rad_s',vmax)))
            local_amax=min(amax,float(node['metadata'].get('max_acceleration_rad_s2',amax)))
        path=compress_collinear(node['q_samples'])
        segments=[]
        for q0,q1 in zip(path,path[1:]):
            d=max(abs(a-b) for a,b in zip(q0,q1))
            if d<1e-9:continue
            # Quintic rest-to-rest: max s'=1.875; max |s''|=10/sqrt(3).
            duration=max(1.875*d/local_vmax,math.sqrt((10/math.sqrt(3))*d/local_amax),.08)
            segments.append({'q0':q0,'q1':q1,'duration':duration})
        motion=sum(s['duration'] for s in segments)
        node['motion_segments']=segments
        node['duration']=max(motion,.08)
        if node['node_type']=='skill':node['duration']+=float(node['metadata'].get('dwell_s',.20))
        node['nominal_duration']=node['duration']
    prediction=simulate_tpg(tpg)
    tpg['nominal_schedule']=prediction['nodes'];tpg['nominal_task_schedule']=prediction['task_schedule']
    tpg['nominal_timestamps']=prediction['timestamps'];tpg['nominal_makespan']=prediction['makespan']
    for node in tpg['nodes']:node.update(prediction['timestamps'][node['id']])
    # Use the SAME segment timing for serial and asynchronous comparisons.
    # Otherwise a timing change could be mistaken for a TPG concurrency gain.
    by_id={n['id']:n for n in tpg['nodes']}
    clock=0.
    for row in tpg['reference_schedule']:
        duration=sum(by_id[n]['duration'] for n in tpg['task_map'][row['task_id']]['node_ids'])
        row.update(start=clock,end=clock+duration,duration=duration)
        tpg['task_map'][row['task_id']]['duration']=duration
        cursor=clock
        for nid in tpg['task_map'][row['task_id']]['node_ids']:
            by_id[nid]['reference_start']=cursor
            cursor+=by_id[nid]['duration'];by_id[nid]['reference_end']=cursor
        clock+=duration
    tpg['reference_makespan']=clock
    tpg['time_parameterization']={'type':'quintic_rest_to_rest_per_original_polyline_segment',
                                  'max_velocity_rad_s':vmax,'max_acceleration_rad_s2':amax,
                                  'extra_runtime_endpoint_settling':True}
    return tpg
