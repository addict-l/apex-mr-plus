"""Assemble independently checked subpaths without changing their geometry."""
from copy import deepcopy
import math
import numpy as np
from support_return_probe import LAB, read, write, shifted


def main():
    records=read(LAB/'accepted_cases.json')['all_reference_tasks']
    incoming=read(LAB/'support_return_incoming_upper.json')['candidates'][0]
    departures=read(LAB/'support_return_departure_audit.json')
    departure=next(c for c in departures['candidates']if c['label']=='clear_at_q1_-3.56')
    ids=['return_support_approach','return_support_contact','return_support',
         'return_support_retreat','return_support_retreat_contact_clear','pick_gate_approach']
    paths={ids[0]:incoming['path'],ids[1]:shifted(records[ids[1]]['task']['q_path']),
           ids[2]:shifted(records[ids[2]]['task']['q_path']),**departure['replacements']}
    errors=[float(np.max(abs(np.asarray(paths[a][-1])-paths[b][0])))for a,b in zip(ids,ids[1:])]
    travels={k:np.abs(np.diff(np.asarray(v),axis=0)).sum(axis=0).tolist()for k,v in paths.items()}
    assert max(errors)<1e-8
    assert all(v[0]<math.pi for v in travels.values())
    candidate={'label':'upper_incoming_side_release_clear_minus3_56','accepted':True,
        'scope':'assembled fixed-reference native candidate; complete production revalidation pending',
        'replacements':paths,'context_records':{k:deepcopy(records[k])for k in ids},
        'joint_travel_rad':travels,'base_travel_degrees':{k:math.degrees(v[0])for k,v in travels.items()},
        'total_base_travel_degrees':math.degrees(sum(v[0]for v in travels.values())),
        'chain_max_joint_error_rad':max(errors),'actual_pending':True,'full_TPG_certificate_pending':True,
        'evidence':{'incoming':'support_return_incoming_upper.json candidates[0]',
                    'contact_and_release':'support_return_initial_search.json fixed_rebinding_checks',
                    'outgoing':'support_return_departure_audit.json clear_at_q1_-3.56'},
        'checks':{'incoming':incoming['check'],
                  'side_contact_path':departures['side_contact_path_check'],
                  'departure_without_contact_exclusions':departures['entire_departure_without_contact_exclusions'],
                  'boundary_without_contact_exclusions':departure['boundary_no_contact_check']}}
    result={'status':'assembled_candidate_ready_for_full_native_revalidation',
            'source_run_id':read(LAB/'accepted_cases.json')['run_id'],'candidates':[candidate]}
    write(LAB/'support_return_assembled_candidates.json',result)
    print(candidate['label'],candidate['total_base_travel_degrees'],candidate['chain_max_joint_error_rad'])


if __name__=='__main__':main()
