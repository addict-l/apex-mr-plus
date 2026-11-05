"""Replay bounded local candidates through the current native geometry oracle.

The stored paths are planning inputs, never certificates for a later run.
The complete rebuilt TPG, nominal rollout and dynamics audit remain mandatory.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .local_route_chain import replace_window

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'apex/local_route_candidates.json'
WINDOWS = {
    'support_pick': ('pick_support_approach', 'support_high_approach'),
    'support_return': ('return_support_approach', 'pick_gate_approach'),
    'gate_lift': ('pick_gate_lift', 'close_gate_approach'),
}
CONTACT_CLEAR_WAITS = frozenset(('pick_support_lift_contact_clear',
    'return_support_retreat_contact_clear', 'pick_gate_lift_contact_clear'))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _dense(path, step=.003):
    yield path[0]
    for start, end in zip(path, path[1:]):
        divisions = max(1, math.ceil(math.sqrt(sum((a-b)**2 for a, b in zip(start, end)))/step))
        for i in range(1, divisions+1):
            yield [a+(b-a)*i/divisions for a, b in zip(start, end)]


def _pose_error(before, after):
    position = math.sqrt(sum((a-b)**2 for a, b in zip(before[:3], after[:3])))
    norms = math.sqrt(sum(v*v for v in before[3:])*sum(v*v for v in after[3:]))
    if not norms:
        raise ValueError('Native FK returned an invalid quaternion')
    orientation = 2*math.acos(min(1., abs(sum(a*b for a, b in zip(before[3:], after[3:]))/norms)))
    return position, orientation


def apply_local_routes(compiler, reference, tasks, asset_path=None):
    """Install three structurally closed windows and freshly check each change."""
    asset_path = Path(asset_path or ASSET)
    asset = json.loads(asset_path.read_text(encoding='utf-8'))
    if asset.get('schema_version') != 1 or asset.get('reference_signature') != _value_hash(reference['tasks']):
        raise ValueError('Local candidate baseline reference identity changed')
    physical = {'scenes/10_Scene3_Dynamic.ttt': _sha(ROOT/'scenes/10_Scene3_Dynamic.ttt'),
                'apex/motion_backend.lua': _sha(ROOT/'apex/motion_backend.lua')}
    physical.update({p.relative_to(ROOT).as_posix(): _sha(p) for p in (ROOT/'config').glob('*.json')})
    if asset.get('physical_input_hashes') != physical:
        raise ValueError('Local candidate physical inputs or native collision backend changed')
    if [row.get('name') for row in asset.get('windows', [])] != list(WINDOWS):
        raise ValueError('Local candidates must cover the three ordered complete windows')
    candidate = deepcopy(tasks)
    structural = []
    declarations = {}
    for window in asset['windows']:
        name = window['name']
        first, last = WINDOWS[name]
        candidate, evidence = replace_window(candidate, first, last, window['paths'])
        structural.append(evidence)
        for ident in window['paths']:
            declarations[ident] = {'window': name, 'method': window.get('methods', {}).get(ident, 'joint_reconnect'),
                'certificate_scope': 'native dense fixed-reference-state', 'require_profile_validation': True}
    originals = {task['id']: task for task in tasks}
    q = deepcopy(compiler.home)
    payloads = {'A': None, 'B': None}
    checked = []
    for task in candidate:
        arm = task['arm']; other = 'B' if arm == 'A' else 'A'
        md = task['metadata']; original = originals[task['id']]
        changed = task['q_path'] != original['q_path']
        if max(abs(a-b) for a, b in zip(q[arm], task['q_path'][0])) > 1e-8:
            raise ValueError('Local route reference joint chain is open: '+task['id'])
        if task.get('payload') != payloads[arm]:
            raise ValueError('Local route payload chain is open: '+task['id'])
        if changed:
            options = deepcopy(md.get('options', {}))
            options.update(other_q=q[other], other_payload=payloads[other] or {},
                contact_exclusions=deepcopy(md.get('contact_exclusions', [])),
                contact_tolerance=md.get('contact_tolerance', .002))
            samples = []
            count = 0
            for joint in _dense(task['q_path']):
                samples.append({arm: joint, other: q[other], 'payloads': deepcopy(payloads),
                                'placed': deepcopy(md['placed']), 'options': deepcopy(options)})
                if len(samples) == 128:
                    answers = compiler.p.validate_samples(samples)
                    if len(answers) != len(samples) or any(r.get('ok') is not True for r in answers):
                        raise ValueError('Local route failed current native geometry: '+task['id'])
                    count += len(samples); samples = []
            if samples:
                answers = compiler.p.validate_samples(samples)
                if len(answers) != len(samples) or any(r.get('ok') is not True for r in answers):
                    raise ValueError('Local route failed current native geometry: '+task['id'])
                count += len(samples)
            record = {'task_id': task['id'], 'native_geometry_valid': True, 'samples': count, 'step_rad': .003}
            if task['kind'] != 'transit' and task['id'] not in CONTACT_CLEAR_WAITS:
                old_fk = compiler.p.fk(arm, original['q_path'][0])
                new_fk = compiler.p.fk(arm, task['q_path'][0])
                if not old_fk.get('ok') or not new_fk.get('ok'):
                    raise ValueError('Native skill endpoint FK failed: '+task['id'])
                dp, dr = _pose_error(old_fk['pose'], new_fk['pose'])
                if dp > 5e-5 or dr > .0002:
                    raise ValueError('Local route changes a physical skill endpoint: '+task['id'])
                record.update(skill_position_error_m=dp, skill_orientation_error_rad=dr)
            if task['id'] in declarations:
                md.update(local_route_candidate=declarations[task['id']],
                    planner='ValidatedLocalRouteCandidate', path_source='native_local_candidate_revalidated',
                    planning_stats={'candidate_asset_sha256': _sha(asset_path), 'native_samples': count,
                                    'planner_called_this_run': False, 'geometry_oracle_called_this_run': True})
            checked.append(record)
        q[arm] = list(task['q_path'][-1])
        payloads.update(deepcopy(md.get('effects', {}).get('payloads', {})))
    compiler.tasks = candidate; compiler.q = q
    report = {'source': 'frozen_local_candidates_with_fresh_native_revalidation',
              'candidate_asset_sha256': _sha(asset_path), 'structural_windows': structural,
              'changed_task_checks': checked, 'samples': sum(r['samples'] for r in checked),
              'continuous_collision_guarantee': False,
              'remaining_checks': ['curved_geometry', 'all_phase_TPG_conflicts', 'nominal_rollout', 'actual_dynamics']}
    return candidate, report
