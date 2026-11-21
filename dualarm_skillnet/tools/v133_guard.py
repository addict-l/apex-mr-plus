"""Check unchanged archives/tasks while reporting explicitly authorized edits."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as batch

BASE = ROOT / 'results/sim_v1_3/v133'
AUTHORIZED = frozenset({
    'skillnet/planning/search.py', 'skillnet/planning/heuristic.py',
    'skillnet/cost/model.py', 'skillnet/report.py', 'skills/blocks_v1.toml',
    'gen/geometry.py', 'gen/generator.py', 'gen/families.py', 'gen/__main__.py',
    'simbridge/backend.py', 'simbridge/__main__.py', 'simbridge/frames.py',
    'simbridge/workcell.py', 'simbridge/runner.py',
    # D native acceptance exposed force-event schema and cross-arm hold edges.
    # The copied validator/controller stack and the seven accepted methods stay
    # immutable; runtime_repair1_backup retains the exact pre-repair sources.
    'simbridge/compiler.py', 'simbridge/controller_ext.lua',
})
SIM_MIGRATIONS = frozenset('tasks/sim/'+name+'.toml' for name in (
    's1_feedback_sim','s1_feedback_sim_attempt2','s1_qualify_sim',
    's2_tower_sim','s3a_cantilever_sim','s3a_platform_sim'))


def reach_removal_only(name, baseline, root):
    saved=baseline.get('reach_table_migrations',{}).get(name)
    if name not in SIM_MIGRATIONS or saved is None:return False
    try:
        before=deepcopy(saved['before_data'])
        after=tomllib.loads((root/name).read_text(encoding='utf-8-sig'))
        if 'reach' in after.get('capabilities',{}):return False
        before.get('capabilities',{}).pop('reach',None)
        for data in (before,after):
            if data.get('capabilities')=={}:data.pop('capabilities')
        return before==after
    except (ValueError,OSError,KeyError):return False


def verify_local(baseline, root=ROOT):
    root = Path(root)
    protected, unexpected, authorized = [], [], []
    expected_documents=dict(baseline.get('protected_files',{}))
    for update in baseline.get('user_document_updates',[]):
        name=update['path']
        if name!='FIX_v1.3.3.md' or expected_documents.get(name)!=update['previous_sha256']:
            protected.append(name+':invalid_user_authorization_chain')
        else:
            expected_documents[name]=update['current_sha256']
    for group, failures in (('protected_files', protected), ('code_hashes', unexpected)):
        for name, before in baseline.get(group, {}).items():
            if group=='protected_files':before=expected_documents[name]
            path = root / name
            if not path.is_file() or batch.sha256(path) != before:
                allowed=(name in AUTHORIZED or group=='protected_files' and reach_removal_only(name,baseline,root))
                (authorized if allowed else failures).append(name)
    old_core = {name for name in baseline.get('code_hashes', {}) if name.startswith('skillnet/')}
    current_core = {p.relative_to(root).as_posix() for p in (root/'skillnet').rglob('*.py')}
    new = sorted(current_core-old_core)
    return {'protected_count': len(baseline.get('protected_files', {})),
            'protected_changes': protected, 'code_changes': unexpected,
            'authorized_changes': sorted(set(authorized)), 'new_core_files': new,
            'passed': not protected and not unexpected and not new}


def check(archive_guard=None):
    baseline = batch.read_json(BASE/'baseline.json')
    result = verify_local(baseline)
    result['archives'] = (archive_guard or batch.ArchiveGuard()).check()
    stack = batch.read_json(BASE/'stack_before.json')
    result['stack_changes'] = [name for name, old in stack.items()
                               if not (ROOT/name).is_file() or batch.sha256(ROOT/name) != old]
    status = batch.read_json(ROOT/'results/sim_v1_3/v132_status_protected_before.json')
    result['accepted_method_changes'] = []
    for name, expected in status.get('accepted_method_ast', {}).items():
        tree = ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
        actual = {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name in expected}
        result['accepted_method_changes'].extend(name+':'+method for method, body in expected.items()
                                                 if actual.get(method) != body)
    result.update(checked_at=batch.utc_now(), authorization='FIX_v1.3.3 + user; baseline retained')
    result['passed'] = (result['passed'] and result['archives']['passed']
                        and not result['stack_changes'] and not result['accepted_method_changes'])
    return result


def preflight(archive_guard):
    environment = batch.preflight(archive_guard)
    if not environment['ok']:
        return environment
    evidence = check(archive_guard)
    if not evidence['passed']:
        return {'ok': False, 'global_stop': True, 'reason': 'v133 immutable archive/task/code check failed',
                'evidence': evidence}
    return {**environment, 'v133_protection': evidence}


def main():
    parser = argparse.ArgumentParser(description='v1.3.3授权变更与不可变归档校验')
    parser.add_argument('--out', type=Path, default=BASE/'archive_check.json')
    args = parser.parse_args()
    result = check()
    batch.atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
