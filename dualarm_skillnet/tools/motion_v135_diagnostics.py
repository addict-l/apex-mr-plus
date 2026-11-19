"""Pinned, resumable local native diagnostics of the fourteen old failures.

Import is inert. Run the supervisor directly in a detached Windows process;
never wrap it in control.run_job (that would fence/lock the supervisor itself).
Only each scene worker belongs to a KILL_ON_JOB_CLOSE Job Object. No network
service, model endpoint, installer, package download or chat callback is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import motion_v135_control as control
from tools.overnight_regression import health

SOURCE = ROOT / 'results/sim_v1_3/v134/manifests'
ORDER = (('sim20', 'v134-fixed20-06'), ('sim20', 'v134-fixed20-08'),
         ('sim20', 'v134-fixed20-00'), ('walls3', 'v134-wall-037'),
         ('sim20', 'v134-fixed20-02'), ('sim20', 'v134-fixed20-07'),
         ('sim20', 'v134-fixed20-12'), ('sim20', 'v134-fixed20-13'),
         ('sim20', 'v134-fixed20-15'), ('sim20', 'v134-fixed20-16'),
         ('sim20', 'v134-fixed20-17'), ('sim20', 'v134-fixed20-18'),
         ('sim20', 'v134-fixed20-19'), ('walls3', 'v134-wall-001'))
TOOLS = ('tools/motion_v135_diagnostics.py', 'tools/motion_v135_control.py',
         'tools/overnight_batch.py', 'tools/overnight_regression.py',
         'tools/motion_v135_single.py', 'tools/motion_v135_preflight.py',
         'tools/motion_v135_detached.py')


def queue_root(queue_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', queue_id):
        raise ValueError('队列ID必须是安全标识')
    return control.BASE / 'diagnostics' / queue_id


def snapshot():
    return {'core': control.version(), 'tools': {p: old.sha256(ROOT/p) for p in TOOLS}}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    """Read append-only sources without instantiating/writing an old Journal."""
    rows = []
    if Path(path).exists():
        for line in Path(path).read_bytes().splitlines(keepends=True):
            if line.endswith(b'\n'):
                rows.append(json.loads(line))
    return rows


def baseline_jobs():
    manifests, commits, sources = {}, {}, {}
    for cohort in ('sim20', 'walls3'):
        path = SOURCE/(cohort+'.json')
        manifests[cohort] = old.read_json(path)
        journal = ROOT/'results/sim_v1_3/v134/states/runtime_fix2'/cohort/'completed.jsonl'
        commits[cohort] = {row['id']: row for row in _rows(journal)}
        sources[str(path)] = old.sha256(path)
        sources[str(journal)] = old.sha256(journal)
    jobs = []
    for index, (cohort, ident) in enumerate(ORDER):
        source = next(j for j in manifests[cohort]['jobs'] if j['id'] == ident)
        prior = commits[cohort][ident]
        _require(prior.get('passed') is False, '清单并非正式旧失败：'+ident)
        sha = source.get('prepared_task_sha256', source.get('task_sha256'))
        _require(sha and old.sha256(source['task']) == sha, '旧任务SHA改变：'+ident)
        _require(source['timeout_seconds'] == 1200., '原场景预算不是1200秒：'+ident)
        baseline = {k: deepcopy(prior.get(k)) for k in ('status', 'passed', 'failure_phase',
            'reason', 'real_planning_status', 'wall_seconds', 'evidence_directory', 'phase_seconds')}
        if ident in {'v134-fixed20-00', 'v134-fixed20-02'}:
            baseline['failure_detail_phase'] = 'tpg_geometry'
        elif ident == 'v134-wall-037':
            baseline['failure_detail_phase'] = 'home_B'
        else:
            baseline['failure_detail_phase'] = prior.get('failure_phase')
        path = Path(prior.get('evidence_directory', ''))/'motion_backend.json'
        if path.is_file():
            audit = old.read_json(path)
            failed = next((r for r in audit.get('validations', []) if r.get('status') != 'VALID'), None)
            baseline['first_failed_action'] = failed.get('action') if failed else None
            baseline['motion_backend'] = str(path)
            sources[str(path)] = old.sha256(path)
        hints = Path(source['task']).parent/'qualify_log.json'
        if hints.is_file():
            sources[str(hints)] = old.sha256(hints)
        jobs.append({'id': 'case-'+str(index).zfill(2), 'phase': 'sim', 'kind': 'sim',
            'family': source['family'], 'baseline_cohort': cohort, 'baseline_job_id': ident,
            'source_job_id': source.get('source_job_id', ident), 'task': source['task'],
            'prepared_task_sha256': sha, 'task_sha256': sha, 'timeout_seconds': source['timeout_seconds'],
            'baseline_result': baseline, 'baseline_output': source['output']})
    return jobs, sources


def dependencies():
    install = Path(os.environ.get('COPPELIASIM_ROOT', r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu'))
    paths = [Path(sys.executable), install/'coppeliaSim.exe', install/'lua.dll',
             install/'models/robots/non-mobile/UR5.ttm',
             install/'programming/zmqRemoteApi/clients/python/src/coppeliasim_zmqremoteapi_client/__init__.py']
    _require(all(p.is_file() for p in paths), '本地Python/CoppeliaSim/UR5/客户端依赖不完整')
    versions = {}
    for module, distribution in (('numpy','numpy'), ('scipy','scipy'), ('zmq','pyzmq'), ('PIL','Pillow')):
        _require(importlib.util.find_spec(module) is not None, '本地依赖不可导入：'+module)
        versions[distribution] = importlib.metadata.version(distribution)
    return {'python': sys.version, 'python_executable': sys.executable,
            'installed_packages': versions, 'local_files': {str(p): old.sha256(p) for p in paths},
            'remote_services_required': [], 'network_requirement': 'local loopback 127.0.0.1:23210 only',
            'video': False, 'headless': True, 'port': 23210}


def select_jobs(jobs, selected_ids=None):
    """A representative rerun is a new cohort, never a changed old denominator."""
    if selected_ids is None:
        return jobs
    by_id = {job['baseline_job_id']: job for job in jobs}
    _require(bool(selected_ids) and len(set(selected_ids)) == len(selected_ids),
             '代表案例必须非空且不能重复')
    _require(all(ident in by_id for ident in selected_ids), '只能选择已登记的原14个失败案例')
    return [by_id[ident] for ident in selected_ids]


def _stop(root, reason):
    row = {'at': old.utc_now(), 'queue_id': root.name, 'reason': reason, 'global_stop': reason}
    old.atomic_json(root/'global_stop.json', row)
    old.atomic_json(control.BASE/'global_stop.json', row)


def _check_gate(name, version):
    from tools.motion_v135_preflight import verify_native
    job = old.read_json(control.BASE/'jobs'/name/'result.json')
    args = job['command']
    output = Path(args[args.index('--out')+1])
    if not output.is_absolute():
        output = ROOT/output
    proof = verify_native('gate_parallel', output, job, version)
    _require(proof['passed'], '当前版本普通取放门未通过：'+str(proof.get('reason')))
    return {'job': name, 'result_path': str(control.BASE/'jobs'/name/'result.json'),
            'result_sha256': old.sha256(control.BASE/'jobs'/name/'result.json'), 'proof': proof}


def prepare(queue_id='failed14_a1', gate_job='m5_parallel_gate_a1', selected_ids=None):
    root = queue_root(queue_id)
    with old.execution_lock(control.BASE/'diagnostic_prepare.lock'):
        if (root/'selection.json').exists():
            selection = verify(root)
            expected = selected_ids if selected_ids is not None else [row[1] for row in ORDER]
            _require(selection.get('selected_ids', [row[1] for row in ORDER]) == expected,
                     '既有队列不能改换代表案例；使用新ID')
            return {'status': 'already_prepared', 'queue_id': queue_id, 'root': str(root)}
        _require(not root.exists() or not any(root.iterdir()), '队列目录已有未封存内容，必须用新ID')
        _require(not (control.BASE/'global_stop.json').exists(), 'global_stop仍存在，禁止准备活动队列')
        _require(control.audit()['passed'], '保护检查未通过')
        current = snapshot()
        gate = _check_gate(gate_job, current)
        jobs, sources = baseline_jobs()
        jobs = select_jobs(jobs, selected_ids)
        local = dependencies()
        for index, job in enumerate(jobs):
            logical = queue_id+'-'+job['id']
            job.update(id=logical, logical_job_id=logical, run_attempt=0, queue_id=queue_id,
                output=str(root/'cases'/str(index).zfill(2)), code_version_sha256=old.fingerprint(current))
        manifest = {'schema_version': 1, 'session_id': 'motion_v135_'+queue_id, 'port': 23210,
            'state_dir': str(root/'state'), 'report_dir': str(root/'reports'),
            'session_budget_seconds': None,
            # Both original source manifests use 18000 seconds. Fourteen
            # task budgets sum to16800; do not extend the original ceiling.
            'phase_budgets_seconds': {'plan': None, 'sim': 18000., 'gates': None}, 'jobs': jobs}
        old.atomic_json(root/'manifest.json', manifest)
        old.load_manifest(root/'manifest.json')
        for name, sha in {**current['core'], **current['tools']}.items():
            dst = root/'code_snapshot'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT/name).read_bytes())
            _require(old.sha256(dst) == sha, '复制期间代码改变：'+name)
        old.atomic_json(root/'dependencies.json', local)
        selection = {'prepared_at': old.utc_now(), 'queue_id': queue_id, 'diagnostic_only': True,
            'formal_run': False, 'requested': len(jobs),
            'selected_ids': [job['baseline_job_id'] for job in jobs],
            'cohorts': dict(Counter(job['baseline_cohort'] for job in jobs)),
            'code': current, 'source_files': sources, 'normal_gate': gate,
            'manifest_sha256': old.sha256(root/'manifest.json'),
            'dependencies_sha256': old.sha256(root/'dependencies.json'),
            'code_fingerprint': old.fingerprint(current),
            'policy': 'exact original failed-task identities, input SHA and budgets; no resampling',
            'launch_policy': 'detached supervisor; never inside a per-task Job Object; scene workers are owned'}
        _require(snapshot() == current, '准备过程中代码变化，保留现场并使用新ID')
        old.atomic_json(root/'selection.json', selection)
        return {'status': 'prepared', 'queue_id': queue_id, 'root': str(root), 'requested': len(jobs)}


def verify(root):
    selection = old.read_json(root/'selection.json')
    _require(selection['code'] == snapshot(), '代码/批次工具指纹变化；当前队列禁止混版本')
    _require(old.sha256(root/'manifest.json') == selection['manifest_sha256'], '诊断清单SHA变化')
    _require(old.sha256(root/'dependencies.json') == selection['dependencies_sha256'], '依赖记录SHA变化')
    gate = selection['normal_gate']
    _require(Path(gate['result_path']).is_file() and old.sha256(gate['result_path']) == gate['result_sha256'],
             '普通取放门证明被改变')
    for name, sha in gate['proof'].get('evidence_files', {}).items():
        _require(Path(name).is_file() and old.sha256(name) == sha, '普通取放门证据SHA变化：'+name)
    for name, sha in selection['source_files'].items():
        _require(Path(name).is_file() and old.sha256(name) == sha, '原输入/正式来源SHA变化：'+name)
    for name, sha in old.read_json(root/'dependencies.json')['local_files'].items():
        _require(Path(name).is_file() and old.sha256(name) == sha, '本地依赖SHA变化：'+name)
    manifest = old.load_manifest(root/'manifest.json')
    ids = selection.get('selected_ids', [row[1] for row in ORDER])
    _require(len(manifest['jobs']) == selection['requested']
             and [job['baseline_job_id'] for job in manifest['jobs']] == ids,
             '诊断队列分母或任务身份改变')
    return selection


def canonical_rows(jobs, records):
    canonical = {j['id'] for j in jobs}
    selected = {}
    for row in records.values():
        logical = row.get('logical_job_id')
        if logical not in canonical:
            logical = row['id'] if row['id'] in canonical else re.sub(r'-resume[1-9][0-9]*$', '', row['id'])
        if logical not in canonical or row.get('status') == 'interrupted':
            continue
        _require(logical not in selected, '同一逻辑任务有重复完整提交：'+logical)
        _require(row.get('passed') is not True or row.get('native_independent_passed') is True,
                 '已提交/恢复的通过缺少原生独立验证证明：'+logical)
        selected[logical] = row
    return selected


def recovery_jobs(root, originals, records):
    completed = canonical_rows(originals, records)
    jobs, receipts = [], []
    for original in originals:
        logical = original['id']
        if logical in completed:
            selected = completed[logical]
            if selected['id'] == logical:
                jobs.append(original)
            else:
                jobs.append(old.read_json(root/'recovery'/(selected['id']+'.json'))['job'])
            continue
        attempts = [row['id'] for row in records.values() if row['id'] == logical or
                    re.fullmatch(re.escape(logical)+r'-resume[1-9][0-9]*', row['id'])]
        if not attempts:
            jobs.append(original)
            continue
        _require(len(attempts) < 3, '同场景已中断三次，保留现场：'+logical)
        retry = deepcopy(original)
        retry.update(id=logical+'-resume'+str(len(attempts)), run_attempt=len(attempts),
                     output=str(root/'recovery_outputs'/logical/str(len(attempts)).zfill(2)))
        receipts.append({'at': old.utc_now(), 'previous_attempts': attempts, 'job': retry})
        jobs.append(retry)
    return jobs, receipts


def execution_manifest(root):
    canonical = old.load_manifest(root/'manifest.json')
    state, journal = old._state(canonical), old.Journal(canonical['state_dir'])
    old._recover_active(canonical, state, journal)
    old.save_state(canonical, state)
    originals = old.read_json(root/'manifest.json')
    jobs, receipts = recovery_jobs(root, originals['jobs'], journal.load())
    for receipt in receipts:
        path = root/'recovery'/(receipt['job']['id']+'.json')
        if path.exists():
            _require(old.read_json(path)['job'] == receipt['job'], '恢复任务定义变化')
        else:
            old.atomic_json(path, receipt)
    value = {**originals, 'jobs': jobs}
    path = root/'execution_manifests'/(old.fingerprint(value)+'.json')
    if not path.exists():
        old.atomic_json(path, value)
    return old.load_manifest(path)


def counts(manifest, selected):
    answer = {}
    for cohort in ('total', 'sim20', 'walls3'):
        jobs = [j for j in manifest['jobs'] if cohort == 'total' or j['baseline_cohort'] == cohort]
        done = [selected[j['id']] for j in jobs if j['id'] in selected]
        passed = sum(r.get('passed') is True for r in done)
        timeout = sum(r.get('status') == 'timeout' for r in done)
        answer[cohort] = {'requested': len(jobs), 'committed': len(done), 'passed': passed,
                         'timeout': timeout, 'failed': len(done)-passed-timeout, 'pending': len(jobs)-len(done),
                         'statuses': dict(Counter(r.get('status', 'missing') for r in done))}
    return answer


def startup_streak(rows):
    streak = 0
    for row in rows:
        if row.get('status') == 'interrupted':
            continue
        phase = row.get('startup_failure_phase') or row.get('failure_phase')
        streak = streak+1 if not row.get('passed') and phase in {'simulator_start', 'simulator_connect'} else 0
    return streak


def check_native(job, result):
    """Credit only complete native execution plus independent validation."""
    value = deepcopy(result)
    value['native_independent_passed'] = False
    if not value.get('passed'):
        return value
    try:
        output = Path(job['output']).resolve()
        evidence = Path(value['evidence_directory']).resolve()
        _require(evidence.is_relative_to(output), '证据越过当前尝试目录')
        names = ('summary.json', 'execution.json', 'validation.json', 'execution_validation.json',
                 'geometry/rollout_verification.json', 'reference_receipt.json', 'tasks.json', 'tpg.json')
        docs = {name: old.read_json(evidence/name) for name in names}
        summary, execution, audit, rollout = (docs[n] for n in
            ('summary.json', 'execution.json', 'validation.json', 'geometry/rollout_verification.json'))
        _require(value.get('status') == 'passed' and summary.get('passed') is True and summary.get('success') is True
                 and summary.get('status') == 'passed' and summary.get('simulator_closed') is True
                 and summary.get('task_sha256') == job['task_sha256']
                 and summary.get('real_planning_status') == 'solved' and not summary.get('fatal_environment'),
                 '正常入口summary不完整或原SHA不符')
        _require(audit.get('success') is True and audit.get('violations') == []
                 and audit.get('expected_nodes', 0) > 0
                 and audit.get('actual_node_intervals') == audit.get('expected_nodes'), '独立验收失败/不完整')
        _require(execution.get('success') is True and execution.get('done') is True
                 and execution.get('independent_validation_passed') is True
                 and execution.get('total', 0) > 0 and execution.get('completed') == execution['total']
                 and execution.get('tracking_failures') == execution.get('collision_failures') == []
                 and not execution.get('timeout') and not execution.get('remote_interface_failure'), '原生执行失败/不完整')
        _require(rollout.get('valid') is True and rollout.get('coverage_complete') is True
                 and rollout.get('no_sampled_collision') is True and rollout.get('stationary_waiting_poses_included') is True
                 and rollout.get('planned_samples', 0) > 0
                 and rollout.get('tested_samples') == rollout['planned_samples'], 'rollout证据不完整')
        value.update(native_independent_passed=True,
            native_evidence_sha256={name: old.sha256(evidence/name) for name in names})
    except (ValueError, KeyError, TypeError, OSError) as exc:
        value.update(passed=False, status='missing_evidence', uncredited_status=result.get('status'),
                     reason='不能计为完整通过：'+str(exc))
    return value


def _progress(root, manifest, state=None):
    state = state if state is not None else _read_if(root/'state/state.json')
    records = old.Journal(manifest['state_dir']).load()
    selected = canonical_rows(manifest['jobs'], records)
    active = state.get('active_job') or {}
    current = {'id': active.get('id'), 'pid': active.get('pid'), 'birth': active.get('birth'),
               'elapsed_seconds': active.get('elapsed_seconds'), 'phase': None}
    if active.get('job'):
        current['baseline_job_id'] = active['job']['baseline_job_id']
        output = Path(active['job']['output'])
        paths = (list(output.rglob('run_progress.json'))+list(output.rglob('executor_progress.json'))
                 +list(output.rglob('motion_progress.json')))
        if paths:
            path = max(paths, key=lambda p: p.stat().st_mtime_ns)
            data = _read_if(path)
            current.update(phase=data.get('phase'), progress_file=str(path),
                           progress_file_mtime_epoch=path.stat().st_mtime,
                           progress_age_seconds=max(0., time.time()-path.stat().st_mtime),
                           details=data)
        else:
            current['phase'] = 'worker_start_or_loading'
        if 'result_path' in active:
            token = Path(active['result_path']).name.replace('.result.json', '.log')
            log = output/token
            current.update(log_path=str(log), log_mtime_epoch=log.stat().st_mtime if log.exists() else None)
    old_progress = _read_if(root/'progress.json')
    previous = old_progress.get('current', {})
    same = previous.get('id') == current.get('id') and previous.get('phase') == current.get('phase')
    now = time.time()
    current['phase_observed_since_epoch'] = previous.get('phase_observed_since_epoch', now) if same else now
    current['same_phase_observed_seconds'] = max(0., now-current['phase_observed_since_epoch'])
    return {'at': old.utc_now(), 'queue_id': root.name, 'counts': counts(manifest, selected),
            'status': state.get('status', 'ready'), 'current': current,
            'supervisor_pid': os.getpid(), 'supervisor_birth': old.process_birth(os.getpid()),
            'result_journal': str(Path(manifest['state_dir'])/'completed.jsonl')}


def _read_if(path):
    return old.read_json(path) if Path(path).is_file() else {}


def rebuild(queue_id='failed14_a1'):
    root = queue_root(queue_id)
    manifest = old.load_manifest(root/'manifest.json')
    records = old.Journal(manifest['state_dir']).load()
    selected = canonical_rows(manifest['jobs'], records)
    cases = []
    for job in manifest['jobs']:
        row = selected.get(job['id'])
        cases.append({'logical_job_id': job['id'], 'baseline_cohort': job['baseline_cohort'],
            'baseline_job_id': job['baseline_job_id'], 'source_job_id': job['source_job_id'],
            'task': job['task'], 'task_sha256': job['task_sha256'], 'timeout_seconds': job['timeout_seconds'],
            'baseline_result': job['baseline_result'], 'result': row,
            'status': row.get('status') if row else 'pending'})
    result = {'at': old.utc_now(), 'queue_id': queue_id, 'diagnostic_only': True, 'formal_run': False,
              'counts': counts(manifest, selected), 'cases': cases,
              'interrupted_attempts': [r for r in records.values() if r.get('status') == 'interrupted']}
    old.atomic_json(root/'results.json', result)
    columns = ('baseline_cohort', 'baseline_job_id', 'source_job_id', 'task_sha256', 'status',
               'prior_failure_phase', 'passed_prior_failure_point', 'passed', 'failure_phase',
               'wall_seconds', 'evidence_directory', 'reason')
    table = []
    for case in cases:
        row = case['result'] or {}
        table.append({**{k: case.get(k) for k in columns}, **{k: row.get(k) for k in columns if k in row},
                      'prior_failure_phase': case['baseline_result']['failure_detail_phase']})
    old.atomic_text(root/'comparison.csv', old._csv_text(table, columns))
    lines = ['# 旧失败14例：原预算正常入口诊断', '', '更新：'+result['at'], '',
             '本队列为诊断，正式成绩需最终代码原批次。所有中断尝试保留；未跑/未知/预算/超时不计通过。', '',
             '|组|请求|已提交|完整通过|未通过|超时|待跑|', '|---|---:|---:|---:|---:|---:|---:|']
    for cohort, count in result['counts'].items():
        lines.append('|'+cohort+'|'+ '|'.join(str(count[k]) for k in ('requested','committed','passed','failed','timeout','pending'))+'|')
    lines += ['', '|任务|旧失败点|本次状态|越过旧失败点|最终阶段|耗时(s)|证据|', '|---|---|---|---|---|---:|---|']
    for case in cases:
        row = case['result'] or {}
        lines.append(f"|{case['baseline_job_id']}|{case['baseline_result']['failure_detail_phase']}|{case['status']}|{row.get('passed_prior_failure_point', '未运行')}|{row.get('failure_phase') or row.get('execution_phase') or ''}|{row.get('wall_seconds', '')}|{row.get('evidence_directory') or row.get('output') or ''}|")
    old.atomic_text(root/'comparison.md', '\n'.join(lines)+'\n')
    return result


def _crossed(job, result):
    if result.get('native_independent_passed'):
        return True, '完整原生执行及独立验收通过'
    phase = job['baseline_result']['failure_detail_phase']
    evidence = Path(result.get('evidence_directory') or Path(job['output'])/'simulation'/Path(job['task']).stem.replace('.qualified', ''))
    summary = _read_if(evidence/'summary.json')
    if phase == 'tpg_geometry':
        return (evidence/'tpg.json').is_file(), 'tpg.json只在完整图构建与参数化完成后保存；不证明动力学'
    if phase == 'home_B':
        return (evidence/'reference_receipt.json').is_file(), '完整参考收据包含必要回位；仍需执行验收'
    if summary.get('real_planning_status') == 'solved' or (evidence/'reference_receipt.json').is_file():
        return True, '本次合法完整动作参考通过（允许装配顺序改变），不代表动力学通过'
    action = job['baseline_result'].get('first_failed_action')
    records = _read_if(evidence/'motion_backend.json').get('validations', [])
    if action and any(r.get('action') == action and r.get('status') == 'VALID' for r in records):
        return True, '本次上下文中同名原失败动作已通过；其余轨迹仍可能失败'
    return False, '尚无越过旧失败点的完整动作/阶段证据'


def run(queue_id='failed14_a1', *, heartbeat_seconds=30.):
    root = queue_root(queue_id)
    cursor = {'queue_id': queue_id, 'status': 'starting', 'pid': os.getpid(),
              'birth': old.process_birth(os.getpid()), 'started_at': old.utc_now(),
              'resume_command': f'python -X utf8 -u tools/motion_v135_diagnostics.py run --queue-id {queue_id}'}
    stop = threading.Event()
    with old.execution_lock(root/'supervisor.lock'):
        old.atomic_json(root/'supervisor.json', cursor)
        try:
            verify(root)
            _require(not (control.BASE/'global_stop.json').exists(), 'global_stop存在')
            manifest = execution_manifest(root)
            cursor['status'] = 'running'
            old.atomic_json(root/'supervisor.json', cursor)
            def heartbeat():
                while not stop.is_set():
                    try:
                        data = _progress(root, old.load_manifest(root/'manifest.json'))
                        old.atomic_json(root/'progress.json', data)
                        with (root/'heartbeat.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(data, ensure_ascii=False)+'\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                    except BaseException as exc:
                        old.atomic_json(root/'heartbeat_error.json', {'at': old.utc_now(), 'reason': repr(exc)})
                    stop.wait(heartbeat_seconds)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()

            def environment(guard):
                if (control.BASE/'global_stop.json').exists():
                    return {'ok': False, 'global_stop': True, 'reason': '本轮整体停止记录存在'}
                try:
                    verify(root)
                    protected = control.audit()
                    if not protected['passed']:
                        raise ValueError('历史/冻结核心保护差异')
                except (ValueError, OSError, KeyError) as exc:
                    _stop(root, str(exc))
                    return {'ok': False, 'global_stop': True, 'reason': str(exc)}
                return old.preflight(guard)

            def factory(command, log_path, start_signal):
                command = list(command)
                command[3] = str(Path(__file__).resolve())
                spec = old.read_json(command[-1])
                state = old.read_json(Path(manifest['state_dir'])/'state.json')
                duration = min(spec['job']['timeout_seconds'], old.remaining_seconds(state, 'sim'))
                spec.update(execution_manifest=manifest['manifest_path'], hard_deadline_epoch=time.time()+duration)
                old.atomic_json(command[-1], spec)
                return old.OwnedProcess(command, log_path, start_signal)

            def adapter(job, result):
                value = check_native(job, result)
                crossed, basis = _crossed(job, value)
                value.update(logical_job_id=job['logical_job_id'], source_job_id=job['source_job_id'],
                    baseline_job_id=job['baseline_job_id'], baseline_cohort=job['baseline_cohort'],
                    run_attempt=job['run_attempt'], prior_failure_phase=job['baseline_result']['failure_detail_phase'],
                    passed_prior_failure_point=crossed, prior_point_basis=basis, code_version_sha256=job['code_version_sha256'])
                cleanup = health()
                value['cleanup'] = cleanup
                if not cleanup['passed']:
                    _stop(root, '场景后自有进程/端口/磁盘复核失败')
                previous = old.Journal(manifest['state_dir']).load()
                if startup_streak([*previous.values(), value]) >= 3:
                    _stop(root, '连续三项CoppeliaSim启动/连接失败')
                old.atomic_json(Path(job['output'])/'diagnostic_result.json', value)
                print(json.dumps({'at': old.utc_now(), 'task_completed': job['baseline_job_id'],
                                  'status': value['status'], 'passed_prior_point': crossed}, ensure_ascii=False), flush=True)
                return value

            state = old.run_phase(manifest, 'sim', environment_check=environment,
                                  process_factory=factory, result_adapter=adapter)
            cursor.update(status='finished' if state['status'] == 'phase_complete' else state['status'], state=state)
            if state.get('global_stop'):
                _stop(root, state['global_stop'])
        except BaseException as exc:
            cursor.update(status='global_stop' if (control.BASE/'global_stop.json').exists() else 'interrupted',
                          reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        finally:
            stop.set()
            if 'thread' in locals():
                thread.join(timeout=5.)
            cursor['updated_at'] = old.utc_now()
            old.atomic_json(root/'supervisor.json', cursor)
            if (root/'manifest.json').is_file():
                result = rebuild(queue_id)
                progress = _progress(root, old.load_manifest(root/'manifest.json'))
                progress.update(status=cursor['status'], current=None)
                old.atomic_json(root/'progress.json', progress)
                cursor['counts'] = result['counts']
                old.atomic_json(root/'supervisor.json', cursor)
            control.record('old_failure_queue_'+queue_id, cursor['status'], evidence=str(root/'supervisor.json'))
    return cursor


def status(queue_id='failed14_a1'):
    root = queue_root(queue_id)
    if not (root/'selection.json').is_file():
        return {'status': 'not_prepared', 'queue_id': queue_id}
    supervisor = _read_if(root/'supervisor.json')
    active = supervisor.get('birth') is not None and old.process_birth(supervisor.get('pid', 0)) == supervisor['birth']
    return {'queue_id': queue_id, 'status': supervisor.get('status', 'prepared'),
            'supervisor_alive': active, 'supervisor': supervisor, 'progress': _read_if(root/'progress.json')}


def worker(path):
    spec = old.read_json(path)
    waited = time.monotonic()
    while not Path(spec['start_signal']).exists():
        if time.monotonic()-waited > 30:
            raise TimeoutError('未收到拥有进程握手')
        time.sleep(.03)
    job = spec['job']
    root = queue_root(job['queue_id'])
    began = time.monotonic()
    deadline = began+max(.001, spec['hard_deadline_epoch']-time.time()-.75)
    try:
        verify(root)
        execution = old.load_manifest(spec['execution_manifest'])
        _require(job == next(j for j in execution['jobs'] if j['id'] == job['id']), 'worker输入与清单不符')
        from simbridge import runner
        original = runner.Simulator
        class NamespaceSimulator(original):
            def __init__(self, ignored_root, *args, **kwargs):
                super().__init__(Path(job['output'])/'instance', *args, **kwargs)
        runner.Simulator = NamespaceSimulator
        try:
            result = old.sim_job({**job, 'timeout_seconds': min(job['timeout_seconds'], max(.001, deadline-time.monotonic()))})
        finally:
            runner.Simulator = original
        unchanged = snapshot() == old.read_json(root/'selection.json')['code']
        result['code_unchanged_during_job'] = unchanged
        if not unchanged:
            result.update(status='invalid_version', passed=False, reason='运行期间代码改变，不计通过')
    except BaseException as exc:
        result = {'status': 'worker_failed', 'passed': False, 'failure_phase': 'worker',
                  'reason': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
    if 'phase' in result:
        result['execution_phase'] = result.pop('phase')
    # The durable worker file may be recovered after supervisor death before
    # the parent adapter commits it. It must already carry native proof.
    result = check_native(job, result)
    crossed, basis = _crossed(job, result)
    result.update(definition_sha256=job['definition_sha256'], logical_job_id=job['logical_job_id'],
        worker_wall_seconds=time.monotonic()-began, diagnostic_only=True, formal_run=False,
        source_job_id=job['source_job_id'], baseline_job_id=job['baseline_job_id'], baseline_cohort=job['baseline_cohort'],
        run_attempt=job['run_attempt'], code_version_sha256=job['code_version_sha256'],
        passed_prior_failure_point=crossed, prior_point_basis=basis,
        prior_failure_phase=job['baseline_result']['failure_detail_phase'])
    old.atomic_json(spec['result_path'], result)
    return 0 if result.get('passed') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='本机旧失败14例可续跑正常入口诊断队列')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest='command')
    for name in ('prepare', 'run', 'status', 'rebuild'):
        command = sub.add_parser(name)
        command.add_argument('--queue-id', default='failed14_a1')
        if name == 'prepare':
            command.add_argument('--gate-job', default='m5_parallel_gate_a1')
            command.add_argument('--only', nargs='+', help='新命名空间内按顺序复测原失败案例ID')
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if not args.command:
        parser.error('需要prepare/run/status/rebuild')
    result = (prepare(args.queue_id, args.gate_job, args.only) if args.command == 'prepare' else
              {'run': run, 'status': status, 'rebuild': rebuild}[args.command](args.queue_id))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get('status') in {None, 'finished', 'prepared', 'already_prepared', 'not_prepared', 'running'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
