"""Resumable v133 workers on the existing owned-process, fsynced supervisor.

No experiment runs on import. Each stage has a frozen manifest and separate
journal; raw and qualified searches are independent, with one shared task cap.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
import traceback
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import v133_guard

BASE = ROOT/'results/sim_v1_3/v133'
REPORT = ROOT/'results/sim_v1_3/v133_report.md'
FAMILIES = ('F1','F2','F3','F4','F5')
MOTION_FILES=('compiler.py','frames.py','primitives.py','backend.py','workcell.py','runner.py','qualifier.py',
              'controller_ext.lua','executor.py')


def selected_stage(logical, *, base=BASE, selection=None):
    from tools.v133_experiments import resolve_stage
    return resolve_stage(logical, base=base, selection=selection)


def input_cohort_sha256(jobs):
    cells=[{'family':j['family'],'index':j['index'],'table':j['table'],
            'task':str(Path(j['task']).resolve()) if j.get('task') else None,
            'sha256':j.get('task_sha256',j.get('prepared_task_sha256')),
            'timeout_seconds':float(j.get('timeout_seconds',60.))} for j in jobs]
    payload=json.dumps(sorted(cells,key=lambda r:(r['family'],r['index'],r['table'])),
                       sort_keys=True,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def planning_fingerprint():
    names=('skillnet/planning/search.py','skillnet/planning/heuristic.py','skillnet/cost/model.py',
           'skills/blocks_v1.toml')
    result={name:old.sha256(ROOT/name) for name in names}
    if (BASE/'calibration.json').is_file():result['calibration.json']=old.sha256(BASE/'calibration.json')
    return result


def motion_fingerprint():
    return {name:old.sha256(ROOT/'simbridge'/name) for name in MOTION_FILES
            if (ROOT/'simbridge'/name).is_file()}


def connection_failures(base=BASE):
    records=[]
    for directory in (base/'states').glob('*'):
        if directory.is_dir():records.extend(old.Journal(directory).load().values())
    failures=0
    for row in sorted(records,key=lambda r:(r.get('completed_at',''),r.get('id',''))):
        if row.get('phase') not in {'sim','gates'} or row.get('status')=='skipped':continue
        startup_phase=row.get('startup_failure_phase') or row.get('original_failure_phase') or row.get('failure_phase')
        if startup_phase in {'simulator_start','simulator_connect'}:failures+=1
        elif row.get('startup_connect_completed') or row.get('simulator_pid'):failures=0
    return failures


def shared_stop_reason(base=BASE):
    ledger=base/'global_stop.json'
    if ledger.exists() and old.read_json(ledger).get('global_stop'):
        return old.read_json(ledger)['global_stop']
    for path in (base/'physical_stage_outcome.json',base/'workflow_state.json',*sorted((base/'states').glob('*/state.json'))):
        if path.exists() and old.read_json(path).get('global_stop'):
            return str(path)+': '+str(old.read_json(path)['global_stop'])
    if connection_failures(base)>=3:return '跨阶段连续三次CoppeliaSim启动/连接失败'
    return None


def environment_check(archive_guard):
    stopped=shared_stop_reason()
    if stopped:return {'ok':False,'global_stop':True,'reason':stopped}
    return v133_guard.preflight(archive_guard)


def adapt_result(job, result):
    result=dict(result)
    if result.get('status')!='timeout' or job.get('kind') not in {'qualify','sim','gate'}:return result
    output=Path(job['output'])
    candidates=(list(output.glob('*/qualify_log.json')) if job.get('kind')=='qualify' else
                list(output.glob('simulation/*/summary.json'))+list(output.glob('simulation/*/run_progress.json')))
    for path in candidates:
        try:progress=old.read_json(path)
        except (OSError,ValueError):continue
        phase=(progress.get('startup_failure_phase') or progress.get('original_failure_phase')
               or progress.get('failure_phase') or progress.get('phase'))
        if phase in {'simulator_start','simulator_connect'}:
            result.update(timeout_phase=result.get('failure_phase'),hard_timeout=True,
                          failure_phase=phase,timeout_checkpoint=str(path))
        if progress.get('simulator_pid') or progress.get('pid'):
            result['simulator_pid']=progress.get('simulator_pid') or progress['pid']
        if phase=='workcell':result['startup_connect_completed']=True
        break
    if list(output.glob('simulation/*/workcell.json')):
        result['startup_connect_completed']=True
    return result


def require_motion_gate(job):
    if job.get('kind') not in {'sim','qualify','gate'}:return
    if job.get('motion_code_sha256') is not None and job['motion_code_sha256']!=motion_fingerprint():
        raise RuntimeError('真实任务冻结的运动代码已变化，须新attempt与输出命名空间')
    if job.get('kind')=='gate' and job.get('test_name')==old.LIVE_TESTS[0]:return
    path=BASE/'last_motion_gate.json'
    if not path.exists():raise RuntimeError('必须先通过修改后的普通取放真实回归门')
    proof=old.read_json(path)
    if proof.get('passed') is not True or proof.get('motion_code_sha256')!=motion_fingerprint():
        raise RuntimeError('运动代码变化后尚未通过新的普通取放回归门')


def append_report(text):
    with REPORT.open('a',encoding='utf-8') as stream:
        stream.write('\n\n'+text+'\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def prepare(stage, jobs, *, base=BASE):
    base=old.confined(base)
    path=base/'manifests'/f'{stage}.json'
    saved=old.read_json(path) if path.exists() else None
    frozen=[]
    for original in jobs:
        job=dict(original)
        job.setdefault('timeout_seconds',1200.)
        if job.get('task'):
            task=old.confined(job['task'])
            job['task']=str(task)
            job['prepared_task_sha256']=old.sha256(task)
        if job.get('kind') in {'qualify','sim','gate'}:
            job.setdefault('motion_code_sha256',motion_fingerprint())
        if job.get('kind') in {'plan','sim','gate'}:
            job.setdefault('planning_code_sha256',planning_fingerprint())
        frozen.append(job)
    if saved:
        if saved['jobs'] != frozen:
            raise ValueError('已有清单定义或输入SHA改变，禁止覆盖：'+stage)
        old.load_manifest(path)
        return path
    manifest={'schema_version':1,'session_id':'v133_'+stage,'port':23210,
        'state_dir':str(base/'states'/stage),'report_dir':str(base/'stage_reports'/stage),
        'session_budget_seconds':None,
        'phase_budgets_seconds':{'plan':None,'sim':18000,'gates':None},'jobs':frozen}
    old.atomic_json(path,manifest)
    old.load_manifest(path)
    return path


def worker_result(job, result):
    result=dict(result)
    if 'phase' in result:
        result['execution_phase']=result.pop('phase')
    result.update({key:job[key] for key in ('kind','table','index') if key in job})
    return result


def paired_planning(manifest, records):
    groups={}
    for job in manifest['jobs']:
        if job.get('kind')!='plan':continue
        key=(job['family'],job['index'])
        row=groups.setdefault(key,{'family':key[0],'index':key[1]})
        prefix=job['table']+'_'
        record=records.get(job['id'],{})
        for field in ('status','makespan','bound','expanded','elapsed_seconds','reason','unknown','auxiliary_skills','output'):
            row[prefix+field]=record.get(field,'pending' if field=='status' else None)
    rows=[groups[key] for key in sorted(groups)]
    for row in rows:
        for alias,original in (('distance_prior','raw'),('ik','qualified')):
            for field in ('status','bound'):
                row[alias+'_'+field]=row.get(original+'_'+field)
        row['bound']=row.get('qualified_bound')
    return rows


def rebuild(stage, *, base=BASE):
    path=base/'manifests'/f'{stage}.json'
    if not path.exists():return
    manifest=old.load_manifest(path)
    records=old.Journal(manifest['state_dir']).load()
    old.rebuild_reports(manifest)
    if any(j.get('kind')=='plan' for j in manifest['jobs']):
        rows=paired_planning(manifest,records)
        stream=io.StringIO(newline='')
        columns=['family','index','bound','distance_prior_status','ik_status','distance_prior_bound','ik_bound']+[f'{table}_{field}' for table in ('raw','qualified')
            for field in ('status','makespan','bound','expanded','elapsed_seconds','reason','unknown','auxiliary_skills','output')]
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        directory=base/'stage_reports'/stage
        old.atomic_text(directory/'paired_planning_200.csv',stream.getvalue())
        old.atomic_json(directory/'paired_planning_200.json',rows)
        if stage==selected_stage('planning200',base=base):
            old.atomic_text(base/'planning_200.csv',stream.getvalue())
            old.atomic_json(base/'planning_200.json',rows)
    old.atomic_json(base/'stage_reports'/stage/'completion_counts.json',{
        'stage':stage,'requested':len(manifest['jobs']),'recorded':len(records),
        'passed':sum(r.get('passed') is True for r in records.values()),
        'pending':sum(j['id'] not in records for j in manifest['jobs'])})


def process_factory(command, log_path, start_signal):
    command=list(command)
    command[3]=str(Path(__file__).resolve())
    specification=Path(command[-1])
    data=old.read_json(specification)
    require_motion_gate(data['job'])
    data['hard_deadline_epoch']=time.time()+data['job']['timeout_seconds']
    old.atomic_json(specification,data)
    return old.OwnedProcess(command,log_path,start_signal)


def run_stage(stage, *, base=BASE):
    manifest=old.load_manifest(base/'manifests'/f'{stage}.json')
    for job in manifest['jobs']:
        if job.get('motion_code_sha256') and job['motion_code_sha256']!=motion_fingerprint():
            raise RuntimeError('已有真实阶段的运动代码指纹改变，保留旧记录并新建attempt')
        if job.get('planning_code_sha256') and job['planning_code_sha256']!=planning_fingerprint():
            raise RuntimeError('规划代码或标定参数改变，保留旧结果并新建attempt')
    phases={j['phase'] for j in manifest['jobs']}
    if len(phases)!=1:raise ValueError('每个v133阶段须为单一任务类型')
    state=old.run_phase(manifest,next(iter(phases)),environment_check=environment_check,
                        process_factory=process_factory,result_adapter=adapt_result)
    stopped=shared_stop_reason(base)
    if stopped:
        state.update(global_stop=stopped,status='global_stop')
        old.save_state(manifest,state)
    rebuild(stage,base=base)
    records=old.Journal(manifest['state_dir']).load()
    for job in manifest['jobs']:
        row=records.get(job['id'],{})
        if (job.get('kind')=='gate' and job.get('test_name')==old.LIVE_TESTS[0]
                and row.get('passed') is True and row.get('motion_code_sha256')):
            old.atomic_json(BASE/'last_motion_gate.json',{
                'passed':True,'source':row.get('output'),'job_id':job['id'],
                'motion_code_sha256':row['motion_code_sha256'],'completed_at':row.get('completed_at')})
    append_report(f'## 后台阶段 {stage}\n\n记录时间 {old.utc_now()}；状态 {state["status"]}；'
        f'请求{len(manifest["jobs"])}，已提交{len(records)}，通过{sum(r.get("passed") is True for r in records.values())}。'
        f'global_stop={state.get("global_stop")}。\n'
        f'清单 `v133/manifests/{stage}.json`；逐项事实 `v133/states/{stage}/completed.jsonl`；'
        f'阶段CSV `v133/stage_reports/{stage}/`。重启同一stage跳过已完成ID。')
    return state


def plan_job(job, deadline):
    from skillnet.core.engine import Engine
    from skillnet.domains.blocks import load_problem
    from skillnet.planning.search import search
    began=time.monotonic()
    engine=Engine(load_problem(job['task'],ROOT/'skills/blocks_v1.toml'))
    result=search(engine,time_limit=max(.001,deadline-time.monotonic()-1.))
    bound=getattr(result,'suboptimality_bound',None)
    if bound is not None and not math.isfinite(bound):bound=None
    steps=[{'action':str(s.action),'skill':s.action.name,'args':list(s.action.args),
            'start':s.start,'end':s.end,'state_before':[str(f) for f in sorted(s.state)]} for s in result.steps]
    skills=sorted({s.action.name for s in result.steps})
    auxiliary=[name for name in skills if name in {'handover','support_on','support_off','detach','co_pick','co_insert'}]
    for step in result.steps:
        bindings=dict(zip(step.action.skill.params,step.action.args))
        loc=engine.hooks.locations.get(bindings.get('loc'),{})
        slot=engine.hooks.slots.get(bindings.get('slot'),{})
        if loc.get('kind')=='buffer' or slot.get('temporary'):
            auxiliary.append(step.action.name)
    value={'task':engine.problem.name if hasattr(engine,'problem') else Path(job['task']).stem,
        'status':result.status,'passed':result.status=='solved','makespan':result.makespan,
        'bound':bound,'suboptimality_bound':bound,'expanded':result.expanded,'generated':result.generated,
        'elapsed_seconds':time.monotonic()-began,'search_elapsed_seconds':result.elapsed,
        'rejected':result.rejected,'unknown':[str(f) for f in sorted(result.unknown)],
        'schedule':steps,'skills':skills,'auxiliary_skills':sorted(set(auxiliary)),
        'auxiliary':bool(auxiliary),'proposal_enabled':False,'witness_loaded':False,
        'stages':list(getattr(result,'stages',())),
        'failure_phase':None if result.status=='solved' else 'planning',
        'reason':None if result.status=='solved' else result.status+': '+', '.join(map(str,sorted(result.unknown)))}
    value['search_stages']=value['stages']
    old.atomic_json(Path(job['output'])/'plan.json',value)
    return value


def worker(specification):
    data=old.read_json(old.confined(specification))
    waiting=time.monotonic()
    while not Path(data['start_signal']).exists():
        if time.monotonic()-waiting>30:raise RuntimeError('worker未获所有权握手')
        time.sleep(.03)
    job=data['job'];began=time.monotonic()
    deadline=began+max(.001,data['hard_deadline_epoch']-time.time()-.75)
    try:
        if job.get('task') and old.sha256(Path(job['task']))!=job['task_sha256']:
            raise ValueError('输入SHA改变')
        kind=job.get('kind',job['phase'])
        require_motion_gate(job)
        fingerprint=motion_fingerprint() if kind in {'qualify','sim','gate'} else None
        if kind=='plan':result=plan_job(job,deadline)
        elif kind=='qualify':
            from simbridge.qualifier import qualify_task
            result=qualify_task(job['task'],out=job['output'],timeout_wall_s=max(.001,deadline-time.monotonic()),
                                deadline=deadline,port=23210)
            result.setdefault('passed',result.get('status')=='completed' and result.get('simulator_closed') is True)
        elif kind=='sim':result=old.sim_job(job)
        elif kind=='gate':result=old.gate_job(job)
        else:raise ValueError('未知worker类型 '+kind)
        if fingerprint is not None:result['motion_code_sha256']=fingerprint
    except BaseException as exc:
        result={'status':'worker_failed','passed':False,'failure_phase':'worker',
                'reason':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
    result=worker_result(job,result)
    result['worker_wall_seconds']=time.monotonic()-began
    result.setdefault('wall_seconds',result['worker_wall_seconds'])
    result['definition_sha256']=job['definition_sha256']
    old.atomic_json(data['result_path'],result)
    return 0 if result.get('passed') else 1


def prepare_qualify(seed=20261115):
    from gen.generator import generate
    jobs=[]
    for family in FAMILIES:
        for index in range(40):
            name=f'gen_{family.lower()}_s{seed}_{index:03d}'
            task=BASE/'generated'/f'{name}.toml'
            if not task.exists():generate(family,seed,index,BASE/'generated')
            jobs.append({'id':f'q-{family.lower()}-{index:03d}','phase':'sim','kind':'qualify',
                'family':family,'index':index,'task':str(task),'timeout_seconds':60.,
                'output':str(BASE/'qualification'/family/f'{index:03d}')})
    return prepare('qualify200',jobs,base=BASE)


def prepare_planning(seed=20261115, *, stage=None, source_manifest=None):
    stage=stage or selected_stage('planning200',base=BASE)
    if Path(stage).name!=stage or stage in {'.','..'}:raise ValueError('无效规划阶段名称')
    if stage!='planning200' or source_manifest is not None:
        from tools.v133_experiments import load_selection
        selection=load_selection(BASE)
        source_path=Path(source_manifest) if source_manifest else BASE/'manifests/planning200.json'
        source=old.load_manifest(source_path)
        original=[j for j in source['jobs'] if j.get('kind')=='plan']
        keys={(j.get('family'),j.get('index'),j.get('table')) for j in original}
        expected={(f,i,t) for f in FAMILIES for i in range(40) for t in ('raw','qualified')}
        if len(original)!=400 or keys!=expected:raise ValueError('原试跑不是同一完整400输入集合')
        source_sha=old.sha256(source_path)
        if selection.get('source_manifest_sha256') not in (None,source_sha):
            raise ValueError('原试跑清单SHA改变，拒绝重用')
        cohort=input_cohort_sha256(original)
        if selection.get('input_cohort_sha256') not in (None,cohort):
            raise ValueError('新试跑输入集合与选择证明不一致')
        jobs=[]
        for previous in original:
            if not previous.get('task') or float(previous['timeout_seconds'])!=60.:
                raise ValueError('重跑必须逐项复用已完成资格输入与原60秒预算')
            job={key:previous[key] for key in ('family','index','table','task','timeout_seconds')}
            job.update(id=stage+'-'+previous['id'],phase='plan',kind='plan',
                output=str(BASE/'experiments'/stage/'planning'/previous['family']/f"{previous['index']:03d}"/previous['table']),
                source_trial=previous['id'],source_stage=source_path.stem,
                source_manifest_sha256=source_sha,input_cohort_sha256=cohort,
                experiment_selection_sha256=selection.get('selection_sha256'))
            jobs.append(job)
        path=prepare(stage,jobs,base=BASE)
        if input_cohort_sha256(old.load_manifest(path)['jobs'])!=cohort:
            raise ValueError('新清单输入集合发生变化')
        return path
    from gen.geometry import distance_prior_copy
    from simbridge.taskio import write_task_data
    jobs=[]
    for family in FAMILIES:
        for index in range(40):
            name=f'gen_{family.lower()}_s{seed}_{index:03d}'
            source=BASE/'generated'/f'{name}.toml'
            prior=distance_prior_copy(tomllib.loads(source.read_text(encoding='utf-8-sig')))
            prior_path=BASE/'distance_prior'/f'{name}.distance_prior.toml'
            if prior_path.exists():
                if tomllib.loads(prior_path.read_text(encoding='utf-8-sig'))!=prior:
                    raise ValueError('已准备距离先验布局/能力表改变，禁止覆盖：'+str(prior_path))
            else:write_task_data(prior_path,prior)
            for table in ('raw','qualified'):
                task=(prior_path if table=='raw' else
                      BASE/'qualification'/family/f'{index:03d}'/name/f'{name}.qualified.toml')
                job={'id':f'p-{table}-{family.lower()}-{index:03d}','phase':'plan','kind':'plan',
                    'family':family,'index':index,'table':table,'task':str(task),'timeout_seconds':60.,
                    'output':str(BASE/'planning'/family/f'{index:03d}'/table)}
                if not task.is_file():
                    job.pop('task')
                    job['skip_reason']='资格任务没有提交有效副本；不把原始表冒充IK表，其余独立工作继续'
                jobs.append(job)
    return prepare(stage,jobs,base=BASE)


def prepare_native(stage, *,attempt=1, tasks=None):
    if attempt<1 or attempt>3:raise ValueError('同一问题最多3次尝试')
    original_stage=stage
    if attempt>1:stage=stage+f'_attempt{attempt}'
    jobs=[]
    if original_stage in {'d_gate','d_gates','gates'}:
        tests=(old.LIVE_TESTS[:1] if original_stage=='d_gate' else
               old.LIVE_TESTS[3:] if original_stage=='d_gates' else old.LIVE_TESTS)
        for name in tests:
            short=name.rsplit('.',1)[-1].removeprefix('test_')
            jobs.append({'id':stage+'-'+short,'phase':'gates','kind':'gate','test_name':name,
                         'output':str(BASE/stage/short),'timeout_seconds':1200.,
                         'motion_code_sha256':motion_fingerprint()})
    elif original_stage=='d_e2e':
        for name in (tasks or ('s3a_cantilever_sim','s3a_platform_sim')):
            jobs.append({'id':stage+'-'+name,'phase':'sim','kind':'sim',
                'task':str(ROOT/'tasks/sim'/f'{name}.toml'),
                'output':str(BASE/stage/name),'timeout_seconds':1200.})
    else:raise ValueError('未知真实阶段')
    return prepare(stage,jobs,base=BASE)


def prepare_sim20(seed=20261116, *, planning_stage=None, stage=None):
    from tools.v133_experiments import load_selection
    selection=load_selection(BASE)
    planning_stage=planning_stage or selected_stage('planning200',base=BASE,selection=selection)
    stage=stage or selected_stage('sim20',base=BASE,selection=selection)
    manifest=old.load_manifest(BASE/'manifests'/f'{planning_stage}.json')
    records=old.Journal(manifest['state_dir']).load();jobs=[];selection={}
    for number,family in enumerate(FAMILIES):
        candidates=[]
        for job in manifest['jobs']:
            row=records.get(job['id'],{})
            if (job.get('family')!=family or job.get('table')!='qualified' or row.get('status')!='solved'
                    or row.get('passed') is not True or row.get('definition_sha256')!=job['definition_sha256']):continue
            plan_path=Path(job['output'])/'plan.json'
            if not plan_path.exists():continue
            proof=old.read_json(plan_path)
            if proof.get('status')!='solved' or proof.get('proposal_enabled') is not False or proof.get('witness_loaded') is not False:continue
            if old.sha256(Path(job['task']))!=job['task_sha256']:raise ValueError('已解IK任务SHA改变')
            candidates.append(job)
        candidates.sort(key=lambda j:j['id'])
        chosen=random.Random(seed+number).sample(candidates,min(4,len(candidates)))
        selection[family]=[job['id'] for job in chosen]
        for index in range(4):
            directory=BASE if stage=='sim20' else BASE/'experiments'/stage
            job={'id':f'{stage}-{family.lower()}-{index}','phase':'sim','kind':'sim','family':family,
                 'output':str(directory/'sim20'/family/str(index)),'timeout_seconds':1200.,
                 'source_planning_stage':planning_stage,'experiment_selection_sha256':selection.get('selection_sha256')}
            if index<len(chosen):
                proof=Path(chosen[index]['output'])/'plan.json'
                job.update(task=chosen[index]['task'],source_plan=chosen[index]['id'],
                           source_plan_path=str(proof),source_plan_sha256=old.sha256(proof))
            else:job['skip_reason']=f'{family} qualified-solved仅{len(candidates)}例，无法补齐4；不以先验或见证补位'
            jobs.append(job)
    path=prepare(stage,jobs,base=BASE)
    old.atomic_json(BASE/'sim20_selection.json',{'seed':seed,'selection':selection})
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description='v1.3.3逐任务持久化、后台可续跑批处理')
    parser.add_argument('--stage')
    parser.add_argument('--prepare',choices=('qualify200','planning200','sim20','d_gate','d_e2e','d_gates','gates'))
    parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker)
    if args.prepare:
        action={'qualify200':prepare_qualify,'planning200':prepare_planning,'sim20':prepare_sim20}.get(args.prepare)
        print(action() if action else prepare_native(args.prepare,attempt=args.attempt));return 0
    if not args.stage:parser.error('需要--stage或--prepare')
    stage=selected_stage(args.stage,base=BASE)
    if args.rebuild:rebuild(stage);return 0
    state=run_stage(stage)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
    return 0 if state['status']=='phase_complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
