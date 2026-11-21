"""Continue qualify200 -> paired planning -> sim20 -> gates after D closes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import overnight_batch as old
from tools import v133_batch as batch
from tools import v133_guard as guard
from tools import v133_experiments as experiments

BASE=batch.BASE
STAGES=('qualify200','planning200','sim20','gates')


class GlobalStop(RuntimeError):pass


def global_stop_reason(base=BASE):
    return batch.shared_stop_reason(base)


def verify_selection_regressions(selection, *, project=ROOT):
    if not selection.get('selection_present'):return
    expected=selection.get('planning_code_sha256')
    if expected!=batch.planning_fingerprint():
        raise RuntimeError('所选实验规划核心指纹与当前版本不一致')
    proofs={}
    for name in ('full','original11'):
        value=selection.get('regressions',{}).get(name)
        if not value:raise RuntimeError('所选新核心缺少回归证明：'+name)
        path=(Path(project)/value).resolve()
        if not path.is_relative_to(Path(project).resolve()) or not path.is_file():
            raise RuntimeError('所选回归证明缺失或越界：'+name)
        proof=old.read_json(path)
        if proof.get('planning_code_sha256',proof.get('code_sha256'))!=expected:
            raise RuntimeError('回归证明的规划核心指纹不匹配：'+name)
        proofs[name]=proof
    if proofs['full'].get('status')!='passed':raise RuntimeError('新核心完整默认集合尚未通过')
    for module in ('tests.test_v11_search','tests.test_v11_admissibility','tests.test_v11_ordering'):
        rows=[r for r in proofs['full'].get('per_test',[]) if r.get('id','').startswith(module+'.')]
        if not rows or not all(r.get('status')=='passed' for r in rows):
            raise RuntimeError('新核心v1.1逐测证明未通过：'+module)
    rows=proofs['original11'].get('rows',[])
    if proofs['original11'].get('passed') is not True or len(rows)!=11 or not all(r.get('passed') is True for r in rows):
        raise RuntimeError('新核心11原任务独立重放尚未通过')


def readiness(*, selection=None, base=BASE):
    stopped=global_stop_reason()
    if stopped:raise GlobalStop(stopped)
    path=base/'physical_stage_outcome.json'
    if not path.exists() or old.read_json(path).get('finished') is not True:
        raise RuntimeError('D与其真实阶段尚未结束，不启动后续批量')
    for label in ('abc','e'):
        path=ROOT/'out'/f'v133_{label}_full.json'
        if not path.exists() or old.read_json(path).get('status')!='passed':
            raise RuntimeError('规划修改后的完整默认回归尚未通过：'+label)
    batch.require_motion_gate({'kind':'qualify'})
    verify_selection_regressions(selection or experiments.load_selection(base))


def prepare(stage, *, physical_stage=None, selection=None):
    selection=selection or experiments.load_selection(BASE)
    physical_stage=physical_stage or experiments.resolve_stage(stage,BASE,selection=selection)
    if stage=='planning200':return batch.prepare_planning(stage=physical_stage)
    if stage=='sim20':
        return batch.prepare_sim20(stage=physical_stage,
            planning_stage=experiments.resolve_stage('planning200',BASE,selection=selection))
    if stage=='qualify200':return batch.prepare_qualify()
    if stage=='gates' and physical_stage=='gates':return batch.prepare_native('gates')
    raise ValueError('不支持的实际阶段映射：'+stage+' -> '+physical_stage)


def machine_check():
    stopped=global_stop_reason()
    if stopped:return {'passed':False,'global_stop':stopped}
    value=guard.check()
    value['c_free_bytes']=shutil.disk_usage('C:\\').free
    value['passed']=value['passed'] and value['c_free_bytes']>=old.MIN_C_FREE
    return value


def report():
    from tools import v133_summary
    return v133_summary.rebuild()


def run_pipeline(*,base=BASE,preparer=None,runner=None,checker=None,reporter=None,ready=None):
    base=Path(base)
    selection=experiments.load_selection(base)
    runner=runner or batch.run_stage
    checker=checker or machine_check;reporter=reporter or report
    ready=ready or (lambda:readiness(selection=selection,base=base))
    path=(base/'experiments'/selection['selection_id']/'workflow_state.json'
          if selection.get('selection_present') else base/'workflow_state.json')
    cursor=old.read_json(path) if path.exists() else {'status':'ready','stages':{},'created_at':old.utc_now(),
        'selection_sha256':selection.get('selection_sha256')}
    try:
        if cursor.get('global_stop'):raise GlobalStop(cursor['global_stop'])
        if cursor.get('selection_sha256')!=selection.get('selection_sha256'):
            raise GlobalStop('既有命名空间的实验选择SHA改变')
        ready()
        cursor['status']='running'
        for stage in STAGES:
            current_selection=experiments.load_selection(base)
            if current_selection.get('selection_sha256')!=selection.get('selection_sha256'):
                raise GlobalStop('运行中的实验选择SHA改变，拒绝混合批次')
            check=checker()
            if not check['passed']:raise GlobalStop(check.get('global_stop') or '归档/保护校验不一致或C盘低于5GiB')
            physical=experiments.resolve_stage(stage,base,selection=selection)
            if preparer is None:prepare(stage,physical_stage=physical,selection=selection)
            else:preparer(physical)
            outcome=runner(physical)
            cursor['stages'][stage]={'status':outcome['status'],'physical_stage':physical,'updated_at':old.utc_now()}
            cursor['updated_at']=old.utc_now();old.atomic_json(path,cursor)
            reporter()
            if outcome.get('global_stop'):raise GlobalStop(outcome['global_stop'])
            if outcome['status']=='environment_blocked':
                raise RuntimeError('环境被占用，保留断点且不改变外部进程')
        final=checker();old.atomic_json(base/'archive_check.json',final)
        if not final['passed']:raise GlobalStop('最终归档/保护校验不一致或C盘低于5GiB')
        cursor.update(status='finished',reason=None)
    except GlobalStop as error:
        cursor.update(status='global_stop',global_stop=str(error),reason=str(error))
        old.atomic_json(base/'global_stop.json',{'global_stop':str(error),'updated_at':old.utc_now()})
    except Exception as error:
        cursor.update(status='interrupted',reason=f'{type(error).__name__}: {error}')
    finally:
        cursor['updated_at']=old.utc_now();old.atomic_json(path,cursor)
        reporter()
    return cursor


def main():
    parser=argparse.ArgumentParser(description='v133后半程后台可续跑监督器')
    parser.add_argument('--rebuild',action='store_true',help='只重建报告，不运行实验')
    args=parser.parse_args()
    if args.rebuild:report();return 0
    with old.execution_lock(BASE/'workflow.lock'):
        result=run_pipeline()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['status']=='finished' else 2


if __name__=='__main__':raise SystemExit(main())
