"""Read-only extraction of the retained formal movement baseline."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.overnight_batch import atomic_json, atomic_text

BASE = ROOT/'results/sim_v1_3/motion_v135'
OLD = ROOT/'results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2'


def main():
    rows = []
    for stage in ('sim20', 'walls3'):
        manifest = json.loads((ROOT/f'results/sim_v1_3/v134/manifests/{stage}.json').read_text(encoding='utf-8'))
        committed = {row['id']: row for line in (Path(manifest['state_dir'])/'completed.jsonl').read_text(encoding='utf-8').splitlines() if line.strip() for row in [json.loads(line)]}
        for job in manifest['jobs']:
            folder = Path(job['output'])/'simulation'/Path(job['task']).stem.removesuffix('.qualified')
            if not folder.exists():
                candidates = list((Path(job['output'])/'simulation').glob('*/summary.json'))
                if len(candidates) != 1:
                    raise ValueError(f'Non-unique retained output: {job["id"]}')
                folder = candidates[0].parent
            summary = (json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                       if (folder/'summary.json').exists() else committed[job['id']])
            backend = json.loads((folder/'motion_backend.json').read_text(encoding='utf-8')) if (folder/'motion_backend.json').exists() else {}
            failures = []
            for i, validation in enumerate(backend.get('validations', [])):
                if validation.get('status') == 'VALID':
                    continue
                failures.append({'index': i, **{k: validation.get(k) for k in
                    ('action', 'status', 'original_status', 'detail', 'classification', 'failed_segment',
                     'start_q', 'held', 'placed', 'ik_seed_limit', 'escalated', 'failure_evidence')}})
            rows.append({'stage': stage, 'id': job['id'], 'source_job_id': job.get('source_job_id'),
                'task': job['task'], 'task_sha256': summary['task_sha256'], 'source': str(folder),
                'summary': {k: summary.get(k) for k in ('status', 'passed', 'reason', 'failure_phase',
                    'real_planning_status', 'phase_seconds', 'wall_seconds')}, 'failures': failures,
                'feedback_rounds': backend.get('feedback_rounds', [])})
    atomic_json(BASE/'m1/baseline_cases.json', rows)
    lines = ['# M1 正式基线抽取', '', '历史输入只读；本表未把有限候选失败作为全局不可达证明。', '',
        '|集合|任务|状态|阶段|运动失败记录数|', '|---|---|---|---|---:|']
    lines.extend(f'|{r["stage"]}|{r["id"]}|{r["summary"]["status"]}|{r["summary"].get("failure_phase") or ""}|{len(r["failures"])}|' for r in rows)
    lines.extend(['', '## 已证实与待验证', '',
        '- 已证实：较早 seed 的碰撞会覆盖末次 IK_branch_discontinuity；公共后端红灯测试见 out/motion_v135_m1_classification_red.log。',
        '- 已证实：资格鉴定分别验证 grasp/approach 点 IK，未验证连续连接；F2/012 两解最大 wrapped 差 2.994851 rad。执行 approach 与资格解一致，另一臂持物和姿态不同。',
        '- 已证实：单次 transit 选一个 IK 终点；动作后段失败没有整体候选回退。当前侧抓俯仰 60°，未发现坐标变换错误证据。',
        '- 已证实：三次动作规避后 reach 降级；碰撞 cut 可导致全局 no_plan；PipelineResult/schedule.status 可不一致。',
        '- 已证实：最终 compile/home 不在动作反馈内，名义时间 evaluate 再编译一次；037 的固定 home_B 有 Finger_pos–b1_3 目标碰撞证据。',
        '- 已证实：两个 timeout 均在 TPG graph 几何检查，已有 skip_ordered_pairs、增量闭包、精确状态 cache 和最多128样本batch；实际 batch 只有约11样本，RPC约一万次。',
        '- 待验证：完整动作候选/端点回溯能救回多少真实场景；TPG耗时中服务端、传输、序列化比例；固定home目标碰撞是否能通过合法动作候选解决。', '',
        '## 单案例命令', '', '以下均为诊断预算，非正式1200秒统计；输出在 motion_v135/m1，保留完整上下文。', '', '```powershell',
        'python -X utf8 tools/motion_v135_replay.py --source results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/sim20/06/simulation/gen_f2_s20261115_014 --mode action --record-index 1 --diagnostic-seconds 120',
        'python -X utf8 tools/motion_v135_replay.py --source results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/sim20/07/simulation/gen_f2_s20261115_012 --mode action --record-index 1 --diagnostic-seconds 120',
        'python -X utf8 tools/motion_v135_replay.py --source results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/sim20/08/simulation/gen_f3_s20261115_021 --mode action --record-index 45 --diagnostic-seconds 120',
        'python -X utf8 tools/motion_v135_replay.py --source results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/walls3/037/simulation/gen_f3_s20261115_037 --mode home --home-arm B --diagnostic-seconds 120',
        'python -X utf8 tools/motion_v135_tpg_profile.py --case 00 --seconds 120 --out results/sim_v1_3/motion_v135/m1/tpg00_a1',
        'python -X utf8 tools/motion_v135_tpg_profile.py --case 02 --seconds 120 --out results/sim_v1_3/motion_v135/m1/tpg02_a1', '```', ''])
    atomic_text(BASE/'m1/baseline_cases.md', '\n'.join(lines))
    print(json.dumps({'cases': len(rows), 'failures': sum(len(r['failures']) for r in rows)}))


if __name__ == '__main__':
    main()
