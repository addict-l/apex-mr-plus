# 最近完整批次的测试结果

两次测量使用不同的冻结版本。当前交接算法 `adjacent_ik_c1` 的定向复验另见[版本记录](recovery_handoff_20261119/validation.json)。

| 任务族 | planning200 规划求解 | 固定 sim20 原生执行及独立验收 |
| --- | ---: | ---: |
| F1 | 40/40 | 4/4 |
| F2 | 40/40 | 4/4 |
| F3 | 36/40 | 3/4 |
| F4 | 40/40 | 2/4 |
| F5 | 40/40 | 0/4 |
| 合计 | 196/200（98%） | 13/20（65%） |

## planning200 · v1.3.4

`c_wall_v134/runtime_fix2` 的规划层批次共 200 个任务，196 个求解成功，2 个耗尽预算，2 个报告无方案。表格采用全部输入作分母。按 v1.3.4 验收口径，F3 排除两个已证明物理不可行的任务后为 36/38。每例的任务哈希、求解状态、耗时、搜索扩展量和调度可在结果中查阅。

[逐例 CSV](v134/planning_200.csv) · [逐例详细 JSON](v134/planning_200.json) · [失败分析](v134/planning_failure_analysis.md) · [验收口径](v134/user_section8_ruling.json)

## 固定 sim20 · mount_registry_a1

批次 `mount_registry_sim20_a1`，候选指纹 `81abae5d08e36e7b3e75a9c7b86daebaea4a05523626565bf420b204f5f8a0f7`。20 个任务均有终态。13 个通过原生执行及独立验收，4 个耗尽预算，2 个返回 unknown，1 个超时。同一组任务的 v1.3.4 基线为 8/20。

[逐例 CSV，含基线对照](motion_v135/diagnostics/mount_registry_sim20_a1/comparison.csv) · [统计与阻塞原因](motion_v135/diagnostics/mount_registry_sim20_a1/summary.json) · [20 条完成记录](motion_v135/diagnostics/mount_registry_sim20_a1/state/completed.jsonl) · [批次审计](motion_v135/diagnostics/mount_registry_sim20_a1/state/final_audit.json)

| 场景 | 终态 | 原始证据 |
| --- | --- | --- |
| F1_014 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_014/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_014/execution_validation.json) |
| F1_028 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_028/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_028/execution_validation.json) |
| F1_024 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_024/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_024/execution_validation.json) |
| F1_035 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_035/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F1_035/execution_validation.json) |
| F2_009 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_009/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_009/execution_validation.json) |
| F2_001 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_001/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_001/execution_validation.json) |
| F2_014 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_014/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_014/execution_validation.json) |
| F2_012 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_012/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F2_012/execution_validation.json) |
| F3_021 | `budget` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_021/summary.json) |
| F3_020 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_020/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_020/execution_validation.json) |
| F3_006 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_006/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_006/execution_validation.json) |
| F3_000 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_000/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F3_000/execution_validation.json) |
| F4_030 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_030/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_030/execution_validation.json) |
| F4_033 | `unknown` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_033/summary.json) |
| F4_039 | `passed` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_039/summary.json) · [独立验收](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_039/execution_validation.json) |
| F4_013 | `unknown` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F4_013/summary.json) |
| F5_028 | `budget` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F5_028/summary.json) |
| F5_037 | `budget` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F5_037/summary.json) |
| F5_033 | `timeout` | [超时收据](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F5_033/committed.json) |
| F5_024 | `budget` | [运行摘要](motion_v135/diagnostics/mount_registry_sim20_a1/cases/F5_024/summary.json) |

原始 JSON 和 CSV 保留采集机器的路径字段。上表提供仓库内可直接打开的证据链接。
