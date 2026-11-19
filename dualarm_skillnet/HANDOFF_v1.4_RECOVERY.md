# 双臂装配在线故障恢复交接单

接手目标是从故障后的实测状态继续完成原装配任务。开发依据是 [Claude 的 v1.4 恢复设计原稿](DESIGN_v1.4_RECOVERY.md)，请先读第 8 节，再读第 2、3、7 节。第 8 节记录了对早期草案的修订。本次交付的是恢复开发的算法起点，恢复模块和恢复成功率属于后续里程碑。

## 领取代码

首次检出可在 PowerShell 运行以下命令。在已有仓库中开发时，也可以从同一远端交接分支创建自己的工作分支。

```powershell
git clone --branch handoff/skillnet-adjacent-ik-c1-20261119 https://github.com/addict-l/apex-mr-plus.git apex-mr-recovery
cd apex-mr-recovery
git switch -c recovery/r-m1
cd dualarm_skillnet
python -B -X utf8 -m unittest tests.test_engine tests.test_bounded_search tests.test_blocks
```

上面的 21 项规划冒烟测试已在交接副本通过。使用 Python 3.11。R-M1 可以离线开展；进入原生仿真阶段前，再按 [仿真环境交接](HANDOFF_v1.3_SIM.md) 配置 CoppeliaSim 和依赖。

## 冻结基线与验证范围

- 采用 2026-11-19 的 `adjacent_ik_c1`。版本指纹为 `141e2ee6cd30103a2cd6633f45f4f392c12e27be7f8a59540b0fd91bea451de6`。125 个登记文件含规划器、运动后端及相应测试和工具。[逐文件 SHA-256](results/sim_v1_3/recovery_handoff_20261119/frozen_files.json)
- 该版继承固定底座碰撞登记，并含相邻动作段 IK 终点排序和预检超时后的回退。本版已有 F3_021、F4_039 两个正常入口原生执行与独立验收通过的记录。上一冻结版本的完整 sim20 为 13/20，其中 F1 和 F2 各 4/4、F3 为 3/4、F4 为 2/4、F5 为 0/4。两个版本分别统计，恢复任务另设分母。[基线验证记录](results/sim_v1_3/recovery_handoff_20261119/validation.json)
- 交接副本已复跑 88 项运动聚焦测试。复核命令为 `python -B -X utf8 -m unittest tests.test_adjacent_ik_deadline tests.test_adjacent_ik_precheck tests.test_motion_v135_action_candidates tests.test_motion_v135_backend_candidates tests.test_motion_v135_candidate_selection tests.test_motion_v135_qualification_hints tests.test_v13_compiler`。
- 本次复跑范围为上述 21 + 88 项。冻结快照还包含一份早期测试草稿 `tests/test_motion_v135_repairs_chain.py`，它引用的 `tools.motion_v135_repairs_chain` 模块尚未随该版本交付。

## 第一项交付：R-M1 离线规划恢复

1. 先做一个能重放的最小例子。选择 F1 平铺任务，在合法执行前缀之后构造一次低处滑落，积木平放在工作台新位置。记录原目标、故障前后持物关系、积木身份及新位姿。估计器应产生动态位置和一致的符号事实；恢复计划应从测得状态出发，保留此前仍成立的装配结果，并达到原目标。
2. 用已有 [Problem](skillnet/core/problem.py)、[规划流水线](skillnet/pipeline.py) 与 [从指定状态搜索的接口](skillnet/planning/search.py) 接入恢复。处理动态位置、能力更新、临时与持续失效、受保护的持物及支撑事实。重试按原稿默认最多 2 次。路径级失败的排除条件限定到实测状态、具体动作和路径版本。记录不可自动恢复、预算耗尽、求解失败各自的原因。
3. 扩展到原稿的 R1–R6。按固定种子生成 500 个符号恢复场景，F1–F5 各 100 个，覆盖适用的故障、时机和持续性。交付逐例清单、种子、可重放输入、恢复状态、计划和校验结果，以及分组求解率、工时增量和扩展数。原稿 R-M1 目标为 F1–F3 求解率 100%，F4–F5 至少 90%。首轮不占用仿真器。

## 后续顺序及协作

按 R-M1 → R-M2（注入与检测）→ R-M3（在线闭环）→ R-M4（矩阵验证）→ R-M5（汇总）串行推进。每一阶段提交实现、测试命令、结果与未决问题，再进入下一阶段。R-M2 以后依原稿核实 M4 前置条件；仿真时与正常装配开发约定实例及端口使用时段。

第一版采取双臂全停恢复。受影响臂立即安全停并维持已有夹持或支撑，另一臂完成当前节点后停；从测得状态重规划，从两臂当前关节配置重新编译，换 TPG 后再执行。按第 8 节，R1–R6 必做，R7 可选；首轮真实仿真 20 例，从正常装配已有端到端成功证据的任务中抽取，与全停重来基线对照，先知基线离线计算。40 例及只重试对照留到第二档。

后续恢复提交放在自己的开发分支。合入更新的正常规划算法时，单独记录所用基线提交，复跑相同恢复用例，便于区分正常规划变化和恢复逻辑变化。
