# db77489 恢复算法离线定量评测报告

- 生成时间：2026-11-23 21:59:27（本机时区）
- 基准脚本：`recovery_demo/run_recovery_benchmark.py` (sha256 `f53fc0ae5c113c5e13583c65026575bb53986bf861abf36877061c359d0d97f3`)
- 厂商快照：`recovery_demo/vendor_db77489/dualarm_skillnet` (upstream commit `db774898ea73b59d4a9327ff49149712f38d2524`)
- 运行耗时：292.9 s
- 隔离：仅写 `recovery_demo/benchmark/`；未启动 CoppeliaSim；未改动冻结基线、`mode_a/`、`mode_b/`、`recovery_demo/bundle/` 与既有产物

## 1. 实验配置

| 项目 | 取值 |
|---|---|
| 域 | blocks-v1 积木域，双臂 A/B |
| 恢复预算 (time_limit) | 30.0 s |
| max_expansions | 200,000 |
| 套件 A 离线包 | 12 个检查点 / 8 个备份 / 4898 个候选图状态 |
| 套件 A 离线准备耗时 | 8.002 s |
| 套件 A 动作域 | gate base_allowed ∩ 故障允许动作 |
| 套件 B 离线准备 | checkpoints=8, backups=2, backup_seconds=1.0, graph_limit=5000 |
| 套件 B 恢复调用 | recovery.recover(..., compare=True)（原版实现） |
| 冷启动基线 | 相同故障域 search()，time_limit=30 s，无候选图 |
| post_first_solution_limit | None（恢复与冷启动一致） |
| 并行度 | 套件 A 主进程顺序执行；套件 B 11 个任务各一个无窗口 (CREATE_NO_WINDOW) 子进程并行执行 |
| 可复现性 | 套件 A 完全确定性；套件 B 的离线准备按 wall-clock 预算 （backup_seconds=1.0 s/检查点）生成备份，并行负载会影响备份条数，重复运行间 L1/L2 层级计数可能浮动 1–2 例 |
| 计时口径 | 恢复决策用 perf_counter 外测；套件 B 在子进程内包裹 recovery.search 精确切分恢复/冷启动两段（只读插桩，不改源码）；冷启动首解取 vendor search 内部计时（Windows monotonic 量化约 15.6 ms） |

## 2. 总体统计摘要

| 指标 | 全部用例 |
|---|---|
| 用例总数 | 116 |
| 恢复成功 | 25 (21.6%) |
| L1 backup 真命中 | 17 (14.7%) |
| L2 graph_search | 8 (6.9%) |
| 无解 (no_plan) | 78 |
| 超时 (budget) | 13 |
| 其他状态 | 0 |
| 冷启动可解用例 | 25 |
| 其中恢复成功 | 25 (100.0%) |
| L1 决策延迟中位 (ms) | 55.11 |
| L1 决策延迟 P95 (ms) | 75.10 |
| L1 决策延迟（模型已加载）中位 (ms) | 9.11 |
| L1 首解中位加速（模型已加载） | 45.9×，n=9 |
| L1 首解中位加速（端到端含加载） | 2.68× |
| L1 多备份择优平均工时节省 | 33.35% |
| L2 平均状态复用率 | 92.53% |
| L2 复用出边 / 新生成行 | 15337 / 41 |
| L2 首解中位加速（模型已加载） | 1.00×，n=8 |
| L2 首解中位加速（端到端含加载） | 0.97× |
| L2 总决策中位加速 | 1.03× |
| L2 与冷启动最优性一致率 | 100.0% |

| 子集 | 用例 | 恢复成功 | L1 | L2 | 无解 | 超时 |
|---|---|---|---|---|---|---|
| 套件 A（乐高桥） | 18 | 10 | 8 | 2 | 8 | 0 |
| 套件 B（11 标准任务） | 98 | 15 | 9 | 6 | 70 | 13 |
| 全部 | 116 | 25 | 17 | 8 | 78 | 13 |

L1 命中备份类型分布：backup=17。

### 2.1 按故障类型

| 故障类型 | 用例 | 成功 | L1 | L2 | 无解 | 超时 |
|---|---|---|---|---|---|---|
| 全部 | 116 | 25 | 17 | 8 | 78 | 13 |
| action | 61 | 17 | 10 | 7 | 37 | 7 |
| skill | 54 | 7 | 7 | 0 | 41 | 6 |
| composite | 1 | 1 | 0 | 1 | 0 | 0 |

### 2.2 冷启动可解子集上的恢复率（剔除故障本身不可恢复的用例）

| 子集 | 冷启动可解 | 其中恢复成功 | 恢复率 | L1 | L2 |
|---|---|---|---|---|---|
| 套件 A（乐高桥） | 10 | 10 | 100.0% | 8 | 2 |
| 套件 B（11 标准任务） | 15 | 15 | 100.0% | 9 | 6 |
| 全部 | 25 | 25 | 100.0% | 17 | 8 |

## 3. 套件 A：乐高桥 9 块砖联合任务

复用 `recovery_demo/bundle/plans.json` 的 12 个停稳检查点与 `run_offline_gate.py` 的 `build_engine`/`base_allowed`；每个检查点注入 nominal 下一步具体接地动作失效，并对无法连带封死桥面 B 臂取放的步骤注入该臂该技能失效；另加 checkpoint 8 对称跨区 reach 复合故障。

| 检查点 | 故障类型 | 故障内容 | 阶段 | 状态 | 决策 (ms) | 决策核心 (ms) | 首解 (s) | 冷启动首解 (s) | 首解加速（核心） | 首解加速（端到端） | 复用状态/新行 | makespan | 冷 makespan | 最优一致 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | action | pick(B, Blue_01, stock_Blue_01) | graph_search | solved | 4871.54 | 4808.36 | 0.2699 | 0.2500 | 1.23× | 1.16× | 2844/1 | 187.295 | 187.295 | yes |
| 1 | action | insert(B, Blue_01, seat_Blue_01) | graph_search | no_plan | 80.42 | 28.45 | — | — | — | — | 0/0 | — | — | n/a |
| 3 | action | insert(A, Red_02, seat_Red_02) | graph_search | no_plan | 96.31 | 27.97 | — | — | — | — | 0/0 | — | — | n/a |
| 3 | skill | A/insert | graph_search | no_plan | 133.47 | 49.48 | — | — | — | — | 0/0 | — | — | n/a |
| 5 | action | pick(A, Red_01, stock_Red_01) | backup | solved | 55.11 | 3.44 | 0.0551 | 0.1250 | 36.3× | 3.51× | — | 154.523 | 154.523 | yes |
| 5 | skill | A/pick | backup | solved | 76.70 | 3.49 | 0.0767 | 0.1560 | 44.8× | 2.68× | — | 214.359 | 214.359 | yes |
| 7 | action | insert(A, Red_01, seat_Red_01) | graph_search | no_plan | 78.89 | 28.82 | — | — | — | — | 0/0 | — | — | n/a |
| 7 | skill | A/insert | graph_search | no_plan | 90.20 | 25.50 | — | — | — | — | 0/0 | — | — | n/a |
| 8 | action | pick(A, Orange_01, stock_Orange_01) | backup | solved | 69.86 | 3.10 | 0.0699 | 0.1100 | 35.5× | 2.19× | — | 104.437 | 104.437 | yes |
| 8 | composite | reach(A, seat_Blue_01)=INVALID, reach(A, seat_Blue_02)=INVALID, reach(A, seat_Orange_01)=INVALID, reach(A, seat_Orange_02)=INVALID, reach(A, stock_Blue_01)=INVALID, reach(A, stock_Blue_02)=INVALID, reach(A, stock_Green_01)=INVALID, reach(A, stock_Green_02)=INVALID, reach(B, seat_Green_01)=INVALID, reach(B, seat_Green_02)=INVALID, reach(B, seat_Red_01)=INVALID, reach(B, seat_Red_02)=INVALID, reach(B, stock_Orange_01)=INVALID, reach(B, stock_Orange_02)=INVALID | graph_search | solved | 2154.79 | 2109.39 | 0.2054 | 0.1880 | 1.21× | 1.17× | 105/1 | 186.027 | 186.027 | yes |
| 8 | skill | A/pick | backup | solved | 67.27 | 2.96 | 0.0673 | 0.1720 | 58.2× | 3.19× | — | 163.802 | 163.802 | yes |
| 9 | action | insert(A, Orange_01, seat_Orange_01) | backup | solved | 66.22 | 3.40 | 0.0662 | 0.1560 | 45.9× | 3.01× | — | 105.955 | 105.955 | yes |
| 9 | skill | A/insert | backup | solved | 73.93 | 3.27 | 0.0739 | 0.1570 | 48.0× | 2.95× | — | 165.271 | 165.271 | yes |
| 11 | action | insert(B, Orange_02, seat_Orange_02) | graph_search | no_plan | 97.09 | 30.40 | — | — | — | — | 0/0 | — | — | n/a |
| 13 | action | pick(B, Green_02, stock_Green_02) | backup | solved | 66.55 | 2.11 | 0.0666 | 0.1250 | 59.3× | 2.55× | — | 69.831 | 69.831 | yes |
| 15 | action | insert(B, Green_02, seat_Green_02) | backup | solved | 74.71 | 2.37 | 0.0747 | 0.1090 | 46.1× | 2.08× | — | 52.253 | 52.253 | yes |
| 16 | action | pick(B, Bridge_deck_12x4, stock_Bridge_deck_12x4) | graph_search | no_plan | 90.87 | 44.17 | — | — | — | — | 0/0 | — | — | n/a |
| 17 | action | insert(B, Bridge_deck_12x4, seat_Bridge_deck_12x4) | graph_search | no_plan | 94.22 | 48.05 | — | — | — | — | 0/0 | — | — | n/a |

技能级故障跳过用例（按设计会直接封死 B 臂对桥面的取放）：

- checkpoint 0：skill `B/pick` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 1：skill `B/insert` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 11：skill `B/insert` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 13：skill `B/pick` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 15：skill `B/insert` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 16：skill `B/pick` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解
- checkpoint 17：skill `B/insert` — 屏蔽 B/pick 或 B/insert 会使单臂独占的桥面区域不可达，故障在构造上即无解

## 4. 套件 B：11 个标准任务

每个任务用原版 `recovery.prepare` 准备（参数见 §1），对每个已解检查点注入单动作失效与臂/技能失效，逐例调用原版 `recovery.recover(..., compare=True)`。标准任务的可达域多为单臂独占，因此大量故障在几何上不可恢复，恢复器在这些用例上正确返回 `no_plan`。

| 任务 | 准备状态 | 准备 (s) | 检查点 | 用例 | 成功 | L1 | L2 | 无解 | 超时 | L1 中位 (ms) | L2 复用率 (%) | L2 中位加速 | L2 最优一致 | 离线图行数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_far_handover | solved | 0.12 | 4 | 8 | 4 | 4 | 0 | 4 | 0 | 20.87 | — | — | — | 19 |
| s1_nearest_arm | solved | 0.08 | 2 | 4 | 3 | 3 | 0 | 1 | 0 | 18.05 | — | — | — | 18 |
| s1_parallel_place | solved | 0.13 | 2 | 4 | 0 | 0 | 0 | 4 | 0 | — | — | — | — | 8 |
| s1_reach_handover | solved | 0.09 | 3 | 6 | 2 | 2 | 0 | 4 | 0 | 51.24 | — | — | — | 22 |
| s2_tower | solved | 2.84 | 4 | 8 | 0 | 0 | 0 | 8 | 0 | — | — | — | — | 118 |
| s2_tower_same_type | solved | 5.49 | 3 | 6 | 2 | 0 | 2 | 1 | 3 | — | 95.0 | 1.13× | 100% | 1022 |
| s3a_cantilever | solved | 5.19 | 6 | 12 | 0 | 0 | 0 | 12 | 0 | — | — | — | — | 864 |
| s3a_cantilever_with_proposal | solved | 4.98 | 6 | 12 | 0 | 0 | 0 | 12 | 0 | — | — | — | — | 864 |
| s3a_platform | solved | 29.58 | 8 | 16 | 2 | 0 | 2 | 9 | 5 | — | 87.8 | 0.99× | 100% | 4159 |
| s3a_platform_demo_needs_support | solved | 0.48 | 3 | 6 | 0 | 0 | 0 | 6 | 0 | — | — | — | — | 65 |
| s3a_platform_with_proposal | solved | 29.43 | 8 | 16 | 2 | 0 | 2 | 9 | 5 | — | 87.8 | 0.99× | 100% | 4177 |

套件 B 每一条用例的完整指标见 `recovery_benchmark.csv`；11 份子进程日志见 `recovery_demo/benchmark/logs/`。

## 5. 指标定义

| 指标 | 定义 |
|---|---|
| L1（backup 真命中） | 故障后仅重放离线备份/相邻检查点计划即得到可行解（stage=backup, status=solved），含 nominal_suffix 与 backup |
| L2（graph_search） | 备份全部失效后加载候选图做增量搜索并得到可行解（stage=graph_search, status=solved） |
| 决策延迟（端到端） | 从收到故障快照到返回计划的墙钟时间（perf_counter），含引擎与离线包加载；套件 A 为进程内实测，套件 B 由 recover(compare=True) 外测并按内部 search 边界切分 |
| 决策延迟（模型已加载） | 端到端延迟减去一次模型构建时间（套件 A 实测；套件 B 用同任务 Engine(load_problem) 实测值近似） |
| 首解加速（核心） | 冷启动 search 内部 first_solution_seconds ÷ 恢复首解核心时间（两侧均从模型就绪起算）；冷启动首解低于 1 ms 的用例视为计时不可分辨，不计入中位，记为 n |
| 首解加速（端到端） | （模型加载 + 冷启动首解）÷ 恢复首解端到端时间，仅统计冷启动首解 ≥1 ms 的用例 |
| 多备份择优工时节省 | 对有效备份重放 ≥2 条的解，(最差有效备份 makespan − 选中 makespan) ÷ 最差有效备份 makespan |
| 状态复用率 | reused_states ÷ (reused_states + generated_rows)，候选图命中行占本次搜索展开行的比例 |
| 复用出边 / 新生成行 | CandidateGraph.stats() 的 reused_edges / generated_rows 汇总 |
| L2 总决策加速 | （模型加载 + 冷启动总时）÷ 恢复端到端延迟 |
| 最优性一致率 | L2 解 makespan 与冷启动解 makespan 相对误差 ≤1e-6 的比例 |
| 无解 / 超时 | status=no_plan（故障域内不可达目标）/ status=budget（30 s 或扩张预算耗尽） |

## 6. 主要结论

- 套件 A + B 共 116 例（A 18 例、B 98 例）；恢复成功 25 例（21.6%），L1 备份真命中 17 例（14.7%）、L2 图修补 8 例（6.9%）；无解 78 例、超时 13 例。
- 剔除故障本身不可恢复的用例后（冷启动可解 25 例），恢复成功率 100.0%（25/25），其中套件 A 为 100.0%、套件 B 为 100.0%。标准任务集的低总成功率来自单臂独占可达域与持件锁定这两类结构性无解，而非恢复器搜索预算不足。
- L1 决策延迟中位 55.11 ms、P95 75.10 ms（端到端含模型加载）；扣除模型加载后的核心延迟中位 9.11 ms；首解核心加速中位 45.9×（可测 n=9），端到端口径 2.68×；多备份择优平均节省工时 33.35%。
- L2 平均状态复用率 92.5%，累计复用出边 15337 条、新生成行 41 行；首解核心加速中位 1.00×（n=8），总决策中位加速 1.03×，最优性一致率 100.0%。图复用主要收益是避免重搜与保持最优性，而非提前首解；候选图上限 5000 时重生成行会抵消部分收益。
- 未恢复用例分布：action/budget=7, action/no_plan=37, skill/budget=6, skill/no_plan=41。经典模式是（i）持件臂的 insert 失效后另一臂既拿不到件、任务里也没有可回退的 place/handover 通道；（ii）单臂独占 stock 的 pick 失效后另一臂不可达。budget 集中在 s3a 平台类高分支任务（30 s / 200k 扩张预算耗尽）。

> 复现：`python -B -X utf8 recovery_demo/run_recovery_benchmark.py`；套件 B 每任务一个无窗口子进程，套件 A 在主进程执行，全部产物位于 `recovery_demo/benchmark/`。
