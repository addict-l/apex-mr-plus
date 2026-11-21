# v1.4 双臂装配在线故障恢复交接单

2026-11-20。恢复开发从固定 sim20 完整通过 15/20 的 `tpg_broadphase_sim20_a1` 起步。先读 [设计原稿第 8 节](DESIGN_v1.4_RECOVERY.md#8-已确认的决定2026-11-15)，再读第 2、3、7 节。第 8 节的已确认决定优先于早期草案。

## 领取代码与离线自检

在 Python 3.11 的 PowerShell 环境执行下面的命令。保留交接分支，在自己的开发分支完成恢复工作。

```powershell
git clone --branch handoff/skillnet-tpg-broadphase-15of20-20261120 https://github.com/addict-l/apex-mr-plus.git
Set-Location apex-mr-plus
git switch -c recovery/r-m1
Set-Location dualarm_skillnet
New-Item -ItemType Directory -Path out -Force | Out-Null
$frozen = Get-Content -LiteralPath handoffs/recovery_20261120/frozen_files.json -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($section in 'files', 'verified_test_fixtures', 'other_packaged_files', 'recovery_task_files') {
    foreach ($entry in $frozen.$section.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash -ne $entry.Value) {
            throw "SHA-256 不匹配 $($entry.Name)"
        }
    }
}
python -B -X utf8 -c "import hashlib,json; from pathlib import Path; m=json.loads(Path('handoffs/recovery_20261120/frozen_files.json').read_text(encoding='utf-8')); assert hashlib.sha256(json.dumps(m['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['fingerprint']; print(m['fingerprint'])"
python -B -X utf8 -m unittest -v tests.test_engine tests.test_bounded_search tests.test_blocks
$validation = Get-Content -LiteralPath handoffs/recovery_20261120/validation.json -Raw -Encoding UTF8 | ConvertFrom-Json
$modules = @($validation.offline_regression.groups.PSObject.Properties | ForEach-Object { $_.Value.modules } | ForEach-Object { "tests.$_" })
python -B -X utf8 -m unittest -v @modules
```

离线门控应得到 **21/21 规划冒烟**和 **208/208 聚焦回归**。冒烟与聚焦集合有重叠，分别计数。聚焦集合覆盖早停、v1.1 最优性与可容许性、规划核心、运动编译与后端、TPG 粗筛、采样器生命周期和停机几何验证。测试仅使用检出代码和随附夹具。原生阶段按 [仿真环境交接](HANDOFF_v1.3_SIM.md) 安装 CoppeliaSim，确认实际模块、场景、仿真器实例及端口，再开始故障注入。

## 冻结基线与仿真样本

版本指纹为 `87dd723bcee7ae34d95daa3359ac911c0383555a428546bb6f0cd7f5cac99e61`。它对应 125 个原冻结文件，包含底座碰撞登记、相邻段 IK 预检、双网格路径校验、在线首解后 2 秒早停，以及 TPG 扫掠 AABB、十字架采样和精确单臂正缓存。补充测试、夹具和任务输入另列 SHA-256。[冻结文件清单](handoffs/recovery_20261120/frozen_files.json)

普通取放门控 `s1_parallel_place` 完成 108/108 节点，独立验收 0 违规。固定 sim20 在 1,200 秒单任务上限下完成 20/20 提交，15 例原生执行及独立验收均通过。15 个任务构成 **R-M3 和 R-M4 的抽样池**。[验证记录与输入哈希](handoffs/recovery_20261120/validation.json)

| 任务族 | 通过数 | 端到端通过的任务 |
|---|---:|---|
| F1 平铺 | 4/4 | F1_014、F1_028、F1_024、F1_035 |
| F2 跨区交接 | 4/4 | F2_009、F2_001、F2_014、F2_012 |
| F3 塔墙 | 4/4 | F3_021、F3_020、F3_006、F3_000 |
| F4 臂托悬臂 | 3/4 | F4_030、F4_033、F4_039 |
| F5 支撑台悬臂 | 0/4 | 无合格样本 |
| **总计** | **15/20** | **75%** |

各任务的 `task_archive_path` 对应检出树中的原资格表 TOML。复现历史场景时，从受保护实验归档按清单取得原 `qualify_log.json`，校验两者 SHA-256，并在自己独立结果目录中将资格日志放在任务文件旁。发布提交只附资格日志的来源和 SHA-256。五个未通过任务及精确停点已登记，F5 真机覆盖仍待验证；当前 20 个恢复事件可从 F1–F4 合格任务抽取不同故障和时机，五任务族覆盖项须单列未满足。R-M1 仍按 F1–F5 五族分别验收。

## 实现约束

- 生产恢复按测量、事实更新、能力与约束更新、重搜、从实测关节重编译、换 TPG 执行的同一路径处理。只消费测得状态、技能前提、设备与接触条件。场景名、积木名、槽位名和预设某只臂只可出现在夹具或分析中。
- R1 抓空或滑落、R2 技能或臂持续失效、R3 放置或插装偏差、R4 支撑丢失、R5 交接失败、R6 路径阻塞必做。R7 速度退化可选。第一版受影响臂立即安全停并维持夹持或支撑，另一臂完成当前节点后停。两臂稳定后读取同一快照，冻结合法已完成节点，从两臂实测关节重编译并换 TPG。
- 掉落积木仅在工作台上平放时自动恢复，偏航不限。R1 在抬起后 5 cm 内或转移前半程注入。侧翻、叠在其它积木上或掉出台面时单列为不可自动恢复。
- R2 以技能级持续失效为主。整臂失效时，空手进入正常恢复；持物则将该块标记不可用，有同类备用块才换块，否则记录不可恢复；正在支撑则保持支撑有效，由另一臂完成装配，终态允许保持该支撑。状态估计与重搜保护仍有效的 `held` 与 `supporting` 事实及结构稳定性。
- 同一 `(arm, skill)` 的暂时性故障最多重试 2 次，之后升级为持续性失效，能力更新限定到已确认的作用范围。路径阻塞的排除键是 `(S_m, action, path_version)`，其中 `S_m` 是刚测得的状态。保留原装配目标和独立验收，单独记录预算耗尽、不可自动恢复、规划无解、编译未知和执行再次失败。

## R-M1 至 R-M5 验收

| 阶段 | 实施顺序与产物 | 验收 |
|---|---|---|
| R-M1 离线规划恢复 | 以 F1 合法前缀中的一次低处滑落起步，估计器从原目标、实测持物关系和新位姿生成动态位置与符号事实。恢复计划保留仍有效的已装配结果并到达原目标。复用 `Problem`、`pipeline.run` 和从指定状态搜索的接口。再用固定种子生成 500 个符号恢复场景，F1–F5 各 100 个，记录适用故障、时机、持续性、状态、计划、校验、扩展数和新增工时。 | 最小 F1 重放通过。F1–F3 各 100/100，F4–F5 各至少 90/100。附逐例 CSV、种子、原任务和恢复输入。 |
| R-M2 注入、检测与安全停 | 按原设计接入注入、监视和安全停。R1–R6 各至少注入一次，以无故障对照测误报并记录故障信号、`t_fault`、`t_detect`、两臂停稳和持物或支撑状态。先核实设计要求的 M4 装配前置。 | 每个注入的检测延迟 `t_detect - t_fault ≤ 0.5 s`，使用仿真时间。对照 0 误报，安全停过程保留有效的夹持和支撑。 |
| R-M3 原生在线闭环 | 从上表选 F1/F2 已通过任务，分别覆盖 R1 与 R2。用 CoppeliaSim 原生动力学执行注入、检测、两臂停稳、从实测状态重规划与重编译、换 TPG、恢复到原目标。 | R1、R2 在 F1、F2 都有成功的原生恢复证据。保存完整时间线、重编译起点、切换记录和独立验收。 |
| R-M4 原生 20 例矩阵 | 从合格池构造 20 个可复现的故障事件，R1–R6 各 3–4 个，覆盖适用的暂时性和持续性故障。对照同输入同注入的全停重来基线，与离线计算的故障先知时间下界。预先锁定场景与注入种子。 | 至少 16/20 完整执行并通过独立验收，按故障族和任务族分层。报告检测、重规划、恢复、完成及未受影响臂空闲时间，与两条基线逐例对照。按原设计核对在线用时是否处于先知下界与全停重来之间，并标出实测偏离。F5 覆盖缺口独立登记。扩至 40 例及只重试基线属于后续档。 |
| R-M5 汇总 | 汇总 `recovery_500.csv`、`recovery_sim_20.csv` 和 `summary.md`，附冻结指纹、种子、命令、逐例证据及失败原因。对比 Scene1 的 R2 持续和暂时失效参考 180.82 s、115.50 s。 | 每项门控给出通过状态、分子分母、计时口径和证据。不可恢复、未解、超时及缺测保留原分类。 |

按 R-M1 → R-M2 → R-M3 → R-M4 → R-M5 顺序提交证据。R-M1 可独立离线推进。后续仿真与正常装配开发协调 CoppeliaSim 实例、端口及时间窗，按照 [项目规则](AGENTS.md) 无窗启动和清理。

## 双线并行开发

恢复线优先新增 `simbridge/monitor.py`、`simbridge/estimator.py`、`simbridge/recovery.py`、注入或拼接文件及对应测试，尽量通过现有接口复用正常流水线。主线仍在优化 `simbridge/compiler.py`、`apexmr/geometry.py` 和 `skillnet/planning/`。共享接口确需改动时，先记录动机与调用边界，独立提交最小接线，并约定合并顺序。跨线同步时记录基线提交和指纹，用同一恢复用例复跑验证。恢复结果写入各自的新结果目录，冻结基线和历史成绩保留。
