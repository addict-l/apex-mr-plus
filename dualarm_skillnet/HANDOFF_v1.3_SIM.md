# v1.3 交接单：接入真实运动后端，在 CoppeliaSim 中跑通完整链路，并验证泛化性

请在 `D:\APEX_MR_Plus\dualarm_skillnet\` 下工作。本轮目标有两个：

1. **跑通完整链路**：任务文件 → 规划（已有）→ 真实运动规划（IK、RRT-Connect、碰撞检查）→ 多模态 TPG → CoppeliaSim 动力学执行 → 验收。
2. **验证泛化性**：用任务生成器批量生成积木任务，在规划层大规模验证，在仿真里抽样验证。

小模型部分本轮**不动**：保留现有接口和实验，不新增工作。

不要修改 `SPEC.md`、`FIX_v1.1.md`、`HANDOFF_v1.2_LLM.md`、`FIX_v1.2.1.md` 和本文件；本轮改动写进 `README.md` 新增的“v1.3 修订”一节。

---

## 1. 本轮的工作规则（覆盖 SPEC 第 0 节中冲突的条款）

| 事项 | 规则 |
|---|---|
| 改动范围 | 只改 `dualarm_skillnet/` 内的文件。**绝对不改动** `apex_mr_scene1/`、`apex_mr_scene3/` 及其结果、场景、备份文件，它们是已验收的归档 |
| 复用 APEX-MR 代码 | 允许，但只能**复制**到本项目的 `apexmr/` 目录（见第 3 节），不能 `import` 原目录。每个复制的文件在 `apexmr/PROVENANCE.json` 中记录来源路径和 SHA256；修改过的文件要写明改了什么 |
| 第三方依赖 | 核心包 `skillnet/` 继续只用标准库。新增的 `simbridge/` 和 `apexmr/` 允许使用 `numpy`、`scipy`、`pyzmq`、`cbor2`、`Pillow`、`matplotlib`；允许用 `pip install` 安装缺失的这几项，不装别的。CoppeliaSim 的 Python 远程客户端从安装目录加载（`sim_connection.py` 已这样做），不单独安装 |
| 仿真器 | CoppeliaSim Edu 已安装在 `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu`。允许以无界面方式启动和关闭由本项目自己启动的实例（`sim_connection.Simulator` 已实现）。**不要**关闭不是本项目启动的 CoppeliaSim 窗口。远程接口端口固定用 **23210**，避开两个旧项目使用的端口 |
| 长时间运行 | 仿真执行可能需要几分钟到几十分钟。每个仿真任务设置墙钟超时（默认 20 分钟），超时记为 `timeout`，继续下一个，不要无限等待 |
| 磁盘 | 仿真输出写到 `results/sim_v1_3/`。录像可选，默认关闭；开启时每个任务一个 mp4，总量超过 2 GB 时停止录像 |
| git、网络 | 仍然禁止 git 操作；除 pip 安装上述依赖外不访问网络 |
| 停下报告 | CoppeliaSim 启动失败、远程接口连不上、依赖装不上、C 盘剩余不足 5 GB、同一问题重试 3 次仍失败 |

---

## 2. 背景：现有的两部分

### 2.1 本项目（规划层，已完成）
- `skillnet/`：9 个参数化技能，事件状态 A* 规划，校验补全，任务图，距离 cost。
- `skillnet/motion/base.py`：`MotionBackend` 接口（`validate(action, state)`、`qualify(fact)`），目前只有占位的 `NullBackend` 和测试用的 `ScriptedBackend`。
- `pipeline.py` 已经有两条反馈回路：运动失败 → 加 cut 重搜；能力未知 → `qualify` 后回填重搜。
- 每个技能定义了 `primitives`（接近、夹取、抬起、放下、对准、下压、拔出等），目前只用于显示。

### 2.2 APEX-MR 执行栈（在 `apex_mr_scene3/`，只读）
| 模块 | 作用 | 关键接口 |
|---|---|---|
| `src/sim_connection.py` | 启动、连接、关闭 CoppeliaSim | `Simulator(root, port).start(headless)`, `.connect() -> (client, sim)` |
| `apex/motion.py` + `apex/motion_backend.lua` | 在停止的几何模型里做 IK、RRT-Connect、碰撞检查 | `MotionPlanner.solve_ik / plan / validate_joint / validate_pair / sample_state_pairs / fk` |
| `apex/tpg.py` | 从参考任务构建多模态 TPG | `build_tpg(tasks, conflict_oracle)`；任务格式：`id, arm, kind, duration, dependencies, metadata, q_path(transit 必需), reference_order` |
| `apex/geometry.py`、`apex/fast_interarm.*` | 双臂运动冲突采样 | `SampledConflictOracle`, `verify_rollout` |
| `apex/program.py` | 速度、加速度、加加速度限制下的时间参数化 | `parameterize(tpg, vmax, amax, jmax)`, `refresh_schedule` |
| `apex/runtime.py` + `apex/controller.lua` | 把 TPG 和控制器嵌入场景并执行 | `install_controller(sim, cfg, tpg, scene_name)`, `execute(sim, client, cfg, outdir, ...)` |
| `apex/validation.py` | 执行结果验收 | `validate_execution(graph, execution, requirements)` |
| `apex/scene.py`、`src/build_scene.py` | 场景构建（双 UR5、工作台、底板、积木） | `build(sim, client, install)`, `resolve(sim)` |

控制器已支持的技能动作：`grasp`、`release`（放下并刚性吸合）、`release_buffer`、`install`、`support_start / support_end`（Scene1 版）、`release_arm_support`（Scene3 版）、`wait`。**不支持**：交接、拆下（解除吸合）、双臂同步共持。

`apex_mr_scene1/recovery/sim_demo.py` 里的 `DemoMotionCompiler` 是“抽象动作 → 运动段 → 轨迹”的现成范例，可参考其结构。

---

## 3. 目录结构（新增部分）

```
dualarm_skillnet/
  apexmr/                     # 从 apex_mr_scene3 复制的执行栈，含 PROVENANCE.json
    sim_connection.py  motion.py  motion_backend.lua  tpg.py  geometry.py
    fast_interarm.py  fast_interarm.lua  program.py  runtime.py  controller.lua
    validation.py  scene_helpers.py（从 scene.py / build_scene.py 抽取需要的函数）
  simbridge/                  # 本项目新写的桥接层（允许第三方依赖）
    __init__.py
    catalog.py                # 积木类型 → 物理尺寸、质量、抓取偏移（读 blocks_catalog.toml）
    workcell.py               # 由任务文件生成 CoppeliaSim 场景：工作台、底板、料区、中转位、交接点、积木、两台 UR5
    frames.py                 # 任务坐标 ↔ 世界坐标；位置、槽位 → 抓取位姿、放置位姿、接近位姿
    primitives.py             # 技能原语 → 带目标位姿的运动请求序列
    compiler.py               # 运动请求 → IK + RRT 轨迹 → APEX-MR 任务列表（含依赖）
    backend.py                # CoppeliaBackend(MotionBackend)：validate / qualify / evaluate
    executor.py               # TPG 构建、时间参数化、嵌入控制器、执行、验收
    runner.py                 # 单任务端到端入口；批量入口
    controller_ext.lua        # 对 controller.lua 的扩展：handover、detach
  skills/blocks_catalog.toml  # 积木物理目录
  gen/                        # 任务生成器
    generator.py  families.py
  results/sim_v1_3/
  tests/                      # 新增测试：不依赖仿真器的用 mock；真实仿真测试用环境变量门控
```

---

## 4. 第一部分：工作站与坐标系

### 4.1 物理目录 `skills/blocks_catalog.toml`
为任务文件中出现的每种积木类型（`2x4`、`tower_base`、`beam`、`pillar`、`counterweight`、`temp`、`cap` 等）定义：尺寸（米）、质量（千克）、颜色、抓取点相对积木中心的偏移和姿态、插装方向（默认竖直向下）。尺寸参考 Scene3：普通砖约 0.12 × 0.14 × 0.076 m，梁约 0.52 × 0.16 × 0.04 m。类型缺失时加载报错。

### 4.2 工作站生成 `simbridge/workcell.py`
由任务文件直接生成场景，**不手工建场景**：
- 工作台和底板沿用 Scene3 的高度：台面 z = 0.76，底板顶面 z = 0.772。两台 UR5 基座在 (±0.70, 0.13, 0.783)，可从 Scene3 的设计预览场景 `backups/design_preview/apex_mr_scene3.ttt` 加载后删除多余物体，或按 `apex/scene.py` 的方式重建；二选一，在 README 里写明。
- **坐标映射**：任务文件中的 `pos = (x, y, z)` 视为相对底板顶面中心的坐标，世界坐标 = `(x, y, 0.772 + z)`。料区、中转位、交接点在对应位置放一块平垫并贴标签；槽位按 `below` 关系叠放，层高取积木高度。
- 积木按 `[[block]]` 的 `at` / `installed_in` 放置，用目录里的尺寸和质量创建为动态刚体；初始已安装的积木按 Scene1/3 的方式刚性吸合。
- 场景保存到 `results/sim_v1_3/<任务名>/workcell.ttt`，并写出 `workcell.json`（所有物体的句柄别名、世界位姿、物理参数）。

### 4.3 连接模型
沿用 Scene1/3 的“在实测位姿处理想刚性吸合”：`insert` 完成时把积木固定到下层；`place` 到料区、中转位时不吸合。**`detach` 需要新增“解除吸合”**：在控制器里把积木从父体上解除固定、恢复为自由刚体，再由臂夹住抬起。临时支撑台顶部与被托积木之间只接触、不吸合。

---

## 5. 第二部分：技能到轨迹

### 5.1 原语 → 运动请求 `simbridge/primitives.py`
每个技能的 `primitives` 展开为一串**运动请求**，每个请求含：臂、类型（自由转移 / 直线段 / 技能动作）、目标位姿或关节目标、手持物体、场景中已放置物体的状态、速度档位（普通 / 接触慢速）。对照表：

| 技能 | 运动请求序列 |
|---|---|
| pick | 自由转移到位置上方 0.10 m（接近位）→ 直线下降到抓取位 → `grasp` → 直线上升 0.10 m |
| place | 自由转移到位置上方 → 直线下降到放置位 → `release` → 直线上升 |
| insert | 自由转移到槽位上方 → 直线下降到插装位（可带 0.03 m 的对准停顿）→ `install`（吸合）→ 直线上升 |
| detach | 自由转移到槽位上方 → 直线下降到抓取位 → `grasp` → `detach`（解除吸合）→ 直线上升 |
| handover | 交出臂：自由转移到交接点交出位；接收臂：自由转移到交接点接收位（两位姿由 `frames.py` 定义，夹爪相对）→ 接收臂 `grasp` → 交出臂 `release`（不吸合）→ 两臂各自后退 0.10 m。两臂的请求之间加显式依赖 |
| support_on | 自由转移到支撑点上方 → 直线下降到接触 → `support_start`（沿用 Scene1 的测力就绪逻辑）|
| support_off | `support_end` → 直线后退 |
| co_pick / co_insert | **本轮不实现**：`validate` 返回 INVALID，原因“共持尚未接入运动后端” |

所有回位：每只臂计划末尾加自由转移回 home。

### 5.2 运动编译 `simbridge/compiler.py`
- 对每个运动请求：接近位和抓取位用 `solve_ik`（带手持物体和已放置物体），自由转移用 `plan`（RRT-Connect，另一只臂取其**当时**的姿态作为障碍），直线段用 IK 插值并逐点 `validate_joint`。
- 顺序：按规划层给出的时间表，依次编译每一步；每一步开始时的臂姿态取上一步结束时的姿态；场景物体状态按规划层的 `state_before` 同步（哪些积木在哪、哪些已装好）。
- 输出：APEX-MR 任务列表（`id, arm, kind, duration, dependencies, metadata.action, q_path`），任务间依赖来自规划层的任务图（因果边、同臂边）加上交接的跨臂依赖。每个任务的 `metadata` 记录它来自哪个规划步骤。
- 预算：每次 IK 最多 N 个种子、每次 RRT 最多 T 秒（默认 10 秒），用完记为 UNKNOWN；明确碰撞或无解记为 INVALID，原因写明是哪一段、碰到什么。

### 5.3 真实运动后端 `simbridge/backend.py`
实现 `CoppeliaBackend(MotionBackend)`：
- `validate(action, state)`：编译该动作的全部运动请求；全部成功返回 VALID，并把编译结果缓存（键：动作、两臂起始姿态、手持物体、已放置物体集合）；失败返回 INVALID 或 UNKNOWN，`detail` 写明具体原因，供 pipeline 加 cut。
- `qualify(fact)`：`reach(arm, place)` → 对该位置的抓取位和接近位各做一次 IK，都成功为 VALID，明确失败为 INVALID，预算用尽为 UNKNOWN；`liftable(arm, block)` → 按目录质量与设定上限比较。
- `evaluate(action, state)`：返回编译后轨迹经时间参数化的实际耗时，第一版只记录、不参与搜索（见第 7.2 节）。
- 后端要能被 pipeline 直接传入，整条反馈回路（运动失败 → cut → 重搜；能力未知 → 验证 → 重搜）在真实后端上要能触发。

---

## 6. 第三部分：TPG 构建与执行 `simbridge/executor.py`
1. `build_tpg(tasks, SampledConflictOracle(...))` 构建多模态 TPG；`parameterize` 做时间参数化（沿用 vmax 0.60、amax 1.20、jmax 8.0）。
2. 扩展控制器：在复制的 `controller.lua` 基础上，新增 `handover`（接收臂 grasp 后交出臂 release）和 `detach`（解除吸合）两种技能动作，其余沿用。扩展写在 `controller_ext.lua`，或在 PROVENANCE 中注明对 `controller.lua` 的修改。
3. `install_controller` 嵌入 TPG 与控制器，保存为 `<任务名>/planned.ttt`；`execute` 执行；`validate_execution` 验收。
4. 验收要求按任务自动生成：每个目标槽位的积木位姿误差 ≤ 5 mm、姿态误差 ≤ 0.02 rad；所有 `installed` 目标成立；临时槽位清空；双臂空手回位；无碰撞或跟踪失败事件。
5. 输出 `<任务名>/`：`plan.json`（规划层）、`tasks.json`（编译后的参考任务）、`tpg.json`、`execution.json`、`validation.json`、`summary.json`（规划耗时、编译耗时、TPG 规模、名义工时、实际工时、每步估算耗时与实际耗时的对照、验收结论）。

### 6.1 端到端入口 `simbridge/runner.py` 与命令行
```bash
python -X utf8 -m simbridge run tasks/s1_reach_handover.toml [--port 23210] [--headless] [--video] [--out DIR]
python -X utf8 -m simbridge batch --manifest results/sim_v1_3/manifest.toml
```
`run` 完成：规划 → 建场景 → 用真实后端重新规划（含反馈回路）→ 编译 → TPG → 执行 → 验收 → 写汇总；任一阶段失败，写明阶段和原因并以非零退出码结束。

---

## 7. 第四部分：泛化验证

### 7.1 任务生成器 `gen/`
按五个任务族随机生成任务文件（固定随机种子，可复现）：

| 族 | 变化的量 | 系统要自己推出的 |
|---|---|---|
| F1 平铺取放 | 积木数 2–6，初始位置、目标位置随机，两臂可达区域随机划分 | 并行、就近选臂 |
| F2 跨区取放 | 至少一块积木的料位和目标位分属两臂的独占区；交接点位置随机 | 交接或中转 |
| F3 塔与墙 | 层数 2–5，每层 1–3 个槽位，同类型或异类型积木随机 | 自下而上顺序、选块 |
| F4 悬臂（臂托） | 悬出长度随机，配重槽位随机；两臂至少一只能托 | 托住与松开时机 |
| F5 悬臂（支撑台） | 两臂都托不到；临时槽位 1–3 层 | 搭台、拆台、归位 |

生成器要保证任务合法（通过加载器校验、初始状态稳定），并给每个任务写一句 `instruction`。

### 7.2 两层验证
- **规划层（大规模）**：每族生成 40 个任务，共 200 个，用现有规划器（占位后端）求解。记录：结论、完成时间、扩展状态数、规划耗时、用到的技能种类、是否出现了辅助步骤。**不改代码**的前提下统计各族求解率。
- **仿真层（抽样）**：每族取 4 个任务，共 20 个，用真实后端跑完整链路。记录：各阶段耗时、反馈回路触发次数及原因、编译失败率、执行成功率、名义工时与实际工时、每步估算耗时与实际耗时的对照。
- **cost 对照**：汇总估算耗时与实际耗时的比值分布；在 README 中给出结论（是否需要修正距离估算的参数或改用 `evaluate`），本轮不改 cost 模型。

### 7.3 输出
`results/sim_v1_3/planning_200.csv`、`sim_20.csv`、`summary.md`（中文：各族求解率、辅助步骤出现率、仿真成功率、失败原因分类、耗时对照、发现的问题），以及每个仿真任务的目录。

---

## 8. 验收标准（全部满足才算完成）

1. 原有 241 项测试全部通过；核心包 `skillnet/` 仍然只依赖标准库（用一条测试检查 `skillnet` 下没有第三方 import）。
2. **mock 测试**（不启动仿真器）：原语展开对 9 个技能各有一条测试；编译器在给定假 IK/RRT 结果时生成正确的任务依赖；`CoppeliaBackend` 对 IK 失败、RRT 超时、碰撞三种情况分别返回 INVALID / UNKNOWN / INVALID 并附原因；pipeline 在真实后端返回 INVALID 时加 cut 并重搜。
3. **真实仿真测试**（`SKILLNET_LIVE_SIM=1` 时运行）：以下 5 个示例任务端到端通过验收：`s1_parallel_place`、`s1_reach_handover`、`s2_tower`、`s3a_cantilever`、`s3a_platform`。其中 `s1_reach_handover` 证明交接可执行，`s3a_platform` 证明拆下可执行，`s3a_cantilever` 证明托住与松开可执行。
4. **反馈回路在真实后端上触发过至少一次**：构造一个任务（例如交接点放在某只臂够不着的位置，但能力表写成 VALID），运行后应出现“运动失败 → cut → 重搜 → 改走中转”的记录，并最终执行成功。
5. 泛化验证：200 个规划任务求解率按族报告，F1–F3 要求 100%，F4、F5 要求 ≥ 90%，未求解的任务逐个写明原因；20 个仿真任务执行成功率 ≥ 80%，失败的逐个写明阶段和原因。
6. 每个仿真任务的汇总里都有“估算耗时 vs 实际耗时”对照；`summary.md` 给出整体比值分布。
7. 真实仿真测试和 20 个仿真任务的总墙钟时间在 README 中记录。
8. `apex_mr_scene1/`、`apex_mr_scene3/` 的文件 SHA 在本轮前后一致（在 README 里给出校验方法和结果）。

---

## 9. 里程碑（按顺序，每个完成后运行完整测试集）

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 复制与连通 | 建 `apexmr/` 并写 PROVENANCE；安装依赖；用 23210 端口启动无界面 CoppeliaSim、连接、关闭；`workcell.py` 能从 `s1_parallel_place` 生成场景并保存 | 场景文件存在，`workcell.json` 中物体位姿与任务文件一致 |
| M2 原语与编译 | `catalog`、`frames`、`primitives`、`compiler`；对 `s1_parallel_place` 编译出全部任务 | mock 测试通过；真实 IK/RRT 对该任务全部成功 |
| M3 执行 | `executor`、控制器扩展（handover、detach）、`runner run` | `s1_parallel_place`、`s1_reach_handover` 端到端通过 |
| M4 真实后端与反馈 | `CoppeliaBackend` 接入 pipeline | 验收标准 2、4；`s2_tower`、`s3a_cantilever`、`s3a_platform` 端到端通过 |
| M5 泛化验证 | 生成器、200 个规划任务、20 个仿真任务、汇总 | 验收标准 5、6、7 |

## 10. 明确不做

- 不接入小模型，不改 `decompose/`。
- 不实现双臂同步共持的运动与执行。
- 不改 cost 模型（只记录估算与实际的对照）。
- 不做执行中的故障注入与在线恢复（Scene1 那类流程留到下一轮）。
- 不建力学模型；结构是否真的稳定由动力学仿真结果体现，规划层规则不变。

## 11. 执行要求

- 严格按里程碑顺序；M1 没有连通仿真器之前不写编译器。
- 每个真实仿真运行都保存完整证据（场景、TPG、执行日志、验收结果），不得只保留汇总。
- 遇到 APEX-MR 代码与本项目假设不一致的地方（例如坐标约定、吸合方式），以在 README 中写明的方式解决，不要悄悄改规划层的语义。
- 做不到的验收标准，不放宽、不绕开：停下来，在 README 里写明卡在哪里、尝试过什么、建议怎么办，然后报告。
