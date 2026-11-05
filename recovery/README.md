# 问题①：从持物与持续支撑状态恢复的离线原型

本原型实现了“已知能力变化 → 失效相关证据 → 从当前状态搜索恢复动作 → 构造 TPG → 回放核查”的闭环，闭环运行在**离散状态与二维准静态模型**中。它借鉴 D-ITAGS 和 WRAP 的思路，不是两篇论文原代码的拼接，也不声称完整复现或保留其理论界。

第一版不启动 CoppeliaSim，不提供关节运动、避碰或动力学执行结果。`VALID` 能力来自示例输入的声明；没有从关节限位自动推导资格，也没有伪造 IK 或 RRT 路径。输出统一标记 `executable_in_coppeliasim: false`。

## 运行

在仓库根目录，使用 Python 3.11（最低 3.10），安装已有依赖中的 NumPy、SciPy 即可：

```sh
python -m pip install numpy scipy
python -m recovery --case support_swap
python -m recovery --suite --policy all
python -m unittest discover -s tests -v
```

macOS 可以直接运行这些 Python 命令，不需要 Windows `.cmd` 或模拟器。若本机已有带 NumPy/SciPy 的环境，直接使用该环境。

生成可编辑的完整 JSON 输入，再运行自定义输入：

```sh
python -m recovery --case support_swap --export-scenario results/recovery/support_swap_input.json
python -m recovery --scenario results/recovery/support_swap_input.json
```

`--no-reuse` 在事件后清空全部证据，用于和定向缓存失效对照；`--max-expansions`、`--time-limit-s` 控制每次搜索预算。默认预算为 20,000 次状态扩展、15 秒。CLI 先从同一快照计算事件前的名义计划以预热证据，再应用事件；**名义计划没有被执行，物理快照不会重置或前移**。

```sh
python -m recovery --case support_swap --no-reuse --output results/recovery/cold
```

单次输出位于 `results/recovery/<case>/<policy>/`：

- `scenario.json`：完整输入，含初始快照、能力事实、二维物理参数和事件。
- `result.json`：搜索动作、事件后快照、抽象时序、缓存统计与独立回放验收。
- `tpg.json`：复用 `normal.tpg.build_tpg()` 构造的抽象时间图。没有 `q_path`，不可以直接交给 Lua 控制器。
- `results/recovery/summary.json`：本次所有场景的简要对照。

CLI 退出码 0 表示返回计划/安全前缀通过抽象回放核查，**不表示每个问题都已解出**。`no_plan`、`unknown`、`partial` 是合法实验结果；回放核查失败时退出码为 2。

## 具体复用与新增内容

| 模块 | 本次使用方式 |
|---|---|
| 原有 `normal.allocation.solve_allocation` | 在声明的候选上调用原 MILP，提供安装/支撑角色的排序提示；不修改原目标函数，也不拿其代价冒充恢复 makespan |
| 原有 `normal.tpg.build_tpg / simulate_tpg` | 复用同臂资源顺序、前驱完成释放、延迟传播；为双臂交接增加双臂预约屏障 |
| 新增 `model.py` | 保存物体位置、持物者、持续支撑、已完成任务；拒绝丢弃进行中技能的快照 |
| 新增 `planner.py` | 枚举通用动作，以事件为搜索状态，选择动作、执行者与开始时间 |
| 新增 `oracles.py` | 三值能力查询、二维受力 LP、带事实依赖的证据缓存 |
| 新增 `schedule.py` | 从原始快照重新回放计划，检查占用、因果、支撑和受力，不信任搜索结果中的通过标记 |

原 MILP 只提供等代价搜索中的角色偏好，**不是融合后的联合求解器**；状态搜索可选择其他绑定。它的固定任务/负载模型不能决定是否需要暂存、交接和支撑接替，这些由新增的状态搜索处理。第一版没有强行把辅助动作塞进旧 MILP。

`normal/` 的 Python 算法、Lua 执行器及已有 `.ttt` 场景均保持原样；只更新了 README 的入口说明。后续可以把新动作交给原 `MotionPlanner` 和经过改造的运动编译器，但本版本没有完成该接线。

## 搜索在做什么

状态包含物体的唯一位置、每只机械臂的持续支撑关系、完成记录，以及正在执行的动作和剩余时间。动作来自以下有限原语，而不是按场景指定一条恢复链：

```text
pick / place / handover / acquire_support / release_support / install / work
```

`work` 代表只占用本臂、只写入完成记录的独立工作，输入必须显式声明 `independent_of_assembly: true`。它不能代表任意几何上独立的机器人运动。

每个搜索状态可以启动满足条件的动作，也可以推进到下一次完成事件。代价是推进的模型时间，因此允许另一只臂在等待期间执行独立工作。持物、持续支撑和运行中动作都会占用机械臂；`acquire_support` 动作完成后，支撑占用一直持续到 `release_support` 完成。

为了让第一版的并行语义可核查，所有会改变装配、物体位置或支撑的技能共用一个互斥资源；只有声明为独立的 `work` 能与其重叠。这是保守建模选择，不是空间避碰证明。

能力查询采用 `VALID / INVALID / UNKNOWN`；未声明的条目为 `UNKNOWN`。未知、求解器超时、搜索预算耗尽都不能作为不可行证明；未知结果不会缓存成永久禁用边。事件仅能修改能力与物理参数，不能改写已完成记录或物体位置。

事件后只失效依赖变化事实的缓存项；未受影响项在依赖签名相同时复用。二维 LP 缓存保守依赖整个物理模型，物理参数变化会重新核查这部分证据。任务影响报告沿前序依赖传播，但**剩余窗口的搜索从头构造**，并未复用原 D-ITAGS 的 open/closed/pruned 搜索前沿。

搜索结果：

- `solved`：全部输入任务在有限模型中完成，双臂空手且支撑已释放。
- `partial`：返回有进展、已结束全部进行中动作的安全前缀，仍有任务或资源释放目标未完成。
- `no_plan`：声明的有限原语/候选中没有找到完整解，搜索未遇到未知检查且预算未耗尽；不代表真实连续机器人问题不可行。
- `unknown`：没有完整解或有进展前缀，且有未知检查或预算限制。
- `invalid_snapshot`：输入的结构在声明的二维模型中没有静态平衡，不能据此启动恢复。

存在有进展前缀时，`partial` 优先于 `unknown`；剩余不确定原因见 `search.unknown_capabilities` 与 `search.budget_exhausted`。输出的 `predicted_makespan_s` 只计返回计划的模型时间，不能把部分计划的短时间与完整解直接比较。

## 二维受力检查

当前物理模型只有一个已连接结构、一个有界连接反力/反力矩，以及若干向上的点支撑。所有坐标以该连接为原点，采用 X-Z 平面内的统一约定：

```text
ΣFx = 0
ΣFz = 0
Σ(x Fz - z Fx) + M_connection + M_external = 0
0 ≤ Fz_contact ≤ normal_limit
|Fx_contact| ≤ friction × Fz_contact
```

连接的水平、竖直反力和力矩均有界；重力、已装物体载荷及当前安装动作的向下插装力写入平衡方程。安装完成才加入新连接的承载变化。LP 用 `scipy.optimize.linprog / HiGHS`，返回力见证，并独立检查平衡、摩擦和边界残差。

对建立支撑、释放支撑、安装都检查动作前/加载时/完成后的声明状态。新支撑只能在 `acquire_support` 完成后参与承载；原支撑释放完成后，剩余接触必须独立承载。这里假定抽象技能能够保持已有接触；连续接触过渡、实际反力确认、三维稳定性、夹持摩擦、关节力矩和动力学冲击都还需要运动层及执行反馈验证。

## 已运行的合成对照

2026-11-05 在本机 Python 3.11 / NumPy / SciPy 环境运行。完整记录见 [../examples/recovery/suite_summary.json](../examples/recovery/suite_summary.json)，可用上面的 `--suite --policy all` 重新生成。原有 15 项测试和新增 26 项测试共 41 项通过。

三种策略使用相同的快照、能力声明、物理模型与验收：`fixed` 保持原任务执行者；`reassign` 允许换执行者、普通源位抓取与支撑建立/解除；`recovery` 再允许暂存、缓存取物与交接。它们是本仓库的功能消融，**不标作原版 D-ITAGS、WRAP 或 APEX-MR 的论文复现对照**。

| 输入场景 | fixed | reassign | recovery |
|---|---|---|---|
| 原例：A 的任务未完成，B 另有独立任务 | partial：先完成 block3 | partial | partial |
| 尚未持物，安装能力变化 | no_plan | solved，模型时间 4 s | solved，4 s |
| B 持物，必须先交接给 A | no_plan | no_plan | solved，4.5 s |
| A 支撑、B 持物，需要支撑接替 | no_plan | no_plan | solved，7 s |
| B 的支撑承载不足 | no_plan | no_plan | no_plan |
| 当前角色关系相同，但没有缓存位 | no_plan | no_plan | no_plan |
| A 的替代安装资格仍未知 | unknown | unknown | unknown |
| 原支撑位承载不足，另有可用接触位 | no_plan | no_plan | solved，7 s |

四个可解恢复场景、两个有限模型无解场景、一个未知场景和一个部分完成场景均通过返回计划/安全前缀的独立验收；这不是八次真实机器人成功执行。回归测试还覆盖了安装/支撑延迟、交接双臂预约、提前释放拒绝、持续占用、能力/物理缓存失效、已完成任务保留与摩擦约束。

在 `support_swap` 示例中，搜索生成：

```text
B 暂存连接砖
B 建立替代支撑（A 继续支撑）
A 释放旧支撑
A 取回连接砖
A 安装（B 全程保持支撑）
B 释放支撑
```

动作链来自搜索。输入与完整输出分别见 [support_swap_input.json](../examples/recovery/support_swap_input.json)、[support_swap_result.json](../examples/recovery/support_swap_result.json)。时间由示例技能时长给定，不是仿真测量或性能提升证据。

## 下一步接入真实场景的接口

1. 在安全技能边界导出当前关节、持物相对位姿、实际连接、支撑确认及已完成任务；当前 `normal.controller.lua.getState()` 的信息不足以构造该快照。
2. 用停止的独立规划场景同步快照。原 `normal.motion.MotionPlanner` 要求仿真停止，不能直接控制一个正在运行的动力学实例。
3. 通过原 `solve_ik / plan / validate_joint / validate_pair` 验证带持物和持续支撑的每个新转移，取代当前声明资格；超时保留 UNKNOWN。
4. 编译真实关节路径，检查对方静止支撑状态与整个支撑切换过程，再构造带几何验证的剩余 TPG。
5. 增加运行时恢复协议和新计划安装入口；新接触由传感器确认后才放行旧支撑退出。

目前未实现以上在线接线，原 `pipeline_manifest.json` 中的 `autonomous_failure_recovery: false` 仍然适用于正常模式。

## 思路来源

- [D-ITAGS](https://arxiv.org/abs/2209.13092)：能力/任务变化后的定向修复、分配与调度的交错处理。本版本只借鉴事件影响和证据复用，没有搬入其分配搜索或原证明。
- [WRAP](https://arxiv.org/abs/2609.29407)：显式装配/持物状态、辅助操作搜索及受力可行性检查。本版本使用更小的动作集合和二维 LP，没有复现其六维 wrench 模型、运动后端或全部搜索策略。
- 现有 [normal/README.md](../normal/README.md)：本仓库的正常模式分配器、TPG 与运动后端，作为尽量复用的工程基础。
