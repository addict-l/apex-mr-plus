# v1.3.2 进度交接（Claude 已完成部分 → Codex 接手）

本文件记录 `FIX_v1.3.2.md` 中**已经由另一方完成**的部分，Codex 接手时请从第 3 节开始，不要重做第 1 节的内容，也不要撤销这些改动。规则仍以 `HANDOFF_v1.3_SIM.md` 第 1 节、`FIX_v1.3.1.md`、`FIX_v1.3.2.md` 为准。

## 1. 已完成（不要重做）

| 项 | 文件 | 内容 |
|---|---|---|
| 侧抓俯仰（FIX_v1.3.2 §2.1） | `simbridge/frames.py` | 新增 `SIDE_GRASP_PITCH = 45°`；`side_payload(block, arm, pitch=SIDE_GRASP_PITCH)` 把工具 z 绕手指闭合轴向下转 45°。TCP 仍距积木中心 `size_y/2 + 0.006`，积木保持水平，取料/交接/放置全程同一绑定。`handover_poses` 偏航选择、`handover_retreat`、目录顶抓路径均未改 |
| 腕部净空测试（§2.3） | `tests/test_handover_frames.py` | 4 项全部通过（原来 3 项失败）。腕部估计点 z ≥ 底板 + 0.12 m、到基座 ≤ 0.65 m；绑定往返一致 |
| 旧断言更新 | `tests/test_v13_primitives.py` | 两处写死"水平侧抓"的断言改为俯仰几何（工具 z 水平投影相对、z 分量 = −sin45°；退让在 x 方向位移 0.10·cos45°、z 方向 +0.10·sin45°） |
| RRT 两阶段搜索（§3） | `simbridge/compiler.py` | 构造参数 `local_box_margin=.6`、`local_box_time=4.`。`transit` 先在局部关节盒内搜 4 秒，失败后用同一 10 秒预算的剩余时间做全范围搜索；明确 INVALID 不重试。证据条目新增 `search_stage`（`local_box` / `global`），失败时 `evidence={"attempts": [...], "last_answer": ...}`，错误信息写明两阶段均未找到解 |
| 两阶段 mock 测试 | `tests/test_v13_compiler.py` | 新增 `TwoStageTransitSearchTests` 4 项：局部失败→全范围成功、两阶段都失败记 UNKNOWN 并保留两次尝试、INVALID 不重试、可关闭局部盒。原 `solve_time == 10` 断言改为"局部阶段 4 秒 + joint_bound_margin=.6"；`test_complete_compile_failure_does_not_leave_successful_prefix` 改为按"第二条自由运动"失败而不是按调用次数 |

聚焦测试结果（机器有游戏负载时运行，仍全部通过）：`tests.test_v13_primitives tests.test_v13_compiler tests.test_handover_frames tests.test_v13_runner tests.test_v13_executor tests.test_v13_workcell tests.test_v13_bootstrap tests.test_pipeline tests.test_motion` → **87 项通过**，3.0 秒。

## 2. 接手时先知道的一个新问题：臂托位姿不可行

`Frames.support_pose(slot, arm)` 当前返回：TCP 在槽位中心正下方（梁底面高度），姿态为单位四元数（A）或绕 z 转 π（B），即**工具 z 竖直向上、托板在 TCP 上方、腕部在梁下方约 0.20 m**。梁在一块柱子高度时底面只有 0.848 m，腕部会落到 0.65 m，低于台面（0.760 m）。这个位姿在任何现实梁高下都无法通过 IK，M4 的 `s3a_*` 任务一定会卡在这里。

归档 Scene3 的做法（`apex_mr_scene3/apex/scene.py` 第 148–151 行，只读参考）：**工具 z 水平、朝向梁**，把 A 的 `PadForceSensor`/`SupportShoe` 相对掌心绕 x 转 90° 改成水平托板，TCP 从侧面伸到梁底下，由托板上表面接触梁底。腕部与托板同高，只要梁底面 ≥ 台面 + 约 0.09 m 就可行。

M4 时请这样处理（写进 README，不要悄悄改）：
- `simbridge/workcell.py` / `build_arms` 之后，把**两只臂**的托板都按 Scene3 的方式转成水平（A、B 各做一次，B 的方向镜像）。
- `support_pose` 改为水平接近：TCP 位于梁底面下方（托板厚度 + 间隙），从**托举臂自己基座一侧**水平伸入到支撑点正下方；支撑点取梁的**悬出端**而不是槽位中心（槽位中心可能在柱子正上方）。`support_retreat` 改为沿 −工具 z 水平退出再下降。
- 仿真版悬臂任务的梁底面高度 ≥ 底板 + 0.152 m（柱子用两块 `pillar_type` 叠放，或单块放在两层平台上），避免腕部贴着底板。

## 3. 剩下要做的（按顺序）

1. **`tasks/sim/`**（FIX_v1.3.1 §3.2）：`s2_tower_sim.toml`、`s3a_cantilever_sim.toml`、`s3a_platform_sim.toml`。约束：料位/中转/交接参考点两两水平间距 ≥ 0.20 m；料垫（0.18 × 0.18）完全落在底板之外、台面之内；料位到所属臂基座 (±0.70, 0.13) 的水平距离 0.30–0.65 m；塔放在 (0.0, 0.12) 附近；悬臂几何物理一致（柱偏离梁中心、悬出 ≥ 0.18 m、配重在柱一侧的梁上、临时台在悬出端正下方且顶面与柱顶等高）。建议料位：A 侧 (−0.50, −0.20)、(−0.50, −0.42)、(−0.72, −0.30)、(−0.72, −0.52)，B 侧镜像。写一个纯几何检查工具 `tools/check_sim_layout.py`（积木、料垫、底板、臂座互不重叠；叠放不超出底板；间距与可达范围），对三个文件运行并保存结果。规划层对三个文件要能求解；机器繁忙触发 budget 时记录、等空闲重跑。
2. **`simbridge/backend.py`**（HANDOFF_v1.3 §5.3 + FIX_v1.3.1 §3.1）：`CoppeliaBackend(planner, cfg, frames)` 实现 `validate(action, state)` / `qualify(fact)` / `evaluate(schedule, graph)`。建议：内部持有一个 `MotionCompiler`；pipeline 每一轮从初始状态逐步调用 `validate`，所以收到"初始状态"（无 `held`、两臂都在 home）时 `reset()`；`validate` 直接调用 `compile_action(action, state)`，`CompileFailure` 按其 `status` 转成 `MotionResult`。为避免最终 `compile()` 重复做 IK/RRT，给编译器加按（臂、种类、目标位姿或目标关节、起始关节、手持、已放置、接触对）为键的路径缓存。`runner.run_task` 在建场景之后用 `run(problem, backend=...)` 重新规划（含反馈回路），`summary.feedback` 记录反馈，再编译、执行。mock 测试：IK 失败 → INVALID、RRT 两阶段超时 → UNKNOWN、碰撞 → INVALID；用 `GeometryFake` 让 B 对 x < −0.1 的位姿 IK 失败、能力表全写 VALID，验证 pipeline 出现"运动失败 → cut → 重搜 → 交接/中转"并最终 solved。
3. **`gen/`**（HANDOFF_v1.3 §7.1）：五族生成器，内置第 1 点的间距、可达范围和几何检查，生成失败重采样；固定种子每族 2 个样例做规划层冒烟，不跑 200 个。
4. 把以上写进 README 的"v1.3.2 修订"一节（第 1 节的内容也一并记录为"已由另一方完成"）。

## 4. 等机器空闲后（FIX_v1.3.2 §4、§5）

完整默认集合连跑两遍 → `s1_reach_handover` 端到端（`m3_attempt_5/`）→ `s1_parallel_place` 重跑 → 汇报 M3 → `tasks/sim/` 三个任务端到端与反馈回路真实触发（M4）→ 200 / 20 泛化验证（M5）。真实仿真与完整测试集不要同时运行。
