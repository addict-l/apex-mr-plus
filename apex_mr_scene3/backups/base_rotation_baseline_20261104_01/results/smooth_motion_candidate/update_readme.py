"""Refresh the project entry documentation from accepted measured artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/apex'
read = lambda name: json.loads((OUT/name).read_text(encoding='utf-8'))
s, e, a, g = map(read, ['summary.json', 'execution.json', 'execution_validation.json', 'tpg.json'])
assert s['success'] and s['main_scene_promoted'] and a['success']
assert s['run_id'] == a['run_id'] == e['run_id']
path = ROOT/'README.md'
old = path.read_text(encoding='utf-8')
tail = old[old.index('## 场景中的问题'):]
gain = (455.6-e['makespan'])/455.6*100
body = f'''# apex_mr_scene3：双 UR5 连续轨迹装配

本场景用两台 UR5 搭建带收口挡板的积木连桥，研究临时支撑的布置、承载交接、机械臂释放和最后撤撑。旧工程位于同级 `apex_mr_scene1`。当前主场景已更新为通过完整动力学验收的连续运动版本。

## 打开当前版本

运行 [Open_Scene3.cmd](Open_Scene3.cmd)，在 CoppeliaSim 点击 ▶。它打开 [apex_mr_scene3.ttt](apex_mr_scene3.ttt)，文件保存初始布局、完整 TPG 和中央控制器，可独立重放。

- [实际运动速度对比](results/smooth_motion_candidate/figures/02_measured_joint_velocity.png)
- [工时与近零速边界对比](results/smooth_motion_candidate/figures/01_actual_time_and_boundary_stops.png)
- [载荷与精度对比](results/smooth_motion_candidate/figures/03_loads_errors_and_conditions.png)
- [本轮录像](results/apex/scene3_execution.mp4)，约 4 倍速，画面标有仿真时间。录像时长不能当作执行工时。
- [完整运行汇总](results/apex/summary.json)、[事件与传感器原始记录](results/apex/execution.json)、[独立验收](results/apex/execution_validation.json)

## 已验收的性能变化

最新运行为 **`{s['run_id']}`**，`success=true`、`main_scene_promoted=true`、`stage=actual_execution_audited`。

| 指标 | 原版基线 | 连续运动版本 |
|---|---:|---:|
| 实际仿真完工时间 | 455.60 s | {e['makespan']:.2f} s |
| 关节几何路径总长度 | 107.853 rad | 98.017 rad |
| 非技能 pose 边界近零速观测 | 318 / 355 | 41 / 326 |
| 默认限速且无依赖等待的 pose 子集 | 205 / 238 | 19 / 210 |
| 当前图实际事件验收 | 419 节点 / 760 边 | 390 节点 / 753 边 |

实际仿真工时减少 **138.52 s（{gain:.2f}%）**。近零速度使用 `max(abs(joint_velocity)) < 0.01 rad/s`，取边界附近真实采样；两次运行均为 100% 边界观测覆盖。路径重新分片使分母改变，因此数量与分母一起报告。近零观测还包含必要的运动端点和依赖等待，不能全部当作可消除的停车。

285 处几何里程碑采用共享时钟连续通过，其中 6 组低速滑动宏动作包含 47 处小角度 C2 衔接。47 个曲线内部节点的真实结束与后继开始同帧发生，邻近速度样本均高于 0.01 rad/s。抓放、承载交接、验载、接触模式切换以及必要端点仍独立确认。

基线完整保存在 [smooth_motion_baseline_20261104_01](backups/smooth_motion_baseline_20261104_01)，运行 ID 为 `20261103T121253Z-d0d0c68e`。比较采用同一动态源场景、相同四份配置、全局与任务局部速度/加速度限制，以及相同载荷、姿态和生命周期验收阈值。

## 优化怎样执行

1. 当前几何资格检查和 MILP 正常求解。已知完整参考的支撑方案与当前无约束最优目标值相同，作为平局优先选择；其全部任务、抓取变换、原始端点和依赖保持闭合。
2. 固定原端点精化 6 条有绕行的路径。旧路径先按当前完整几何上下文重新检查，再与有界 RRT-Connect / shortcut 候选比较，选更短有效路径。95 条已是直线或零位移的运动不重搜 IK。该策略避免局部较近 IK 分支破坏后续可达性。
3. 共线片段共享一条五次时间曲线。小角度滑动采用经几何与末端走廊复核的 C2 Bezier 曲线。所有外部依赖在连续段入口取得，原始任务守卫保留，节点由实测进度确认。
4. 以新曲线重新构建全相位跨臂冲突，再检查完整名义时序与等待位姿，最后执行真实关节动力学并独立审计。

6 个宏通过 1317 个真实机器人/载荷配置检查和 2629 次 FK 检查。最大末端偏离约 0.0588 mm，低于 0.5 mm 走廊上限；完整名义时序的 5589 个样本全部通过。依据见 [曲线验证](results/apex/curve_smoothing_validation.json) 和 [独立曲线审计](results/smooth_motion_candidate/independent_curve_audit.json)。

命令轨迹采用原有 `vmax=0.6 rad/s`、`amax=1.2 rad/s²`，新增 `jmax=8 rad/s³`。这些是命令曲线解析上界；位置伺服和接触动力学的实测速率并不自动满足同一 jerk 上界。实测不等间隔差分的 jerk 峰值估计仍约 A/B 13.30/11.46 rad/s³，不能宣传“实际 jerk 小于 8”。

## 本轮实际验收

- 390 个节点及 753 条依赖均通过实际事件检查，所有物料、承载交接和撤撑守卫满足。
- 工具装配合力峰值 10.010 N、力矩峰值 0.9406 N·m，低于原实验阈值 40 N / 8 N·m。传感器合力包含工具和载荷的重力、惯性，没有将其当作去皮后的纯接触力。
- 最终最大位置误差 {a['max_goal_position_error_m']*1000:.3f} mm，最大姿态误差 {a['max_goal_orientation_error_rad']:.6f} rad。
- 永久承载证明包含实际施加 8 N 的物理子步回调，6 个连续合格记录覆盖 0.50 s。控制器的连续合格驻留要求另为至少 0.65 s。
- K1、K2、Gate、Bridge 四个最终连接均动态启用，临时支撑归仓，挡板关闭，双臂空手。
- 131 项软件回归测试通过，涵盖生产 Lua 控制器、周期关节、轨迹连续性、依赖守卫、路径候选与验收失败条件。软件测试不替代本次实际仿真验收。

独立报告见 [最终审计](results/apex/independent_final_audit.json)、[重规划比较](results/smooth_motion_candidate/replanned_comparison.json)、[冷加载检查](results/smooth_motion_candidate/delivery_validation.json)。旧节点逐 ID 比较的原始拒绝结果仍保留在比较 JSON 中；新路径重分片后另按任务语义、新几何证书与实际全边事件审核，均通过。

**结果范围：** 这是相同模型下的两次完整仿真比较。抓取和连接仍是测得位姿处的理想刚性约束，有限碰撞采样不构成连续时间证明。本次采用有界候选选优，不宣称全局最优轨迹或实体 UR5 的实验性能。

## 运行与结果保护

- [Run_Scene3_APEX.cmd](Run_Scene3_APEX.cmd)：尝试复用签名匹配的参考与 TPG，再检查并执行动力学。
- [Replan_Scene3_APEX.cmd](Replan_Scene3_APEX.cmd)：重做资格、分配、路径精化、C2 宏、冲突图、全程检查和动力学执行。
- [Build_Scene3.cmd](Build_Scene3.cmd)：只重建动态源场景，不更新主场景。
- [Plan_Scene3.cmd](Plan_Scene3.cmd)：只规划与检查，不执行，不发布。

```powershell
python -m apex.pipeline --reuse-reference --reuse-tpg
python -m apex.pipeline --replan --no-video
python -m apex.pipeline --replan --plan-only
```

可用 `--vmax`、`--amax`、`--jmax` 设置正且有限的命令轨迹限制。修改限值会使缓存失效。签名覆盖动态源字节、生产 Python/Lua、配置、相关依赖、采样/时间参数及作为优化输入的历史参考文件。

源场景为 `scenes/10_Scene3_Dynamic.ttt`，已规划初始场景为 `scenes/11_Scene3_APEX_Planned.ttt`。只有本轮完整实际执行和独立验收通过、冻结输入仍一致时，才复制为主场景 `apex_mr_scene3.ttt`。失败与 plan-only 均不发布。

历史设计预览仍通过 `Open_Scene3_Preview.cmd` 打开，其按钮直接设置姿态；`Plan_Scene3_Abstract.cmd` 是离散枚举，均不能替代真实关节执行记录。

'''
path.write_text(body+tail, encoding='utf-8')
print(json.dumps({'updated': str(path), 'run_id': s['run_id'], 'actual_makespan_s': e['makespan']}))
