"""Update delivery notes only after the accepted dynamic run and cold load."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


summary = read(ROOT/'results/apex/summary.json')
audit = read(LAB/'independent_audit.json')
cold = read(LAB/'delivery_validation.json')
assert summary['success'] and summary['main_scene_promoted'] and audit['accepted'] and cold['success']
assert summary['run_id'] == audit['run_id'] == cold['run_id']
before = audit['metrics']['before']['actual_makespan_s']
after = summary['actual_makespan_s']
actual = read(LAB/'candidate_actual_rotation.json')
focus = {row['task']: row for row in actual['focus_tasks']}
angles = {name: row['measured']['total_variation_deg'] for name, row in focus.items()}
nominal = read(ROOT/'results/apex/nominal_geometry_validation.json')
force = audit['metrics']['after']['assembly_contact']['B']
old_return = audit['rotation_comparison']['before']['semantic_groups']['return_support']['actual_elapsed_s']
new_return = actual['semantic_groups']['return_support']['actual_elapsed_s']
readme = ROOT/'README.md'
tail = readme.read_text(encoding='utf-8').split('## 场景中的问题', 1)[1]
prefix = f'''# apex_mr_scene3：双 UR5 上方运输与连续轨迹装配

两台 UR5 完成临时支撑、承载交接、连桥装配和支撑回收。当前主场景已经过完整动力学验收。旧 Scene 1 位于同级 `apex_mr_scene1`。

## 打开当前版本

运行 [Open_Scene3.cmd](Open_Scene3.cmd)，在 CoppeliaSim 点击 ▶。主文件 [apex_mr_scene3.ttt](apex_mr_scene3.ttt) 保存初始布局、完整 TPG 和中央控制器。

- [本轮实际转角与工时对比](results/base_rotation_investigation/base_rotation_comparison.png)
- [本轮加速录像](results/apex/scene3_execution.mp4)，画面标注仿真时间，影片时长不是执行工时。
- [运行汇总](results/apex/summary.json)、[实际事件与载荷验收](results/apex/execution_validation.json)
- [本轮独立审核](results/base_rotation_investigation/independent_audit.json)、[冷加载检查](results/base_rotation_investigation/delivery_validation.json)

## 本轮实际结果

最新运行 **`{summary['run_id']}`** 已通过完整实际动力学、独立审计和初始场景冷加载。工时 **{before:.2f} → {after:.2f} s**，再减少 **{before-after:.2f} s（{100*(before-after)/before:.2f}%）**。相对最初 455.60 s 版本，累计减少 {100*(455.60-after)/455.60:.2f}%。

以下转角来自实际关节位置连续采样的累计绝对变化，保留真实 winding，不用目标值或速度积分代替测量：

- 取支架接近：237.43° → **{angles['pick_support_approach']:.2f}°**。
- 带载运往支撑区域：284.75° → **{angles['support_route_front']:.2f}°**。
- 放回支架的接近：221.47° → **{angles['return_support_approach']:.2f}°**。
- 最后 Gate 抬升：197.88° → **{angles['pick_gate_lift']:.5f}°**，不再在原地翻转基座。
- Gate 抬起后运往装配区：251.39° → **{angles['close_gate_approach']:.2f}°**。

Gate 抓取前的接近仍有约 **{angles['pick_gate_approach']:.2f}°** 转动，本轮未证明它已达到最短轨迹。没有将所有运输旋转都宣传为可消除。

回库阶段是明确的代价：按相同任务起止定义，时间 **{old_return:.2f} → {new_return:.2f} s**。上方避障和接触慢速退让使这部分更慢；整条任务因取架与 Gate 的收益而更快。取架、回库、最后取装三个扩大邻接窗口的总角程都下降，完整任务中没有超过 180° 的单个 transit。阶段累积转角仍可以超过 180°，不与单段统计混用。

## 路径如何修复

1. 正常进行几何资格检查和 ILP 求解。已验收 317.08 s 的完整参考只作为相同目标值下的方案与闭合链输入。
2. 对取支架、回库和最后 Gate 三个首尾闭合窗口做原生 IK / 有界 RRT-Connect / shortcut 搜索。支架采用侧向接近、上方运输和释放后侧向退离；Gate 抬升改为 100 mm 后接装配上方，避免 150 mm 直抬时载荷碰肘部。
3. 选中的场景专用路径冻结于 [local_route_candidates.json](apex/local_route_candidates.json)，作为有来源的规划输入。当前运行重新检查 15 个改变的任务、9319 个密集原生几何样本，并重新核对取架与回库技能端姿。复用候选不被计为本轮新调用 RRT，也不当成实际执行证据。
4. 6 个原滑动宏和 1 个新的 Gate 抬升任务使用经过真实几何/FK走廊验证的 C2 曲线，共 66 处小角衔接。大角 RRT 折点仍保留停止；本轮未把所有折点强行平滑。
5. 按新曲线重建全相位双臂冲突图，完整名义几何与等待位姿 **{nominal['tested_samples']} 个采样**全部通过，再执行动力学。冻结输入与主场景 SHA 在发布和冷加载时再次检查。

## 实际验收与边界

- 134 个参考任务、{summary['tpg']['nodes']} 个节点和 {summary['actual_edges_checked']} 条实际依赖全部通过。
- 支架承载交接、永久结构实际 8 N 验载、卸载、取出、归仓和 Gate 关闭均通过，双臂最终空手。
- 工具装配合力峰值 {force['max_force_N']:.4f} N、力矩 {force['max_torque_Nm']:.4f} N·m，原阈值 40 N / 8 N·m 保持。
- 最终最大位置误差 {1000*summary['max_goal_position_error_m']:.4f} mm、姿态误差 {summary['max_goal_orientation_error_rad']:.6f} rad。
- 147 项软件测试通过。软件测试与本轮实际仿真分别保留证据。

原全局限速 `0.6 rad/s`、限加速度 `1.2 rad/s²`，以及接触慢速 `0.12 rad/s` 保持；命令 jerk 限值为 `8 rad/s³`。命令曲线导数界已独立重算，不等同于实测伺服 jerk 保证。

这是场景专用有界候选选优和一次完整仿真比较，不宣称全局最优或实体 UR5 性能。抓取/连接采用测得位姿处的理想刚性约束，有限采样不构成连续时间无碰撞证明。

## 运行与历史结果

使用 [Run_Scene3_APEX.cmd](Run_Scene3_APEX.cmd) 重放并验收，或 [Replan_Scene3_APEX.cmd](Replan_Scene3_APEX.cmd) 重新验证和构图。`--plan-only` 不执行、不发布。输入签名覆盖生产 Python/Lua、候选 JSON、四份物理配置、源场景、时间参数和相关历史参考。物理布局或碰撞后端改变后，旧局部候选会被拒绝，需要重新搜索。

上一版 317.08 s 完整快照在 [base_rotation_baseline_20261104_01](backups/base_rotation_baseline_20261104_01)，117 个文件逐一 SHA 校验。最初 455.60 s 基线在 [smooth_motion_baseline_20261104_01](backups/smooth_motion_baseline_20261104_01)。上一轮速度图和报告仍在 `results/smooth_motion_candidate`，只属于其注明的旧运行，不作为本轮证据。Scene 1 归档未修改。

## 场景中的问题'''
readme.write_text(prefix+tail, encoding='utf-8')
project = ROOT.parent/'README.md'
old = project.read_text(encoding='utf-8')
rest = old[old.index('- [Open_Scene3.cmd]'):]
rest = rest[:rest.index('本轮抓取与永久连接')] if '本轮抓取与永久连接' in rest else rest
intro = f'''# APEX MR 双臂积木：Scene 3 上方运输与连续运动

当前工程：[apex_mr_scene3](apex_mr_scene3/README.md)。最新运行 **`{summary['run_id']}`** 已通过完整动力学与独立审计，主场景已更新。

工时 **317.08 → {after:.2f} s**，再减少 {before-after:.2f} s。取架接近基座 237.43° → {angles['pick_support_approach']:.2f}°，回库接近 221.47° → {angles['return_support_approach']:.2f}°，最后 Gate 抬升 197.88° → 接近 0°。回库阶段本身更慢，完整流程更快；当前路径不宣称全局最优。

134 个参考任务、{summary['tpg']['nodes']} 个实际节点和 {summary['actual_edges_checked']} 条依赖全部通过；临时支撑已回收，Gate 关闭，双臂空手，永久结构通过实际 8 N 验载。147 项软件测试、{nominal['tested_samples']} 个完整名义几何采样和主场景冷加载也通过。

双击 [Open_Scene3.cmd](apex_mr_scene3/Open_Scene3.cmd) 并点击 ▶。
可查看 [实测转角与工时图](apex_mr_scene3/results/base_rotation_investigation/base_rotation_comparison.png)、[加速回放](apex_mr_scene3/results/apex/scene3_execution.mp4)、[本轮独立审核](apex_mr_scene3/results/base_rotation_investigation/independent_audit.json)。

'''
project.write_text(intro+rest, encoding='utf-8')
print(json.dumps({'readmes_updated': [str(readme), str(project)], 'run_id': summary['run_id']}))
