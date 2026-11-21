# v1.3.4 批次比较

更新：2026-11-17T13:48:18.469363+00:00。命名空间 `c_wall_v134`。

|族|已提交|solved|原分母|明确物理拒绝|可行分母|可行求解率|
|---|---:|---:|---:|---:|---:|---:|
|F1|40|40|40|0|40|1.0|
|F2|40|40|40|0|40|1.0|
|F3|40|36|40|2|38|0.9473684210526315|
|F4|40|40|40|0|40|1.0|
|F5|40|40|40|0|40|1.0|

## 同任务 IK 规划：v133 → v134

|族|v133 solved/40|v134 solved/40|v134明确物理拒绝|状态转移|
|---|---:|---:|---:|---|
|F1|37/40|40/40|0|{'solved→solved': 37, 'budget→solved': 3}|
|F2|40/40|40/40|0|{'solved→solved': 40}|
|F3|16/40|36/40|2|{'solved→solved': 16, 'budget→solved': 19, 'unknown→budget': 2, 'unknown→no_plan': 2, 'unknown→solved': 1}|
|F4|40/40|40/40|0|{'solved→solved': 40}|
|F5|40/40|40/40|0|{'solved→solved': 40}|

物理不可行不静默删除：仅D五任务有明确目标reach全臂INVALID证明且搜索no_plan时，从可行分母单列扣除。所有其它失败仍保留。

UNKNOWN/no_plan超5秒或缺耗时：0；solved首解超15秒或缺耗时：0。

|阶段|请求|已提交|状态|
|---|---:|---:|---|
|motion_gate|1|1|{'passed': 1}|
|qualify5|5|5|{'completed': 4, 'partial': 1}|
|planning200|200|200|{'solved': 196, 'budget': 2, 'no_plan': 2}|
|sim20|20|20|{'timeout': 2, 'passed': 8, 'failed': 10}|
|walls3|3|3|{'failed': 2, 'passed': 1}|
|gates|5|5|{'passed': 5}|

- 已正确拒绝 `v134-plan-f3-013`：['filled(s5_1)']，0.15700000000651926秒。

- 已正确拒绝 `v134-plan-f3-015`：['filled(s5_1)']，0.2029999999795109秒。

逐任务v133/当前状态、bound、扩展、耗时及两项首解字段见planning_200.csv；缺失数据保持空值。真实阶段逐任务CSV由状态journal重建，固定样本/墙样本分开记录。

## 固定同一组20项真实任务：v133 → v134

按原manifest source_job_id、原任务SHA及原清单SHA配对；缺失和失败不抽样补齐，固定分母20。

|版本|真实通过/20|待证据|UNKNOWN直接终止|cut后UNKNOWN|结构化cut|实际8→24升级|反馈轮次|
|---|---:|---:|---:|---:|---:|---:|---:|
|v133|5/20|0|14|0|0|0|0|
|v134|8/20|0|0|1|53|3|71|

v133 结构化 cut 原因：{}；直接终止证据不足 0 项。

v133 结构化cut原因：{}。


v134 结构化 cut 原因：{'ik_collision': 53}；直接终止证据不足 0 项。

v134 结构化cut原因：{'ik_collision': 53}。


源链异常 0 项；无法映射的新任务 0 项。
逐样本状态、失败原因及native链接：sim20_comparison.csv / sim20_comparison.json。

|版本|直接终止分类|数量（分母20）|
|---|---|---:|
|v133|real_planning:ik_exhausted|13|
|v133|real_planning:rrt_unknown|1|
|v133|execution:tracking_failure|1|
|v134|evidence_missing|2|
|v134|real_planning:ik_exhausted|10|

IK/RRT 候选碰撞不等于执行碰撞，有限种子UNKNOWN不等于绝对不可达。旧15项失败应按终端证据区分IK耗尽、RRT未知和真实跟踪失败。
cut/升级/轮次只取 motion_backend.json.feedback_rounds；runner motion_feedback_count 与全量预资格调用不当作cut。旧报告无结构化轮次时不补造次数，plan.feedback只作辅助。

仅成功且源链有效样本的最终plan.makespan/actual：n=8，median=1.045684262728591。

## walls3 单列（不补入固定20分母）

|任务|提交|真实通过|失败阶段|分类|证据|
|---|---|---|---|---|---|
|v134-wall-001|True|False|real_planning|real_planning:ik_exhausted|D:\APEX_MR_Plus\dualarm_skillnet\results\sim_v1_3\v134\experiments\c_wall_v134\runtime_fix2\walls3\001\simulation\gen_f3_s20261115_001\summary.json|
|v134-wall-009|True|True|None|passed|D:\APEX_MR_Plus\dualarm_skillnet\results\sim_v1_3\v134\experiments\c_wall_v134\runtime_fix2\walls3\009\simulation\gen_f3_s20261115_009\summary.json|
|v134-wall-037|True|False|compile|compile:unclassified|D:\APEX_MR_Plus\dualarm_skillnet\results\sim_v1_3\v134\experiments\c_wall_v134\runtime_fix2\walls3\037\simulation\gen_f3_s20261115_037\summary.json|


<!-- v134-frozen-final -->
## 冻结版最终验收（FIX §7、§8）

报告时间：2026-11-17 08:49:21 EST（America/New_York）。正式实验命名空间 c_wall_v134 / runtime_fix2；未重跑完整测试、planning200或登记。

规划层按第8节裁定通过。统计保留F3 36/38=94.737%，005/012借槽中转budget及legacy reference全域HMax作为已知限制/豁免；候选仅归档，不混入正式证据。

|验收项|结果|实际/证据|
|---|---|---|
|规划层（第8节裁定）|通过|planning_200.csv + user_section8_ruling.json|
|原11任务与最优性回归|通过|{'full': 'D:\\APEX_MR_Plus\\dualarm_skillnet\\results\\sim_v1_3\\v134\\regressions\\full_runtime_fix2.json', 'original11': 'D:\\APEX_MR_Plus\\dualarm_skillnet\\results\\sim_v1_3\\v134\\original11_four_fixes_runtime_fix2.json'}|
|固定20成功率≥70%|未通过|8/20|
|无运动UNKNOWN直接退出|通过|feedback_trigger_audit.json|
|3墙至少2项执行通过|未通过|1/3|
|五项真实门控|通过|5/5|
|估算/实际中位数0.7–1.3|通过|{'n': 8, 'median': 1.045684262728591}|
|归档、冻结指纹、来源与清理|通过|['archive_final_frozen.json', 'frozen_restore_audit.json', 'final_cleanup.json', 'native_final_source_audit.json']|

### 同任务规划 v1.3.3 → v1.3.4

|族|v133 solved/40|v134 solved/40|单列物理拒绝|可行率|
|---|---:|---:|---:|---:|
|F1|37|40|0|40/40|
|F2|40|40|0|40/40|
|F3|16|36|2|36/38|
|F4|40|40|0|40/40|
|F5|40|40|0|40/40|

### 固定20逐样本对比

|原source_job_id|族|v133|v134|失败分类|
|---|---|---|---|---|
|sim20-f1-0|F1|通过|timeout|timeout:TPG_geometry|
|sim20-f1-1|F1|通过|通过|passed|
|sim20-f1-2|F1|legacy_terminal:ik_collision|timeout|timeout:TPG_geometry|
|sim20-f1-3|F1|legacy_terminal:rrt_exhausted|通过|passed|
|sim20-f2-0|F2|通过|通过|passed|
|sim20-f2-1|F2|通过|通过|passed|
|sim20-f2-2|F2|legacy_terminal:ik_collision|failed|real_planning:no_plan_after_ik_collision|
|sim20-f2-3|F2|legacy_terminal:ik_collision|failed|real_planning:no_plan_after_ik_collision|
|sim20-f3-0|F3|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|
|sim20-f3-1|F3|legacy_terminal:ik_collision|通过|passed|
|sim20-f3-2|F3|通过|通过|passed|
|sim20-f3-3|F3|execution:tracking_failure|通过|passed|
|sim20-f4-0|F4|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|
|sim20-f4-1|F4|legacy_terminal:ik_collision|failed|real_planning:no_plan_after_ik_collision|
|sim20-f4-2|F4|legacy_terminal:ik_collision|通过|passed|
|sim20-f4-3|F4|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|
|sim20-f5-0|F5|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|
|sim20-f5-1|F5|legacy_terminal:ik_collision|failed|real_planning:unknown_after_ik_collision|
|sim20-f5-2|F5|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|
|sim20-f5-3|F5|legacy_terminal:ik_collision|failed|real_planning:budget_after_ik_collision|

### 新增墙逐任务结果

|任务|结果|失败分类|原因|
|---|---|---|---|
|v134-wall-001|failed|real_planning:unknown_after_ik_collision|RuntimeError: 真实运动后端重规划未求解：unknown|
|v134-wall-009|passed|passed||
|v134-wall-037|failed|compile:unclassified|CompileFailure: 运动段 home_B 的 RRT-Connect 失败（local_box 均未找到解）：Invalid goal; {'pair': ['/B_UR5/B_Finger_pos', '/b1_3'], 'reason': 'block_robot_collision:b1_3:B'}|

固定20：v133 5/20 → v134 8/20；walls3 1/3另列，不补固定20分母。

### 真实反馈触发（按调用/反馈轮记录计数）

|阶段|碰撞UNKNOWN→INVALID|8→24升级|规避cut|实际重搜轮|能力降级事件/事实|
|---|---:|---:|---:|---:|---:|
|sim20|54|3|54|54|10/10|
|walls3|5|1|5|5|1/1|
|gates|0|0|0|0|0/0|

升级仅数运动验证8→24，不将D资格32种子或runner计数当成反馈触发。重复验证按真实调用计数；能力事实和降级事件分开。缺失的超时证据不补造零触发的结论。

估算/实际：成功且源链有效的固定20样本 n=8，中位数=1.045684262728591；标定未改。

旧版末端原因按冻结§4.1规则做报告重分类，原始状态与证据不改。详细失败原因、逐样本原生证据与阶段耗时见 sim20_final_comparison.csv、sim20_comparison.json；分类汇总见 sim20_failure_categories.csv；反馈明细见 feedback_trigger_audit.json。

归档/历史保护=True；登记核心SHA一致=True；最终无CoppeliaSim残留且23210空闲=True。候选保存在 scope_trials/stopped_by_user_v1/，不进入此次验收。

最终结论：执行流程完成；以上未通过项如实保留，本轮不再修复冻结代码。

<!-- /v134-frozen-final -->
