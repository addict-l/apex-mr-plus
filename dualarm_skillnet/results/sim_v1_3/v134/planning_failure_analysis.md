# runtime_fix2 planning200 已提交失败分析

读取UTC：2026-11-17T09:06:51.200349+00:00；已提交 200/200；状态 {'solved': 196, 'budget': 2, 'no_plan': 2}。workflow=finished，stage=phase_complete。

仅读已提交证据，不改变原始记录；按10分钟间隔读取，全部提交或workflow中断/整体停止后结束。未导入规划器、运行测试/搜索/仿真或操作进程。

搜索秒=search_elapsed_seconds；job秒含加载/返回，worker墙钟另列。层数/层宽直接来自slot.below，不把其他结构推断成墙。首解和bound缺失不作为额外根因证据。

unknown/no_plan ≤5秒检查：2项；超出搜索/job门限 []；缺失字段见JSON。

| 族/索引 | 状态 | 块/槽 | 最大层宽×层 | expanded | bound | 首解s/扩展 | 搜索/job/worker s | 末阶段 | v133 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| F3/005 | budget | 10/10 | 2×5 | 75270 | — | —/— | 56.969/57.032/57.922 | weighted/budget/w3.0 | unknown |
| F3/012 | budget | 10/10 | 2×5 | 93263 | — | —/— | 56.922/56.985/57.953 | weighted/budget/w3.0 | unknown |
| F3/013 | no_plan | 10/10 | 2×5 | 0 | — | —/— | 0.110/0.157/0.985 | —/—/w— | unknown |
| F3/015 | no_plan | 10/10 | 2×5 | 0 | — | —/— | 0.140/0.203/1.141 | —/—/w— | unknown |

## F3/005：gen_f3_s20261115_005

预算内未找到现任解，bound=None不能计算次优比值。

阶段观察：['末阶段达到所记墙钟帽']；层宽：{1: 2, 2: 2, 3: 2, 4: 2, 5: 2}；底层根数：2。

明确表内UNKNOWN：[]。不把这些事实认定全部必需。正目标双臂INVALID证据：[]。

证据：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/planning/F3/005/worker_81cf2a1923e840e5bbb0c01a43c37be0.result.json；输入：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/qualification/gen_f3_s20261115_005/gen_f3_s20261115_005/gen_f3_s20261115_005.qualified.toml。

## F3/012：gen_f3_s20261115_012

预算内未找到现任解，bound=None不能计算次优比值。

阶段观察：['末阶段达到所记墙钟帽']；层宽：{1: 2, 2: 2, 3: 2, 4: 2, 5: 2}；底层根数：2。

明确表内UNKNOWN：[]。不把这些事实认定全部必需。正目标双臂INVALID证据：[]。

证据：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/planning/F3/012/worker_34c5ad674ad1495fb42e1f6634541c09.result.json；输入：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/qualification/gen_f3_s20261115_012/gen_f3_s20261115_012/gen_f3_s20261115_012.qualified.toml。

## F3/013：gen_f3_s20261115_013

正目标双臂INVALID证据成立，限定于当前资格表的物理拒绝。

阶段观察：[]；层宽：{1: 2, 2: 2, 3: 2, 4: 2, 5: 2}；底层根数：2。

明确表内UNKNOWN：[]。不把这些事实认定全部必需。正目标双臂INVALID证据：[{'positive_goal_kind': 'filled', 'place': 's5_1', 'reach_statuses': {'A': 'INVALID', 'B': 'INVALID'}}]。

证据：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/planning/F3/013/worker_9070b0a2b7db4fc6a4997db144f9578a.result.json；输入：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/qualification/gen_f3_s20261115_013/gen_f3_s20261115_013/gen_f3_s20261115_013.qualified.toml。

## F3/015：gen_f3_s20261115_015

正目标双臂INVALID证据成立，限定于当前资格表的物理拒绝。

阶段观察：[]；层宽：{1: 2, 2: 2, 3: 2, 4: 2, 5: 2}；底层根数：2。

明确表内UNKNOWN：[]。不把这些事实认定全部必需。正目标双臂INVALID证据：[{'positive_goal_kind': 'filled', 'place': 's5_1', 'reach_statuses': {'A': 'INVALID', 'B': 'INVALID'}}]。

证据：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/planning/F3/015/worker_5dc54d5c3aaf4c449294633ea86984b6.result.json；输入：results/sim_v1_3/v134/experiments/c_wall_v134/runtime_fix2/qualification/gen_f3_s20261115_015/gen_f3_s20261115_015/gen_f3_s20261115_015.qualified.toml。

完整字段及证据路径见同名JSON。此文件只是可覆盖的派生分析，不改变原始结果。

## 监测结束

最终200/200全部提交，workflow finished、stage phase_complete；已停止读取。196 solved、2 budget、2 no_plan，无UNKNOWN。F1/F2/F4/F5各40/40 solved；F3为36 solved、2 budget、2有正目标双臂INVALID证据的no_plan，单列拒绝后的求解率36/38=94.737%。两项no_plan的搜索/job均≤5秒。未运行solver或启动后续任务。
