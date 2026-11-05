# APEX MR 双臂积木：Scene 3 上方运输与连续运动

当前工程：[apex_mr_scene3](apex_mr_scene3/README.md)。最新运行 **`20261105T121655Z-b7a8b9ea`** 已通过完整动力学与独立审计，主场景已更新。

工时 **317.08 → 309.42 s**，再减少 7.66 s。取架接近基座 237.43° → 144.21°，回库接近 221.47° → 162.58°，最后 Gate 抬升 197.88° → 接近 0°。回库阶段本身更慢，完整流程更快；当前路径不宣称全局最优。

134 个参考任务、353 个实际节点和 719 条依赖全部通过；临时支撑已回收，Gate 关闭，双臂空手，永久结构通过实际 8 N 验载。147 项软件测试、5403 个完整名义几何采样和主场景冷加载也通过。

双击 [Open_Scene3.cmd](apex_mr_scene3/Open_Scene3.cmd) 并点击 ▶。

克隆后安装 CoppeliaSim Edu 4.10 和 Python 3.11，再运行 `python -m pip install -r apex_mr_scene3/requirements.txt`。默认模拟器目录为 `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu`；其他安装位置可通过 `COPPELIASIM_ROOT` 指定。仓库可放在其他目录，文档中的 `D:/APEX_MR_Plus` 为原开发路径。
可查看 [实测转角与工时图](apex_mr_scene3/results/base_rotation_investigation/base_rotation_comparison.png)、[加速回放](apex_mr_scene3/results/apex/scene3_execution.mp4)、[本轮独立审核](apex_mr_scene3/results/base_rotation_investigation/independent_audit.json)。

- [Open_Scene3.cmd](apex_mr_scene3/Open_Scene3.cmd)：打开已发布的可执行主场景。
- `apex_mr_scene3/Build_Scene3.cmd`：构建正式动态源场景，不覆盖主场景。
- `apex_mr_scene3/Plan_Scene3.cmd`：正式 ILP → RRT-Connect → TPG 规划与名义几何验证，不执行动力学。
- **[Run_Scene3_APEX.cmd](apex_mr_scene3/Run_Scene3_APEX.cmd)**：运行新场景的 ILP → RRT-Connect → TPG → 动力学流水线，尝试复用签名匹配的计划。
- [Replan_Scene3_APEX.cmd](apex_mr_scene3/Replan_Scene3_APEX.cmd)：完整重算新场景；可附加 `--plan-only` 只做规划、`--no-video` 关闭录像、`--rebuild-scene` 重建动态源场景。
- `apex_mr_scene3/Open_Scene3_Preview.cmd / Build_Scene3_Preview.cmd`：单独打开或重建历史五状态设计预览，不覆盖主场景或动态源场景。
- `apex_mr_scene3/Plan_Scene3_Abstract.cmd`：保留的纯离散生命周期枚举基线，不生成机器人路径。
- `apex_mr_scene1/Open_APEX_MR_Plus.cmd`：打开归档的旧场景。
- `apex_mr_scene1/Run_Normal_Mode.cmd`：运行旧场景原有的动力学流程。

2026-11-03 已将原工程完整移动至 `apex_mr_scene1`；318 个文件逐一核对 SHA-256，内容保持一致。清单见 `archive_manifest.json`。Git 历史仍在本目录。旧报告中的历史绝对路径保留，旧命令示例里的 `cd D:\APEX_MR_Plus` 应改为 `cd D:\APEX_MR_Plus\apex_mr_scene1`；官方启动脚本自动使用新目录。

Scene 1 的动力学结果只属于原场景。Scene 3 的状态预览与离散计划不能视为新场景的动力学验收。
