# APEX_MR_Plus：双臂积木协作环境预览

使用 **CoppeliaSim Edu 4.10** 打开 `APEX_MR_Plus.ttt`，或双击 `Open_APEX_MR_Plus.cmd`。场景内的中文控制面板可以切换预设和观察视角。

- **① 正常布局**：橙色 A 臂、蓝色 B 臂、左右料盘、共享暂存区、城堡连桥与黄色临时桥墩。
- **② A 臂能力变化**：A 持连接砖，B 在桥下处于支撑端点；A 的肘关节通过软件缩小允许运动范围，当前构型仍在该范围内。
- **③ 临时支撑到位**：桥墩已放在桥下，B 处于连接砖安装端点。此预设用于理解恢复后的布局。

三个预设也分别保存在 `scenes/01_Workcell_Overview.ttt`、`02_Capability_Change.ttt` 和 `03_Temporary_Support.ttt`。

## 当前版本范围

当前是**运动学环境与 IK 端点预览**，切换预设直接设置物体和机械臂端点状态。所有场景结构均采用静态设置，积木采用放大后的模块尺寸，非真实 LEGO 尺寸。

尚未实现自主恢复搜索、碰撞自由的连接轨迹、真实 LEGO 卡扣连接、接触力控制和支撑承载动力学。预设③由场景作者指定，**不是算法生成的恢复结果**。关节范围缩小也未证明目标任务在该范围下整体不可行；这些内容需要后续规划与验证模块确认。

后续研究目标是从当前持物、连接和支撑状态出发，根据能力变化自动组合通用动作，并生成可执行的剩余计划。`config/experiment_definition.json` 记录动作接口与未来验证要求，不包含固定恢复动作链。

## 文件

- `APEX_MR_Plus.ttt`：主场景，纹理已嵌入。
- `Open_APEX_MR_Plus.cmd`：打开主场景。
- `src/`：场景构建、中文控制面板与纹理生成脚本。
- `assets/`：中文标牌 PNG 纹理。
- `scenes/`：三个预设快照。
- `previews/`：从 CoppeliaSim 实际渲染得到的预览图。
- `config/scene_manifest.json`：生成后的几何、端点和场景元数据。
- `config/experiment_definition.json`：研究目标与未来算法接口。
- `logs/`：仿真器运行日志。
