# Frozen v1.3.5 Dual-Arm SkillNet + APEX-MR Algorithm & Presentation Bundle

本目录打包归档当前已冻结验收的 **双臂积木技能网络与 APEX-MR 联合规划算法（v1.3.5 冻结候选基线）**、**9 块积木桥双模式实战（Scene A / Scene B）**、**F1–F4 泛化录像**、**`db77489` 在线恢复联调与 116 例基准证据**，以及最终 **第三次汇报 PPT（含内嵌高清视频与论文级架构图）**。

## 目录结构

```text
frozen_v135_algorithm_and_ppt/
├── README.md                                        # 本说明文件
├── manifest.json                                    # 全量文件清单与 SHA-256 校验表
├── algorithm/
│   ├── frozen_candidate_125files/                   # 逐字节无改动的 125 文件冻结候选基线
│   │   ├── isolation_receipt.json                   # 候选指纹 87dd723b... 校验收据
│   │   └── candidate/                               # skillnet/, apexmr/, simbridge/, skills/, tasks/, gen/, tests/, tools/
│   ├── lego_bridge_showcase/                        # 9 块积木桥双模式运动编译、TPG 与独立回放代码及收据
│   │   ├── *.py                                     # 42 个核心编译/标定/测试/出图脚本
│   │   ├── assets/scene_manifest.json               # 场景零件与位姿清单
│   │   ├── outboard_park_a1/                        # 外侧停靠位与桥面抓取预检标定收据
│   │   ├── mode_a/                                  # Scene A 输入、计划、汇总与 0 碰撞验收收据
│   │   └── mode_b/                                  # Scene B 输入、计划、交接闸门与 0 碰撞验收收据
│   └── recovery_demo_evidence/                      # 在线恢复 (db77489) 端到端联调脚本、收据与 116 例定量基准
│       ├── run_offline_gate.py / run_physical_recovery.py / run_recovery_benchmark.py
│       ├── offline_gate_receipt.json / physical_recovery_receipt.json
│       └── benchmark/recovery_benchmark_report.md & recovery_benchmark.csv
└── ppt_and_media/
    ├── DualArm_SkillNet_APEX_MR_Presentation.pptx   # 最终 11 页汇报 PPT（已内嵌 F1–F4 + Scene A/B 全部 7 段高清视频）
    ├── DualArm_SkillNet_APEX_MR_Presentation.pdf    # 同步导出高清 PDF 预览版
    ├── paper_figures/                               # 3 组复刻原论文 Figure 2 风格高清架构图（300 DPI PNG + 矢量 PDF + 脚本）
    ├── technical_diagrams/                          # 3 张三层技术时序/TPG 对比图（300 DPI PNG）
    ├── videos/                                      # 8 段独立高清实拍 MP4（F1–F4 + Scene A/B/交接特写 + 在线恢复）
    └── ppt_scripts/                                 # PPT 及定制学术配图复现脚本
```

## 核心冻结验收数据汇总

| 场景 / 基准 | 规模与策略 | 串行 → TPG 并行 | TPG 加速比 / 双臂并发 | 独立动力学/网格验收 |
| :--- | :--- | :---: | :---: | :---: |
| **F1–F4 泛化基准** | 并行 / 交接 / 堆叠 / 悬臂托举 | 4/4 原生节点全过 | — | **4/4 通过，0 违规** |
| **Scene A（Mode A）** | 9 块积木桥 · 18 步自由跨区 | 44.60 s → **38.41 s** | 1.161× · 6.20 s (16.1%) | 4,788 帧回放 **0 碰撞** · 误差 `1.13e-13 m` |
| **Scene B（Mode B）** | 9 块积木桥 · 22 步含 4/4 空中交接 | 77.81 s → **62.25 s** | **1.250×** · **15.56 s (25.0%)** | 8,155 帧回放 **0 碰撞** · 误差 `3.31e-13 m` |
| **Online Recovery** | Mode A 前缀 → `t=14.81s` 故障屏障 → `db77489` → 4/4 交接 | 75.42 s → **69.76 s** | 1.081× · 恢复决策 `1.937 s`（图复用 `105/106`） | 6,019 帧回放 **0 碰撞** · 误差 `2.91e-13 m` |
| **恢复定量基准（116 例）** | 积木桥 18 例 + 标准 11 任务 98 例 | 可解子集恢复率 **100% (25/25)** | L1 中位 `9.11 ms`（`45.9×` 核心加速）；L2 平均状态复用 **92.53%** | 100% 最优性一致 |

> 说明：为保持仓库轻量、避免超出 GitHub 单文件体积限制，原始多 GB 级仿真临时日志、CoppeliaSim 二进制 `.ttt` 场景以及逐帧数十 MB 浮点轨迹 `replay.json` 保留于本地档案；本目录包含完整的冻结源码、任务定义、验收收据、全部实拍视频与可直接汇报的 PPTX/PDF 交付物。
