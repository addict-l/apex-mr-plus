# v1.3.4 根因实验脚本（只读实验，不改规划核心）

用法统一为 `python -X utf8 tools/v134_experiments/<脚本> <模式> <秒数> <qualified.toml ...>`，任务文件取
`results/sim_v1_3/v133/qualification/F*/NNN/*/*.qualified.toml`。脚本通过替换 `heuristic.build_heuristic` 注入实验启发式，
搜索代码本身不变。

| 脚本 | 模式 | 作用 |
|---|---|---|
| `fast_hmax.py` | `record` | 前提计数版 HMax 与现有 HMax 在真实搜索状态上的数值等价与耗时对比 |
| `fast_hmax.py` | `fast` / `pdb` / `fasthmax` | 只换启发式跑完整 anytime 搜索 |
| `hadd.py` | `hadd` / `goalcount` | 非可容许 h_add / 目标计数 |
| `hadd_match.py` | `match` | h_add + 分配可行性剪枝（FIX_v1.3.4 §2 的首解阶段原型） |
| `hadd_diag.py` | （秒数 任务…） | 记录贪心 h_add 的最优 h 状态与未完成目标，用于看卡在哪 |
| `lifo.py` | `lifo` | 同 f 后进先出平局（结论：无效） |

注意：这些脚本里 `bound` 字段在非可容许模式下没有意义。
