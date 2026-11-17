# 技能网络框架实现规格（双臂积木场景 · v1）

> 本文件是交给 Codex 的实现说明。请严格按本规格实现；遇到本规格没有覆盖的细节，选最简单、可测试的做法，并在 README 的“实现说明”里写明。

## 0. 项目位置与工作方式

| 项目 | 内容 |
|---|---|
| 项目名 | `dualarm_skillnet`（双臂技能网络规划器） |
| 项目文件夹 | `D:\APEX_MR_Plus\dualarm_skillnet\`（下文简称“项目根目录”，本文件就在其中） |
| Python 包名 | `skillnet` |
| 所在仓库 | 项目文件夹位于 `D:\APEX_MR_Plus` 这个 git 仓库内，但它是一个**独立项目** |

工作规则：

1. **所有新建和修改的文件都只能在项目根目录内**。不要改动、移动或删除项目根目录以外的任何文件，包括仓库根目录的 `README.md`、`.gitignore`，以及 `apex_mr_scene1/`、`apex_mr_scene3/`、`ppt_output/`、`local_backups/` 等文件夹。
2. 不要引用、导入或复制项目根目录以外的代码。
3. 在项目根目录内新建自己的 `.gitignore`，忽略 `__pycache__/`、`out/` 等运行产物。
4. 不要执行 `git commit`、`git push`、切换分支等 git 操作，除非用户明确要求。仓库里已有其他未提交的改动，不要碰它们。
5. 所有命令都在项目根目录下运行。

## 0.1 给实现者的总要求

- 在项目根目录下实现一个独立的 Python 包 `skillnet`。
- Python 3.11，**只用标准库**（`tomllib`、`dataclasses`、`heapq`、`itertools`、`math`、`json`、`argparse`、`time`、`collections`）。不要安装第三方包。
- 测试用 `unittest`，从项目根目录运行：`python -X utf8 -m unittest discover -s tests -t .`
- 不访问网络，不调用任何大模型 API，不连接仿真器。
- 所有面向用户的输出（命令行报告、`report.md`）用中文；代码标识符、注释用英文。
- 按第 12 节的里程碑顺序实现，每个里程碑完成后测试必须全部通过。

## 1. 要做出来的东西

一个**双臂积木任务规划器**：输入一个积木任务，系统自己推出完整步骤，决定两只臂各做什么、按什么顺序、哪些同时做，并保证每一步都可行。所有任务共用同一个参数化技能库，**换任务只换输入文件，不改代码**。

### 输入

| 输入 | 内容 |
|---|---|
| 任务文件（`tasks/*.toml`） | 目标；积木、位置、槽位及坐标；每只臂的初始位置和速度；能力表；可选的“小模型方案”（模拟小模型输出，用于测试校验与补全） |
| 技能库文件（`skills/blocks_v1.toml`） | 9 个参数化技能，所有任务共用 |

### 输出

| 输出 | 内容 |
|---|---|
| 完整步骤 | 小模型方案经校验补全后的步骤，每一步标注“保留 / 补出 / 删除 / 多余”及原因 |
| 双臂时间表 | 每一步由哪只臂做、开始和结束时间；两只臂可以同时干活 |
| 任务顺序图 | 每一步依赖前面哪一步、每一步的作用；导出 Mermaid 和 JSON |
| 结论 | `solved`（找到方案）、`no_plan`（确定无解）、`unknown`（遇到未验证的能力，无法下结论，并列出这些能力）、`budget`（超出搜索预算） |

### 第一版范围

- 只做规划层：推步骤、分配两只臂、排序、检查。
- 只考虑运动学层面：臂能不能到、要走多久。**不建力学模型**，不计算任何受力。
- cost 按距离估算（第 6.5 节）。
- 运动后端只有接口和占位实现；小模型只有提示词生成、输出解析和接口。

## 2. 算法链路

```
技能库（9 个参数化技能，所有任务共用）
   ┆ 各步都要查
   ▼
① 任务输入：读任务文件，得到初始状态
   ▼
② 小模型拆解 → 初步步骤（第一版：取任务文件里的 [proposal]；另提供提示词生成）
   ▼                                              ◄─┐ ③ 补不上时，把问题反馈给小模型（只做接口）
③ 校验与补全：逐条检查，往回推补出缺的步骤      ─┘
   ▼  完整步骤（作为参考）
④ 搜索与排序：事件状态 UCS ＋ 三项检查 ＋ 距离 cost   ◄─┐ 运动失败 / 能力未知
   ▼  双臂时间表                                       │
⑤ 运动验证：运动后端接口（第一版用占位后端）          ─┘
   ▼
输出：完整步骤 ＋ 双臂时间表 ＋ 任务顺序图 ＋ 报告
```

分工：
- ③ 回答“小模型的方案对不对、缺不缺”，输出一个能做通的顺序方案，并说明改了什么。规则层能修的直接修，修不了才反馈给小模型。
- ④ 回答“在所有可行做法里哪个最快”。③ 的结果只作为参考（耗时相同时优先选更接近它的方案），臂的分配、辅助方式、并行都由 ④ 决定。

## 3. 目录结构

```
dualarm_skillnet/              # 项目根目录
  SPEC.md                      # 本文件
  .gitignore                   # 只作用于本项目
  README.md                    # 中文：用途、运行方法、文件格式、实现说明
  skills/blocks_v1.toml
  tasks/                       # 第 11 节的示例任务
  skillnet/
    __init__.py
    __main__.py                # 命令行
    core/
      logic.py                 # 事实、条件的解析与代入
      skill.py                 # Skill（带参数）与 Action（参数已代入）
      library.py               # 读取技能库；自动生成技能网络连线；导出 Mermaid
      problem.py               # Problem、Proposal 数据结构
      engine.py                # 通用推理引擎
      hooks.py                 # 领域接口（Protocol）
    capability/oracle.py       # 三值能力表
    cost/model.py              # cost 接口与距离估算
    domains/blocks.py          # 积木领域：读任务文件、推导条件、稳定规则
    planning/
      search.py                # 事件状态 UCS
      relevance.py             # 从目标往回推
      completion.py            # 校验并补全小模型方案
      taskgraph.py             # 任务顺序图
    decompose/
      base.py                  # Decomposer 接口、FixedDecomposer
      prompt.py                # 提示词生成、输出解析、反馈消息
    motion/base.py             # MotionBackend 接口、NullBackend、ScriptedBackend
    pipeline.py                # 串起整条链路与反馈
    report.py                  # 中文报告、plan.json、taskgraph.mmd
  tests/
```

## 4. 技能库

### 4.1 文件格式（`skills/blocks_v1.toml`）

```toml
version = "blocks-v1"

[[skill]]
name = "pick"                  # 标识符
title = "抓取"                 # 显示名
stage = "1"                    # 引入的阶段
description = "单臂从某个位置抓起一块积木"
params = { arm = "arm", block = "block", loc = "loc" }    # 类型只能是 arm / block / loc / slot；顺序即参数顺序
pre = ["free(?arm)", "at(?block, ?loc)", "not two_arm_part(?block)"]   # 前置条件；只有这里允许 not
capability = ["reach(?arm, ?loc)", "liftable(?arm, ?block)"]           # 三值能力检查
add = ["held(?arm, ?block)", "empty(?loc)"]                            # 结束时添加
delete = ["free(?arm)", "at(?block, ?loc)"]                            # 开始时删除
occupies = ["?arm"]            # 执行期间占用的臂；至少一个；必须是 arm 类型参数
move_to = { "?arm" = "?loc" }  # 每只参与的臂移动到哪个参数（loc 或 slot）；空表示原地
op_time = 2.0                  # 操作时间（秒），> 0
structural = false             # true：会改变结构，同一时刻只允许一个此类动作在执行
primitives = ["approach(?arm, ?loc)", "grasp(?arm, ?block)", "lift(?arm)"]   # 内部原语，只用于显示和以后接运动后端
```

读取时必须校验并给出清楚的错误信息（指出技能名和出错的那一条）：名称是标识符且不重复；参数类型合法；所有条件里的变量都在 `params` 里；`not` 只出现在 `pre`；`occupies` 和 `move_to` 的键都是 arm 类型参数；`move_to` 的值是 loc 或 slot 类型参数；`op_time > 0`。

### 4.2 九个技能（必须完全按此定义）

| 技能 | params | pre | capability | add | delete | occupies | move_to | op_time | structural |
|---|---|---|---|---|---|---|---|---|---|
| pick 抓取 | arm, block, loc | `free(?arm)`, `at(?block, ?loc)`, `not two_arm_part(?block)` | `reach(?arm, ?loc)`, `liftable(?arm, ?block)` | `held(?arm, ?block)`, `empty(?loc)` | `free(?arm)`, `at(?block, ?loc)` | arm | arm→loc | 2.0 | 否 |
| place 放置 | arm, block, loc | `held(?arm, ?block)`, `empty(?loc)`, `placeable(?loc)` | `reach(?arm, ?loc)` | `at(?block, ?loc)`, `free(?arm)` | `held(?arm, ?block)`, `empty(?loc)` | arm | arm→loc | 2.0 | 否 |
| handover 交接 | giver: arm, taker: arm, block, loc | `held(?giver, ?block)`, `free(?taker)`, `handover_point(?loc)` | `reach(?giver, ?loc)`, `reach(?taker, ?loc)` | `held(?taker, ?block)`, `free(?giver)` | `held(?giver, ?block)`, `free(?taker)` | giver, taker | giver→loc, taker→loc | 3.0 | 否 |
| insert 插装 | arm, block, slot | `held(?arm, ?block)`, `fits(?block, ?slot)`, `not filled(?slot)`, `slot_ready(?slot)`, `insert_supported(?slot)` | `reach(?arm, ?slot)` | `installed(?block, ?slot)`, `filled(?slot)`, `free(?arm)` | `held(?arm, ?block)` | arm | arm→slot | 3.0 | 是 |
| detach 拆下 | arm, block, slot | `free(?arm)`, `installed(?block, ?slot)`, `slot_clear(?slot)` | `reach(?arm, ?slot)`, `liftable(?arm, ?block)` | `held(?arm, ?block)` | `free(?arm)`, `installed(?block, ?slot)`, `filled(?slot)` | arm | arm→slot | 3.0 | 是 |
| support_on 托住 | arm, slot | `free(?arm)`, `support_point(?slot)` | `reach(?arm, ?slot)` | `supporting(?arm, ?slot)` | `free(?arm)` | arm | arm→slot | 2.0 | 否 |
| support_off 松开 | arm, slot | `supporting(?arm, ?slot)` | — | `free(?arm)` | `supporting(?arm, ?slot)` | arm | 原地 | 1.0 | 是 |
| co_pick 共同抓取 | arm1: arm, arm2: arm, block, loc | `arm_pair(?arm1, ?arm2)`, `free(?arm1)`, `free(?arm2)`, `at(?block, ?loc)`, `two_arm_part(?block)` | `reach(?arm1, ?loc)`, `reach(?arm2, ?loc)` | `coheld(?arm1, ?arm2, ?block)`, `empty(?loc)` | `free(?arm1)`, `free(?arm2)`, `at(?block, ?loc)` | arm1, arm2 | 两臂→loc | 3.0 | 否 |
| co_insert 共同插装 | arm1: arm, arm2: arm, block, slot | `coheld(?arm1, ?arm2, ?block)`, `fits(?block, ?slot)`, `not filled(?slot)`, `slot_ready(?slot)`, `insert_supported(?slot)` | `reach(?arm1, ?slot)`, `reach(?arm2, ?slot)` | `installed(?block, ?slot)`, `filled(?slot)`, `free(?arm1)`, `free(?arm2)` | `coheld(?arm1, ?arm2, ?block)` | arm1, arm2 | 两臂→slot | 4.0 | 是 |

“拿着积木”（held / coheld）和“托着部件”（supporting）都会让臂持续不空闲（没有 `free`），直到放下、交出或松开。

### 4.3 技能网络（自动生成，不手写）

规则：技能 X 的某个 `add` 条件，与技能 Y 的某个肯定前置条件谓词相同，就连一条 X → Y，标注该谓词。Y 的前置条件是推导条件时，若 X 的 `add` 含有该推导条件依赖的谓词（第 7.1 节的 `derived_basis`），也连线，标注为“推导条件（由 X 的谓词推出）”。允许自环（例如 insert → insert，经由 slot_ready ← filled）。

## 5. 任务文件格式（`tasks/*.toml`）

注意 TOML 规则：所有顶层键必须写在第一个 `[...]` 或 `[[...]]` 之前。

```toml
name = "s1_reach_handover"
description = "红色积木只有 B 够得着，目标位置只有 A 够得着"
arms = ["A", "B"]                  # 顺序决定 arm_pair 的顺序
goal = ["at(red, board_l1)"]       # 额外目标（基础事实，可带 not）
fill_all_slots = true              # 默认 true：所有非临时槽位都要装好

[[arm]]
id = "A"
home = [-0.30, 0.30, 0.20]         # 初始位置（米）
speed = 0.25                       # 平均移动速度（米/秒），> 0

[[location]]
id = "stock_r1"
kind = "stock"                     # stock 料区 / buffer 中转 / board 底板 / handover 交接点
pos = [0.40, -0.20, 0.00]

[[block]]
id = "red"
type = "2x4"
at = "stock_r1"                    # 与 installed_in 二选一
# installed_in = "t1"              # 一开始就装在某槽位
# two_arms = true                  # 需要两只臂拿

[[slot]]
id = "beam"
type = "beam"
pos = [0.00, 0.00, 0.10]
below = ["p1"]                     # 下面直接连着的槽位；空表示直接在底板上
overhang = false                   # 是否悬空
anchors = []                       # 悬空槽位：这些槽位装好后就稳了（overhang=true 时必须非空）
support_point = false              # 能否用臂托
needs_support_during = []          # 插装本槽位时，这些槽位必须有支撑
temporary = false                  # 是否属于临时支撑台
platform_for = ""                  # 临时槽位：为哪个槽位提供支撑

[capabilities.reach]               # 每种能力一张表
default = "INVALID"                # VALID / INVALID / UNKNOWN
valid = { A = ["board_l1", "hand_mid"], B = ["stock_r1", "hand_mid"] }   # 键是第一个参数
# invalid = [["A", "stock_r1"]]    # 也支持“参数列表的列表”写法
# unknown = { A = ["hand_far"] }

[capabilities.liftable]
default = "VALID"

[proposal]                         # 可选：模拟小模型输出
instruction = "把红色积木放到底板左区"
steps = ["pick(B, red, stock_r1)", "place(B, red, board_l1)"]
```

### 加载器自动生成的事实

| 类别 | 事实 |
|---|---|
| 静态 | `fits(b, s)`（积木类型等于槽位类型）、`below(t, s)`、`placeable(loc)`（kind ≠ handover）、`handover_point(loc)`（kind = handover）、`two_arm_part(b)`、`arm_pair(a_i, a_j)`（arms 中 i < j）、`support_point(s)`、`overhang(s)`、`anchor(s, a)`、`temporary(s)`、`platform_part(t, s)`、`needs_support_during(s, t)` |
| 初始 | `free(arm)`；`arm_at(arm, home_<arm>)`；`at(b, loc)`；`installed(b, s)` 与 `filled(s)`（installed_in）；`empty(loc)`（所有 placeable 且没有积木的位置） |
| 默认目标 | `free(arm)`（每只臂）；`filled(s)`（fill_all_slots 时，每个非临时槽位）；`not filled(t)`（每个临时槽位）；再加上 `goal` 里写的 |

加载时校验并给出清楚的错误：所有 id 全局唯一；引用的位置、槽位存在；一个位置最多一块积木；`installed_in` 的类型匹配；`overhang=true` 时 `anchors` 非空；`platform_for` 只能出现在临时槽位；`below` 不能成环。

## 6. 核心规则

### 6.1 条件分类
- **动态条件**：谓词出现在某个技能的 `add` / `delete` 中，或是 `arm_at`。
- **推导条件**：领域提供的 `derived_predicates`（第 7.1 节），每次现算。
- **静态条件**：其余全部，只在生成具体动作时检查一次。
- 同一谓词不能既是动态又是推导，否则加载时报错。

### 6.2 生成具体动作
对每个技能，枚举参数的全部取值组合（按类型取对象）。**同类型的两个参数必须取不同值**。静态的肯定条件不在静态事实里、或静态的否定条件在静态事实里，丢弃。动作以 `(name, args)` 为键，可以从字符串 `pick(B, red, stock_r1)` 解析。

### 6.3 执行语义
- 动作**开始**时删除 `delete` 中的事实；**结束**时添加 `add` 中的事实，并把每只移动的臂的 `arm_at` 更新为 `move_to` 的目标（删旧加新）。
- 动作执行期间占用 `occupies` 中的臂；被占用的臂不能开始新动作。
- `structural = true` 的动作，同一时刻最多一个在执行。
- 同一个动作（同一键）不能同时执行两次。

### 6.4 三值能力
- 能力表返回 `Evidence(fact, status, source)`；未声明的能力谓词一律 `UNKNOWN`。
- 动作的能力结论：任一 INVALID → INVALID；否则任一 UNKNOWN → UNKNOWN；否则 VALID。
- 只有 VALID 的动作能执行。UNKNOWN 必须记录（搜索结果里列出）。
- 能力表支持 `update(fact, status, source)`，用于反馈回路。

### 6.5 cost（距离估算）
- 每个位置、槽位和 `home_<arm>` 都有坐标。
- 一个动作的耗时 = `max(每只移动的臂：欧氏距离(当前位置, 目标位置) / 该臂速度)` + `op_time`。没有移动的臂取 0。
- 耗时在动作**开始时**按当时的 `arm_at` 计算，所以同一个动作在不同状态下耗时不同。
- 整个方案的 cost 是双臂时间表的**完成时间**（并行部分不重复计算）。
- 接口：`CostModel.estimate(action, state) -> float`。预留 `evaluate(action, state) -> float | None`（以后接真实轨迹时间，第一版返回 None，不使用）。

### 6.6 结构稳定规则（不涉及力）
见第 7.2 节。每个动作**开始后**和**结束后**的状态都要检查，不通过则该动作不能执行。

## 7. 积木领域（`domains/blocks.py`）

### 7.1 推导条件

| 推导条件 | 何时为真 | derived_basis |
|---|---|---|
| `slot_ready(s)` | `below` 中 s 下面的每个槽位都 `filled` | filled |
| `slot_clear(s)` | 没有任何已 `filled` 的槽位 u 满足 `below(s, u)` | （无） |
| `has_support(s)` | 存在臂 a 有 `supporting(a, s)`；或者 s 有临时槽位且所有 `platform_part(t, s)` 的 t 都 `filled` | supporting, filled |
| `insert_supported(s)` | 对每个 `needs_support_during(s, t)`，`has_support(t)` 为真 | supporting, filled |

临时支撑台与被托积木之间只是接触、没有连接：临时槽位不写进被托槽位的 `below`，所以不影响 `slot_ready` / `slot_clear`。

### 7.2 稳定规则
对每个 `overhang` 槽位 s：若 `filled(s)`，则必须“它的所有 anchors 都 filled”或者 `has_support(s)`。否则返回违规说明（中文，指明槽位）。

### 7.3 领域接口（`core/hooks.py` 的 Protocol）
- `derived_predicates: frozenset[str]`、`derived_basis: dict[str, frozenset[str]]`
- `resource_predicates: frozenset[str]`（积木领域为 `{"free", "arm_at"}`，任务图里不作为因果边）
- `evaluate(pred, args, state) -> bool`
- `invariant(state) -> str | None`
- `expand(fact) -> list[Fact]`：推导条件可能依赖的全部基础事实（往回推用）
- `justify(fact, state) -> list[Fact]`：在给定状态下使推导条件成立的基础事实（任务图用）
- `invariant_needs(action) -> list[Fact]`：执行该动作后，稳定规则要求的事实（例如装入悬空槽位 → `has_support(s)` 和各 anchor 的 `filled`）
- `position(place) -> tuple[float, float, float]`、`speed(arm) -> float`
- `scene_summary() -> list[str]`（提示词用）

## 8. 规划

### 8.1 事件状态 UCS（`planning/search.py`）

```python
search(engine, *, start=None, goal=None, hint=None, cuts=(), protected=(),
       allowed=None, max_expansions=200_000, time_limit=60.0) -> SearchResult
```

- 节点 = （当前事实集合，正在执行的动作及剩余时间的有序元组）。
- 两种转移：
  1. **开始一个动作**：耗时 0。条件：臂不忙；若是 structural 则当前没有 structural 在执行；前置条件满足；能力 VALID；`(当前事实, 动作键)` 不在 `cuts`；动作的 `delete` 与 `protected` 不相交；开始后的状态通过稳定规则。
  2. **推进到最近一个动作完成**：耗时 Δt = 最小剩余时间；所有剩余时间 ≤ Δt 的动作同时结束，依次执行结束效果；结束后的状态必须通过稳定规则。
- 优先队列键：（累计耗时，偏离 hint 的次数，序号）。“偏离”指开始了一个不在 `hint` 动作键集合里的动作；`hint` 为空时恒为 0。
- 目标：没有正在执行的动作，且所有目标条件（可含推导条件、否定条件）成立。
- 重复检测：同一节点只保留（耗时，偏离）字典序最小的。
- 超出 `max_expansions` 或 `time_limit` 返回 `budget`；队列耗尽时，若途中遇到过 UNKNOWN 返回 `unknown`，否则 `no_plan`。
- 性能要求：为每个动作选一个“触发事实”（它的动态肯定前置条件中出现次数最少的那个），按当前状态中的事实查索引得到候选动作，不要每次遍历全部动作。
- `SearchResult`：`status`、`steps`（每步含动作、开始时间、结束时间、开始前的状态）、`makespan`、`expanded`、`generated`、`rejected`（按原因计数：capability_invalid / capability_unknown / invariant / cut / protected）、`unknown`（遇到的 UNKNOWN 能力事实）、`elapsed`。

### 8.2 往回推（`planning/relevance.py`）

```python
relevant_actions(engine, goal=None) -> tuple[frozenset[ActionKey], dict[ActionKey, list[Fact]]]
```

从目标出发：肯定目标展开为基础事实（推导条件用 `expand` 递归展开）加入“需要”集合；否定目标的事实加入“需要删除”集合。反复扫描动作：`add` 命中“需要”、或 `delete` 命中“需要删除”的动作算相关，记录命中的事实；再把它的肯定前置条件（展开后）和 `invariant_needs` 加入“需要”。直到不再增加。

用途：(1) 作为 `search` 的 `allowed`，缩小搜索范围；若缩小后结论不是 `solved`，pipeline 自动改用完整搜索并在反馈记录里写明；(2) 报告里说明每个相关动作“为了什么”。

### 8.3 校验与补全（`planning/completion.py`）

```python
complete(engine, steps: list[str], goal=None) -> CompletionReport
```

从初始状态起，逐条处理小模型的步骤（顺序执行语义：开始效果加结束效果一起应用）：

1. 解析失败（技能不存在、参数个数不对、对象不存在或类型不对、不满足静态条件）→ **删除**，原因写明。
2. 能力不是 VALID → **删除**，原因列出每条不通过的能力及其结论。
3. **多余**判断：失败的肯定前置条件中，有涉及积木、且已被前面的步骤删除过（之后没有再加回）的事实；或失败的否定前置条件对应的事实，是前面的步骤加上去的 → 标为**多余**，原因写明。
4. 前置条件不满足 → 以失败的条件为目标，从当前状态调用 `search` 求一段子计划；`protected` 为这一步当前已成立的动态肯定前置条件。成功则把子计划按开始时间顺序加入，标为**补出**，原因写“为满足 X 的前置条件：…”；失败则**删除**这一步，原因写明。
5. 补完后再次检查；执行后不通过稳定规则 → **删除**。否则执行，标为**保留**。
6. 全部处理完后，若总目标未达成，从当前状态搜索到总目标，加入的步骤标为**补出**（原因“补齐任务目标：…”）；仍不可达则记录未解决的目标。

`CompletionReport`：`entries`（种类、动作文本、原因）、`plan`（最终动作序列）、`goal_reached`、`unresolved`、`feedback_for_model`（第 9.2 节，补不上时生成）。

### 8.4 任务顺序图（`planning/taskgraph.py`）

```python
build_task_graph(engine, steps) -> TaskGraph
```

- 节点：时间表中的每一步（按开始时间排序）。
- **因果边** i → j：j 的肯定前置条件需要的基础事实 f（推导条件用 `justify(f, j 开始前的状态)` 拆开；排除静态谓词和 `resource_predicates`），由在 j 开始前结束、且最后一个添加 f 的步骤 i 提供。边上标注事实。
- **同臂边**：同一只臂上前后相邻的两步之间（若已有因果边则不重复）。
- 每个节点的“作用”：为后面哪些步骤提供了哪些事实；达成了哪些目标事实。
- 导出：`to_mermaid()`（flowchart TD，节点文字含时间、臂、技能显示名和动作）、`to_json()`。

## 9. 小模型与运动后端

### 9.1 小模型接口（`decompose/`）
- `Proposal(instruction, goal, steps, source)`。
- `Decomposer` 接口：`decompose(problem, feedback: list[str] | None = None) -> Proposal`。
- `FixedDecomposer`：返回任务文件里的 `[proposal]`。
- `build_prompt(problem, mode)`，`mode` 取：
  - `"minimal"`：任务、场景（积木及类型、位置、槽位及上下关系）、技能清单（签名、显示名、前置条件、效果）；
  - `"with_capabilities"`（默认）：在 minimal 基础上加每只臂的能力摘要，例如“A 能到达：stock_l1, board_l1, hand_mid”“需要两只臂拿的积木：…”。
  提示词要求只输出一个 JSON 对象：`{"goal": [...], "steps": ["pick(B, red, stock_r1)", ...]}`，只能用技能库里的技能和场景里的对象。
- `parse_model_output(text) -> Proposal`：从文本中提取第一个 JSON 对象并校验；格式错误时抛出带中文说明的异常。
- `build_feedback(report) -> list[str]`：把 ③ 补不上的问题写成给小模型的中文反馈，例如“第 2 步 place(B, red, board_l1)：B 够不着 board_l1”。
- 第一版**不调用任何模型**；pipeline 支持传入实现了 `Decomposer` 的对象，以后接真实模型时只需新增实现。

### 9.2 反馈给小模型的规则
只有当 ③ 结束后仍有 `unresolved`（补不上）时，pipeline 才调用 `decomposer.decompose(problem, feedback)` 再试，最多 `max_model_rounds` 次（默认 2）。`FixedDecomposer` 收到反馈时原样返回同一方案（因此不会改善），这是预期行为。

### 9.3 运动后端（`motion/base.py`）
- `MotionResult(status, detail, segments)`。
- `MotionBackend` 接口：`validate(action, state) -> MotionResult`；`qualify(fact) -> Status | None`（按需验证某条能力，返回 None 表示无法验证）。
- `NullBackend`：`validate` 恒为 VALID，`segments` 为代入参数后的 `primitives`，`detail` 写“占位后端：未做几何检查”；`qualify` 返回 None。
- `ScriptedBackend(fail_once=[动作文本...], qualify={事实文本: 结论})`：`validate` 对列出的动作第一次返回 INVALID、之后 VALID；`qualify` 按表返回。仅用于测试。

## 10. 链路与反馈（`pipeline.py`）

```python
run(problem, *, decomposer=None, backend=None, prune=True, max_rounds=6,
    max_model_rounds=2) -> PipelineResult
```

1. ② 取方案：`decomposer` 为空时用 `FixedDecomposer`；没有方案时跳过 ③。
2. ③ `complete`；若有 `unresolved` 且还有模型轮次，按第 9.2 节反馈重试。
3. ④ 计算相关动作（`prune=True` 时）；以 ③ 的动作键为 `hint` 调用 `search`。缩小范围后不是 `solved` → 改用完整搜索，记录反馈。
4. **能力未知回路**：对结果中每条 UNKNOWN 能力调用 `backend.qualify`；有结论的写回能力表（source 写“运动后端按需验证”），记录反馈，回到第 3 步重新搜索。
5. ⑤ 按开始时间逐步调用 `backend.validate(action, 该步开始前的状态)`。某步 INVALID → 把（开始前状态，动作键）加入 `cuts`，记录反馈（中文，写明动作、时间、原因），回到第 3 步。
6. 全部通过 → 生成任务图，返回结果。
7. 第 3–5 步最多循环 `max_rounds` 轮，超过则以当前结论返回。

`PipelineResult`：`problem`、`proposal`、`completion`、`schedule`、`graph`、`motion`（每步验证结果）、`feedback`（按发生顺序的中文记录）、`status`。

## 11. 示例任务与验收

### 11.1 标准布局（所有示例共用，单位米；可在各任务里增删位置）

| 名称 | 坐标 | 说明 |
|---|---|---|
| home_A | (-0.30, 0.30, 0.20) | A 初始位置，速度 0.25 m/s |
| home_B | (0.30, 0.30, 0.20) | B 初始位置，速度 0.25 m/s |
| stock_l1 / l2 / l3 | (-0.40, -0.20 / -0.10 / 0.00, 0.00) | 左料区 |
| stock_r1 / r2 / r3 | (0.40, -0.20 / -0.10 / 0.00, 0.00) | 右料区 |
| stock_c | (-0.10, -0.20, 0.00) | 中间料位 |
| buffer_mid | (0.00, -0.30, 0.00) | 中转位 |
| hand_mid | (0.00, 0.10, 0.20) | 交接点（近） |
| hand_far | (0.00, 1.20, 0.20) | 交接点（远），只在 s1_far_handover 中使用 |
| board_l1 / board_r1 / board_c | (-0.15 / 0.15 / 0.00, 0.20, 0.00) | 底板平放位置 |
| 塔、桥的槽位 | 底板中部 (0.00, 0.10, z)，z 按层高 0.03 递增 | 各任务自定 |

能力表：A 能到达所有 x ≤ 0 的位置和槽位，B 能到达所有 x ≥ 0 的位置和槽位（交接点、中转位、底板中部的槽位两者都能到），除非任务另有说明；`liftable` 默认 VALID。

### 11.2 示例任务

| 文件 | 场景 | 验收 |
|---|---|---|
| `s1_parallel_place.toml` | red 在 stock_l1 → board_l1；blue 在 stock_r1 → board_r1 | `solved`；两只臂的动作在时间上有重叠；完成时间小于把两组动作串行执行的时间 |
| `s1_nearest_arm.toml` | 一块积木在 stock_c → board_c，两只臂都够得着（本任务把 stock_c、board_c 设为两臂都可达） | 抓取由离得近的 A 完成 |
| `s1_reach_handover.toml` | red 在 stock_r1（只有 B 能到）→ board_l1（只有 A 能到）；附带小模型方案 `pick(B, red, stock_r1)`、`place(B, red, board_l1)` | 最终方案使用 handover（经 hand_mid）。③ 中 `place(B, red, board_l1)` 被**删除**，原因含 `reach(B, board_l1)` 与 INVALID；之后补出的步骤使目标达成 |
| `s1_far_handover.toml` | 同上，但唯一交接点是 hand_far | 最终方案不用交接，改经 buffer_mid 中转；完成时间大于 s1_reach_handover |
| `s2_tower.toml` | 三层塔 t1 ← t2 ← t3；b1、b2 在左料区，b3 在右料区；附带错误顺序的小模型方案：`pick(B, b3, stock_r1)`, `insert(B, b3, t3)`, `pick(A, b1, stock_l1)`, `insert(A, b1, t1)`, `pick(A, b2, stock_l2)`, `insert(A, b2, t2)` | 时间表中插装顺序为 t1、t2、t3。③ 中 `insert(B, b3, t3)` 之前补出若干步骤使 t1、t2 装好，`insert(B, b3, t3)` 被**保留**，后面四步都标为**多余** |
| `s3a_cantilever.toml` | 柱 p1（左料区）→ 梁 beam（右料区，`below=["p1"]`，overhang，anchors=["cw"]，support_point）→ 配重 cw（右料区，`below=["beam"]`）；另有两块临时积木与临时槽位 tp1、tp2（`platform_for="beam"`） | 方案包含 support_on 与 support_off，且 support_off 在 cw 插装结束之后；不使用临时槽位 |
| `s3a_platform.toml` | 同上，但设 `reach(A, beam)` 与 `reach(B, beam)` 中只有 B 为 VALID，且 tp1、tp2 只有 A 能到 | 方案不含 support_on；临时槽位先装后拆；结束时临时槽位全部清空，所有臂空闲 |
| `s3a_platform_demo_needs_support.toml`（可选） | 用 `needs_support_during` 表示“插装 cap 时 beam 必须有支撑” | 插装 cap 时 beam 处于有支撑状态 |

### 11.3 其他测试

| 测试 | 验收 |
|---|---|
| 技能库读取 | 读入 9 个技能；技能网络包含 pick → place、pick → insert、insert → insert（经 slot_ready）、support_on → insert（经 insert_supported 或稳定规则相关谓词） |
| 定义校验 | 技能里使用未声明变量、在 add 里写 not、occupies 指向非臂参数时，读取报错并指出技能名与条目 |
| 任务校验 | 位置重复放积木、引用不存在的槽位、overhang 没有 anchors、below 成环时报错 |
| cost 随位置变化 | 把 board_l1 挪远后，s1_parallel_place 的完成时间变长 |
| 运动失败回路 | s1_reach_handover 配 `ScriptedBackend(fail_once=["handover(B, A, red, hand_mid)"])` → 最终方案经 buffer_mid 中转；反馈记录恰有 1 条运动失败 |
| 能力未知回路 | s1_reach_handover 中把 `reach(A, hand_mid)` 设为 UNKNOWN，配 `ScriptedBackend(qualify={"reach(A, hand_mid)": "VALID"})` → 反馈记录含一次按需验证；最终方案使用 handover |
| 能力未知但无法验证 | 同上但用 NullBackend → 最终方案经 buffer_mid；结果里列出 `reach(A, hand_mid)` 为 UNKNOWN |
| 往回推剪枝的回退 | 构造一个剪枝后找不到方案的情况（或用 mock 使 `allowed` 过小）→ pipeline 改用完整搜索并记录 |
| 提示词 | `minimal` 不含能力摘要；`with_capabilities` 含每只臂的可达位置；`parse_model_output` 能从夹杂文字的文本中取出 JSON，格式错误时报中文错误 |
| 任务图 | s2_tower 中 insert(t1) → insert(t2) → insert(t3) 之间有因果边（经 filled）；导出的 Mermaid 文本非空 |

## 12. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 核心 | logic、skill、library（含网络导出）、engine、capability、hooks | 技能库读取、定义校验 |
| M2 领域、cost、搜索 | domains/blocks、cost、search | 任务校验；s1 四个任务、s2 的顺序；cost 随位置变化 |
| M3 往回推、补全、任务图 | relevance、completion、taskgraph | s1_reach_handover 与 s2_tower 的 ③ 结果；任务图测试 |
| M4 链路与反馈 | motion、pipeline、report、命令行 | 运动失败、能力未知（两种）、剪枝回退 |
| M5 支撑场景与提示词 | s3a 任务、decompose | s3a 全部验收；提示词测试 |

## 13. 命令行与输出

```bash
python -X utf8 -m skillnet network [--skills PATH] [--out DIR]
python -X utf8 -m skillnet run TASK [--skills PATH] [--out DIR] [--no-proposal] [--no-prune]
python -X utf8 -m skillnet prompt TASK [--mode minimal|with_capabilities]
python -X utf8 -m skillnet all [--out DIR]
```

- `network`：打印每个技能的签名与连线，写出 `skill_network.mmd`。
- `run`：打印中文报告，并在 `DIR/<任务名>/` 写出 `report.md`、`plan.json`（时间表与完整步骤）、`taskgraph.mmd`、`taskgraph.json`。
- `prompt`：打印给小模型的提示词。
- `all`：依次运行 `tasks/` 下全部任务，最后打印汇总表（任务、结论、完成时间、步骤数、扩展状态数、耗时）。
- 命令行启动时把标准输出设为 UTF-8，保证 Windows 终端中文正常。

报告结构：

```
任务：<名称> —— <描述>
① 初始状态：…
② 小模型方案（来源）：…
③ 校验与补全：保留 / 补出 / 删除 / 多余，逐条附原因
④ 搜索结果：<结论>，完成时间 X s，扩展 N 个状态，用时 T s
   t=0.00–3.10  B    抓取  pick(B, red, stock_r1)
   …
⑤ 运动验证与反馈：…
⑥ 任务顺序图：k 个节点、m 条依赖；每一步的作用
未验证的能力：…（如有）
```

## 14. 明确不做

- 不建力学模型：没有受力 LP、插装下压力、支撑力。
- 不连接 CoppeliaSim 或 APEX-MR，不生成真实轨迹，不用 RRT 计算 cost。
- 不调用任何大模型，不访问网络。
- 不构建 TPG，不执行。
- 共持的两个技能只写定义和单元测试（生成动作、能力检查），不提供示例任务。

## 15. 已定的默认选择（如需修改请先改本文件）

- 目录：本仓库内的独立项目文件夹 `dualarm_skillnet/`，见第 0 节的工作规则。
- 定义文件格式：TOML。
- 保留“结构类动作同一时刻只允许一个”。
- 示例坐标使用第 11.1 节的标准布局。
- 提示词默认带能力摘要；规则层能修的直接修，修不了才反馈给小模型。
