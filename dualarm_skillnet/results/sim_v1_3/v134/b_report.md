# v1.3.4 B：松弛图求值提速

FIX §3 的实现与聚焦证据已完成。未运行完整默认集合、真实 IK、仿真或泛化批次；没有修改 search.py、cost 标定、技能库、旧任务和 v1.3.3 结果。

## 接口与语义

- `HMaxBound(engine, ..., allowed=None)` 仅用允许且能力 VALID 的动作建立规则、前提/额外依赖和负目标清理；None 保持全域。已运行的动作即使在 allowed 外，其承诺效果仍会注入。
- `evaluate(state, running=())` 一次松弛返回 `(h_max, h_add)`；`__call__` / `estimate` 仍返回旧 HMax。`hadd` 仅用于非可容许的首解排序，不能构成 lower_bound 或 bound 证书。正目标 earliest 之和加上原 HMax 的负目标、running 剩余和未来清理 max 约束；共用缓存，不二次求闭包。
- 前提采用去重计数，在最后一个前提出堆时触发。支撑/锚点等额外依赖在规则已启用后出堆仍重算；自定义 optimistic 回调造成已出堆事实变早时重新传播，不重复扣计数。
- 标准 BlocksHooks 在无 overhang 效果时的恒等回调可跳过。仅对精确类、导入时保存的原始方法身份及原动作代理启用；实例覆盖、类覆盖、子类、自定义 hooks 均保留泛型回调。overhang 原回调和额外依赖不变。
- `build_heuristic(..., allowed=None)` 可构造子问题 HMax；PDB 仍为全域下界。A 阶段需自行调用受限实例，最终全空间证书仍用 allowed=None。B 没有改搜索调用。

## 红灯与聚焦绿灯

- `out/v134_b_dual_red.txt`：旧接口缺少 evaluate，随后 dual green。
- `out/v134_b_allowed_red_corrected.txt`：旧构造器不支持 allowed；校正测试夹具后仅该预期错误。
- `out/v134_b_identity_guard_red.txt`：类回调被覆盖时不能跳过，旧优化反例得到 4 而应为 42；已修成冻结原函数身份。
- 最终 `out/v134_b_focused_final_green.txt`：43 项全部通过，57.056 秒，包含新边界、bounds、v11_search、v11_ordering、v11_admissibility、cost_calibration。原断言没有删除或放宽。

## 真实状态等价与性能

对指定的四份 v1.3.3 qualified 输入，各用冻结旧 HMax 引导真实事件搜索，记录最先 300 个不同的实际生成状态（含 running），达到 300 即停止；每任务采样上限 60 秒，不以此宣称任务求解。未使用 proposal/witness。每个状态独立关闭缓存重算新旧 HMax，**1,200/1,200 完全相等，数值差为 0**。

| 输入 | 状态数 | running 状态 | 全规则 / goal_domain 规则 | 数值不同 | 最终均值 ms | 中位 ms | 最大 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| F3_024 | 300 | 267 | 1110 / 420 | 0 | 5.908 | 5.658 | 10.817 |
| F3_001 | 300 | 249 | 228 / 114 | 0 | 1.080 | 0.899 | 5.039 |
| F3_029 | 300 | 275 | 168 / 54 | 0 | 0.692 | 0.618 | 2.048 |
| F1_006 | 300 | 267 | 168 / 20 | 0 | 0.751 | 0.619 | 8.728 |

15 块墙完整 1,110 条规则的最终无缓存平均 5.908 ms，达到每状态平均 ≤8 ms 的性能目标。最大 10.817 ms 仍超过 8，**不宣称单次最坏耗时≤8 ms**；OS/运行负载可引入抖动。原型提到的180条规则并非本次全部任务的实际 goal_domain，本次15块墙为420条，按真实数量记录。

第一轮计数实现均值10.766 ms，未达8 ms，保留 `b_equivalence_fix1.json` 和 `out/v134_b_equivalence_fix1.txt`。第二轮同进程配对测量：冻结旧实现14.687 ms、新实现5.057 ms，受限图1.683 ms；完整记录在 `out/v134_b_equivalence_fix2.txt`。最终仅函数身份守卫进一步收紧后，重放原1,200状态（不新增搜索），最终源指纹与逐项性能在 `b_equivalence.json` / `out/v134_b_final_replay_green.txt`。不同轮次的性能值不能当作同一次配对测量。

## 证据与后续

- `b_before/manifest.json`：D/B修订前 bounds.py 和 heuristic.py 原字节及 SHA256；冻结旧实现作为独立数值基准。
- `b_states_f3_024.json`、`b_states_f3_001.json`、`b_states_f3_029.json`、`b_states_f1_006.json`：全部实际状态、running动作/剩余时间、旧/新值。
- `b_equivalence.json`：输入 SHA、采样范围、规则数、最终源码 SHA、性能及测试证据。
- `out/v134_b_verify.py` 是证据采集脚本，不触发模拟器或批次；仅报告本次状态，不充作独立最优性或真实运动证明。

B 到此冻结。A/C/D 及四项完成后的串行默认全集和11任务逐项对照由主流程继续，当前B聚焦不能代替那些后续验收。
