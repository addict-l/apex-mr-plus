# v1.3.4 C：运动 UNKNOWN 升级与规避重搜

已完成，仅修改 `skillnet/pipeline.py`、`simbridge/backend.py`，新增 `tests/test_v134_motion_feedback.py`。未改 compiler、runner、frames、primitives、APEX、标定、旧任务或 v133 结果。

先保存红灯 `out/v134_c_red.log`（16 tests，11 fail + 7 error，包括子测试），再实现。6 个 pipeline seam 先绿；最终 **17 个新测试全部通过（0.449 s）**，日志 `out/v134_c_green.log`。相关 pipeline/backend/compiler/remote mock 联合 **74/74 通过（7.042 s）**，其中包含之前的16个新测试和58个现有测试，日志 `out/v134_c_existing_focused.log`。合计75个不同测试通过；没有运行完整集合、真实仿真或规划实验。

终端独立 IK 条目中的规定原因前缀或碰撞对映射 INVALID，保存原始 UNKNOWN、原始种子列表及映射路径。RRT 两阶段证据中的 pair、branch 包装内的 answer、timeout 包装及非终止的历史恢复碰撞不被误改。非碰撞 UNKNOWN 在同一回滚后的几何上下文升级到24；升级仍未知才 cut 并重搜。

实际次数反例确认初始8次加升级24次共32次调用，升级的24个seed全部不同，重复调用升级不再查询。`deterministic_ik_seeds(cfg,count)` 不改输入配置，已有seed优先、用确定性Halton补齐，可供后续D的32种子资格验证复用。

backend 内的极窄编译器适配按 action.key 保留24种子覆盖，临时配置把 seed bank SHA 绑定到缓存版本，缓存 limits 同时为24。升级成功保留已验证前缀，原 runner compiler 对象不替换；最终编译与scratch都能复用证书。其他臂位姿改变会使缓存失效，测试没有靠伪造8种子缓存恢复。

同一动作累计三个不同状态/action cut 后，只降级本次运行的对应reach，来源明确为“运动级三次规避”；不降级liftable。support_off虽无声明capability，按arm/slot映射reach以避免后续再次选择相同托举角色。真实资格回填仍同步调用方oracle；运动降级留在工作副本，不污染输入oracle。

真实后端默认12轮，其他后端默认6轮，显式max_rounds优先。轮数耗尽为budget。若未知运动cut使受限搜索穷尽，整体保留unknown，同时保留该次搜索原no_plan，不能把有限运动采样未知改写为全域无解。失联仍按fatal路径停止，不做升级或重搜。

每轮结构化证据在 `motion_backend.json.feedback_rounds`，包括搜索状态、初始/升级结果、分类、cut、累计次数、能力降级及终态；plan.feedback保存可读文本。**旧runner的motion_feedback_count仍只统计INVALID验证，不能充当新增UNKNOWN cut总数**，批次统计应读取feedback_rounds。

只读复核已确认映射、种子独立性、前缀回滚、缓存配置及原编译器对象兼容，无blocker。源码SHA、日志SHA及接口详情见同名JSON。D、完整回归、原11对照与同20仿真复跑均未在本项执行，交由root按顺序推进。
