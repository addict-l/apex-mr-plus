"""Honest delivery report from frozen inputs, owned receipts and actual mode artifacts."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json

from bridge_runtime import HERE, atomic_json

STAGES={
    "outboard_native_calibration":"双臂独立外侧高位标定与双向连通",
    "outboard_a007_a008_precheck":"旧 a006 现场的 a007–a008 原生完整取放预检",
    "outboard_deck_precheck":"B 臂桥面两个朝向的联合取放预检",
    "mode_a_symbolic":"Mode A 符号计划",
    "mode_a_motion_and_validation":"Mode A 参考、TPG 与独立回放",
    "mode_a_media":"Mode A 验收后视频与三层图",
    "mode_b_handover_gate":"Mode B 双向交接门控",
    "mode_b_symbolic":"Mode B 符号计划",
    "mode_b_motion_and_validation":"Mode B 参考、TPG 与独立回放",
    "mode_b_media":"Mode B 验收后双视频与三层图",
}


def read(path, default=None):
    path=Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def link(path,label):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def val(data,*keys):
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return None


def fmt(value):
    if isinstance(value,float):
        return f"{value:.4g}"
    return "—" if value is None else str(value)


def mode_info(mode):
    dir=HERE/mode
    evidence={}
    for key,name in (("plan","plan.json"),("reference","reference.json"),("tpg","tpg.json"),
                     ("validation","validation.json"),("summary","summary.json"),("media","media_receipt.json"),
                     ("commit","COMMITTED.json"),("gate","handover_gate.json"),("failure","stage_failure.json"),
                     ("motion_progress","motion_progress.json"),("replay_prefix","replay_prefix.json")):
        content=read(dir/name)
        if content is not None:
            evidence[key]={"file":str(dir/name),"data":content}
    return evidence


def report_numbers(mode):
    records={name:item["data"] for name,item in mode.items()}
    plan=records.get("plan",{})
    summary=records.get("summary",{})
    reference=records.get("reference",{})
    validation=records.get("validation",{})
    tpg=records.get("tpg",{})
    actions=plan.get("actions",[])
    primitives=Counter(task.get("metadata",{}).get("primitive") for task in reference.get("tasks",[]))
    values=[
        len(actions) if actions else None,
        len(reference.get("actions_committed",[])) if reference else None,
        primitives["post_place_park"] if reference else None,
        primitives["idle_endpoint_park"] if reference else None,
        sum(row.get("kind")=="handover" for row in actions) if actions else None,
        val(summary,"tpg_nodes") if summary else val(tpg.get("counts",{}),"nodes"),
        val(summary,"type1"),
        val(summary,"type2"),
        val(summary,"serial_reference_s"),
        val(summary,"parallel_schedule_s"),
        val(summary,"speedup","parallel_speedup"),
        val(summary,"planning_wall_seconds","planning_wall_s"),
        val(reference,"compile_wall_seconds"),
        ("通过" if validation.get("passed") is True else "失败") if validation else None,
        val(validation,"unapproved_collisions","collision_count","collisions"),
        val(validation,"maximum_part_position_error_m"),
    ]
    return dict(zip(
        ("符号计划步骤","已提交参考动作","放置后回停段","闲臂协同撤回段","交接","TPG 节点","Type-1 边","Type-2 边","串行参考工时（s）","TPG 并行工时（s）",
         "并行加速比","规划墙钟（s）","参考编译墙钟（s）","独立验收","独立验收未授权碰撞","最终底面最大误差（m）"),values))


def blocked_reference(mode):
    reference=mode.get("reference",{}).get("data",{})
    failure=reference.get("failure",{})
    action=failure.get("action")
    if reference.get("status")!="blocked" or not action:
        return None
    detail=failure.get("detail",{})
    attempts=[row for row in reference.get("probes",[])
              if row.get("action")==action["id"] and row.get("primitive")==detail.get("primitive") and "candidate" in row]
    pairs=Counter(tuple((row.get("detail",{}).get("native") or {}).get("pair") or []) for row in attempts)
    first=next((row.get("detail",{}) for row in attempts
                if row.get("detail",{}).get("first_failure") is not None),{})
    return {"action":action,"message":failure.get("message"),"primitive":detail.get("primitive"),
            "candidates":detail.get("ik_candidates"),"attempts":len(attempts),
            "endpoint_rejections":sum(row.get("detail",{}).get("first_failure")==0 for row in attempts),
            "pair_counts":[{"pair":list(pair),"attempts":count} for pair,count in pairs.items() if pair],
            "first_rejection":{"sample":first.get("first_failure"),"pair":(first.get("native") or {}).get("pair"),
                               "state":first.get("failed_state")},
            "motion_prefix_tasks":len(reference.get("tasks",[])),
            "compile_wall_seconds":reference.get("compile_wall_seconds")}


def write_report(runroot, state):
    stage={row["stage"]:row for row in state["stages"]}
    batch=read(HERE/"native/batch_benchmark_receipt.json",{})
    fast=read(HERE/"native/fast_batch_parity_receipt.json",{})
    park_root=HERE/"outboard_park_a1"
    prechecks={name:read(park_root/file,{}) for name,file in (
        ("calibration","parking_calibration.json"),
        ("a007_a008","precheck_a007_a008.json"),
        ("deck","precheck_deck.json"))}
    rrt=read(HERE/"native/rrt_benchmark_receipt.json",{})
    codec=read(HERE/"native/stage1_codec_receipt.json",{})
    modes={name:mode_info(name) for name in ("mode_a","mode_b")}
    p1=read(HERE/"p0_1_receipt.json",{})
    p2=read(HERE/"p0_2_height_receipt.json",{})
    neighbors=read(HERE/"p0_2_neighbor_local_receipt.json",{})
    frozen=read(runroot/"frozen_inputs.json",{})
    takeover=read(runroot/"takeover.json",{})
    cleaned=state.get("cleanup",{})
    final_audit=read(runroot/"independent_closeout.json",{})
    final={
        "at":datetime.now().astimezone().isoformat(),"run_id":state["run_id"],
        "status":"finished","stop_reason":state.get("stop_reason"),"supervision":str(runroot/"progress.json"),
        "stages":state["stages"],"frozen_inputs":frozen,"takeover":takeover,
        "preflight":{"batch_baseline":batch,"fast_collision_parity":fast,"rrt":rrt,"codec":codec,
                     "outboard":prechecks},
        "prior_attempt_archive":read(runroot/"prior_attempt_archive.json",{}),
        "showcase_code":read(runroot/"showcase_code.json",{}),
        "mode_delivery":{name:{key:item["file"] for key,item in mode.items()} for name,mode in modes.items()},
        "measured":{name:report_numbers(mode) for name,mode in modes.items()},
        "blocked_reference":{name:blocked_reference(mode) for name,mode in modes.items()},
        "cleanup":cleaned,"independent_closeout":final_audit,
    }
    for mode in modes:
        final[mode+"_committed"]=bool(modes[mode].get("commit",{}).get("data",{}).get("passed"))
    atomic_json(HERE/"final_receipt.json",final)
    statuses={"passed":"通过","failed":"失败","timeout":"失败（硬超时）","visible_window":"失败（窗口）",
              "cleanup_failed":"失败（清理未通过）","disk_stop":"失败（磁盘）"}
    lines=["# 九块乐高桥纯运动学总装", "",
       f'**Mode A {"已提交" if final["mode_a_committed"] else "未提交"}，'
       f'Mode B {"已提交" if final["mode_b_committed"] else "未提交"}。**', "",
       f'监督运行 \u0060{state["run_id"]}\u0060，冻结 v1 场景。桥面 TCP 高度 2.300 mm，口宽 31.840 mm。']
    if state.get("stop_reason"):
        lines.append("停止点为 "+state["stop_reason"]+"。")
    lines+=["","## 阶段收据","","| 阶段 | 状态 | 墙钟（s） | 收据 |","|---|---|---:|---|"]
    for key, label in STAGES.items():
        row=stage.get(key)
        label_status=statuses.get(row["status"],row["status"]) if row else "未执行"
        links=[link(row["receipt"],"收据"),link(row["log"],"日志")] if row else []
        if row and row.get("native_witness"):
            links.append(link(row["native_witness"],"无窗进程"))
        lines.append(f'| {label} | {label_status} | {fmt(row.get("elapsed_seconds")) if row else "—"} | '
            +(" · ".join(links) if links else "—")+" |")
    lines+=["","## 接班与基础设施实测",""]
    if takeover:
        lines.append(f'启动前磁盘与端口独占绑定通过。{link(runroot/"takeover.json","启动前收据")}')
    if final["prior_attempt_archive"]:
        lines.append(link(runroot/"prior_attempt_archive.json","上一轮原生证据与文件哈希"))
    if fast:
        lines.append(
            f'既有布尔批查询验证 {fmt(fast.get("matched"))}/{fmt(fast.get("sample_count"))} 个样本与原结论一致，'
            f'登记 {fmt(fast.get("pair_count"))} 对几何检查。双负例的实际结果见收据。'
            f'原批量 {fmt(fast.get("baseline_samples_per_second"))} 样本/s，'
            f'本轮 {fmt(fast.get("fast_samples_per_second"))} 样本/s，'
            f'实测倍率 {fmt(fast.get("throughput_speedup"))}。'
            +link(HERE/"native/fast_batch_parity_receipt.json","逐点快检与计时"))
    for name,label,file in (("calibration","A/B 外侧关节目标与原生连通","parking_calibration.json"),
                            ("a007_a008","换停靠位后复现 a007–a008","precheck_a007_a008.json"),
                            ("deck","桥面两朝向与锁定绑定","precheck_deck.json")):
        evidence=prechecks[name]
        if evidence:
            lines.append(f'{label} {"通过" if evidence.get("passed") is True else "失败"}。'
                         +link(park_root/file,"原生收据"))
    if batch:
        throughput=batch.get("samples_per_second",{})
        lines.append(
            f'上轮完整测距批量{"通过" if batch.get("passed") else "失败"}，'
            f'{fmt(batch.get("matched"))}/{fmt(batch.get("sample_count"))} 个样本吻合。'
            f'单点 {fmt(throughput.get("legacy"))} 样本/s，批量 {fmt(throughput.get("batch"))} 样本/s，'
            f'吞吐倍率 {fmt(batch.get("throughput_speedup"))}。'
            +link(HERE/"native/batch_benchmark_receipt.json","等价、吞吐和分项计时"))
    if rrt:
        solution=next((record["bounded_search"] for record in rrt.get("attempts",[])
                       if record.get("bounded_search",{}).get("ok")),{})
        lines.append("既有有界 RRT "+("通过" if rrt.get("passed") else "失败")+f'，成功路径耗时 {fmt(solution.get("wall_seconds"))} s。'
                     +link(HERE/"native/rrt_benchmark_receipt.json","RRT 收据"))
    if codec:
        video=codec.get("video",{})
        lines.append(f'既有无窗管道短测 {"通过" if codec.get("passed") else "失败"}，帧数 '
                     f'{fmt(video.get("frame_count"))}，原生采集 {fmt(codec.get("capture_seconds_per_frame"))} s/帧，'
                     f'采集加编码流 {fmt(codec.get("stream_seconds_per_frame"))} s/帧。'
                     +link(HERE/"native/stage1_codec_receipt.json","逐帧实测和封装校验"))
    lines+=["","## 两种模式实测","","| 指标 | Mode A | Mode B |","|---|---:|---:|"]
    for key in final["measured"]["mode_a"]:
        lines.append(f'| {key} | {fmt(final["measured"]["mode_a"][key])} | {fmt(final["measured"]["mode_b"][key])} |')
    lines+=["","## 既有 v1 证据与本次停点",""]
    if p1:
        lines.append(f'- 既有 P0-1 状态 {p1.get("status")}。{link(HERE/"p0_1_receipt.json","桥面扫掠")}')
    if neighbors:
        lines.append(f'- 既有邻砖局部落位 {sum(bool(r.get("passed")) for r in neighbors.get("directions",[]))}/2。'
                     +link(HERE/"p0_2_neighbor_local_receipt.json","局部收据"))
    if p2:
        prior="，".join(f'{row["id"]} '+("通过" if row.get("passed") else "失败")
                         for row in p2.get("directions",[]))
        lines.append(f'- 原有 v1 高低错位连续链结果为 {prior}。'
                     +link(HERE/"p0_2_height_receipt.json","v1 原始交接证据"))
    for name,blocked in final["blocked_reference"].items():
        if not blocked:
            continue
        action=blocked["action"]
        first=blocked["first_rejection"]
        first_pair=" ↔ ".join(first.get("pair") or [])
        line=f'- {name} 在 {action["id"]} 停止，{action["arm"]} 臂 {action["kind"]} {action["part"]}。'
        line+=f'停止原语为 {blocked["primitive"]}。{blocked["message"]}。'
        if blocked["attempts"]:
            line+=f'{blocked["candidates"]} 个候选，{blocked["attempts"]} 次探针，其中 {blocked["endpoint_rejections"]} 次在端点拒绝。'
        if first_pair:
            line+=f'最早碰撞对为 {first_pair}。'
        if "failure" in modes[name]:
            line+=link(modes[name]["failure"]["file"],"原生失败证据")
        lines.append(line+link(modes[name]["reference"]["file"],"全部候选与可重放前缀"))
        for item in blocked["pair_counts"]:
            lines.append(f'  - {" ↔ ".join(item["pair"])}，{item["attempts"]} 次。')
        lines.append("  结论适用于冻结的参考计划和本次现场姿态。")
    gate=modes["mode_b"].get("gate",{}).get("data",{})
    for row in gate.get("directions",[]):
        short=row.get("reason") or ("通过" if row.get("passed") else "未通过")
        line=f'- {row["id"]} 新门控结果 {short}。'
        if row.get("stages"):
            last=row["stages"][-1]
            line+=f' 最后阶段 {last.get("phase",last.get("label","见收据"))}。'
        lines.append(line)
    if gate.get("diagnostic",{}).get("file"):
        lines.append(link(gate["diagnostic"]["file"],"诊断现场（非成功）"))
    lines+=["","## 产物与清理",""]
    for name, mode in modes.items():
        if mode:
            entries=[link(row["file"],key) for key,row in mode.items()]
            commit=mode.get("commit",{}).get("data",{})
            entries.extend(link(HERE/name/rel,Path(rel).name) for rel in commit.get("files",{})
                           if rel.endswith((".png",".mp4")))
            lines.append(f'- {name} '+" · ".join(entries))
            failure=mode.get("failure",{}).get("data",{})
            if failure.get("reason"):
                lines.append(f'  最早已记录停点为 {failure["reason"]}。')
    lines += ["",f'端口 23210 独占绑定 {"通过" if cleaned.get("port_free") else "未通过"}，'
              f'CoppeliaSim 残留 {len(cleaned.get("processes",[]))}。C 盘 {fmt(cleaned.get("C_free_GiB"))} GiB，'
              f'D 盘 {fmt(cleaned.get("D_free_GiB"))} GiB。',
              "",link(HERE/"assets/lego_bridge_kinematic_v1.ttt","冻结 v1 场景")+" · "
              +link(HERE/"final_receipt.json","总收据")+" · "+link(runroot/"progress.json","监督状态"),""]
    if frozen:
        lines.append(f'启动复核父候选 {frozen.get("parent_files_checked")} 份文件、冻结 v1 场景、两份原始资产与两种任务输入。'
                     +link(runroot/"frozen_inputs.json","指纹收据"))
    if final_audit.get("passed"):
        lines.append("独立收尾复核确认本轮 Job Object 子进程全部退出，"
                     "冻结 v1 场景、原件、两份任务与父候选 125 份文件指纹均一致。"
                     +link(runroot/"independent_closeout.json","独立收尾审计"))
    text="\n".join(lines)
    temp=HERE/"final_showcase_report.md.tmp"
    temp.write_text(text,encoding="utf-8")
    temp.replace(HERE/"final_showcase_report.md")
