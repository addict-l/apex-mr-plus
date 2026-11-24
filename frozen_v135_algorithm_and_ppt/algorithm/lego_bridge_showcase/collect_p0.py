"""Assemble the Phase 0 result from actual native receipts and audit artifacts."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json

from PIL import Image

from bridge_runtime import HERE, ROOT, atomic_json, health


def read(path, default=None):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return default


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def link(path, label):
    path = Path(path).resolve()
    return f"[{label}](<{path.as_posix()}>)"


def mm(distance):
    return "—" if distance is None else f"{distance * 1000:.6f}"


def pairstr(pair):
    return " / ".join(pair or ()) or "—"


def status(passed):
    return "通过" if passed else "未通过"


def sample_line(prefix, row, *, p1):
    floor = row.get("minimum_clearance_m" if p1 else "minimum_distance_m")
    pair = (row.get("closest") or {}).get("geometry", {}).get("pair") if p1 else row.get("pair")
    active = row.get("sample_count" if p1 else "samples", 0)
    return f'| {prefix} | {row["phase"]} | {active} | {mm(floor)} | {pairstr(pair)} | {status(row.get("status")=="passed" if p1 else row.get("passed"))} |'


def main():
    manifest = read(HERE/"assets/scene_manifest.json", {})
    asset_origin = read(HERE/"assets/input_receipt.json", [])
    preserved = [{"original": row["original"], "copy": row["copy"], "sha256": row["sha256"].lower(),
                  "source_matches": sha(row["original"]).lower() == row["sha256"].lower(),
                  "copy_matches": sha(row["copy"]).lower() == row["sha256"].lower()}
                 for row in asset_origin]
    history = ROOT/"results/sim_v1_3/motion_v135/diagnostics/tpg_broadphase_sim20_a1"
    frozen = read(history/"isolation_receipt.json")
    history_hashes = {name:sha(history/"candidate"/name)==digest for name,digest in frozen["files"].items()}
    audit = health()
    lifecycles = {name: read(HERE/"native"/(name+"_lifecycle.json"))
                  for name in ("original_inspection","build","world_preflight","p0_1","p0_2")}
    first = lifecycles.get("original_inspection") or {}
    visibility_ok = all(scope and scope.get("stopped") and scope.get("visible_windows")==[]
                     and scope.get("simulator_log_bytes",0)>0 and scope.get("cleanup",{}).get("passed")
                     for scope in lifecycles.values())
    p1 = read(HERE/"p0_1_receipt.json", {})
    p2 = read(HERE/"p0_2_receipt.json", {})
    height_scan = read(HERE/"p0_2_height_receipt.json", {})
    height_directions = height_scan.get("directions", [])
    local = read(HERE/"p0_2_neighbor_local_receipt.json", {})
    for label, value in (("p0_2_height",height_scan),("p0_2_neighbor_local",local)):
        if value:
            scope = read(HERE/"native"/(label+"_lifecycle.json"))
            lifecycles[label] = scope
            visibility_ok = visibility_ok and bool(scope and scope.get("stopped")
                 and scope.get("visible_windows")==[] and scope.get("cleanup",{}).get("passed"))
    tilted = read(HERE/"p0_2_tilted_receipt.json", {})
    tilted_directions = tilted.get("directions", [])
    if tilted:
        lifecycles["p0_2_tilted"] = read(HERE/"native/p0_2_tilted_lifecycle.json")
        visibility_ok = visibility_ok and bool(lifecycles["p0_2_tilted"]
            and lifecycles["p0_2_tilted"].get("stopped")
            and lifecycles["p0_2_tilted"].get("visible_windows")==[]
            and lifecycles["p0_2_tilted"].get("cleanup",{}).get("passed"))
    index = p1.get("chosen_attempt", 0)
    trials = p1.get("attempts", [])
    good_deck = trials[index] if isinstance(index,int) and 0 <= index < len(trials) else {}
    all_directions = p2.get("directions", [])
    negatives = [
        {"id":"deck_narrow_opening", "passed":(p1.get("negative") or {}).get("status")=="correctly_rejected",
         **(p1.get("negative") or {})},
        *p2.get("negative_cases", [])
    ]
    capture_paths = [(good_deck.get("preview_path"), "桥面连续扫掠中的抓持")]
    hand = HERE/"p0_previews/hand_mid_orthogonal.png"
    if hand.is_file():
        capture_paths.append((hand, "正交交接过程已实扫到的交接姿态"))
    else:
        endpoint = HERE/"p0_previews/hand_mid_endpoint_candidate.png"
        if endpoint.is_file():
            capture_paths.append((endpoint, "通过端点几何校验的正交抓持设点"))
    if len(capture_paths) == 1:
        preview_dir = HERE/"p0_previews"
        probes = [*preview_dir.glob("hand_mid_tilted_*.png"), *preview_dir.glob("hand_mid_height_*.png")]
        actual = sorted(p for p in probes if "endpoint_only" not in p.name)
        static = sorted(p for p in probes if "endpoint_only" in p.name)
        blocked = preview_dir/"hand_mid_orthogonal_collision.png"
        if actual:
            capture_paths.append((actual[0], "倾斜接方连续扫掠已到达的正交交接姿态"))
        elif static:
            capture_paths.append((static[0], "通过全臂端点检查的倾斜交接设点"))
        elif blocked.is_file():
            capture_paths.append((blocked, "正交交接被拒设点的碰撞诊断图"))
    previews = []
    for path, scope in capture_paths:
        if path is not None and Path(path).is_file():
            with Image.open(path) as image:
                width,height = image.size
                image.verify()
            previews.append({"path":str(path), "scope":scope, "width":width,"height":height,"sha256":sha(path)})
    build_gate = manifest.get("build_gate_passed", False)
    p1_ok = p1.get("status") == "passed"
    direction_coverage = {row["id"] for row in [*all_directions, *tilted_directions, *height_directions] if row.get("passed")}
    p2_ok = direction_coverage == {"A_to_B", "B_to_A"}
    freeze_ok = all(row["copy_matches"] and row["source_matches"] for row in preserved) and all(history_hashes.values())
    scene_ok = bool(manifest and sha(manifest["scene"])==manifest["scene_sha256"])
    physical_gate = all((build_gate,p1_ok,p2_ok,freeze_ok,visibility_ok,audit["passed"],scene_ok,
                         len(negatives)==3,all(row.get("passed") for row in negatives),len(previews)==2))
    neighbor = {}
    parts = manifest.get("parts", {})
    if parts:
        for label, x,y in (("左桥墩","Green_01","Green_02"), ("右桥墩","Orange_01","Orange_02")):
            delta = abs(parts[x]["target_bottom_m"][1]-parts[y]["target_bottom_m"][1])
            actual_gap = delta-(parts[x]["bounds_max_m"][1]-parts[x]["bounds_min_m"][1])
            neighbor[label] = actual_gap
    result = {
        "at":datetime.now().astimezone().isoformat(), "phase":"0", "passed":physical_gate,
        "policies":{"pure_stopped_kinematics":manifest.get("pure_kinematics"),
                    "dynamics_enabled":manifest.get("dynamics_enabled"),"tcp_near_sampling_maximum_m":.001},
        "assets":preserved,"candidate_scene":manifest.get("scene"),
        "candidate_scene_sha256":manifest.get("scene_sha256"),"candidate_scene_hash_matches":scene_ok,
        "build_gate_passed":build_gate, "manifest":{
            "path":str(HERE/"assets/scene_manifest.json"),
            "SHA256":sha(HERE/"assets/scene_manifest.json") if manifest else None,
            "arms":manifest.get("arms"),"parts":parts,"support_regions":manifest.get("support_regions"),
            "support_retained":manifest.get("support_precheck")},
        "neighbor_seam_m":{"nominal":.0002,"native_measured":neighbor},
        "p0_1":{"passed":p1_ok,"state":p1.get("status"),"sweep":good_deck,
                "source":str(HERE/"p0_1_receipt.json"),"blocker":p1.get("blocking_issue")},
        "p0_2":{"passed":p2_ok,"directions":all_directions,"source":str(HERE/"p0_2_receipt.json"),
                "trace_path":p2.get("trace_path"),"error":p2.get("error"),
                "tilted_directions":tilted_directions,
                "tilted_source":str(HERE/"p0_2_tilted_receipt.json") if tilted else None,
                "tilted_trace_path":tilted.get("trace_path"),"tilted_error":tilted.get("error"),
                "height_directions":height_directions, "height_source":str(HERE/"p0_2_height_receipt.json") if height_scan else None,
                "height_trace_path":height_scan.get("trace_path"), "height_error":height_scan.get("error"),
                "local_neighbor":local},
        "negative_cases":negatives, "previews":previews,
        "audit":{"history_core_fingerprint":frozen["candidate_fingerprint"],
                 "history_core_files_checked":len(history_hashes),
                 "history_mismatches":[name for name,value in history_hashes.items() if not value],
                 "reference_hashes_passed":freeze_ok,"native_lifecycles":lifecycles,
                 "no_visible_native_windows":visibility_ok,"first_witness":first,"cleanup":audit}
    }
    atomic_json(HERE/"p0_receipt.json",result)
    lines = ["# 九块乐高桥 Phase 0 预检", "",
             f"**Phase 0 {status(physical_gate)}。** 场景构建{status(build_gate)}，P0-1 {status(p1_ok)}，P0-2 {status(p2_ok)}。", "",
             "资产口径为 8 mm 粒距、16 mm 定制桥墩砖。测试全程保持仿真停止，以关节位置和夹爪移动副驱动几何，按实测位姿校验重挂工件。九块砖保留原始网格。", "",
             "## 场景和夹爪", "",
             "| 臂 | 实测四档开口（mm） | 压块净空（mm） | 夹指宽 × 厚（mm） | 外壳 X × Y（mm） | 轨盖 X × Y（mm） |",
             "|---|---|---:|---|---|---|"]
    for side,arm in manifest.get("arms", {}).items():
        jaw = arm["jaw_evidence"]
        sizes = jaw["sizes"]
        size = lambda name, indexes: " × ".join(f'{sizes[name][0]["size_m"][idx]*1000:.2f}' for idx in indexes)
        apertures = " / ".join(format(row["measured_m"]*1000, ".3f") for row in jaw["openings"])
        lines.append(f'| {side} | {apertures} | '
                     f'{mm(jaw["press_tcp_clearance_m"])} | {size("Finger",(1,0))} | {size("Housing",(0,1))} | {size("Rail_cover",(0,1))} |')
    if manifest:
        lines += ["", "两把夹爪全行程分别采样 37 组，保留原机械步进从属关系。两臂默认脚本禁用，候选场景重载后核对脚本状态、限位和对象路径。",
                  "底板上表面为 2 mm。完成态通过四处独立承托测量。初始桥面设计底面 y 为实物资产中的 −96 mm。",
                  "模型名义邻缝 0.2 mm，原网格实测 " + "、".join(f"{name} {mm(gap)} mm" for name,gap in neighbor.items()) + "。本轮沿用实测网格。", "",
                  link(HERE/"assets/scene_manifest.json","场景清单、九块底面与四处承托坐标") + " · " + link(HERE/"assets/lego_bridge_kinematic_v1.ttt","候选场景")]
    lines += ["", "## P0-1 桥面连续扫掠", ""]
    if good_deck:
        phases = good_deck.get("phases",[])
        exact_samples = sum(row["sample_count"] for row in phases)
        largest_step = max((row.get("max_measured_position_increment_m",0.) for row in phases),default=0.)
        lines += [f'{good_deck["side"]} 臂累计 {exact_samples} 个原生样本，最大实测 TCP 步长 {mm(largest_step)} mm。'
                  f'桥面侧夹实测开口 {mm(good_deck.get("deck_grasp_opening_m"))} mm，指尖底面距底板 '
                  f'{mm(good_deck.get("fingertip_bottom_above_baseplate_m"))} mm。'
                  f'挂接世界位姿平移误差 {good_deck.get("attach_world_drift_m",0.):.3g} m。',
                  "", "| 方向 | 阶段 | 样本数 | 最小净距（mm） | 形状对 | 结果 |",
                  "|---|---|---:|---:|---|---|"]
        lines.extend(sample_line("桥面 "+good_deck["side"],row,p1=True) for row in phases)
    if p1.get("blocking_issue"):
        lines.append(f'P0-1 停点为 {p1["blocking_issue"]}。')
    lines += ["", link(HERE/"p0_1_receipt.json","桥面收据与逐阶段关节状态"),
              "", "## P0-2 双向交接与邻砖落位", ""]
    for row in all_directions:
        trials = row.get("handoff_endpoint_trials",[])
        feasible = len(row.get("handoff_endpoint_candidates",[]))
        lines.append(f'- {row["id"]}，{row["part"]} → {row["slot"]}。{len(trials)} 组正交端点参数，'
                     f'{feasible} 个通过验证的关节组合。全链{status(row.get("passed"))}。'
                     + (f'停点为 {row["reason"]}。' if row.get("reason") else ""))
    if tilted_directions:
        lines += ["", "接方倾斜复核沿用 ±X 夹持轴，夹持绑定维持至竖直落位与退爪。"]
        for row in tilted_directions:
            trials = row.get("tool_trials", [])
            candidates = row.get("tool_positives", [])
            arm_positives = row.get("arm_positives", [])
            lines.append(f'- {row["id"]}，{len(trials)} 组工具相对姿态，{len(candidates)} 组通过工具与就座端点检查，'
                         f'{len(arm_positives)} 组通过完整双臂端点检查，全链{status(row.get("passed"))}。'
                         + (f'停点为 {row["reason"]}。' if row.get("reason") else ""))
        lines.append(link(HERE/"p0_2_tilted_receipt.json", "接方倾斜复核收据"))
        if tilted.get("error"):
            lines.append(f'倾斜复核执行器错误为 {tilted["error"]}。')
    if height_directions:
        lines += ["", "夹持高度分离复核沿用同一改造场景，先以工具网格筛选，再检查完整双臂。"]
        for row in height_directions:
            trials = row.get("tool_trials",[])
            tool_positives = row.get("tool_positives",[])
            arm_positives = row.get("arm_positives",[])
            lines.append(f'- {row["id"]}，{len(trials)} 组高度和倾角，{len(tool_positives)} 组通过工具端点'
                         f'，{len(arm_positives)} 组通过双臂端点，全链{status(row.get("passed"))}。'
                         + (f'停点为 {row["reason"]}。' if row.get("reason") else ""))
        lines.append(link(HERE/"p0_2_height_receipt.json", "夹持高度分离复核收据"))
    lines.append("两套端点筛选使用实际原生网格。被拒设点的碰撞证据限于已测姿态。")
    p2_rows = [(row["id"], stage) for row in [*all_directions,*tilted_directions,*height_directions]
               for stage in row.get("stages",[])]
    for direction in local.get("directions",[]):
        chosen = next((entry for entry in direction["trials"] if entry.get("passed")), None)
        if chosen is None and direction["trials"]:
            chosen = max(direction["trials"],key=lambda item: len(item["stages"]))
        if chosen:
            lines.append(f'- {direction["id"]} 的 ±X 邻砖局部扫掠{status(chosen.get("passed"))}。'
                         '工件从经核验的局部持物姿态开始。')
            p2_rows.extend((f'{direction["id"]} 局部',stage) for stage in chosen["stages"])
    if local:
        lines.append(link(HERE/"p0_2_neighbor_local_receipt.json", "局部邻缝落位与被拒交接复现"))
    if p2_rows:
        lines += ["", "| 方向 | 阶段 | 样本数 | 最小净距（mm） | 形状对 | 结果 |",
                  "|---|---|---:|---:|---|---|"]
        lines.extend(sample_line(label,row,p1=False) for label,row in p2_rows)
    if p2.get("error"):
        lines.append(f'执行器错误为 {p2["error"]}。')
    lines += ["", link(HERE/"p0_2_receipt.json","交接收据与关节解")]
    if p2.get("trace_path") and Path(p2["trace_path"]).is_file():
        lines.append(link(p2["trace_path"],"逐样本原生碰撞与距离"))
    lines += ["","## 负例与无窗取证",""]
    for row in negatives:
        native = row.get("native",{})
        lines.append(f'- {row["id"]} {("正确拒绝" if row.get("passed") else "未获预期拒绝证据")}。'
                     f'{pairstr(native.get("pair") or row.get("native_pair"))}。')
    for preview in previews:
        lines.append(f'- {preview["scope"]}，{preview["width"]} × {preview["height"]}。'
                     + link(preview["path"],"原生 PNG"))
    lines += ["", "## 清理与保护", "",
              f'C 盘剩余 {audit["C_free_GiB"]:.3f} GiB，23210 '
              f'{"空闲" if audit["port_free"] else "占用"}。现存 CoppeliaSim 进程 {len(audit["processes"])} 个。'
              f'两份原件与参考副本哈希核验{status(freeze_ok)}，历史候选源码复核 {sum(history_hashes.values())}/{len(history_hashes)}。'
              f'首例原生 PID {first.get("simulator_pid","—")}，记录的可见窗口数 {len(first.get("visible_windows",[]))}。', "",
              link(HERE/"p0_receipt.json","结构化总收据") + " · " + link(HERE/"native/original_inspection_lifecycle.json","首例无窗与日志证据"), ""]
    (HERE/"p0_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"phase_0":physical_gate,"build":build_gate,"p0_1":p1_ok,
                      "p0_2":p2_ok,"negatives":[row["passed"] for row in negatives],
                      "previews":len(previews),"cleanup":audit["passed"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
