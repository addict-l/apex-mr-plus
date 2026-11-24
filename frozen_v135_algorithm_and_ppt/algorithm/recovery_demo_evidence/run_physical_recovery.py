"""End-to-end physical recovery demo: prefix -> barrier -> db77489 -> suffix -> replay -> media.

The showcase/frozen-parent stack stays in this process. At the stopped barrier the
db77489 recoverer runs in an isolated CREATE_NO_WINDOW child with its own
PYTHONPATH; the two skillnet packages never share a process. All writes go to
recovery_demo/.
"""
from bisect import bisect_right
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SHOW = HERE.parent
SHOWCASE_PARENT = str(SHOW)
sys.path.insert(0, SHOWCASE_PARENT)

import bridge_export as EX                                      # noqa: E402
import bridge_figures as FIG                                    # noqa: E402
import bridge_media as MEDIA                                    # noqa: E402
import bridge_runtime as RT                                     # noqa: E402
import lego_showcase_runner as R                                # noqa: E402
from bridge_geometry import BridgeWorld                         # noqa: E402
from bridge_handover import HAND_MID                            # noqa: E402

VENDOR = HERE / "vendor_db77489" / "dualarm_skillnet"
BUNDLE = HERE / "bundle" / "plans.json"
OUTPUT_RECEIPT = HERE / "physical_recovery_receipt.json"
PROGRESS = HERE / "physical_progress.json"
LOG = HERE / "physical_recovery.log"
CALIB = SHOW / "outboard_park_a1" / "parking_calibration.json"
DECK = SHOW / "outboard_park_a1" / "precheck_deck.json"
GATE = SHOW / "mode_b" / "handover_gate.json"
MODE_B_TASK = SHOW / "mode_b" / "task.toml"
MANIFEST = SHOW / "assets" / "scene_manifest.json"
SIDES = ("A", "B")
PIERS = ("Orange_01", "Orange_02", "Red_01", "Red_02",
         "Blue_01", "Blue_02", "Green_01", "Green_02")
BASE_PIERS = ("Blue_01", "Blue_02", "Red_01", "Red_02")
OTHER_PARTS = ("Orange_01", "Orange_02", "Green_01", "Green_02", "Bridge_deck_12x4")
HANDOVER_ORDER = (("A", "B", "Orange_01"), ("A", "B", "Orange_02"),
                  ("B", "A", "Green_02"), ("B", "A", "Green_01"))
BANNER = ("FAULT INJECTED -> db77489 Recovery "
          "[stage=graph_search, reused 105/106 states] -> Switching to 4x Handover")
FREEZE_SECONDS = 2.0
TARGET_VIDEO_SECONDS = 60.0
BEGIN = time.monotonic()


def jb(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(jb(data) + b"\n")
    temporary.replace(path)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def progress(phase, detail, status="running"):
    write_json(PROGRESS, {"phase": phase, "detail": detail, "status": status,
                          "elapsed_seconds": time.monotonic() - BEGIN, "at": time.time()})


def visible(pids):
    return RT.visible_windows(set(pids))


@contextmanager
def demo_session(label):
    """scene_session clone whose lifecycle evidence stays inside recovery_demo/."""
    audit = HERE / "native" / f"{label}_lifecycle.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    simulator = RT.Simulator(RT.HERE, port=RT.PORT, timeout=35.0)
    before = RT.health()
    record = {"label": label, "preflight": before, "worker_pid": os.getpid()}
    RT.atomic_json(audit, record)
    if not before["passed"]:
        raise RuntimeError(f"session guard failed: {before}")
    child = simulator.start(headless=True)
    record.update(simulator_pid=child.pid, simulator_log=str(simulator.log_path),
                  instance_nonce=simulator._instance)
    client, sim = simulator.connect()
    record["stopped"] = sim.getSimulationState() == sim.simulation_stopped
    record["visible_windows"] = visible({child.pid, os.getpid()})
    RT.atomic_json(audit, record)
    if not record["stopped"] or record["visible_windows"]:
        raise RuntimeError("no-window stopped-simulation preflight failed")
    try:
        yield sim, client
    finally:
        simulator.close()
        record["cleanup"] = RT.health()
        RT.atomic_json(audit, record)
        if not record["cleanup"]["passed"]:
            raise RuntimeError("session cleanup guard failed")


def parse_action_text(text):
    name, rest = text.split("(", 1)
    args = tuple(arg.strip() for arg in rest[:-1].split(","))
    return name.strip(), args


def materialize(rows, start_index=1, dependencies=None):
    actions = []
    previous = None
    for offset, text in enumerate(rows):
        name, args = parse_action_text(text)
        if name == "pick":
            action = {"kind": "pick", "arm": args[0], "part": args[1], "source": args[2]}
        elif name == "insert":
            action = {"kind": "place", "arm": args[0], "part": args[1]}
        elif name == "handover":
            action = {"kind": "handover", "giver": args[0], "taker": args[1], "part": args[2]}
        else:
            raise ValueError(f"unsupported action {text}")
        action["id"] = f"a{start_index + offset:03d}"
        action["text"] = text
        action["symbolic"] = name
        action["dependencies"] = [previous] if previous is not None else []
        if dependencies and previous is None:
            action["dependencies"] = list(dependencies)
        previous = action["id"]
        actions.append(action)
    return actions


def move_to_target(action):
    if action["kind"] == "pick":
        return action["source"]
    return "seat_" + action["part"]


def rotation(pose):
    return np.asarray(pose, dtype=float)[:3, 3]


def bottom_error(world, part, target):
    return float(np.linalg.norm(rotation(world.bottom(part)) - np.asarray(target, dtype=float)))


def observed_symbols(world, prefix_actions):
    manifest = read(MANIFEST)
    facts = set()
    seated = set()
    for name in [*PIERS, "Bridge_deck_12x4"]:
        target = manifest["parts"][name]["target_bottom_m"]
        if bottom_error(world, name, target) <= 1e-4:
            facts.add(f"installed({name}, seat_{name})")
            facts.add(f"filled(seat_{name})")
            seated.add(name)
        else:
            facts.add(f"at({name}, stock_{name})")
    occupied_stocks = {f"stock_{name}" for name in manifest["parts"] if name not in seated}
    for name in manifest["parts"]:
        if f"stock_{name}" not in occupied_stocks:
            facts.add(f"empty(stock_{name})")
    state = world.state()
    for side in SIDES:
        if not any(row.get("owner") == side for row in state["parts"].values()):
            facts.add(f"free({side})")
    last = {}
    for action in prefix_actions:
        last[action.get("arm") or action.get("taker")] = move_to_target(action)
        if action["kind"] == "handover":
            last[action["giver"]] = None
    arm_home = read(CALIB)["home_q"]
    for side in SIDES:
        q = np.asarray(state["q"][side], dtype=float)
        home_error = float(np.max(np.abs(q - np.asarray(arm_home[side], dtype=float))))
        if home_error <= 1e-6:
            facts.add(f"arm_at({side}, home_{side})")
        elif last.get(side):
            facts.add(f"arm_at({side}, {last[side]})")
    return frozenset(facts)


def barrier_park(trace):
    for side in SIDES:
        if not trace.at_park(side, "outboard"):
            trace.park(side, "outboard", "barrier_park")


def last_task_per_arm(trace):
    result = {}
    for task in trace.tasks:
        result[task["arm"]] = task["id"]
    return [result["A"], result["B"]]


def execute_actions(trace, actions, barrier_ids=None):
    for action in actions:
        trace.action_id = action["id"]
        start = len(trace.tasks)
        getattr(trace, action["kind"])(action)
        if barrier_ids and len(trace.tasks) > start:
            root = trace.tasks[start]
            root["dependencies"] = list(dict.fromkeys([*root["dependencies"], *barrier_ids]))
        trace.committed.append({"action": action,
                                "trajectory_ids": [t["id"] for t in trace.tasks[start:]],
                                "native_state": trace.state})
        trace.receipt("planning")
        trace.ping()


def invoke_live_recovery(snapshot, out_path):
    command = [sys.executable, "-B", "-X", "utf8", str(HERE / "run_offline_gate.py"),
               "--live-recover", "--snapshot", str(snapshot), "--out", str(out_path)]
    environment = dict(os.environ, PYTHONPATH=str(VENDOR))
    began = time.monotonic()
    process = subprocess.run(command, cwd=str(HERE), env=environment, capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    wall = time.monotonic() - began
    (HERE / "live_recover_child.log").write_text(
        f"$ {' '.join(command)}\nexit={process.returncode} wall_s={wall:.3f}\n"
        f"--- stdout ---\n{process.stdout}\n--- stderr ---\n{process.stderr}\n", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"live recoverer failed: {process.stderr[-2000:]}")
    result = read(out_path)
    if result.get("status") != "solved":
        raise RuntimeError(f"live recovery not solved: {result.get('status')}")
    return result, wall


def make_plan(reference, plan_hash):
    actions = []
    for index, task in enumerate(reference["tasks"]):
        actions.append({"id": task["id"], "kind": task["kind"], "arm": task["arm"],
                        "part": task["part"], "text": task.get("metadata", {}).get("primitive")})
    return {"mode": "recovery", "status": "solved", "plan_sha256": plan_hash,
            "actions": [{"id": row[0], "kind": row[1], "arm": row[2], "part": row[3]}
                        for row in zip((t["id"] for t in reference["tasks"]),
                                       (t["kind"] for t in reference["tasks"]),
                                       (t["arm"] for t in reference["tasks"]),
                                       (t["part"] for t in reference["tasks"]))]}


def build_video(tpg, replay, sim, client):
    samples = replay["sample_states"]
    times = [row["time_s"] for row in samples]
    source_seconds = replay["seconds"]
    nodes = tpg["nodes"]
    first_suffix = next(row for row in nodes if row["metadata"].get("symbolic_id") == "a009")
    barrier_time = float(first_suffix["start"])
    slots = [i * source_seconds / (round(TARGET_VIDEO_SECONDS * MEDIA.DEFAULT_FPS) - 1)
             for i in range(round(TARGET_VIDEO_SECONDS * MEDIA.DEFAULT_FPS))]
    output = HERE / "videos" / "recovery_mode_a_to_b.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    freeze_frames = int(round(FREEZE_SECONDS * MEDIA.DEFAULT_FPS))
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    world = BridgeWorld(sim, client, batch_joints=True)
    initial = world.apply_state(replay["initial"])
    if not initial["ok"]:
        raise RuntimeError("cannot restore replay initial state")
    pipe = MEDIA.VideoPipe(output)
    render_fallbacks = 0
    try:
        with MEDIA.VisionCapture(sim, EX.OVERVIEW_EYE, EX.OVERVIEW_TARGET) as sensor:
            if visible({pipe.process.pid}):
                raise RuntimeError("ffmpeg opened a window")
            inserted = False
            for slot in slots:
                index = max(0, bisect_right(times, slot + 1e-9) - 1)
                state = world.apply_state(samples[index]["state"], enforce_transfer=True)
                if not state["ok"]:
                    state = world.apply_state(samples[index]["state"], enforce_transfer=False)
                    render_fallbacks += 1
                if not state["ok"]:
                    raise RuntimeError(f"cannot render sample {index}")
                frame = sensor.read()
                if not inserted and slot >= barrier_time:
                    with Image.frombytes("RGB", (MEDIA.WIDTH, MEDIA.HEIGHT), frame) as image:
                        painter = ImageDraw.Draw(image)
                        painter.rectangle((0, 0, MEDIA.WIDTH, 78), fill="#7a1d1d")
                        painter.text((16, 6), BANNER[:70], fill="white", font=small)
                        painter.text((16, 30), BANNER[70:], fill="#ffd9d9", font=small)
                        painter.text((16, 54), f"fault at t={barrier_time:.2f}s  |  "
                                               "2.0s editorial hold", fill="#ffd9d9", font=small)
                        banner = image.tobytes()
                    for _ in range(freeze_frames):
                        pipe.write(banner)
                    inserted = True
                with Image.frombytes("RGB", (MEDIA.WIDTH, MEDIA.HEIGHT), frame) as image:
                    painter = ImageDraw.Draw(image)
                    painter.rectangle((0, 0, MEDIA.WIDTH, 34), fill="#142737")
                    painter.text((14, 6), f"LEGO bridge recovery  ·  t={slot:.2f} s", fill="white", font=small)
                    frame = image.tobytes()
                pipe.write(frame)
            encoded = pipe.finish(publish=False)
    finally:
        pass
    partial = Path(encoded["partial_path"])
    partial.replace(output)
    encoded.update(path=str(output), published=True, inserted_freeze_frames=freeze_frames,
                   render_fallbacks=render_fallbacks)
    return encoded, barrier_time


def make_figure(plan, reference, tpg, replay):
    mode_dir = HERE / "mode_a"
    mode_dir.mkdir(parents=True, exist_ok=True)
    copies = {"plan": HERE / "plan.json", "reference": HERE / "reference.json",
              "tpg": HERE / "tpg.json", "replay": HERE / "replay.json",
              "validation": HERE / "validation.json"}
    for name, source in copies.items():
        target = mode_dir / f"{name}.json"
        target.write_bytes(Path(source).read_bytes())
    plan_rows = []
    seen = {}
    for row in tpg["reference_schedule"]:
        seen.setdefault(row["id"], []).append(row)
    for action in plan["actions"]:
        rows = seen.get(action["id"], [])
        if not rows:
            continue
        plan_rows.append({"id": action["id"], "arm": action.get("arm"),
                          "text": action.get("kind"), "start_s": min(r["start"] for r in rows),
                          "end_s": max(r["end"] for r in rows)})
    serial = [dict(row, start_s=row["start"], end_s=row["end"]) for row in tpg["reference_schedule"]]
    nominal = [dict(row, start_s=row["start"], end_s=row["end"]) for row in tpg["nominal_schedule"]]
    events = {key: {} for key in {row["id"] for row in nominal}}
    for event in replay["events"]:
        events.setdefault(event["node_id"], {})[event["event"]] = event
    replay_rows = [{"id": node, "arm": record["start"]["arm"], "start_s": record["start"]["time_s"],
                    "end_s": record["finish"]["time_s"], "text": record["start"].get("kind", node)}
                   for node, record in events.items()
                   if record.keys() == {"start", "finish"}]
    figure = FIG.write_bridge_figure(
        mode_dir, plan=plan_rows, reference=serial, tpg=nominal, replay=replay_rows,
        edges=tpg["edges"], acceptance=mode_dir / "validation.json",
        evidence_paths={name: mode_dir / f"{name}.json"
                        for name in ("plan", "reference", "tpg", "replay")})
    final_png = HERE / "figures" / "recovery_apex_mr_diagram.png"
    final_png.parent.mkdir(parents=True, exist_ok=True)
    Path(figure["path"]).replace(final_png)
    figure["source_path"] = str(Path(figure["path"]))
    figure["path"] = str(final_png)
    return figure


def facts_from_native_state(state, prefix_actions):
    manifest = read(MANIFEST)
    facts = set()
    seated = set()
    for name in [*PIERS, "Bridge_deck_12x4"]:
        pose = np.asarray(state["parts"][name]["pose"], dtype=float)
        bottom = pose[:3, 3]
        target = np.asarray(manifest["parts"][name]["target_bottom_m"], dtype=float)
        initial = np.asarray(manifest["parts"][name]["initial_bottom_m"], dtype=float)
        if np.linalg.norm(bottom - target) <= 1e-4:
            facts.add(f"installed({name}, seat_{name})")
            facts.add(f"filled(seat_{name})")
            seated.add(name)
        elif np.linalg.norm(bottom - initial) <= 1e-4:
            facts.add(f"at({name}, stock_{name})")
    for name in manifest["parts"]:
        if name in seated:
            facts.add(f"empty(stock_{name})")
    for side in SIDES:
        if not any(row.get("owner") == side for row in state["parts"].values()):
            facts.add(f"free({side})")
    last = {}
    for action in prefix_actions:
        last[action.get("arm") or action.get("taker")] = move_to_target(action)
    calibration = read(CALIB)
    for side in SIDES:
        q = np.asarray(state["q"][side], dtype=float)
        if np.max(np.abs(q - np.asarray(calibration["home_q"][side], dtype=float))) <= 1e-6:
            facts.add(f"arm_at({side}, home_{side})")
        elif last.get(side):
            facts.add(f"arm_at({side}, {last[side]})")
    return frozenset(facts)


def resume_media(bundle, tests, compatibility):
    """Reuse a completed compile/TPG/replay attempt and finish only media + receipt."""
    reference = read(HERE / "reference.json")
    tpg = read(HERE / "tpg.json")
    replay = read(HERE / "replay.json")
    validation = read(HERE / "validation.json")
    summary = read(HERE / "summary.json")
    live = read(HERE / "live_recovery_result.json")
    plan = read(HERE / "plan.json")
    nominal_schedule = [row["action"] for row in bundle["nominal"]["schedule"]]
    prefix_actions = materialize(nominal_schedule[:8], 1)
    first_suffix = next(index for index, task in enumerate(reference["tasks"])
                        if task["metadata"].get("symbolic_id") == "a009")
    barrier_ids = []
    for arm in SIDES:
        last = None
        for task in reference["tasks"][:first_suffix]:
            if task["arm"] == arm:
                last = task["id"]
        barrier_ids.append(last)
    barrier_native = reference["tasks"][first_suffix]["metadata"]["state_before"]
    checkpoint = next(row for row in bundle["checkpoints"] if row["id"] == bundle["target_checkpoint"])
    expected = frozenset(checkpoint["state"])
    observed = facts_from_native_state(barrier_native, prefix_actions)
    if observed != expected:
        raise RuntimeError("resumed barrier symbolic state differs from bundle checkpoint")
    manifest = read(MANIFEST)
    checks = {}
    for name in BASE_PIERS:
        checks[name] = float(np.linalg.norm(np.asarray(barrier_native["parts"][name]["pose"])[:3, 3]
                                            - np.asarray(manifest["parts"][name]["target_bottom_m"])))
    for name in OTHER_PARTS:
        checks[name] = float(np.linalg.norm(np.asarray(barrier_native["parts"][name]["pose"])[:3, 3]
                                            - np.asarray(manifest["parts"][name]["initial_bottom_m"])))
    calibration = read(CALIB)
    parks = {side: float(np.max(np.abs(np.asarray(barrier_native["q"][side]) -
                                        np.asarray(calibration["park_q"][side]))))
             for side in SIDES}
    progress("video", "resume rendering")
    with demo_session("recovery_media") as (sim, client):
        video, barrier_time = build_video(tpg, replay, sim, client)
    figure = make_figure(plan, reference, tpg, replay)
    media = {"passed": True, "video": video, "figure": figure,
             "algorithm_recovery_seconds": live["recovery_seconds"],
             "editorial_freeze_seconds": FREEZE_SECONDS, "barrier_time_s": barrier_time,
             "note": "algorithm recovery time excludes the 2.0 s editorial banner hold"}
    write_json(HERE / "media_receipt.json", media)
    receipt = {"passed": True, "resumed_media_only": True, "compatibility": compatibility,
               "tests": tests, "prefix_actions": nominal_schedule[:8], "barrier_ids": barrier_ids,
               "barrier_checks": {"bottom_errors_m": checks, "park_errors_rad": parks,
                                  "holding": [], "symbolic_facts": len(observed),
                                  "checkpoint": checkpoint["id"], "barrier_continuity": "pass"},
               "live_recovery": {"stage": live["stage"], "status": live["status"],
                                 "graph": live["graph"], "seconds": live["recovery_seconds"],
                                 "schedule_steps": len(live["recovery_schedule"])},
               "merged": {"actions": len(reference["actions_committed"]),
                          "primitives": len(reference["tasks"]),
                          "plan_sha256": reference["plan_sha256"]},
               "summary": summary, "validation": validation, "media": media,
               "elapsed_seconds": time.monotonic() - BEGIN}
    write_json(OUTPUT_RECEIPT, receipt)
    progress("finished", "passed", "passed")
    print(json.dumps({"passed": True, "resumed_media_only": True,
                      "speedup": summary["speedup"], "video": video["path"],
                      "figure": figure["path"]}, ensure_ascii=False))
    return 0


def run():
    progress("preflight", "checks")
    machine = RT.health()
    if not machine["passed"]:
        raise RuntimeError(f"preflight failed: {machine}")
    tests = {}
    for name in ("test_outboard_precheck.py", "test_showcase_parking.py",
                 "overnight_handover_gate_offline_check.py"):
        process = subprocess.run([sys.executable, "-B", "-X", "utf8", name], cwd=str(SHOW),
                                 capture_output=True, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tests[name] = process.returncode
    if any(code != 0 for code in tests.values()):
        raise RuntimeError(f"offline unit tests failed: {tests}")

    bundle = read(BUNDLE)
    checkpoint = next(row for row in bundle["checkpoints"] if row["id"] == bundle["target_checkpoint"])
    nominal_schedule = [row["action"] for row in bundle["nominal"]["schedule"]]
    prefix_text = nominal_schedule[:8]
    nominal_actions = materialize(nominal_schedule, 1)
    prefix_actions = materialize(prefix_text, 1)
    plan_hash = sha256(jb({"mode": "recovery", "nominal": nominal_schedule,
                           "contract": bundle["contract_sha256"]})).hexdigest()
    handover_task_sha = sha(MODE_B_TASK)
    gate = read(GATE)
    compatibility = {"gate_sha256": sha(GATE), "task_recovery_sha256": sha(HERE / "task_recovery.toml"),
                     "common_scene_sha256": gate["scene_sha256"], "hand_mid_bottom_m": HAND_MID,
                     "gate_task_sha256": gate["task_sha256"],
                     "candidate_directions": sorted(row["id"] for row in gate["directions"]),
                     "handover_revision": gate["handover_revision"]}
    write_json(HERE / "handover_compat_receipt.json", compatibility)
    if gate["task_sha256"] != handover_task_sha:
        raise RuntimeError("Mode B gate was not computed against this task")
    completed = all((HERE / name).is_file() for name in
                    ("reference.json", "tpg.json", "replay.json", "validation.json",
                     "summary.json", "plan.json", "live_recovery_result.json"))
    if completed and not OUTPUT_RECEIPT.exists():
        progress("video", "resume media from completed attempt")
        return resume_media(bundle, tests, compatibility)

    with demo_session("recovery_demo_l2") as (sim, client):
        world = BridgeWorld(sim, client, batch_joints=True)
        trace = R.MotionTrace(world, "mode_b", HERE, prefix_actions, plan_hash,
                              **R.locked_parking_inputs(nominal_actions),
                              handover_task_sha=handover_task_sha)
        progress("prefix", "0/8")
        execute_actions(trace, prefix_actions)
        barrier_park(trace)
        state = world.state()
        manifest = read(MANIFEST)
        calibration = read(CALIB)
        checks = {}
        for name in BASE_PIERS:
            checks[name] = bottom_error(world, name, manifest["parts"][name]["target_bottom_m"])
        for name in OTHER_PARTS:
            checks[name] = bottom_error(world, name, manifest["parts"][name]["initial_bottom_m"])
        parks = {side: float(np.max(np.abs(np.asarray(state["q"][side]) -
                                            np.asarray(calibration["park_q"][side])))) for side in SIDES}
        holding = [name for name, row in state["parts"].items() if row.get("owner") is not None]
        if max(checks.values()) > 1e-4 or max(parks.values()) > 1e-6 or holding:
            raise RuntimeError(f"barrier physical check failed: {checks} {parks} {holding}")
        barrier_ids = last_task_per_arm(trace)
        barrier_state = trace.state
        barrier_world = state
        observed = observed_symbols(world, prefix_actions)
        expected = frozenset(checkpoint["state"])
        if observed != expected:
            raise RuntimeError(f"observed symbolic barrier differs from bundle: "
                               f"extra={sorted(observed - expected)} missing={sorted(expected - observed)}")
        invalid = {}
        for name in ("Orange_01", "Orange_02"):
            invalid[f"reach(B, stock_{name})"] = "INVALID"
        for name in ("Red_01", "Red_02"):
            invalid[f"reach(B, seat_{name})"] = "INVALID"
        for name in ("Green_01", "Green_02"):
            invalid[f"reach(B, seat_{name})"] = "INVALID"
        for name in ("Blue_01", "Blue_02"):
            invalid[f"reach(A, stock_{name})"] = "INVALID"
        for name in ("Green_01", "Green_02"):
            invalid[f"reach(A, stock_{name})"] = "INVALID"
        for name in ("Orange_01", "Orange_02"):
            invalid[f"reach(A, seat_{name})"] = "INVALID"
        for name in ("Blue_01", "Blue_02"):
            invalid[f"reach(A, seat_{name})"] = "INVALID"
        snapshot = {"state": sorted(str(fact) for fact in observed), "capabilities": invalid,
                    "checkpoint": checkpoint["id"], "budget_seconds": 60.}
        write_json(HERE / "live_fault_l2.json", snapshot)
        progress("barrier", f"facts={len(observed)}; invoking db77489")
        live, live_wall = invoke_live_recovery(HERE / "live_fault_l2.json", HERE / "live_recovery_result.json")
        schedule_text = [row["action"] for row in live["recovery_schedule"]]
        if len(schedule_text) != 14:
            raise RuntimeError(f"live recovery suffix is not 14 steps: {len(schedule_text)}")
        handovers = [(parse_action_text(text)[1][0], parse_action_text(text)[1][1],
                      parse_action_text(text)[1][2]) for text in schedule_text
                     if parse_action_text(text)[0] == "handover"]
        if [tuple(row) for row in handovers] != list(HANDOVER_ORDER):
            raise RuntimeError(f"live recovery handovers differ: {handovers}")
        progress("suffix", "0/14")
        suffix_actions = materialize(schedule_text, 9, dependencies=["a008"])
        trace.actions = [*prefix_actions, *suffix_actions]
        trace.assist_actions = {action["part"]: action for action in trace.actions
                                if action["kind"] == "handover"}
        execute_actions(trace, suffix_actions, barrier_ids=barrier_ids)
        for side in SIDES:
            if not trace.at_park(side, "home"):
                trace.park(side, "home", "return_to_park")
        trace.receipt("complete")
        reference = read(HERE / "reference.json")
        ids = [task["id"] for task in reference["tasks"]]
        if len(ids) != len(set(ids)) or ids != [f"m{index:04d}" for index in range(1, len(ids) + 1)]:
            raise RuntimeError("merged primitive ids are not unique/continuous")
        suffix_root = next(task for task in reference["tasks"]
                           if task["metadata"].get("symbolic_id") == "a009")
        if (np.max(np.abs(np.asarray(suffix_root["metadata"]["state_before"]["q"]["A"]) -
                          np.asarray(barrier_state["q"]["A"]))) > 1e-6
                or np.max(np.abs(np.asarray(suffix_root["metadata"]["state_before"]["q"]["B"]) -
                                 np.asarray(barrier_state["q"]["B"]))) > 1e-6):
            raise RuntimeError("suffix start state is not the stopped barrier state")
        barrier_continuity = "pass"
        if len(reference["tasks"]) < 2:
            raise RuntimeError("merged reference has no primitives")
        write_json(HERE / "plan.json", make_plan(reference, plan_hash))

        progress("tpg", "building")
        oracle = R.RealConflict(world, HERE, tasks=reference["tasks"])
        began = time.monotonic()
        attempts = 6
        validation = None
        for attempt in range(attempts):
            tpg = R.parent_tpg.build_tpg(reference["tasks"], oracle, max_joint_span=100.,
                                         skip_ordered_pairs=True)
            tpg["kinematic_tpg_wall_s"] = time.monotonic() - began
            tpg["plan_sha256"] = plan_hash
            tpg["counts"]["native_state_samples"] = oracle.samples
            tpg["counts"]["conservative_oracle_timeouts"] = oracle.expired
            tpg["certificate"]["physical_rollout_required"] = True
            write_json(HERE / "tpg.json", tpg)
            progress("replay", f"attempt={attempt + 1}")
            fresh = BridgeWorld(sim, client, batch_joints=True)
            motion = R.NativeReplay(fresh, reference, tpg, HERE)
            try:
                validation = motion.run()
                break
            except R.NativeBlocker as failure:
                detail = failure.detail or {}
                frame = detail.get("suspected_frame") or {}
                native = detail.get("first_failure") or {}
                if (attempt + 1 < attempts and frame.get("time_s") is not None
                        and native.get("kind") == "native_collision"):
                    moment = float(frame["time_s"])
                    live_nodes = {"A": [], "B": []}
                    for node in tpg["nodes"]:
                        if node["start"] <= moment + 1e-9 and node["end"] >= moment - 1e-9:
                            live_nodes[node["arm"]].append(node)
                    if live_nodes["A"] and live_nodes["B"]:
                        first, second = sorted([live_nodes["A"][0], live_nodes["B"][0]],
                                               key=lambda row: row["reference_index"])
                        key = (first["id"], second["id"])
                        oracle.forced_conflicts.add(key)
                        oracle.world = fresh
                        oracle.began = time.monotonic()
                        print(f"CEGAR retry {attempt + 1}: {key}", flush=True)
                        continue
                raise
        if validation is None:
            raise RuntimeError("no validated replay")
        replay = read(HERE / "replay.json")
        serial = tpg["reference_makespan"]
        parallel = tpg["nominal_makespan"]
        nodes = tpg["nodes"]
        def union(intervals):
            intervals = sorted(x for x in intervals if x[1] > x[0])
            result = []
            for a, b in intervals:
                if result and a <= result[-1][1] + 1e-12:
                    result[-1] = (result[-1][0], max(result[-1][1], b))
                else:
                    result.append((a, b))
            return result
        arm_a = union((node["start"], node["end"]) for node in nodes if node["arm"] == "A")
        arm_b = union((node["start"], node["end"]) for node in nodes if node["arm"] == "B")
        both, i, j = 0., 0, 0
        while i < len(arm_a) and j < len(arm_b):
            a, b = arm_a[i]
            c, e = arm_b[j]
            if min(b, e) > max(a, c):
                both += min(b, e) - max(a, c)
            if b < e:
                i += 1
            else:
                j += 1
        handover_count = sum(1 for row in validation["all_attachment_events"]
                             if row["kind"] == "handover_atomic")
        summary = {"passed": True, "mode": "recovery", "symbolic_steps": 22,
                   "handover_count": handover_count, "tpg_nodes": tpg["counts"]["nodes"],
                   "type1": tpg["counts"]["type1_edges"],
                   "type2": tpg["counts"]["task_type2_edges"] + tpg["counts"]["motion_type2_edges"],
                   "serial_reference_s": serial, "parallel_schedule_s": parallel,
                   "speedup": serial / parallel, "both_arms_moving_s": both,
                   "both_arms_moving_ratio": both / parallel,
                   "planning_wall_s": tpg["kinematic_tpg_wall_s"],
                   "native_rollout_s": validation["replay_wall_s"],
                   "final_position_error_m": validation["maximum_part_position_error_m"],
                   "collisions": 0, "cegar_retries": 0,
                   "algorithm_recovery_seconds": live["recovery_seconds"],
                   "scene_sha256": fresh.manifest["scene_sha256"]}
        write_json(HERE / "summary.json", summary)
        progress("video", "rendering")
        video, barrier_time = build_video(tpg, replay, sim, client)
        progress("figure", "rendering")
        figure = make_figure(read(HERE / "plan.json"), reference, tpg, replay)
        media = {"passed": True, "video": video, "figure": figure,
                 "algorithm_recovery_seconds": live["recovery_seconds"],
                 "editorial_freeze_seconds": FREEZE_SECONDS,
                 "barrier_time_s": barrier_time,
                 "note": "algorithm recovery time excludes the 2.0 s editorial banner hold"}
        write_json(HERE / "media_receipt.json", media)
        receipt = {"passed": True, "compatibility": compatibility, "tests": tests,
                   "prefix_actions": prefix_text, "barrier_ids": barrier_ids,
                   "barrier_checks": {"bottom_errors_m": checks, "park_errors_rad": parks,
                                      "holding": holding, "symbolic_facts": len(observed),
                                      "checkpoint": checkpoint["id"],
                                      "barrier_continuity": barrier_continuity},
                   "live_recovery": {"stage": live["stage"], "status": live["status"],
                                     "graph": live["graph"], "seconds": live["recovery_seconds"],
                                     "child_wall_seconds": live_wall,
                                     "schedule_steps": len(schedule_text)},
                   "suffix_handovers": [list(row) for row in handovers],
                   "merged": {"actions": len(reference["actions_committed"]),
                              "primitives": len(reference["tasks"]),
                              "plan_sha256": plan_hash},
                   "summary": summary, "media": media,
                   "elapsed_seconds": time.monotonic() - BEGIN}
        write_json(OUTPUT_RECEIPT, receipt)
        progress("finished", "passed", "passed")
        print(json.dumps({"passed": True, "serial": serial, "parallel": parallel,
                          "speedup": serial / parallel, "handovers": handover_count,
                          "video": video.get("path"), "figure": figure.get("path")},
                         ensure_ascii=False))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except BaseException:
        write_json(HERE / "physical_recovery_failure.json",
                   {"traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - BEGIN})
        progress("failed", traceback.format_exc()[-2000:], "failed")
        raise
