"""Publish videos and a timing figure from a fully accepted stopped-scene replay."""
from __future__ import annotations

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont

from bridge_media import DEFAULT_FPS, HEIGHT, WIDTH, VideoPipe, VisionCapture, sha256
from bridge_runtime import HERE, atomic_json, scene_session, visible_windows


OVERVIEW_EYE = (.62, -1.00, .69)
OVERVIEW_TARGET = (0., 0., .19)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_mode_evidence(mode):
    folder = HERE / mode
    paths = {name: folder / f"{name}.json"
             for name in ("plan", "reference", "tpg", "replay", "validation", "summary")}
    data = {name: read(path) for name, path in paths.items()}
    validation, summary, replay = data["validation"], data["summary"], data["replay"]
    if (not all(data[name].get("passed") is True for name in ("validation", "replay", "summary"))
            or validation.get("unapproved_collisions") != 0 or validation.get("collisions") != 0
            or validation != replay["validation"] or summary["mode"] != mode):
        raise ValueError("Media export requires a passed native mode validation with zero unapproved collisions")
    phash = sha256(paths["plan"])
    if (data["plan"].get("status") != "solved" or data["reference"].get("status") != "complete"
            or any(data[name]["plan_sha256"] != phash for name in ("reference", "tpg"))
            or validation["reference_plan_sha256"] != phash):
        raise ValueError("Symbolic plan, reference, TPG and replay validation disagree")
    manifest_path = HERE / "assets" / "scene_manifest.json"
    manifest = read(manifest_path)
    scene = HERE / "assets" / "lego_bridge_kinematic_v1.ttt"
    scene_hash = sha256(scene)
    if (not manifest["build_gate_passed"] or not manifest["pure_kinematics"] or manifest["dynamics_enabled"]
            or any(item["scene_sha256"] != scene_hash for item in (replay, summary, manifest))
            or validation["part_positions"].keys() != manifest["parts"].keys()):
        raise ValueError("Mode inputs do not match the signed kinematic scene and its part roster")
    if len(validation["handover_events"]) != summary["handover_count"]:
        raise ValueError("Handover evidence count differs from the mode summary")
    source_paths = {**paths, "scene": scene, "scene_manifest": manifest_path}
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    return folder, paths, data, source_paths, hashes


def _tpg_rows(data):
    src = data["tpg"]
    nominal = [dict(row, start_s=row["start"], end_s=row["end"]) for row in src["nominal_schedule"]]
    serial = [dict(row, start_s=row["start"], end_s=row["end"]) for row in src["reference_schedule"]]
    if [row["id"] for row in serial] != [task["id"] for task in data["reference"]["tasks"]]:
        raise ValueError("TPG reference rows do not match certified serial tasks")
    nodes = {row["id"]: row for row in nominal}
    events = {key: {} for key in nodes}
    for event in data["replay"]["events"]:
        node_id, kind = event["node_id"], event["event"]
        if node_id not in nodes or kind not in {"start", "finish"} or kind in events[node_id]:
            raise ValueError(f"Unexpected or duplicate recorded event for {node_id}: {kind}")
        if event["arm"] != nodes[node_id]["arm"]:
            raise ValueError(f"TPG arm and replay arm differ on {node_id}")
        events[node_id][kind] = event
    replay = []
    for node_id, record in events.items():
        if record.keys() != {"start", "finish"}:
            raise ValueError(f"Replay node {node_id} lacks its recorded start or finish")
        start, finish = record["start"], record["finish"]
        replay.append({"id": node_id, "arm": start["arm"], "start_s": start["time_s"],
                       "end_s": finish["time_s"], "text": start.get("kind", node_id)})
    return serial, nominal, replay, src["edges"]


def _timeline(data):
    replay = data["replay"]
    samples = replay["sample_states"]
    times = [row["time_s"] for row in samples]
    if (len(times) != data["validation"]["native_samples"] or len(times) < 2
            or not all(math.isfinite(value) for value in times)
            or not all(b > a for a, b in zip(times, times[1:]))
            or abs(times[0]) > 1e-8 or abs(times[-1] - replay["seconds"]) > 1e-7):
        raise ValueError("Accepted replay state trace is incomplete or has an invalid time axis")
    return samples, times


def _requests(times, start, end, fps, speed):
    count = max(2, round((end - start) * fps / speed))
    slots = [start + (end - start) * i / (count - 1) for i in range(count)]
    return Counter(bisect_right(times, target + 1e-9) - 1 for target in slots)


def _closeup_requests(validation, times, *, include_deck=False):
    windows = []
    wanted = Counter()
    events = list(validation["handover_events"])
    if include_deck:
        seat = [event for event in validation["all_attachment_events"]
                if event["part"] == "Bridge_deck_12x4" and event["kind"] == "detach_at_place"]
        if len(seat) != 1:
            raise ValueError("Mode B close-ups require one validated bridge-deck seating event")
        events.extend(seat)
    for event in sorted(events, key=lambda row:row["time_s"]):
        moment = event["time_s"]
        half = 2.5 if event["part"] == "Bridge_deck_12x4" else 1.25
        first, last = max(times[0], moment - half), min(times[-1], moment + half)
        if windows and first < windows[-1]["end_s"]:
            first = windows[-1]["end_s"]
        if first >= last:
            raise ValueError("A close-up event is outside the validated time axis")
        windows.append({"part": event["part"], "kind": event.get("kind", "handover_atomic"),
                        "event_time_s": moment, "start_s": first, "end_s": last})
        wanted.update(_requests(times, first, last, DEFAULT_FPS, 1.))
    return windows, wanted


def _camera_focus(world, now, windows):
    event = min(windows, key=lambda window: abs(window["event_time_s"] - now))
    # A held brick moves with its TCP while its saved world pose may still be from the transfer.
    focus = world.bottom(event["part"])[:3, 3].tolist()
    eye = [focus[0] + .17, focus[1] - .22, focus[2] + .13]
    return eye, focus, event


def _overlay(frame, headline, observation):
    with Image.frombytes("RGB", (WIDTH, HEIGHT), frame) as image:
        painter = ImageDraw.Draw(image)
        painter.rectangle((0, 0, WIDTH, 60), fill="#142737")
        painter.text((18, 7), headline, fill="white", font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20))
        painter.text((18, 33), observation, fill="#d6e9f4", font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16))
        return image.tobytes()


def _current_actions(arms, world_time):
    active = []
    for side, nodes in arms.items():
        label = next((row.get("text") or row["id"] for row in nodes
                      if row["start_s"] <= world_time < row["end_s"]), "Idle")
        active.append(f"{side}  {label}")
    return "   |   ".join(active)


def render_videos(folder, mode, data, recorded, source_paths, hashes):
    from bridge_geometry import BridgeWorld

    samples, times = _timeline(data)
    seconds = data["replay"]["seconds"]
    speed = seconds / 60.
    target_frames = _requests(times, 0., seconds, DEFAULT_FPS, speed)
    closeups, closeup_frames = _closeup_requests(data["validation"], times, include_deck=mode=="mode_b")
    if (mode == "mode_b") != bool(closeups):
        raise ValueError("Mode B requires accepted handover events and Mode A requires zero handovers")
    main_name = ("mode_a_free_reach.mp4" if mode == "mode_a" else "mode_b_strict_zones_handover.mp4")
    video_dir = folder / "videos"
    outputs = {"overview": video_dir / main_name}
    if closeups:
        outputs["handovers_closeup"] = video_dir / "mode_b_handovers_closeup.mp4"
    arms = {side: [row for row in recorded if row["arm"] == side] for side in ("A", "B")}
    priming = time.monotonic()
    captured = Counter()
    results = {}
    with scene_session(f"{mode}_media") as (sim, client):
        world = BridgeWorld(sim, client, batch_joints=True)
        if world.manifest["scene_sha256"] != hashes["scene"]:
            raise ValueError("Live replay scene disagrees with the signed source")
        initial = world.apply_state(data["replay"]["initial"])
        if not initial["ok"]:
            raise RuntimeError(f"Cannot restore signed initial replay state: {initial}")
        with ExitStack() as stack:
            sensor = stack.enter_context(VisionCapture(sim, OVERVIEW_EYE, OVERVIEW_TARGET))
            pipes = {key: stack.enter_context(VideoPipe(path)) for key, path in outputs.items()}
            visible = visible_windows({pipe.process.pid for pipe in pipes.values()})
            if visible:
                raise RuntimeError(f"FFmpeg displayed a window: {visible}")
            for index, row in enumerate(samples):
                state = world.apply_state(row["state"], enforce_transfer=True)
                if not state["ok"]:
                    raise RuntimeError(f"Cannot render sample {index} at {row['time_s']:.3f}s: {state}")
                active = _current_actions(arms, row["time_s"])
                if target_frames[index]:
                    sensor.set_camera(OVERVIEW_EYE, OVERVIEW_TARGET)
                    clip = _overlay(sensor.read(), f"LEGO bridge  ·  {mode[-1].upper()}  ·  "
                                    f"t={row['time_s']:.2f} s  ·  {speed:.2f}× replay", active)
                    for _ in range(target_frames[index]):
                        pipes["overview"].write(clip)
                    captured["overview"] += 1
                if closeup_frames[index]:
                    eye, focus, event = _camera_focus(world, row["time_s"], closeups)
                    sensor.set_camera(eye, focus)
                    action = "Deck seating" if event["kind"] == "detach_at_place" else "Native handover"
                    clip = _overlay(sensor.read(), f"{action}  ·  {event['part']}  ·  t={row['time_s']:.2f} s", active)
                    for _ in range(closeup_frames[index]):
                        pipes["handovers_closeup"].write(clip)
                    captured["handovers_closeup"] += 1
                if index % 100 == 0:
                    print(json.dumps({"mode": mode, "media_samples": index + 1, "total": len(times),
                                      "source_time_s": row["time_s"],
                                      "overview_frames": pipes["overview"].frame_count}), flush=True)
            for key, pipe in pipes.items():
                results[key] = pipe.finish(publish=False)
    if any(sha256(path) != hashes[name] for name, path in source_paths.items()):
        raise RuntimeError("Media replay source files changed while rendering")
    for key, path in outputs.items():
        result = results[key]
        partial = Path(result["partial_path"])
        partial.rename(path)
        result.update(path=str(path), published=True, unique_captured_states=captured[key])
    return {
        "videos": results, "capture_mode": "validated native sampled kinematic state trace",
        "source_kinematic_seconds": seconds, "overview_speed_ratio": speed, "closeup_windows": closeups,
        "duration_target_45_75_met": 45 <= results["overview"]["duration_seconds"] <= 75,
        "wall_seconds": time.monotonic() - priming, "source_samples_replayed": len(samples),
        "sensor_count": 1, "mode_media_lifecycle": str(HERE / "native" / f"{mode}_media_lifecycle.json"),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("mode_a", "mode_b"))
    args = parser.parse_args()
    folder, paths, data, source_paths, hashes = load_mode_evidence(args.mode)
    out = folder / "media_receipt.json"
    if out.exists():
        raise FileExistsError(out)
    serial, nominal, replay, edges = _tpg_rows(data)
    media = render_videos(folder, args.mode, data, replay, source_paths, hashes)
    from bridge_figures import write_bridge_figure
    figure = write_bridge_figure(
        folder, plan=data["plan"]["actions"], reference=serial, tpg=nominal, replay=replay,
        edges=edges, acceptance=paths["validation"],
        evidence_paths={key: paths[key] for key in ("plan", "reference", "tpg", "replay")},
    )
    result = {
        "passed": True, "mode": args.mode, "figure": figure,
        "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in source_paths.items()},
        **media,
    }
    atomic_json(out, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

