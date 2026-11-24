"""Render and encode 120 native frames in one owned stopped session."""
from __future__ import annotations

import json
import time

import numpy as np

from bridge_media import VideoPipe, VisionCapture, sha256
from bridge_runtime import HERE, atomic_json, scene_session, visible_windows


def main():
    manifest = json.loads((HERE / "assets/scene_manifest.json").read_text(encoding="utf-8"))
    scene = HERE / "assets/lego_bridge_kinematic_v1.ttt"
    digest = sha256(scene)
    if digest != manifest["scene_sha256"] or not manifest.get("build_gate_passed"):
        raise ValueError("Stage 1 requires the signed kinematic scene")
    native = HERE / "native"
    result_path = native / "stage1_codec_receipt.json"
    output = native / "stage1_codec_120.mp4"
    label = "stage1_codec"
    if any(path.exists() for path in (result_path, output, native / f"{label}_lifecycle.json")):
        raise FileExistsError(f"Existing Stage 1 attempt at {native}")
    started = time.monotonic()
    with scene_session(label) as (sim, _):
        sim.loadScene(str(scene))
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("The native codec test requires a stopped simulation")
        with VisionCapture(sim, [.62, -1.00, .69], [0., 0., .19]) as sensor, VideoPipe(output) as pipe:
            frame_started = time.monotonic()
            capture_seconds = 0.
            for index in range(120):
                before = time.monotonic()
                frame = sensor.read()
                capture_seconds += time.monotonic() - before
                if index == 0:
                    contrast = float(np.frombuffer(frame, dtype=np.uint8).std())
                    if contrast < 3:
                        raise ValueError("Native sensor returned a nearly uniform first frame")
                    visible = visible_windows({pipe.process.pid})
                    if visible:
                        raise RuntimeError(f"FFmpeg displayed a window: {visible}")
                    print(json.dumps({"native_frame": index + 1, "sensor": sensor.handle,
                                      "ffmpeg_pid": pipe.process.pid, "ffmpeg_visible_windows": visible}), flush=True)
                pipe.write(frame)
                if index % 30 == 29:
                    print(json.dumps({"native_frames": index + 1, "seconds": time.monotonic() - frame_started}),
                          flush=True)
            stream_elapsed = time.monotonic() - frame_started
            sensor_handle = sensor.handle
            record = pipe.finish(publish=True)
    lifecycle = json.loads((native / f"{label}_lifecycle.json").read_text(encoding="utf-8"))
    if (not lifecycle["cleanup"]["passed"]
            or record["frame_count"] != 120 or abs(record["duration_seconds"] - 5.) > 1e-6):
        raise RuntimeError("120-frame/native-lifecycle receipt did not pass")
    result = {
        "passed": True, "capture": "native stopped CoppeliaSim RGB24", "vision_sensor_count": 1,
        "sensor": sensor_handle, "scene_sha256": digest, "video": record, "first_frame_contrast": contrast,
        "elapsed_seconds": time.monotonic() - started, "native_capture_seconds": capture_seconds,
        "capture_seconds_per_frame": capture_seconds / 120,
        "stream_seconds_per_frame": stream_elapsed / 120,
        "lifecycle_sha256": sha256(native / f"{label}_lifecycle.json"),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

