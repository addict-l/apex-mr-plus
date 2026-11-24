"""Owned headless CoppeliaSim sessions for the Phase 0 showcase."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import ctypes
import json
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "SPEC.md").is_file())
PORT = 23210
PARENT = ROOT / "results/sim_v1_3/motion_v135/diagnostics/tpg_broadphase_sim20_a1/candidate"
if str(PARENT) in sys.path:
    sys.path.remove(str(PARENT))
sys.path.insert(0, str(PARENT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from apexmr.sim_connection import Simulator
from tools.overnight_batch import execution_lock, process_birth


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 3.0
    while True:
        try:
            tmp.replace(path)
            return
        except PermissionError as error:
            # A Windows status reader can briefly deny replacement of the file.
            if getattr(error, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(0.03)


def port_free():
    with socket.socket() as test:
        try:
            test.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            test.bind(("0.0.0.0", PORT))
            return True
        except OSError:
            return False


def simulator_processes():
    command = """@(Get-CimInstance Win32_Process -Filter "Name='coppeliaSim.exe'" |
      Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine) | ConvertTo-Json -Compress"""
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True,
                          capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(proc.stdout or "[]")
    return [rows] if isinstance(rows, dict) else rows


def visible_windows(pids):
    api = ctypes.windll.user32
    result = []

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def collect(handle, _):
        if api.IsWindowVisible(handle):
            pid = ctypes.c_ulong()
            api.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            if pid.value in pids:
                text = ctypes.create_unicode_buffer(512)
                api.GetWindowTextW(handle, text, len(text))
                result.append({"pid": pid.value, "title": text.value})
        return 1

    api.EnumWindows(collect, 0)
    return result


def health():
    free = shutil.disk_usage("C:\\").free
    data_drive = shutil.disk_usage("D:/").free
    processes = simulator_processes()
    free_port = port_free()
    return {"at": datetime.now().astimezone().isoformat(), "port": PORT, "port_free": free_port,
            "C_free_bytes": free, "C_free_GiB": free / 1024**3, "D_free_bytes": data_drive, "D_free_GiB": data_drive / 1024**3, "processes": processes,
            "passed": free >= 5 * 1024**3 and data_drive >= 5 * 1024**3 and not processes and free_port}


@contextmanager
def scene_session(label="bridge"):
    """Start one stopped instance and retain identity and cleanup evidence."""
    audit = HERE / "native" / (label + "_lifecycle.json")
    audit.parent.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(HERE, port=PORT, timeout=35.0)
    import os
    with execution_lock(ROOT / "results/sim_v1_3/overnight_execution.lock"):
        before = health()
        record = {"label": label, "preflight": before, "worker_pid": os.getpid(), "worker_birth": process_birth(os.getpid()), "stage_nonce": os.environ.get("BRIDGE_STAGE_NONCE"), "headless_arg": "-h"}
        atomic_json(audit, record)
        if not before["passed"]:
            raise RuntimeError(f"Existing simulator, port conflict or insufficient disk: {before}")
        try:
            child = simulator.start(headless=True)
            record.update(simulator_pid=child.pid, simulator_birth=process_birth(child.pid), instance_nonce=simulator._instance, simulator_log=str(simulator.log_path))
            client, sim = simulator.connect()
            record["stopped"] = sim.getSimulationState() == sim.simulation_stopped
            record["visible_windows"] = visible_windows({child.pid, os.getpid()})
            record["simulator_log_bytes"] = simulator.log_path.stat().st_size
            record["process_witness"] = [p for p in simulator_processes() if p["ProcessId"] == child.pid]
            atomic_json(audit, record)
            if not record["stopped"] or record["visible_windows"] or not record["simulator_log_bytes"]:
                raise RuntimeError("No-window stopped-simulation preflight failed")
            print(f"Connected stopped headless simulator PID {child.pid} ({label})", flush=True)
            idle_before = sim.getInt32Param(sim.intparam_idle_fps)
            sim.setInt32Param(sim.intparam_idle_fps, 0)
            record["idle_dispatch"] = {"previous_fps": idle_before, "active_fps": sim.getInt32Param(sim.intparam_idle_fps)}
            atomic_json(audit, record)
            try:
                yield sim, client
            finally:
                sim.setInt32Param(sim.intparam_idle_fps, idle_before)
                record["idle_dispatch"]["restored"] = sim.getInt32Param(sim.intparam_idle_fps) == idle_before
            if sim.getSimulationState() != sim.simulation_stopped:
                raise RuntimeError("A Phase 0 operation started dynamics")
        finally:
            simulator.close()
            record["cleanup"] = health()
            atomic_json(audit, record)
        if not record["cleanup"]["passed"]:
            raise RuntimeError("Simulator cleanup guard failed")


def transform(sim, handle, relative=-1):
    value = np.eye(4)
    value[:3] = np.asarray(sim.getObjectMatrix(handle, relative), dtype=float).reshape(3, 4)
    return value


def set_transform(sim, handle, pose, relative=-1):
    sim.setObjectMatrix(handle, np.asarray(pose, dtype=float)[:3].ravel().tolist(), relative)


def capture(sim, path, eye, target, width=1440, height=900, fov=44):
    import math

    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 0, 1.], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, up, forward))
    pose[:3, 3] = eye
    sensor = sim.createVisionSensor(1+2+128, [width, height, 0, 0],
        [.004, 10, math.radians(fov), .01, 0, 0, 0, .82, .82, .82, 0, 0])
    try:
        set_transform(sim, sensor, pose)
        sim.handleVisionSensor(sensor)
        buffer, size = sim.getVisionSensorImg(sensor)
        picture = Image.frombytes("RGB", tuple(size), buffer).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if np.asarray(picture, dtype=float).std() < 3:
            raise ValueError("Vision sensor returned an essentially flat frame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        picture.save(path)
        return str(path)
    finally:
        sim.removeObjects([sensor])
