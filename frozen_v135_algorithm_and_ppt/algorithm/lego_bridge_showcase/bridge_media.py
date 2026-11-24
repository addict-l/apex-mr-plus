"""Read one native vision sensor and stream RGB directly to a Windows-safe encoder."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np


WIDTH, HEIGHT, CHANNELS = 1280, 720, 3
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS
DEFAULT_FPS = 24
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class VisionCapture:
    """A single explicit-handling sensor, shared by all frames and camera poses."""

    def __init__(self, sim, eye, target, *, fov_degrees=44):
        self.sim = sim
        self.handle = sim.createVisionSensor(
            1 + 2 + 128, [WIDTH, HEIGHT, 0, 0],
            [.004, 10, math.radians(fov_degrees), .01, 0, 0, 0, .82, .82, .82, 0, 0],
        )
        try:
            self.set_camera(eye, target)
        except Exception:
            self.close()
            raise

    def set_camera(self, eye, target):
        eye = np.asarray(eye, dtype=float)
        forward = np.asarray(target, dtype=float) - eye
        if eye.shape != (3,) or forward.shape != (3,) or not np.isfinite(forward).all():
            raise ValueError("Expected finite xyz eye and target")
        distance = np.linalg.norm(forward)
        if distance <= 0:
            raise ValueError("Camera eye equals target")
        forward /= distance
        right = np.cross((0, 0, 1.), forward)
        width = np.linalg.norm(right)
        if width < 1e-10:
            raise ValueError("Camera viewing direction is parallel to world up")
        right /= width
        up = np.cross(forward, right)
        matrix = np.column_stack((right, up, forward, eye))
        self.sim.setObjectMatrix(self.handle, matrix.ravel().tolist(), -1)

    def read(self) -> bytes:
        self.sim.handleVisionSensor(self.handle)
        buffer, resolution = self.sim.getVisionSensorImg(self.handle)
        if tuple(resolution) != (WIDTH, HEIGHT) or len(buffer) != FRAME_BYTES:
            raise ValueError(f"Vision sensor produced invalid RGB24 frame dimensions {resolution}")
        return np.frombuffer(buffer, np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)[::-1].tobytes()

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self.sim.removeObjects([handle])

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class VideoPipe:
    """Feed a fixed-rate RGB24 stream to hidden FFmpeg, retaining all partial failures."""

    def __init__(self, output, *, fps=DEFAULT_FPS, finalize_timeout_s=90):
        self.output = Path(output).resolve()
        self.partial = self.output.with_suffix(".partial.mp4")
        self.stderr_path = self.output.with_suffix(".ffmpeg.log")
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("Output must be a .mp4")
        self.fps = Fraction(str(fps)).limit_denominator(100_000)
        if not 0 < self.fps <= 120:
            raise ValueError("Invalid frame rate")
        self.finalize_timeout_s = finalize_timeout_s
        self.output.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.output, self.partial, self.stderr_path):
            if path.exists():
                raise FileExistsError(path)
        ffmpeg, self.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not self.ffprobe:
            raise RuntimeError("ffmpeg and ffprobe must both be on PATH")
        self.command = [
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-xerror", "-nostdin", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(self.fps), "-i", "pipe:0", "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.partial),
        ]
        self.started_at = time.monotonic()
        self.frame_count = 0
        self.write_seconds = 0.
        self.sealed = False
        # A file sink continuously accepts stderr and avoids pipe-deadlock.
        with self.stderr_path.open("xb") as errors:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=errors, creationflags=CREATE_NO_WINDOW,
            )

    def write(self, frame: bytes):
        if self.sealed or self.process.stdin.closed:
            raise RuntimeError("Encoder input has already closed")
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"Expected {FRAME_BYTES} rgb24 bytes, received {len(frame)}")
        began = time.monotonic()
        try:
            written = self.process.stdin.write(frame)
        except BrokenPipeError as error:
            raise RuntimeError(f"FFmpeg stopped input; see {self.stderr_path}") from error
        if written != FRAME_BYTES:
            raise IOError(f"Short FFmpeg write: {written}/{FRAME_BYTES}")
        self.write_seconds += time.monotonic() - began
        self.frame_count += 1

    def _probe(self) -> dict:
        probe = subprocess.run(
            [self.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration,size",
             "-of", "json", str(self.partial)],
            stdin=subprocess.DEVNULL, capture_output=True, creationflags=CREATE_NO_WINDOW,
            text=True, encoding="utf-8", errors="replace", timeout=self.finalize_timeout_s, check=False,
        )
        if probe.returncode or probe.stderr.strip():
            raise RuntimeError(f"ffprobe decode failed: {probe.stderr.strip()}")
        details = json.loads(probe.stdout)
        [stream] = details["streams"]
        size = self.partial.stat().st_size
        count = int(stream["nb_read_frames"])
        rate = Fraction(stream["avg_frame_rate"])
        duration = float(details["format"]["duration"])
        if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
                or (stream["width"], stream["height"]) != (WIDTH, HEIGHT)
                or count != self.frame_count or count == 0 or rate != self.fps
                or size != int(details["format"]["size"]) or size <= 0
                or abs(duration - count / float(rate)) > 0.002):
            raise ValueError(f"FFmpeg frames, clock, or format failed verification: {details}")
        return {
            "codec": stream["codec_name"], "pix_fmt": stream["pix_fmt"], "width": WIDTH,
            "height": HEIGHT, "fps": float(rate), "exact_fps": str(rate), "frame_count": count,
            "duration_seconds": duration, "size_bytes": size, "sha256": sha256(self.partial),
        }

    def finish(self, *, publish: bool):
        """Close input and verify complete decode before atomically promoting an accepted result."""
        if self.sealed:
            raise RuntimeError("Encoder already finished")
        self.process.stdin.close()
        try:
            status = self.process.wait(timeout=self.finalize_timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            raise
        self.sealed = True
        if status:
            raise RuntimeError(f"FFmpeg exited {status}; see {self.stderr_path}")
        elapsed = time.monotonic() - self.started_at
        result = self._probe()
        final = self.partial
        if publish:
            self.partial.rename(self.output)
            final = self.output
        return {
            **result, "passed": True, "published": bool(publish), "path": str(final),
            "partial_path": str(self.partial), "ffmpeg_pid": self.process.pid,
            "ffmpeg_stderr": str(self.stderr_path), "ffmpeg_command": self.command,
            "encode_elapsed_seconds": elapsed, "write_seconds": self.write_seconds,
            "write_seconds_per_frame": self.write_seconds / self.frame_count,
        }

    def abort(self):
        """Close only this encoder without promoting its partial artifact."""
        if self.sealed:
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.sealed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.abort()

