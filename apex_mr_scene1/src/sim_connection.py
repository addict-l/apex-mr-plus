"""Launch an isolated CoppeliaSim instance and use its bundled ZeroMQ client.

No installation files, simulator preferences, network settings, or unrelated
processes are changed.  ``close`` never quits an instance attached by connect.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, BinaryIO
import uuid


DEFAULT_INSTALL = Path(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu")
_INSTANCE_PARAM = "dualArmBlocks.instance"
_LOG = logging.getLogger(__name__)


class Simulator:
    """A project-owned simulator, or a non-owning connection to a manual one.

    ``root`` is the project directory, not the simulator installation directory.
    Set ``COPPELIASIM_ROOT`` to override the discovered installation.
    ``start(headless=False)`` opens the normal simulator GUI.
    """

    def __init__(
        self,
        root: Path | str,
        port: int = 23010,
        host: str = "127.0.0.1",
        timeout: float = 8.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.install = Path(os.environ.get("COPPELIASIM_ROOT", DEFAULT_INSTALL))
        self.executable = self.install / "coppeliaSim.exe"
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        if not 1 <= self.port <= 65535 or self.timeout <= 0:
            raise ValueError("port must be 1..65535 and timeout must be positive")
        self.process: subprocess.Popen[bytes] | None = None
        self.client: Any = None
        self.sim: Any = None
        self.log_path: Path | None = None
        self._log_file: BinaryIO | None = None
        self._instance: str | None = None

    def _listening(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.2):
                return True
        except (ConnectionError, TimeoutError, OSError):
            return False

    def start(self, headless: bool = True) -> subprocess.Popen[bytes]:
        """Start our executable; refuse to replace or reuse an occupied port."""
        if self.process is not None:
            if self.process.poll() is None:
                raise RuntimeError("This Simulator already owns a running process")
            raise RuntimeError("Create a new Simulator object before restarting")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("start() can only launch a local simulator")
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"CoppeliaSim not found: {self.executable}. Set COPPELIASIM_ROOT."
            )
        if self._listening():
            raise RuntimeError(
                f"Port {self.host}:{self.port} is occupied. "
                "No existing simulator was changed; select a free port."
            )

        results = self.root / "results"
        results.mkdir(parents=True, exist_ok=True)
        self._instance = uuid.uuid4().hex
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = results / f"coppeliasim-{stamp}-{self._instance[:8]}.log"
        self._log_file = self.log_path.open("wb")
        args = [str(self.executable)]
        if headless:
            args.append("-h")
        args.extend(
            [
                f"-GzmqRemoteApi.rpcPort={self.port}",
                f"-G{_INSTANCE_PARAM}={self._instance}",
            ]
        )
        try:
            self.process = subprocess.Popen(
                args,
                cwd=str(self.install),
                # The Python sandbox interprets EOF on stdin as a request to
                # exit. Keep a pipe open for this process's lifetime; DEVNULL
                # makes a successfully booted headless instance quit at once.
                stdin=subprocess.PIPE,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self._log_file.close()
            self._log_file = None
            self._instance = None
            raise
        return self.process

    def connect(self) -> tuple[Any, Any]:
        """Wait at most timeout seconds for the endpoint, then obtain ``sim``.

        Direct connect without start is allowed but never grants ownership.
        The client's default 600-second server lease is shortened, and its
        send/receive calls have a finite timeout as well.
        """
        if self.client is not None:
            return self.client, self.sim
        client_dir = (
            self.install
            / "programming"
            / "zmqRemoteApi"
            / "clients"
            / "python"
            / "src"
        )
        if not (client_dir / "coppeliasim_zmqremoteapi_client").is_dir():
            raise FileNotFoundError(f"Bundled Remote API client missing: {client_dir}")
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        import zmq

        deadline = time.monotonic() + self.timeout
        while not self._listening():
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"CoppeliaSim exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"CoppeliaSim did not listen on {self.host}:{self.port} "
                    f"within {self.timeout:g}s; see {self.log_path}"
                )
            time.sleep(0.1)

        client = RemoteAPIClient(host=self.host, port=self.port)
        client.timeout = self.timeout
        # The bundled client multiplies initialTimeout by 1000 internally.
        # A zero value skips that special case; our socket timeout applies to
        # every call, rather than only its first reply.
        client.initialTimeout = 0
        timeout_ms = max(1, int(self.timeout * 1000))
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.LINGER, 0)
        try:
            sim = client.require("sim")
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError("The project-owned CoppeliaSim has exited")
                if not self._is_owned_api(sim):
                    raise RuntimeError(
                        "The API endpoint is not the project-owned instance. "
                        "Refusing to control or close it."
                    )
        except BaseException:
            self._disconnect(client)
            raise
        self.client, self.sim = client, sim
        return client, sim

    @staticmethod
    def _disconnect(client: Any) -> None:
        # Closing sockets/context is idempotent; the bundled __del__ may call
        # the same operations again when Python releases the client.
        client.socket.close(linger=0)
        client.context.term()

    def _is_owned_api(self, sim: Any) -> bool:
        value = sim.getNamedStringParam(_INSTANCE_PARAM)
        # String parameters are buffer/bytes in the 4.10 Python API.
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8")
        return self._instance is not None and value == self._instance

    def close(self) -> None:
        """Request API quit only from our still-running, nonce-matched process.

        A failed connection or quit is logged. No force-kill fallback is used.
        Manually launched instances are only disconnected.
        """
        if (
            self.process is not None
            and self.process.poll() is None
            and self.client is not None
            and self.sim is not None
        ):
            try:
                if self._is_owned_api(self.sim):
                    self.sim.quitSimulator()
                else:
                    _LOG.warning("Ownership mismatch: simulator left running")
            except Exception as exc:
                _LOG.warning("Could not request simulator quit: %s", exc)
        if self.client is not None:
            self._disconnect(self.client)
            self.client, self.sim = None, None
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                _LOG.warning(
                    "Owned simulator PID %s is still running; no process was killed. "
                    "Log: %s",
                    self.process.pid,
                    self.log_path,
                )
        if (
            self.process is not None
            and self.process.poll() is not None
            and self.process.stdin is not None
        ):
            self.process.stdin.close()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "Simulator":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
