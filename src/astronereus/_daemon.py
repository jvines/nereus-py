"""Supervise a warm Julia daemon and talk to it over a unix socket.

Deliberately NOT juliacall: embedding deadlocks under Julia's `@threads`, which
every sampler uses. A separate process also means a crashing job cannot take the
Python interpreter with it.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import _runtime

_HDR = struct.Struct(">Q")


def _enc(o):
    """JSON fallback for the types a scientific payload actually carries.

    Channels are built with `dataclasses.asdict`, which leaves numpy arrays
    untouched, so every realistic call ships ndarrays. Non-finite floats pass
    through as NaN/Infinity — `daemon.jl` reads with `allow_inf = true`.
    """
    if hasattr(o, "tolist"):            # ndarray, numpy scalar
        return o.tolist()
    if hasattr(o, "item"):              # any remaining 0-d numpy scalar
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"cannot send {type(o).__name__} to Julia")


class DaemonError(RuntimeError):
    pass


class JuliaDaemon:
    def __init__(self, version: str = _runtime.JULIA_VERSION,
                 project: str | os.PathLike | None = None,
                 threads: str | int = "auto",
                 preload: str | None = "Nereus",
                 api_file: str | list[str] | None = None,
                 startup_timeout: float = 300.0,
                 idle_timeout: float = 1800.0):
        self.version = version
        self.project = str(project) if project else None
        self.threads = threads
        self.preload = preload
        # Extra Julia file to include after `using` — the public API surface.
        # In the shipped package this lives inside Nereus itself; during
        # bring-up it is loaded from a path so it can be iterated on.
        self.api_file = api_file
        self.startup_timeout = startup_timeout
        # The daemon exits on its own if we die abnormally (atexit cannot run
        # on SIGKILL/OOM) or if nothing talks to it for this long. Verified:
        # without this a -9'd parent leaves a ~530 MB Julia process behind.
        self.idle_timeout = idle_timeout
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._tmp = Path(tempfile.mkdtemp(prefix="nereus-daemon-"))
        self.sock_path = self._tmp / "daemon.sock"
        self.log_path = self._tmp / "daemon.log"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "JuliaDaemon":
        if self._proc is not None:
            return self
        julia, env = _runtime.julia_env(self.version)
        entry = Path(__file__).parent / "julia" / "daemon.jl"

        cmd = [str(julia), f"-t{self.threads}", "--startup-file=no"]
        if self.project:
            cmd.append(f"--project={self.project}")
        # Preload inside the daemon so the ~20 s `using` cost is paid once, at
        # boot, rather than on the first job.
        ppid = os.getpid()
        if self.preload:
            _files = ([self.api_file] if isinstance(self.api_file, str)
                      else list(self.api_file or []))
            api_inc = "".join(f'include("{p}"); ' for p in _files)
            cmd += ["-e", f"@eval using {self.preload}; "
                          f"{api_inc}"
                          f"include(\"{entry}\"); "
                          f"serve(ARGS[1]; parent_pid=parse(Int,ARGS[2]), "
                          f"idle=parse(Float64,ARGS[3]))",
                    str(self.sock_path), str(ppid), str(self.idle_timeout)]
        else:
            cmd += [str(entry), str(self.sock_path), str(ppid),
                    str(self.idle_timeout)]

        log = self.log_path.open("wb")
        self._proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        atexit.register(self.stop)

        # Julia's output goes to the log, not the terminal -- it is noisy and
        # would interleave with the caller's own. But startup can take MINUTES
        # when a package needs recompiling, and a silent multi-minute wait with
        # the log in an unannounced mkdtemp is indistinguishable from a hang.
        # So: say where the log is, and tick while waiting.
        tty = sys.stderr.isatty()
        t0 = time.time()
        announced = False
        deadline = t0 + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise DaemonError(
                    f"daemon exited with {self._proc.returncode}\n"
                    f"  log: {self.log_path}\n"
                    f"--- log ---\n{self.log_path.read_text(errors='replace')[-4000:]}")
            if self.sock_path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(str(self.sock_path))
                    self._sock = s
                    if announced and tty:
                        print(f"\r  daemon ready ({time.time() - t0:.0f}s)"
                              + " " * 30, file=sys.stderr, flush=True)
                    return self
                except OSError:
                    pass
            waited = time.time() - t0
            # 3 s: long enough that a warm start stays quiet, short enough that
            # nobody wonders whether it is stuck.
            if tty and waited > 3.0:
                if not announced:
                    announced = True
                    print(f"astronereus: starting the Julia daemon "
                          f"(log: {self.log_path})", file=sys.stderr, flush=True)
                print(f"\r  waiting … {waited:.0f}s of {self.startup_timeout:.0f}s",
                      end="", file=sys.stderr, flush=True)
            time.sleep(0.1)
        self.stop()
        raise DaemonError(
            f"daemon did not become ready within {self.startup_timeout}s.\n"
            "  If it was still precompiling, that is not a failure, just a\n"
            "  longer wait than the default allows -- raise it with\n"
            "  astronereus.session(startup_timeout=1200).\n"
            f"  log: {self.log_path}\n"
            f"--- log ---\n{self.log_path.read_text(errors='replace')[-4000:]}")

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._send({"action": "shutdown"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    # -- transport ---------------------------------------------------------
    def _send(self, obj) -> None:
        body = json.dumps(obj, default=_enc).encode()
        self._sock.sendall(_HDR.pack(len(body)) + body)

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise DaemonError("daemon closed the connection")
            buf += chunk
        return bytes(buf)

    def call(self, action: str, payload=None, timeout: float | None = None):
        with self._lock:
            # start() INSIDE the lock: two threads that both saw _sock is None
            # would otherwise each spawn a Julia process.
            if self._sock is None:
                self.start()
            try:
                self._sock.settimeout(timeout)
                self._send({"action": action, "payload": payload or {}})
                (n,) = _HDR.unpack(self._recvn(8))
                resp = json.loads(self._recvn(n))
            except Exception:
                # Framing is length-prefixed, so a half-read response leaves
                # the stream desynchronised — the next call would read the
                # tail of this one as an 8-byte header. There is no
                # resynchronisation point, so drop the connection; the next
                # call reconnects.
                #
                # NOTE: a timeout here does NOT stop the Julia job. It keeps
                # running in the daemon. Call stop() if you need it dead.
                self._drop()
                raise
        if not resp.get("ok"):
            raise DaemonError(resp.get("error", "unknown error") + "\n"
                              + resp.get("backtrace", ""))
        return resp.get("result")

    def _drop(self) -> None:
        """Close the socket and forget it, so the next call reconnects."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
