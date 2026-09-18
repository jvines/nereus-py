"""The user-facing API.

`JuliaDaemon.call("run_job", cfg)` is transport plumbing and should never
appear in user code: it is stringly-typed, gives no completion, no signature,
no docstring, and any typo becomes a runtime error from Julia. This module is
the typed facade over it.

Three levels, in increasing order of control:

    import astronereus
    summary = nereus.run_job(cfg)                  # 1. just do the thing

    with astronereus.session() as s:                    # 2. reuse one warm daemon
        a = s.run_job(cfg_a)
        b = s.run_job(cfg_b)
        pg = s.detect.rv_periodogram(t, rv, err)

    with astronereus.session() as s:                    # 3. escape hatch
        s.raw("some.new.action", payload)

Level 1 lazily starts a process-wide daemon on first use and reuses it, so the
~20 s `using Nereus` is paid once per interpreter, not per call.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any, Sequence

from ._daemon import JuliaDaemon
from ._result import JobResult, JobFailed
from . import _features as _F

_shared: "Session | None" = None
_shared_lock = threading.Lock()


class _Namespace:
    """Groups related actions, e.g. ``s.detect.transits`` -> "detection.transits"."""

    def __init__(self, session: "Session", prefix: str, actions: dict[str, str]):
        self._s, self._prefix, self._actions = session, prefix, actions

    def __dir__(self):
        return sorted(self._actions)

    def __getattr__(self, name: str):
        if name not in self._actions:
            raise AttributeError(
                f"{self._prefix}.{name} is not a Nereus action. "
                f"Available: {', '.join(sorted(self._actions))}")
        action = self._actions[name]

        def _call(**payload):
            return self._s.raw(action, payload)

        _call.__name__ = name
        _call.__doc__ = f"Nereus compute action {action!r}."
        return _call



class Session:
    """A warm Julia session. Use as a context manager."""

    def __init__(self, **daemon_kw):
        self._daemon = JuliaDaemon(**daemon_kw)
        g = lambda: self
        self.detect = _F._Group(g, "detect", _F.DETECT)
        self.detrend = _F._Group(g, "detrend", _F.DETREND)
        self.tomogram = _F._Group(g, "tomogram", _F.TOMOGRAM)
        self.diagnostics = _F._Group(g, "diagnostics", _F.DIAGNOSTICS)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "Session":
        self._daemon.start()
        return self

    def close(self) -> None:
        self._daemon.stop()

    def __enter__(self) -> "Session":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the actual API ----------------------------------------------------
    def run_job(self, config: dict[str, Any] | str, *,
                check: bool = True,
                timeout: float | None = None) -> JobResult:
        """Run a full Nereus job and return its summary.

        `config` is a JOB_CONFIG dict (or a path to one). Nereus writes
        `summary.json`, `chains.nc` and the plot tree into `config["output_dir"]`
        and returns the summary, which includes a `figures` manifest mapping
        logical names to paths.
        """
        # Validate client-side: a missing output_dir otherwise surfaces as an
        # opaque Julia error several minutes into a run.
        if isinstance(config, dict) and not config.get("output_dir"):
            raise ValueError("job config needs an 'output_dir' — Nereus writes "
                             "summary.json, chains.nc and the plot tree there")
        res = JobResult(self.raw("run_job", config, timeout=timeout))
        return res.check() if check else res

    def ping(self) -> dict[str, Any]:
        """Liveness + which Julia/CPU/thread count the daemon actually has."""
        return self.raw("ping", {})

    def raw(self, action: str, payload: Any = None, *,
            timeout: float | None = None) -> Any:
        """Escape hatch: call an action by name.

        Only for actions newer than this client. If you find yourself using it
        routinely, the action belongs in the typed API above.
        """
        return self._daemon.call(action, payload, timeout=timeout)

    @property
    def log_path(self):
        """Where the daemon's stdout/stderr goes — read this when a job dies."""
        return self._daemon.log_path


def session(**daemon_kw) -> Session:
    """A new warm Julia session (not the process-wide shared one)."""
    return Session(**daemon_kw)


def _shared_session(**kw) -> Session:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = Session(**kw).start()
            atexit.register(_shared.close)
    return _shared


def run_job(config: dict[str, Any] | str, *, check: bool = True,
            timeout: float | None = None, **session_kw) -> JobResult:
    """Run a job on a lazily-started, process-wide daemon.

    Convenience for one-shot use. Prefer `with astronereus.session() as s:` when
    making several calls, so the lifetime is explicit.
    """
    return _shared_session(**session_kw).run_job(config, check=check, timeout=timeout)


def ping(**session_kw) -> dict[str, Any]:
    return _shared_session(**session_kw).ping()


# --- per-technique entry points, bound to a session -------------------------
def _bind(name):
    from . import _fit
    fn = getattr(_fit, name)
    def method(self, *a, **kw):
        kw.setdefault("session", self)
        return fn(*a, **kw)
    method.__name__ = name
    method.__doc__ = fn.__doc__
    return method


for _n in ("fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint"):
    setattr(Session, _n, _bind(_n))

# post-hoc operations that act on a finished run
for _n, _act in (("evidence", "evidence"),
                 ("detection_limits", "detection_limits"),
                 ("select_planets", "select_planets"),
                 ("select_noise", "select_noise"),
                 ("pre_white", "pre_white")):
    def _mk(action):
        def m(self, **payload):
            return self.raw(action, payload)
        return m
    setattr(Session, _n, _mk(_act))
