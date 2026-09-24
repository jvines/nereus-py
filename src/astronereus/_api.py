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
import os
import sys
import threading
from typing import Any, Sequence

from ._daemon import JuliaDaemon
from ._result import JobResult, JobFailed
from ._runtime import RUNTIME_VERSION
from . import _features as _F

_shared: "Session | None" = None
_shared_lock = threading.Lock()


def _ver(v: str) -> tuple[int, ...]:
    """"0.6.0" -> (0, 6, 0). Stops at the first non-numeric part, so a
    pre-release such as "0.7.0-DEV" compares as (0, 7, 0) -- close enough for
    "is the cached runtime behind", and never a reason to raise at start-up."""
    out = []
    for part in v.strip().lstrip("v").split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    if not out:
        raise ValueError(f"unparseable version {v!r}")
    return tuple(out)


#: Contract version this client speaks. Must match `Nereus.PY_API_VERSION` in
#: the cached runtime; see the comment on that constant for the bump rule.
PY_API_VERSION = 3


class RuntimeVersionError(RuntimeError):
    """The cached Julia runtime speaks a different API than this client.

    Raised at `Session.start()` rather than left to surface mid-fit as an
    opaque Julia error. The runtime is a compiled Nereus cached under
    ~/.cache/nereus and is NOT refreshed by `pip install -U astronereus`, so
    upgrading the client alone leaves the two out of step.
    """


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
        self._check_api()
        return self

    def _check_api(self) -> None:
        """Fail fast when the cached runtime predates this client."""
        if os.environ.get("NEREUS_SKIP_API_CHECK"):
            return
        try:
            pong = self.ping()
            got = int(pong.get("api", 0))
        except Exception:
            # A daemon too broken to answer `ping` has a bigger problem than a
            # version skew; let the real failure surface on the first call.
            return
        if got == PY_API_VERSION:
            self._warn_stale_runtime(pong.get("nereus"))
            return
        try:
            self.close()   # don't leave a useless daemon running
        except Exception:
            pass           # a failure here must not mask the version error
        how = ("older than" if got < PY_API_VERSION else "newer than")
        fix = ("astronereus.install(force=True)" if got < PY_API_VERSION
               else "pip install -U astronereus")
        raise RuntimeVersionError(
            f"the cached Julia runtime speaks API v{got}, which is {how} this "
            f"client's v{PY_API_VERSION}.\n"
            f"`pip install -U astronereus` does not refresh the runtime — it is "
            f"a separate ~700 MB download cached under ~/.cache/nereus.\n\n"
            f"Fix it with:\n    {fix}\n\n"
            f"(refetch + precompile takes ~10 min. Set NEREUS_SKIP_API_CHECK=1 "
            f"to bypass this check — the mismatch will then surface as a Julia "
            f"error mid-fit instead.)")

    def _warn_stale_runtime(self, got: Any) -> None:
        """Say so when the cached runtime is older than the one we ship for.

        A COMPATIBLE skew, so it cannot be an error: the contract version is
        equal or `_check_api` would already have raised. But equal contracts do
        not mean equal science. Nereus v0.6.0 is the case that made this
        necessary -- its whole content is that default fits converge and
        `prior_rail` stops crying wolf, at contract v3, unchanged. A returning
        user who ran `pip install -U astronereus` kept a v0.5.3 runtime in the
        cache and got none of it, silently.

        Say it once per session, name the command, and get out of the way.
        """
        if not got:
            return
        try:
            if _ver(str(got)) >= _ver(RUNTIME_VERSION):
                return
        except ValueError:
            return          # unparseable version: not worth a scary message
        print(f"astronereus: the cached Julia runtime is Nereus {got}; this "
              f"client ships against {RUNTIME_VERSION}. The contract is "
              f"compatible, so this is not fatal — but fixes in the newer "
              f"runtime are not present.\n"
              f"             Refresh it with: astronereus.install(force=True)"
              f"   (~700 MB + one precompile; NEREUS_SKIP_API_CHECK=1 silences "
              f"this)", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._daemon.stop()

    #: The daemon calls it `stop`, so accept both here.
    stop = close

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
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint",
           # not a fit: renders figures from chains already on disk
           "replot"):
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


# --- data loading -----------------------------------------------------------
# Nereus exports a loader suite (load_vizier_rv, load_orvara_rv,
# load_orvara_relast, load_tess_lc, load_hip_iad, load_gost, load_gaia_dr3,
# load_hgca_row) that had no route out to Python, so users hand-parsed files
# the package already knew how to read. These bind it to the session.
def _bind_reader(name):
    from . import channels
    fn = getattr(channels, name)

    def method(self, *a, **kw):
        kw.setdefault("session", self)
        return fn(*a, **kw)
    method.__name__ = name
    method.__doc__ = fn.__doc__
    return method


for _n in ("read_rv", "read_photometry", "read_relastrom"):
    setattr(Session, _n, _bind_reader(_n))


def _bind_dataset(name):
    def method(self, *a, **kw):
        from . import datasets as _ds
        kw.setdefault("session", self)
        return getattr(_ds, name)(*a, **kw)
    method.__name__ = name
    return method


Session.dataset = _bind_dataset("dataset")
Session.list_datasets = _bind_dataset("list_datasets")

# The astrometry loaders return whole structs rather than channel data, so they
# stay raw ops rather than getting a typed wrapper each.
for _n, _act in (("load_iad", "load_iad"),
                 ("load_gost", "load_gost"),
                 ("load_gaia_dr3", "load_gaia_dr3"),
                 ("load_hgca", "load_hgca")):
    def _mkl(action):
        def m(self, **payload):
            return self.raw(action, payload)
        return m
    setattr(Session, _n, _mkl(_act))


def dataset(name: str, **kw):
    """Load a shipped dataset by name on the process-wide daemon."""
    from .datasets import dataset as _d
    return _d(name, **kw)


def list_datasets(**kw):
    """Name every dataset Nereus ships."""
    from .datasets import list_datasets as _l
    return _l(**kw)
