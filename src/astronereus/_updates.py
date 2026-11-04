"""Tell the user when a newer astronereus or a newer Nereus has been released.

Two comparisons, both against public sources:

* the installed client against the latest non-yanked, non-pre-release
  astronereus on PyPI;
* the Nereus this client runs against (`RUNTIME_VERSION`, which is pinned by the
  client -- a runtime cannot move past it on its own) against the latest
  Nereus.jl GitHub release.

Nothing here may cost the user anything. No network at import; the check runs
once per process, from `Session.start()`, on a daemon thread, with a short
timeout; any failure is silent; the answer is cached for a day in the runtime
cache directory. `ASTRONEREUS_NO_UPDATE_CHECK=1` (or a CI environment) switches
it off, and `astronereus.check_updates()` runs it on demand.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

from . import _runtime

PYPI_URL = "https://pypi.org/pypi/astronereus/json"
NEREUS_URL = "https://api.github.com/repos/jvines/Nereus.jl/releases/latest"
TIMEOUT = 2.0
CACHE_TTL = 24 * 3600.0
RETRY_TTL = 3600.0       # a cached failure is retried sooner than a success
_CACHE_NAME = "update-check.json"

_PLAIN = re.compile(r"^\d+(\.\d+)*$")


class UpdateAvailableWarning(UserWarning):
    """A newer astronereus or Nereus has been released.

    Silence it with `warnings.simplefilter("ignore", astronereus.UpdateAvailableWarning)`
    or `ASTRONEREUS_NO_UPDATE_CHECK=1`.
    """


@dataclass(frozen=True)
class UpdateStatus:
    """What `check_updates` found. A `latest_*` of None means it could not be fetched."""
    client_installed: str | None
    client_latest: str | None
    nereus_runtime: str | None      # the Nereus this client runs against; None for a dev runtime
    nereus_latest: str | None
    dev_runtime: bool = False

    @property
    def client_outdated(self) -> bool:
        return _newer(self.client_latest, self.client_installed)

    @property
    def nereus_outdated(self) -> bool:
        return _newer(self.nereus_latest, self.nereus_runtime)


def _newer(a: str | None, b: str | None) -> bool:
    """a > b, False when either is missing or not a plain release version."""
    if not a or not b or not _PLAIN.match(a) or not _PLAIN.match(b):
        return False
    return _runtime._ver(a) > _runtime._ver(b)


def disabled() -> bool:
    if os.environ.get("ASTRONEREUS_NO_UPDATE_CHECK"):
        return True
    ci = os.environ.get("CI", "")
    return bool(ci) and ci.lower() not in ("0", "false")


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "astronereus-update-check"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read(4_000_000).decode("utf-8"))


def _latest_pypi(get: Callable[[str], Any]) -> str | None:
    try:
        data = get(PYPI_URL)
        best = None
        for ver, files in data["releases"].items():
            if not _PLAIN.match(ver):                    # a/b/rc/dev/post
                continue
            if not files or all(f.get("yanked") for f in files):
                continue
            if best is None or _runtime._ver(ver) > _runtime._ver(best):
                best = ver
        return best
    except Exception:
        return None


def _latest_nereus(get: Callable[[str], Any]) -> str | None:
    try:
        rel = get(NEREUS_URL)           # /releases/latest: never a draft or pre-release
        if rel.get("draft") or rel.get("prerelease"):
            return None
        tag = str(rel["tag_name"]).lstrip("v")
        return tag if _PLAIN.match(tag) else None
    except Exception:
        return None


def _installed_client() -> str | None:
    try:
        return metadata.version("astronereus")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return None


def _dev_runtime() -> bool:
    """Whether the Nereus in use is the user's own, so comparing it means nothing."""
    return bool(os.environ.get("NEREUS_JULIA") or os.environ.get("NEREUS_BUNDLE_URL"))


def _cache_path():
    return _runtime.cache_root() / _CACHE_NAME


def _read_cache(now: float) -> tuple[str | None, str | None] | None:
    try:
        d = json.loads(_cache_path().read_text())
        age = now - float(d["checked"])
        c, n = d.get("client"), d.get("nereus")
        ttl = CACHE_TTL if (c and n) else RETRY_TTL
        if 0 <= age < ttl:
            return c, n
    except Exception:
        pass
    return None


def _write_cache(client: str | None, nereus: str | None, now: float) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"checked": now, "client": client, "nereus": nereus}))
        os.replace(tmp, p)
    except Exception:
        pass


def _status(use_cache: bool, get: Callable[[str], Any]) -> UpdateStatus:
    now = time.time()
    cached = _read_cache(now) if use_cache else None
    if cached is not None:
        client_latest, nereus_latest = cached
    else:
        client_latest, nereus_latest = _latest_pypi(get), _latest_nereus(get)
        _write_cache(client_latest, nereus_latest, now)
    dev = _dev_runtime()
    return UpdateStatus(_installed_client(), client_latest,
                        None if dev else _runtime.RUNTIME_VERSION, nereus_latest, dev)


def messages(st: UpdateStatus) -> list[str]:
    out = []
    if st.client_outdated:
        out.append(
            f"astronereus {st.client_installed} is installed; {st.client_latest} is the "
            f"latest release. Upgrade with: pip install -U astronereus  "
            f"(a new client may also fetch a newer Nereus runtime on its next start). "
            f"Silence with ASTRONEREUS_NO_UPDATE_CHECK=1.")
    if st.nereus_outdated:
        if st.client_outdated:
            out.append(
                f"The latest Nereus is {st.nereus_latest}; the installed astronereus "
                f"runs against Nereus {st.nereus_runtime}.")
        else:
            out.append(
                f"Nereus {st.nereus_latest} is released, but astronereus "
                f"{st.client_installed} (the latest client) runs against Nereus "
                f"{st.nereus_runtime}: the runtime version is pinned by the client, so "
                f"there is nothing to update yet. A client release that ships Nereus "
                f"{st.nereus_latest} has not been published; `pip install -U "
                f"astronereus` will pick it up when it is.")
    return out


def _emit(st: UpdateStatus) -> None:
    for m in messages(st):
        warnings.warn(m, UpdateAvailableWarning, stacklevel=2)


def check_updates(*, warn: bool = True, refresh: bool = True) -> UpdateStatus:
    """Compare the installed astronereus and its Nereus with the latest releases.

    Queries PyPI (astronereus) and the Nereus.jl GitHub releases, ignoring
    pre-releases and yanked files. A version that could not be fetched is None
    in the result; this never raises on network trouble. `refresh=False` accepts
    a result cached within the last 24 hours. Runs even with
    ASTRONEREUS_NO_UPDATE_CHECK set, since you asked.

    The Nereus compared is the one this client runs against (`RUNTIME_VERSION`),
    because a client pins its runtime. With NEREUS_JULIA or NEREUS_BUNDLE_URL
    set you are running your own Nereus and only the client is compared.
    """
    st = _status(not refresh, _http_json)
    if warn:
        _emit(st)
    return st


_started = False
_lock = threading.Lock()
_thread: threading.Thread | None = None


def start_background_check() -> None:
    """Once per process: look for updates without making the caller wait."""
    global _started, _thread
    if disabled():
        return
    with _lock:
        if _started:
            return
        _started = True

    def work():
        try:
            _emit(_status(True, _http_json))
        except Exception:
            pass

    _thread = threading.Thread(target=work, name="astronereus-update-check", daemon=True)
    _thread.start()
