"""A cached runtime older than RUNTIME_VERSION is refetched. No Julia, no download.

The bug this exists for: the runtime directory is named by Julia version alone,
so after `pip install -U astronereus` the old runtime still counted as installed
and every fit ran the previous Nereus until someone remembered
`install(force=True)`.
"""
from __future__ import annotations

import pytest

from astronereus import _runtime


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("NEREUS_HOME", str(tmp_path))
    for var in ("NEREUS_BUNDLE_URL", "NEREUS_NO_AUTO_INSTALL", "NEREUS_JULIA"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_runtime, "RUNTIME_VERSION", "0.8.6")
    return tmp_path


def cache_runtime(nereus_version):
    """A runtime tree that passes is_installed, carrying `nereus_version`."""
    d = _runtime.runtime_dir()
    (d / "julia" / "bin").mkdir(parents=True, exist_ok=True)
    (d / "julia" / "bin" / "julia").write_text("")
    (d / "julia" / "bin" / "julia").chmod(0o755)
    (d / "julia" / "share" / "julia").mkdir(parents=True, exist_ok=True)
    proj = d / "depot" / "dev" / "Nereus"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "Project.toml").write_text(
        f'name = "Nereus"\nuuid = "c7a1e4b5"\nversion = "{nereus_version}"\n')
    return d


class Fetched(Exception):
    pass


def refuse_to_fetch(*a, **kw):
    raise Fetched


def test_installed_version(home):
    assert _runtime.installed_version() is None
    cache_runtime("0.8.5")
    assert _runtime.is_installed()
    assert _runtime.installed_version() == "0.8.5"


@pytest.mark.parametrize("cached, stale", [("0.8.5", True), ("0.6.1", True),
                                           ("0.8.6", False), ("0.9.0", False),
                                           ("0.8.6-DEV", False)])
def test_is_stale(home, cached, stale):
    cache_runtime(cached)
    assert _runtime.is_stale() is stale


def test_a_supplied_bundle_is_never_stale(home, monkeypatch):
    cache_runtime("0.8.5")
    monkeypatch.setenv("NEREUS_BUNDLE_URL", "/Volumes/NEREUS/bundle.tar.zst")
    assert not _runtime.is_stale()


def test_install_refetches_a_stale_runtime(home, monkeypatch):
    monkeypatch.setattr(_runtime, "default_bundle", refuse_to_fetch)
    cache_runtime("0.8.5")
    with pytest.raises(Fetched):
        _runtime.install()


def test_install_keeps_a_current_runtime(home, monkeypatch):
    monkeypatch.setattr(_runtime, "default_bundle", refuse_to_fetch)
    d = cache_runtime("0.8.6")
    assert _runtime.install() == d


def test_julia_env_refreshes_a_stale_runtime(home, monkeypatch):
    calls = []

    def install(version):
        calls.append(version)
        cache_runtime("0.8.6")

    monkeypatch.setattr(_runtime, "install", install)
    cache_runtime("0.8.5")
    julia, env = _runtime.julia_env(_runtime.JULIA_VERSION)
    assert calls == [_runtime.JULIA_VERSION]
    assert _runtime.installed_version() == "0.8.6"
    assert julia == _runtime.runtime_dir() / "julia" / "bin" / "julia"

    _runtime.julia_env(_runtime.JULIA_VERSION)
    assert len(calls) == 1                     # current now: no second fetch


def test_julia_env_runs_a_stale_runtime_when_auto_install_is_off(home, monkeypatch):
    monkeypatch.setattr(_runtime, "install", refuse_to_fetch)
    monkeypatch.setenv("NEREUS_NO_AUTO_INSTALL", "1")
    cache_runtime("0.8.5")
    julia, _ = _runtime.julia_env(_runtime.JULIA_VERSION)
    assert julia.exists()
