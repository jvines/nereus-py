"""Update warnings: mocked HTTP only."""
from __future__ import annotations

import json
import time
import warnings

import pytest

from astronereus import _runtime, _updates
from astronereus._updates import UpdateAvailableWarning


def pypi(*versions, yanked=()):
    return {"releases": {v: [{"yanked": v in yanked}] for v in versions}}


def gh(tag, **kw):
    return {"tag_name": tag, "draft": False, "prerelease": False, **kw}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEREUS_HOME", str(tmp_path))
    for v in ("ASTRONEREUS_NO_UPDATE_CHECK", "CI", "NEREUS_JULIA", "NEREUS_BUNDLE_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(_runtime, "RUNTIME_VERSION", "0.8.10")
    monkeypatch.setattr(_updates, "_installed_client", lambda: "0.4.38")
    monkeypatch.setattr(_updates, "_started", False)
    calls = []

    def serve(pypi_doc, gh_doc):
        def get(url):
            calls.append(url)
            doc = pypi_doc if "pypi" in url else gh_doc
            if isinstance(doc, Exception):
                raise doc
            return doc
        monkeypatch.setattr(_updates, "_http_json", get)
        return calls
    return serve


def run(warn=True, refresh=True):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st = _updates.check_updates(warn=warn, refresh=refresh)
    return st, [x for x in w if issubclass(x.category, UpdateAvailableWarning)]


def test_warning_is_a_userwarning():
    assert issubclass(UpdateAvailableWarning, UserWarning)


def test_newer_client_warns_with_command(env):
    env(pypi("0.4.37", "0.4.38", "0.5.0"), gh("v0.8.10"))
    st, w = run()
    assert st.client_latest == "0.5.0" and st.client_outdated and not st.nereus_outdated
    assert len(w) == 1
    msg = str(w[0].message)
    assert "0.4.38" in msg and "0.5.0" in msg and "pip install -U astronereus" in msg


def test_same_version_is_silent(env):
    env(pypi("0.4.38"), gh("v0.8.10"))
    st, w = run()
    assert not w and not st.client_outdated and not st.nereus_outdated


def test_installed_newer_than_pypi_is_silent(env):
    env(pypi("0.4.30"), gh("v0.8.10"))
    assert not run()[1]


def test_prerelease_and_yanked_ignored(env):
    env(pypi("0.4.38", "0.5.0rc1", "0.5.0.dev2", "0.4.39", "0.4.40", yanked=("0.4.40",)),
        gh("v0.8.10"))
    st, _ = run()
    assert st.client_latest == "0.4.39"


def test_github_prerelease_ignored(env):
    env(pypi("0.4.38"), gh("v0.9.0", prerelease=True))
    st, w = run()
    assert st.nereus_latest is None and not w


def test_nereus_newer_with_current_client(env):
    env(pypi("0.4.38"), gh("v0.9.0"))
    st, w = run()
    assert st.nereus_outdated and len(w) == 1
    msg = str(w[0].message)
    assert "0.9.0" in msg and "0.8.10" in msg and "pinned by the client" in msg


def test_both_outdated_two_warnings(env):
    env(pypi("0.5.0"), gh("v0.9.0"))
    st, w = run()
    assert len(w) == 2


def test_dev_runtime_not_compared(env, monkeypatch):
    monkeypatch.setenv("NEREUS_JULIA", "/somewhere/julia")
    env(pypi("0.4.38"), gh("v0.9.0"))
    st, w = run()
    assert st.dev_runtime and st.nereus_runtime is None and not w


def test_bundle_override_not_compared(env, monkeypatch):
    monkeypatch.setenv("NEREUS_BUNDLE_URL", "/usb/bundle.tar.zst")
    env(pypi("0.4.38"), gh("v0.9.0"))
    assert not run()[1]


@pytest.mark.parametrize("bad", [OSError("offline"), TimeoutError(), ValueError("junk"),
                                 {"nope": 1}, None, []])
def test_failures_are_silent(env, bad):
    env(bad, bad)
    st, w = run()
    assert not w and st.client_latest is None and st.nereus_latest is None


def test_one_source_down_other_still_works(env):
    env(OSError("offline"), gh("v0.9.0"))
    st, w = run()
    assert st.client_latest is None and st.nereus_latest == "0.9.0" and len(w) == 1


def test_cache_hit_skips_network(env):
    calls = env(pypi("0.5.0"), gh("v0.8.10"))
    run()
    assert len(calls) == 2
    st, w = run(refresh=False)
    assert len(calls) == 2 and st.client_latest == "0.5.0" and len(w) == 1


def test_refresh_default_bypasses_cache(env):
    calls = env(pypi("0.5.0"), gh("v0.8.10"))
    run()
    run()
    assert len(calls) == 4


def test_cache_expires_after_a_day(env):
    calls = env(pypi("0.5.0"), gh("v0.8.10"))
    run()
    p = _updates._cache_path()
    d = json.loads(p.read_text())
    d["checked"] = time.time() - 25 * 3600
    p.write_text(json.dumps(d))
    run(refresh=False)
    assert len(calls) == 4


def test_failed_fetch_is_retried_within_the_hour_not_the_minute(env):
    calls = env(OSError(), OSError())
    run()
    run(refresh=False)
    assert len(calls) == 2            # cached failure
    d = json.loads(_updates._cache_path().read_text())
    d["checked"] = time.time() - 2 * 3600
    _updates._cache_path().write_text(json.dumps(d))
    run(refresh=False)
    assert len(calls) == 4


def test_corrupt_cache_is_ignored(env):
    calls = env(pypi("0.5.0"), gh("v0.8.10"))
    _updates._cache_path().parent.mkdir(parents=True, exist_ok=True)
    _updates._cache_path().write_text("{not json")
    st, _ = run(refresh=False)
    assert st.client_latest == "0.5.0" and len(calls) == 2


def test_unwritable_cache_is_silent(env, monkeypatch):
    env(pypi("0.5.0"), gh("v0.8.10"))
    monkeypatch.setattr(_updates, "_cache_path", lambda: (_ for _ in ()).throw(OSError()))
    assert run()[0].client_latest == "0.5.0"


@pytest.mark.parametrize("var, val", [("ASTRONEREUS_NO_UPDATE_CHECK", "1"), ("CI", "true")])
def test_background_check_opt_out(env, monkeypatch, var, val):
    calls = env(pypi("0.5.0"), gh("v0.9.0"))
    monkeypatch.setenv(var, val)
    _updates.start_background_check()
    assert calls == [] and _updates._started is False


def test_ci_false_does_not_disable(env, monkeypatch):
    monkeypatch.setenv("CI", "false")
    assert not _updates.disabled()


def test_explicit_call_ignores_opt_out(env, monkeypatch):
    env(pypi("0.5.0"), gh("v0.8.10"))
    monkeypatch.setenv("ASTRONEREUS_NO_UPDATE_CHECK", "1")
    assert len(run()[1]) == 1


def test_background_check_warns_once_per_process(env, monkeypatch):
    calls = env(pypi("0.5.0"), gh("v0.8.10"))
    monkeypatch.setattr(_updates, "_thread", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _updates.start_background_check()
        _updates._thread.join(10)
        first = _updates._thread
        _updates.start_background_check()      # second Session.start in the process
        assert _updates._thread is first
    assert len([x for x in w if issubclass(x.category, UpdateAvailableWarning)]) == 1
    assert len(calls) == 2


def test_import_does_no_network(monkeypatch):
    import importlib, sys, urllib.request
    def boom(*a, **k):
        raise AssertionError("network at import")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    for m in [m for m in sys.modules if m.startswith("astronereus")]:
        monkeypatch.delitem(sys.modules, m)
    importlib.import_module("astronereus")
