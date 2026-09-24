"""End-to-end smoke test. Proves the pipe works, not that the science is right.

This exists because astronereus 0.2.0 shipped with a default engine that named
a Julia backend removed at the 0.1.0 rename, a JSON encoder that could not
serialise numpy, and three fit_* entry points that called keyword-only Julia
functions positionally. None of it was caught, because nothing here ever ran a
fit. `build_bundle.sh` stopped at `using Nereus`, which proves the runtime
imports and nothing else.

Run against a working copy of Nereus.jl (no bundle needed):

    NEREUS_JULIA=$(which julia) \
    NEREUS_PROJECT=/path/to/Nereus.jl \
    python -m pytest tests/test_smoke.py -x -q

or against an installed runtime bundle by leaving both unset.
"""
from __future__ import annotations

import os
import numpy as np
import pytest

import astronereus
from astronereus import engines


# 4 rounds is the documented smoke-test setting: it does NOT give a usable
# posterior. The default PT(n_rounds=12) is ~1 h and has no place in a test.
SMOKE = engines.PT(n_rounds=4, n_chains=6)


@pytest.fixture(scope="module")
def session():
    kw = {}
    if proj := os.environ.get("NEREUS_PROJECT"):
        # Mandatory under NEREUS_JULIA: julia_env sets JULIA_LOAD_PATH="@:@stdlib"
        # and pops JULIA_PROJECT, so `using Nereus` would resolve against the
        # user's default v1.11 environment. daemon.jl catches that with a bare
        # @warn, so the daemon comes up healthy and every fit then dies.
        kw["project"] = proj
    with astronereus.session(**kw) as s:
        yield s


def test_daemon_answers(session):
    assert session.ping()


def test_every_engine_option_is_accepted(session):
    """Julia validates option names against Base.kwarg_decl and throws on the
    first unknown one. Instantiating each engine with no arguments sends an
    empty option set, so this checks the ENGINE NAMES resolve; the field names
    are checked statically against kwarg_decl in tests/test_engine_fields.py."""
    for cls in (engines.PT, engines.PTHMC, engines.PTWhitening,
                engines.PTEmcee, engines.Nested, engines.NestedINS,
                engines.NestedDynamic, engines.MoMS, engines.Daedalus,
                engines.RJMCMC, engines.TransdimPTEmcee, engines.NUTS,
                engines.MAP, engines.SMC, engines.Ensemble, engines.ESS,
                engines.PA, engines.OFTI):
        wire = cls().to_wire()
        assert wire["engine"], f"{cls.__name__} has no Julia sampler name"
        assert isinstance(wire["options"], dict)


def test_numpy_crosses_the_wire(session):
    """dataclasses.asdict leaves ndarrays untouched, so every realistic payload
    ships numpy. json.dumps without an encoder raises TypeError before a byte
    is sent."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 120.0, 40)
    rv = 12.0 * np.sin(2 * np.pi * t / 4.23) + rng.normal(0, 2.0, t.size)
    err = np.full(t.size, 2.0)

    pg = session.detect.rv_periodogram(t=t, rv=rv, rv_err=err)
    assert pg is not None


def test_non_finite_does_not_kill_the_connection(session):
    """A bare NaN is legal in Python's json output and was fatal to JSON3 until
    allow_inf=true. The failure mode was a silently closed socket that killed
    every subsequent call in the session, so the assertion that matters is that
    the daemon is still answering afterwards."""
    t = np.array([0.0, 1.0, 2.0, np.nan])
    rv = np.array([1.0, 2.0, np.inf, 4.0])
    err = np.ones(4)
    try:
        session.detect.rv_periodogram(t=t, rv=rv, rv_err=err)
    except astronereus.DaemonError:
        pass          # rejecting the data is fine
    assert session.ping(), "daemon died on non-finite input"


def test_one_real_rv_fit(session, tmp_path):
    """The actual pipe: channel -> wire -> Julia -> sampler -> summary back."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 300.0, 60))
    rv = 25.0 * np.sin(2 * np.pi * t / 11.7) + rng.normal(0, 3.0, t.size)

    ch = astronereus.RV(data={"SIM": {"t": t, "rv": rv,
                                      "rv_err": np.full(t.size, 3.0)}})
    res = session.fit_rv(ch, planets=1, engine=SMOKE,
                         output_dir=str(tmp_path))
    assert res is not None
    assert res.status != "failed", getattr(res, "error", res)
