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
        # user's default v1.x environment. daemon.jl catches that with a bare
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


# RV + HGCA through fit_joint. Until Nereus 0.8.8 the HGCA and GOST blocks
# reached Julia's `Data` as raw dicts and died in its constructor, so no
# Python fit could use either.
_HGCA_PRIORS = {"plx": {"type": "normal", "mu": 50.0, "sigma": 0.05},
                "M_pri": {"type": "normal", "mu": 1.0, "sigma": 0.05}}


def _sim_rv(seed=11):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(50000.0, 58000.0, 40))
    rv = 50.0 * np.sin(2 * np.pi * t / 3000.0) + rng.normal(0, 3.0, t.size)
    return astronereus.RV(data={"SIM": {"t": t, "rv": rv,
                                        "rv_err": np.full(t.size, 3.0)}})


def test_rv_plus_inline_hgca(session, tmp_path):
    hgca = {"values": {
        "t": [48348.6, 53176.5, 57388.5],                     # MJD
        "pmra": [10.0, 10.5, 11.0], "pmdec": [-3.0, -3.1, -3.2],
        "cov_ep": [[[0.25, 0.0], [0.0, 0.25]],
                   [[4e-4, 0.0], [0.0, 4e-4]],
                   [[1e-3, 0.0], [0.0, 1e-3]]],
        "plx": 50.0, "plx_err": 0.05, "hip_id": 1}}
    # hip_scans=False: the row's hip_id would otherwise fetch its Hipparcos
    # scans (Nereus newer than 0.8.10), a ~350 MB download for a smoke test.
    res = session.fit_joint(_sim_rv(), astronereus.Astrometry(hgca=hgca, hip_scans=False),
                            planets=1, engine=SMOKE, priors=_HGCA_PRIORS,
                            output_dir=str(tmp_path))
    assert res.status != "failed", getattr(res, "error", res)


def _hgca_catalogue():
    base = os.environ.get("NEREUS_DATA_CACHE",
                          os.path.join(os.path.expanduser("~"), ".nereus"))
    return os.path.join(base, "HGCA", "HGCA_vEDR3.fits")


@pytest.mark.skipif(not os.path.isfile(_hgca_catalogue()),
                    reason="HGCA_vEDR3.fits not in the Nereus cache")
def test_rv_plus_hgca_from_the_loader(session, tmp_path):
    """The user's path: read the row with the session's loader, hand the dict
    it returns straight to the channel."""
    hgca = session.load_hgca(path=_hgca_catalogue(), hip=85653)
    # hip_scans=False: the row's hip_id would otherwise fetch its Hipparcos
    # scans (Nereus newer than 0.8.10), a ~350 MB download for a smoke test.
    res = session.fit_joint(_sim_rv(), astronereus.Astrometry(hgca=hgca, hip_scans=False),
                            planets=1, engine=SMOKE, priors=_HGCA_PRIORS,
                            output_dir=str(tmp_path))
    assert res.status != "failed", getattr(res, "error", res)


def _sim_rv_transit(seed=3):
    """A hot-Jupiter-like system: P = 2.5 d, K = 80 m/s, 1% deep transits."""
    rng = np.random.default_rng(seed)
    P, T0 = 2.5, 59000.3
    t = np.sort(rng.uniform(59000.0, 59030.0, 40))
    rv = 80.0 * np.sin(2 * np.pi * (t - T0) / P) + rng.normal(0, 4.0, t.size)
    tp = np.arange(59000.0, 59005.0, 2.0 / 1440.0)
    ph = ((tp - T0 + 0.5 * P) % P) - 0.5 * P
    flux = 1.0 - 0.01 * (np.abs(ph) < 0.04) + rng.normal(0, 3e-4, tp.size)
    return (astronereus.RV(data={"SIM": {"t": t, "rv": rv,
                                         "rv_err": np.full(t.size, 4.0)}}),
            astronereus.Transit(data={"TESS": {
                "t": tp, "flux": flux, "flux_err": np.full(tp.size, 3e-4)}}))


@pytest.mark.parametrize("planets", [1, [1]])
def test_rv_plus_transit_fit_joint(session, tmp_path, planets):
    """Regression: an int `planets` built planets with no parameters and the
    engine refused with "planet must declare a mass parameter"."""
    rv, tr = _sim_rv_transit()
    eng = engines.PTEmcee(n_temps=4, n_walkers=24, n_steps=60, n_burnin=30,
                          seed=1)
    res = session.fit_joint(rv, tr, planets=planets, M_s=1.0, R_s=1.0,
                            engine=eng, output_dir=str(tmp_path),
                            priors={"P_k1": {"type": "uniform", "lo": 2.4, "hi": 2.6}})
    assert res.status != "failed", res.raw.get("error")
    names = set(res.params)
    assert any(n.startswith("K_k1") for n in names) and any(n.startswith("rr_k1") for n in names), names


def test_rv_plus_transit_fit_joint_use_rho_s(session, tmp_path):
    """The stellar density as a sampled parameter in an RV + transit fit_joint,
    through the daemon, with its prior from `priors`."""
    rv, tr = _sim_rv_transit()
    eng = engines.PTEmcee(n_temps=4, n_walkers=24, n_steps=60, n_burnin=30,
                          seed=1)
    res = session.fit_joint(
        rv, tr, planets=1, use_rho_s=True, M_s=1.0, R_s=1.0, engine=eng,
        output_dir=str(tmp_path),
        priors={"P_k1": {"type": "uniform", "lo": 2.4, "hi": 2.6},
                "rho_s": {"type": "normal", "mu": 1.0, "sigma": 0.1,
                          "lo": 0.3, "hi": 3.0}})
    assert res.status != "failed", res.raw.get("error")
    p = res.raw["params"]["rho_s"]
    assert 0.3 <= p["lo3sig"] and p["hi3sig"] <= 3.0
