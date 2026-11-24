"""Channel options reach the wire whole, or raise. No Julia, no daemon.

The daemon used to hand Julia's fit_rv the RV channel's `data` alone, so
`fit_rv(rv, trend_order=1)` fitted no trend and jitter, gamma_per_instrument,
share and marginalize_gamma were dropped with it. The runtime half is checked
in tests/test_smoke.py; this is what the client sends and refuses.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import astronereus
from astronereus import RM, RV, Transit, _fit


class _Session:
    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}


_D = {"A": {"t": [0.0, 1.0], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]},
      "B": {"t": [0.5, 1.5], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]}}
_P = {"T": {"t": [0.0, 1.0], "flux": [1.0, 1.0], "flux_err": [1e-3, 1e-3]}}


def _wire(fn, *args, **kw):
    s = _Session()
    fn(*args, session=s, **kw)
    return s.payload


def test_fit_rv_keywords_ride_on_the_channel():
    p = _wire(_fit.fit_rv, _D, planets=[], trend_order=1, jitter=2.0)
    (ch,) = p["channels"]
    assert ch["source"] == "RV"
    assert ch["trend_order"] == 1 and ch["jitter"] == 2.0
    assert "trend_order" not in p and "jitter" not in p


def test_every_rv_option_is_on_the_wire():
    rv = RV(data=_D, jitter={"A": 1.5}, trend_order=2, gamma_per_instrument=False,
            share={"sigma": [["A", "B"]]}, marginalize_gamma=True)
    (ch,) = _wire(_fit.fit_rv, rv)["channels"]
    assert ch == {"source": "RV", "data": _D, "jitter": {"A": 1.5}, "trend_order": 2,
                  "gamma_per_instrument": False, "share": {"sigma": [["A", "B"]]},
                  "marginalize_gamma": True}


def test_every_transit_option_is_on_the_wire():
    tr = Transit(data=_P, exposure_time=1800.0, dilution={"T": 0.1}, jitter=1e-4,
                 limb_darkening="kipping")
    (ch,) = _wire(_fit.fit_transit, tr)["channels"]
    assert ch["source"] == "PM"
    assert (ch["exposure_time"], ch["dilution"], ch["jitter"], ch["limb_darkening"]) == \
        (1800.0, {"T": 0.1}, 1e-4, "kipping")


def test_keywords_next_to_a_channel_raise():
    with pytest.raises(TypeError, match="trend_order"):
        _fit.fit_rv(RV(data=_D), trend_order=1, session=_Session())
    with pytest.raises(TypeError, match="jitter"):
        _fit.fit_rv(RV(data=_D), jitter=3.0, session=_Session())
    with pytest.raises(TypeError, match="limb_darkening"):
        _fit.fit_transit(Transit(data=_P), limb_darkening="kipping", session=_Session())


def test_options_that_cannot_be_honoured_raise():
    with pytest.raises(ValueError, match="indicators"):
        RV(data=_D, indicators={"fwhm": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="trend_order"):
        RV(data=_D, trend_order=-1)
    with pytest.raises(ValueError, match="trend_order"):
        RV(data=_D, trend_order=True)
    with pytest.raises(ValueError, match="beta_p"):
        RM(data=_D, flavour="reloaded", beta=3000.0)
    RM(data=_D, flavour="arome", beta=3000.0, vsini=5000.0, lambda_prior=0.1)


def test_daemon_passes_fit_rv_the_whole_channel():
    """The daemon ships with this package; its fit_rv branch must not strip the
    channel to its `data`, and no fit op may fall through to a branch that does."""
    src = (Path(astronereus.__file__).parent / "julia" / "daemon.jl").read_text()
    branch = re.search(r'elseif op == "fit_rv"(.*?)\n    else\n', src, re.S)
    assert branch, "daemon.jl has no fit_rv branch"
    assert "Base.invokelatest(f, chs[1];" in branch.group(1)
    assert 'get(chs[1], "data"' not in src


def test_api_version_bumped_for_the_channel_contract():
    assert astronereus.PY_API_VERSION >= 4
