"""fit_joint(use_rho_s=...): what reaches the wire. No Julia, no daemon."""
from __future__ import annotations

import pytest

from astronereus import RV, Transit, _fit


class _Session:
    """Captures the payload instead of talking to a daemon."""

    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}


_RV = RV(data={"A": {"t": [0.0, 1.0], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]}})
_TR = Transit(data={"T": {"t": [0.0, 1.0], "flux": [1.0, 1.0], "flux_err": [1e-3, 1e-3]}})


def test_use_rho_s_reaches_the_daemon():
    s = _Session()
    _fit.fit_joint(_RV, _TR, use_rho_s=True, session=s)
    assert s.payload["op"] == "fit_joint"
    assert s.payload["use_rho_s"] is True


def test_default_sends_nothing():
    """An older runtime has no such keyword on Julia's fit_joint."""
    s = _Session()
    _fit.fit_joint(_RV, _TR, session=s)
    assert "use_rho_s" not in s.payload


def test_it_needs_a_transit_channel():
    rv2 = RV(data={"B": {"t": [0.0, 1.0], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]}})
    with pytest.raises(ValueError, match="Transit channel"):
        _fit.fit_joint(_RV, rv2, use_rho_s=True, session=_Session())


# --- fit_transit, the Transit channel, the star keywords --------------------

def _wire(fn, *args, **kw):
    s = _Session()
    fn(*args, session=s, **kw)
    return s.payload


def test_fit_transit_use_rho_s_reaches_the_daemon():
    p = _wire(_fit.fit_transit, _TR.data, use_rho_s=True)
    assert p["op"] == "fit_transit" and p["use_rho_s"] is True
    assert "use_rho_s" not in _wire(_fit.fit_transit, _TR.data)


def test_fit_transit_rho_star_points_at_the_new_spelling():
    with pytest.raises(TypeError, match="use_rho_s"):
        _fit.fit_transit(_TR.data, rho_star=(1.79, 0.11), session=_Session())


@pytest.mark.parametrize("kw", [{"use_rho_s": True}, {"use_rho_s": False},
                                {"rho_star": (1.79, 0.11)}])
def test_transit_channel_refuses_what_it_never_read(kw):
    with pytest.raises(TypeError, match="fit_joint"):
        Transit(data=_TR.data, **kw)
    with pytest.raises(TypeError, match="fit_joint"):
        Transit.from_arrays([0.0, 1.0], [1.0, 1.0], [1e-3, 1e-3], **kw)


def test_transit_channel_wire_carries_neither():
    w = Transit.from_arrays([0.0, 1.0], [1.0, 1.0], [1e-3, 1e-3]).to_wire()
    assert "use_rho_s" not in w and "rho_star" not in w


_STAR = dict(T_eff=4730, J_mag=10.1, K_mag=9.4, Ab=0.3,
             sigma_M_s=0.025, sigma_R_s=0.013, sigma_T_eff=80)


@pytest.mark.parametrize("name,call", [
    ("fit_rv", lambda s, kw: _fit.fit_rv(_RV.data, session=s, **kw)),
    ("fit_transit", lambda s, kw: _fit.fit_transit(_TR.data, session=s, **kw)),
    ("fit_astrometry", lambda s, kw: _fit.fit_astrometry(iad="x", session=s, **kw)),
    ("fit_joint", lambda s, kw: _fit.fit_joint(_RV, _TR, session=s, **kw)),
])
def test_star_keywords_reach_the_four_that_read_them(name, call):
    s = _Session()
    call(s, dict(M_s=0.767, R_s=0.754, **_STAR))
    for k, v in _STAR.items():
        assert s.payload[k] == v
    assert s.payload["M_s"] == 0.767


@pytest.mark.parametrize("call", [
    lambda s: _fit.fit_rm([], phot=_TR.data, session=s, T_eff=4730),
    lambda s: _fit.fit_ttv([], session=s, sigma_M_s=0.02),
    lambda s: _fit.fit_binary({}, session=s, Ab=0.3),
])
def test_the_rest_do_not_accept_them(call):
    with pytest.raises(TypeError, match="unknown keyword"):
        call(_Session())


# --- the Transit channel's other options ------------------------------------

_OPTS = dict(exposure_time=1800.0, dilution={"T": 0.1}, jitter={"type": "uniform", "lo": 0.0, "hi": 0.01})


@pytest.mark.parametrize("field,value", list(_OPTS.items()))
def test_each_transit_option_is_on_the_wire(field, value):
    ch = Transit(data=_TR.data, **{field: value})
    assert ch.to_wire()[field] == value
    p = _wire(_fit.fit_joint, _RV, ch)
    assert p["channels"][1][field] == value
    p = _wire(_fit.fit_transit, ch)
    assert p["channels"][0][field] == value


def test_defaults_send_what_julia_defaults_to():
    w = Transit(data=_TR.data).to_wire()
    assert w["limb_darkening"] == "quadratic" and w["gravity_darkening"] is False
    assert w["jitter"] == "default"
    for absent in ("exposure_time", "dilution", "supersample"):
        assert absent not in w


@pytest.mark.parametrize("kw,pat", [
    ({"limb_darkening": "linear"}, "limb_darkening"),
    ({"limb_darkening": "nonlinear"}, "limb_darkening"),
    ({"gravity_darkening": True}, "gravity_darkening"),
    ({"supersample": 5}, "exposure_time"),
    ({"exposure_time": -1.0}, "positive"),
    ({"exposure_time": {"T": 0.0}}, "positive"),
])
def test_what_nereus_lacks_raises(kw, pat):
    with pytest.raises(ValueError, match=pat):
        Transit(data=_TR.data, **kw)


def test_kipping_is_accepted():
    assert Transit(data=_TR.data, limb_darkening="kipping").limb_darkening == "kipping"


def test_fit_transit_gravity_darkening_raises():
    with pytest.raises(ValueError, match="gravity_darkening"):
        _fit.fit_transit(_TR.data, gravity_darkening=True, session=_Session())
