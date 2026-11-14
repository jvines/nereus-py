"""gaia_id / name on the astrometry fits: what reaches the wire. No Julia, no daemon."""
from __future__ import annotations

import pytest

from astronereus import Astrometry, RV, _fit


class _Session:
    """Captures the payload instead of talking to a daemon."""

    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}

    def run_job(self, config, **kw):
        self.payload = config
        return {"status": "ok", "summary": {}}


_RV = RV(data={"A": {"t": [0.0, 1.0], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]}})
_AS = Astrometry(hgca={"epochs": [0.0, 1.0, 2.0]})


def test_fit_astrometry_sends_gaia_id():
    s = _Session()
    _fit.fit_astrometry(hgca={"x": 1}, gaia_id=6774368627975155968, session=s)
    assert s.payload["op"] == "fit_astrometry"
    assert s.payload["gaia_id"] == 6774368627975155968
    assert "name" not in s.payload


def test_fit_astrometry_sends_name():
    s = _Session()
    _fit.fit_astrometry(hgca={"x": 1}, name="HIP 103836", session=s)
    assert s.payload["name"] == "HIP 103836"
    assert "gaia_id" not in s.payload


def test_fit_joint_sends_both_and_keeps_other_extras():
    s = _Session()
    _fit.fit_joint(_RV, _AS, gaia_id="Gaia DR3 6774368627975155968",
                   name="HIP 103836", session=s)
    assert s.payload["op"] == "fit_joint"
    assert s.payload["gaia_id"] == "Gaia DR3 6774368627975155968"
    assert s.payload["name"] == "HIP 103836"


def test_gaia_keys_coexist_with_use_rho_s():
    from astronereus import Transit
    tr = Transit(data={"T": {"t": [0.0, 1.0], "flux": [1.0, 1.0], "flux_err": [1e-3, 1e-3]}})
    s = _Session()
    _fit.fit_joint(_RV, _AS, tr, use_rho_s=True, gaia_id=5, session=s)
    assert s.payload["use_rho_s"] is True and s.payload["gaia_id"] == 5


def test_default_sends_nothing():
    """A runtime that predates the lookup has no such keyword."""
    for call in (lambda s: _fit.fit_astrometry(hgca={"x": 1}, session=s),
                 lambda s: _fit.fit_joint(_RV, _AS, session=s)):
        s = _Session()
        call(s)
        assert "gaia_id" not in s.payload and "name" not in s.payload


@pytest.mark.parametrize("bad", [1.5, True, ["1"], {}])
def test_gaia_id_type_is_checked(bad):
    with pytest.raises(TypeError, match="gaia_id"):
        _fit.fit_astrometry(hgca={"x": 1}, gaia_id=bad, session=_Session())


@pytest.mark.parametrize("bad", ["", "  ", 7])
def test_name_type_is_checked(bad):
    with pytest.raises(TypeError, match="name"):
        _fit.fit_astrometry(hgca={"x": 1}, name=bad, session=_Session())


@pytest.mark.parametrize("fn", [_fit.fit_rv, _fit.fit_transit])
def test_single_technique_fits_without_astrometry_do_not_take_them(fn):
    """No astrometric planet there: nothing to look up, and the keyword is refused."""
    data = _RV.data
    with pytest.raises(TypeError, match="unknown keyword"):
        fn(data, gaia_id=5, session=_Session())


def test_run_job_star_block_passes_through_untouched():
    """The star block of a job config is not interpreted client-side."""
    from astronereus import Session
    s = Session.__new__(Session)
    seen = {}
    s.raw = lambda action, payload=None, **kw: seen.update(action=action, payload=payload) or {
        "status": "ok", "summary": {}}
    cfg = {"output_dir": "out", "star": {"M_s": 0.9, "gaia_id": "Gaia DR3 5", "name": "X"}}
    try:
        s.run_job(cfg, check=False)
    except Exception:
        pass
    assert seen["action"] == "run_job"
    assert seen["payload"]["star"] == {"M_s": 0.9, "gaia_id": "Gaia DR3 5", "name": "X"}
