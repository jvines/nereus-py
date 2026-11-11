"""`hip_scans` on the astrometry channel: what reaches the wire. No Julia, no daemon."""
from __future__ import annotations

from astronereus import Astrometry, RV, _fit


class _Session:
    """Captures the payload instead of talking to a daemon."""

    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}


_RV = RV(data={"A": {"t": [0.0, 1.0], "rv": [1.0, 2.0], "rv_err": [1.0, 1.0]}})


def test_absent_by_default_so_nereus_decides():
    # None is dropped: Nereus then takes the IAD's scans or fetches them.
    assert "hip_scans" not in Astrometry(hgca={"hip_id": 85653}).to_wire()


def test_false_opts_out():
    w = Astrometry(hgca={"hip_id": 85653}, hip_scans=False).to_wire()
    assert w["hip_scans"] is False


def test_fit_astrometry_puts_it_on_the_channel():
    s = _Session()
    _fit.fit_astrometry(hgca={"hip_id": 85653}, hip_scans={"hip_id": 85653}, session=s)
    assert s.payload["channels"][0]["hip_scans"] == {"hip_id": 85653}
    s = _Session()
    _fit.fit_astrometry(hgca={"hip_id": 85653}, session=s)
    assert "hip_scans" not in s.payload["channels"][0]


def test_fit_joint_carries_it_on_the_channel():
    s = _Session()
    vals = {"values": {"t": [48000.0] * 5, "psi": [0.1, 0.5, 1.0, 1.5, 2.0],
                       "parallax_factor": [0.1] * 5, "abscissa_err": [1.0] * 5}}
    _fit.fit_joint(_RV, Astrometry(hgca={"hip_id": 85653}, hip_scans=vals), session=s)
    ch = [c for c in s.payload["channels"] if c["source"] == "AS"][0]
    assert ch["hip_scans"] == vals
