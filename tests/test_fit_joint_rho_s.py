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
