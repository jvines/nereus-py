"""fit_binary takes velocities: an instrument map, or an RV channel's map."""
from __future__ import annotations
import pytest
from astronereus import RV, SB2

_MAP = {"SPEC": {"t": [1.0, 2.0, 3.0], "rv": [10.0, -5.0, 2.0], "rv_err": [1.0, 1.0, 1.0]}}


def test_an_instrument_map_passes_through():
    assert SB2(primary=_MAP).primary == _MAP


def test_an_rv_channel_gives_its_map():
    ch = SB2(primary=RV(data=_MAP), secondary=RV(data=_MAP))
    assert ch.primary == _MAP and ch.secondary == _MAP


def test_an_rv_channel_with_options_is_refused():
    with pytest.raises(TypeError, match="trend_order"):
        SB2(primary=RV(data=_MAP, trend_order=1))
