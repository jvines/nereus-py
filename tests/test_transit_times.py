"""Measured mid-transit times: the client hands the runtime the per-planet
blocks of `data.transit_times`, whichever of the accepted shapes it was given."""
from __future__ import annotations
import pytest
from astronereus import TTV, _fit
from astronereus._result import _unit_of


class _Session:
    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}


_TABLE = {"epoch": [0, 1, 5], "tc": [2459000.1, 2459003.2, 2459015.6],
          "tc_err": [1e-3, 2e-3, 1e-3]}
_BLOCK = {"planet": 2, "name": "b", "time_system": "BJD_TDB",
          "values": dict(_TABLE, source=["A", "B", "C"]),
          "reference_ephemeris": None}


def test_a_table_becomes_planet_1s_block():
    (b,) = TTV(transit_times=_TABLE).transit_times
    assert b == {"planet": 1, "time_system": "BJD_TDB", "values": _TABLE}


def test_a_dataframe_and_a_source_column():
    pd = pytest.importorskip("pandas")
    (b,) = TTV(transit_times=pd.DataFrame(dict(_TABLE, source=["x", "y", "z"]))).transit_times
    assert b["values"]["epoch"] == [0, 1, 5] and b["values"]["source"] == ["x", "y", "z"]
    assert all(isinstance(e, int) for e in b["values"]["epoch"])


def test_blocks_pass_through():
    assert TTV(transit_times=_BLOCK).transit_times == [_BLOCK]
    assert TTV(transit_times=[_BLOCK, _BLOCK]).transit_times == [_BLOCK, _BLOCK]
    assert TTV(transit_times=[]).transit_times == []


def test_anything_else_says_what_it_takes():
    with pytest.raises(TypeError, match="epoch, tc"):
        TTV(transit_times={"t": [1.0]})


def test_fit_ttv_sends_the_blocks_and_the_slot_options():
    s = _Session()
    _fit.fit_ttv(_TABLE, session=s, planets=1, ttv_epochs={1: [0, 1, 5]})
    assert s.payload["channels"][0]["transit_times"][0]["values"] == _TABLE
    assert s.payload["ttv_epochs"] == {1: [0, 1, 5]}
    assert "ttv_n_transits" not in s.payload
    s2 = _Session()
    _fit.fit_ttv(_TABLE, session=s2, planets=1)
    assert "ttv_epochs" not in s2.payload and "ttv_n_transits" not in s2.payload


def test_transit_time_jitter_and_offsets_are_in_days():
    assert _unit_of("sigma_tt_k1") == "d"
    assert _unit_of("ttv_k1_t3") == "d"
    assert _unit_of("sigma_HARPS") == "m/s"


def test_nbody_ttv_needs_rv_and_the_channel_has_no_such_option():
    # N-body TTVs need RV for the masses: fit_ttv takes it, and refuses without.
    import inspect
    assert "nbody" in inspect.signature(_fit.fit_ttv).parameters
    with pytest.raises(ValueError, match="needs rv"):
        _fit.fit_ttv(_TABLE, session=_Session(), nbody=True)
    with pytest.raises(TypeError):
        TTV(transit_times=_TABLE, nbody=True)
