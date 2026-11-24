"""Per-entry units: the unit Julia sends wins; the name map is only a fallback.

No Julia, no daemon: a fake `params` payload with and without `unit`.
"""
from __future__ import annotations

from astronereus._result import Estimate, Params, _unit_of

STATS = {"median": 1.0, "lo16": 0.9, "hi84": 1.1}


def _p(**units):
    """A params block; `units` maps name -> unit (None = send no unit key)."""
    out = {}
    for name, u in units.items():
        d = dict(STATS)
        if u is not None:
            d["unit"] = u
        out[name] = d
    return out


def test_unit_from_julia_beats_the_name_map():
    # gamma_/sigma_ are m/s by name; a photometric instrument sends "".
    p = Params(_p(gamma_TESS="", sigma_TESS="", gamma_HARPS="m/s"))
    assert _unit_of("gamma_TESS", p["gamma_TESS"].raw) == ""
    assert _unit_of("sigma_TESS", p["sigma_TESS"].raw) == ""
    assert _unit_of("gamma_HARPS", p["gamma_HARPS"].raw) == "m/s"


def test_empty_unit_is_dimensionless_not_missing():
    e = Estimate({**STATS, "unit": ""}, "gamma_TESS")
    assert "m/s" not in e.text()
    row = next(r for r in Params({"gamma_TESS": {**STATS, "unit": ""}})._rows())
    assert row[4] == ""


def test_unit_from_julia_beats_the_name_map_both_ways():
    # name says dimensionless (unknown name), Julia says km/s
    e = Estimate({**STATS, "unit": "km/s"}, "tomo_sigma_line_HARPS_map")
    assert e.text().endswith("km/s") or "km/s" in e.text()
    row = next(r for r in Params({"tomo_sigma_line_x": {**STATS, "unit": "km/s"}})._rows())
    assert row[4] == "km/s"


def test_missing_unit_falls_back_to_the_name_map():
    # an older runtime sends no unit key at all
    p = Params(_p(P_k1=None, K_k1=None, gamma_HARPS=None, inc_k1=None))
    assert _unit_of("P_k1", p["P_k1"].raw) == "d"
    assert _unit_of("K_k1", p["K_k1"].raw) == "m/s"
    assert _unit_of("gamma_HARPS", p["gamma_HARPS"].raw) == "m/s"
    assert _unit_of("inc_k1", p["inc_k1"].raw) == "rad"


def test_non_string_unit_is_ignored():
    d = {**STATS, "unit": None}
    assert _unit_of("P_k1", d) == "d"


def test_raw_radian_unit_gets_the_degree_reading():
    # an angle sent as rad is shown with its degree equivalent, from the unit
    # Julia sent rather than from the name
    rows = list(Params({"mystery_angle": {"median": 3.141592653589793,
                                          "lo16": 3.0, "hi84": 3.2,
                                          "unit": "rad"}})._rows())
    assert "180.00 deg" in rows[0][5]


def test_mixed_old_and_new_entries():
    p = Params({"gamma_TESS": {**STATS, "unit": ""},
                "gamma_HARPS": dict(STATS)})          # old-style: no unit
    rows = {r[0]: r for r in p._rows()}
    assert rows["gamma_TESS"][4] == ""
    assert rows["gamma_HARPS"][4] == "m/s"


def test_derived_rows_have_units_without_a_runtime_unit():
    from astronereus._result import _unit_of
    assert _unit_of("trend_P_min") == "d"
    assert _unit_of("trend_K_min") == "m/s"
    assert _unit_of("trend_Msini_min") == "M_jup"
    assert _unit_of("logg_s") == "dex"
    assert _unit_of("M_s_rho") == "M_sun"
    assert _unit_of("sep_mas_k1") == "mas"
    assert _unit_of("omega_p_deg_k1") == "deg"
    # the runtime's unit wins
    assert _unit_of("trend_P_min", {"unit": "d"}) == "d"
