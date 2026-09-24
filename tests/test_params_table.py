"""The params table must always print the interval. No Julia, no daemon.

Regression: `Estimate.text()` and `Params._rows()` replaced the uncertainties
with "multimodal — percentile range not meaningful" whenever a heuristic on the
summary numbers fired. The quantiles were in the payload the whole time, the
LaTeX table written by the same fit printed them, and the reader was left with
a bare median they could not tell from a tight measurement.
"""
from __future__ import annotations

from astronereus._result import Estimate, Params

# Real numbers, from a Gaia astrometric fit: Omega has two modes pi apart, so
# the equal-tailed interval is enormous on one side and tiny on the other.
OMEGA = {"median": 3.0802938996007936,
         "lo16": 3.019068787228358, "hi84": 6.195986306791852}
# secosw: what the Julia LaTeX writer renders as 0.484^{+0.082}_{-1.027}
SECOSW = {"median": 0.4836, "lo16": -0.5434, "hi84": 0.5656}
A_K1 = {"median": 1.1797, "lo16": 1.1757, "hi84": 1.1838}


def test_multimodal_still_prints_its_interval():
    e = Estimate(OMEGA, "Omega_k1")
    assert e.multimodal, "fixture should trip the heuristic"
    t = e.text()
    assert "+3.116" in t and "-0.061" in t, t
    assert "not meaningful" not in t


def test_multimodal_is_flagged_not_hidden():
    assert "[multimodal]" in Estimate(OMEGA, "Omega_k1").text()
    row = next(r for r in Params({"Omega_k1": OMEGA})._rows())
    assert row[2] and row[3], "up/dn columns must not be blank"
    assert "multimodal" in row[5]


def test_unimodal_is_not_flagged():
    assert "multimodal" not in Estimate(A_K1, "a_k1").text()


def test_rounding_follows_the_smaller_error():
    """`_fmt3` (Nereus.jl science_tables.jl) rounds on min(|lo|, |hi|); the two
    renderers must not disagree about the same numbers."""
    t = Estimate(SECOSW, "secosw_k1").text()
    assert "0.484" in t and "+0.082" in t and "-1.027" in t, t


def test_every_estimate_row_has_six_columns():
    rows = list(Params({"Omega_k1": OMEGA, "a_k1": A_K1, "secosw_k1": SECOSW})._rows())
    assert len(rows) == 3
    assert all(len(r) == 6 for r in rows)
    assert all(r[1] and r[2] and r[3] for r in rows), "no blank value/err cells"
