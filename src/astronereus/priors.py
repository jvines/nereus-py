"""Prior dicts for the `priors=` argument of every fit_*, built and checked.

`priors` maps a parameter name to a prior; a prior is the wire's dict
(`{"type": ..., ...}`), a plain number (fixed) or one of these constructors,
which return that dict. Every prior Nereus's `fit_*` route accepts has one:

    from astronereus import priors as pr
    fit_rv(rv, planets=1, priors={
        "P_k1": pr.loguniform(5, 50),
        "K_k1": pr.modjeffreys(knee=1.0, upper=100.0),   # Gregory's modified Jeffreys
        "secosw_k1": pr.normal(0.0, 0.1, lo=-0.5, hi=0.5),   # bounded (truncated) normal
        "q1": pr.beta(2.0, 5.0),                             # Beta on [0, 1]
        "plx": pr.normal(21.6, 0.1)})

`normal` with `lo` and/or `hi` is the truncated normal; `beta` takes `lo`/`hi`
to rescale its support. The constructors check their arguments client-side, so
an empty range or a negative sigma fails here rather than minutes into a run.
"""
from __future__ import annotations

import math
from typing import Any


def _num(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number; got {x!r}")
    return float(x)


def uniform(lo: float, hi: float) -> dict[str, Any]:
    lo, hi = _num(lo, "lo"), _num(hi, "hi")
    if not lo < hi:
        raise ValueError(f"uniform: need lo < hi; got {lo}, {hi}")
    return {"type": "uniform", "lo": lo, "hi": hi}


def loguniform(lo: float, hi: float) -> dict[str, Any]:
    lo, hi = _num(lo, "lo"), _num(hi, "hi")
    if not 0 < lo < hi:
        raise ValueError(f"loguniform: need 0 < lo < hi; got {lo}, {hi}")
    return {"type": "loguniform", "lo": lo, "hi": hi}


def modjeffreys(knee: float, upper: float) -> dict[str, Any]:
    """Modified Jeffreys, p(x) ~ 1 / (x + knee), on [0, upper]."""
    knee, upper = _num(knee, "knee"), _num(upper, "upper")
    if not (knee > 0 and upper > 0):
        raise ValueError(f"modjeffreys: knee and upper must be positive; got {knee}, {upper}")
    return {"type": "modjeffreys", "knee": knee, "upper": upper}


def normal(mu: float, sigma: float, *, lo: float | None = None,
           hi: float | None = None) -> dict[str, Any]:
    """Normal; with `lo` and/or `hi` truncated to that range (a bounded normal)."""
    mu, sigma = _num(mu, "mu"), _num(sigma, "sigma")
    if not sigma > 0:
        raise ValueError(f"normal: sigma must be positive; got {sigma}")
    d: dict[str, Any] = {"type": "normal", "mu": mu, "sigma": sigma}
    if lo is not None:
        d["lo"] = _num(lo, "lo")
    if hi is not None:
        d["hi"] = _num(hi, "hi")
    if lo is not None and hi is not None and not d["lo"] < d["hi"]:
        raise ValueError(f"normal: need lo < hi; got {lo}, {hi}")
    return d


def beta(a: float, b: float, *, lo: float = 0.0, hi: float = 1.0) -> dict[str, Any]:
    a, b = _num(a, "a"), _num(b, "b")
    lo, hi = _num(lo, "lo"), _num(hi, "hi")
    if not (a > 0 and b > 0):
        raise ValueError(f"beta: a and b must be positive; got {a}, {b}")
    if not lo < hi:
        raise ValueError(f"beta: need lo < hi; got {lo}, {hi}")
    return {"type": "beta", "a": a, "b": b, "lo": lo, "hi": hi}


def fixed(value: float) -> dict[str, Any]:
    return {"type": "fixed", "value": _num(value, "value")}


def sine() -> dict[str, Any]:
    """p(i) ~ sin i, the isotropic inclination prior."""
    return {"type": "sine"}


def wrapped_uniform(lo: float, hi: float) -> dict[str, Any]:
    """Uniform on a circle (an angle with no wall at the ends)."""
    lo, hi = _num(lo, "lo"), _num(hi, "hi")
    if not lo < hi:
        raise ValueError(f"wrapped_uniform: need lo < hi; got {lo}, {hi}")
    return {"type": "wrappeduniform", "lo": lo, "hi": hi}


__all__ = ["uniform", "loguniform", "modjeffreys", "normal", "beta", "fixed",
           "sine", "wrapped_uniform"]
