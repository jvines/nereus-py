"""Non-inference features. Each is its own operation, not a mode of `fit`."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


# Operations that read a posterior. Given `result=` (what a fit_* call returned)
# they answer from the block that fit already wrote (`loo`, `ppc`, `fit_health`,
# `evidence`, `detection_limits`) unless `recompute=True`, or the block is
# missing, skipped or failed; recomputing replays the fit's own request to
# rebuild the model, as `replot` does, and evaluates on its `chains.nc`.
_BLOCK_OF = {
    "diagnostics.loo": "loo",
    "diagnostics.ppc": "ppc",
    "diagnostics.fit_health": "fit_health",
    "detection_limits": "detection_limits",
    "evidence": "evidence",
}
# Operations that re-evaluate the model, so need the fit's request, not only
# its chains.
_NEEDS_MODEL = frozenset(_BLOCK_OF)
# What of a recorded request is NOT the model: transport, sampling, output.
_REQUEST_DROP = frozenset({"op", "stopping", "stop_file", "plots", "plot_kwargs",
                           "save_pdf", "science", "output", "transdim"})
_CONFIG_ROUTE = frozenset({"fit_rm", "fit_ttv", "fit_binary", "fit_tomography"})


def _usable(block: Any) -> bool:
    return isinstance(block, Mapping) and block.get("status") not in ("skipped", "failed")


def _from_result(action: str, result: Any) -> Any:
    """The block the fit already produced for `action`, or None."""
    key = _BLOCK_OF.get(action)
    raw = getattr(result, "raw", None)
    if key is None or not isinstance(raw, Mapping):
        return None
    block = raw.get(key)
    if not _usable(block):
        return None
    if action == "evidence":
        return {"log_z": raw.get("log_z"), "evidence": block}
    return block


def result_payload(action: str, result: Any) -> dict[str, Any]:
    """The payload that makes the daemon evaluate `action` on `result`."""
    raw = getattr(result, "raw", None)
    if not isinstance(raw, Mapping):
        raise TypeError(f"result= takes what a fit_* call returned; got {type(result).__name__}")
    req = dict(getattr(result, "_request", None) or {})
    out_dir = raw.get("output_dir") or req.get("output_dir")
    if not out_dir:
        raise ValueError(
            "this result has no output_dir, so there are no chains on disk to "
            "evaluate; run the fit with output_dir=...")
    if action in _NEEDS_MODEL:
        op = req.get("op") or raw.get("op")
        if op in _CONFIG_ROUTE:
            raise ValueError(
                f"{action}: {op} runs as a job config, which cannot be rebuilt "
                "from its request; use the block the fit wrote "
                f"(result.raw[{_BLOCK_OF[action]!r}]) or run_job with the "
                "output block set")
        if not req.get("channels"):
            raise ValueError(
                f"{action} re-evaluates the model, and this result carries no "
                "record of the fit's channels (it did not come from a typed "
                "fit_* call in this session). Pass channels=[...], planets=, "
                "priors= as for replot")
    payload = {k: v for k, v in req.items() if k not in _REQUEST_DROP}
    payload["output_dir"] = str(out_dir)
    return payload


class _Group:
    def __init__(self, session_getter, prefix: str, actions: dict[str, str]):
        self._get, self._prefix, self._actions = session_getter, prefix, actions

    def __dir__(self):
        return sorted(self._actions)

    def __getattr__(self, name):
        if name not in self._actions:
            raise AttributeError(f"{self._prefix}.{name} is not a Nereus "
                                 f"operation. Available: {', '.join(sorted(self._actions))}")
        action = self._actions[name]

        def _call(*, result=None, recompute: bool = False, **payload):
            return run_op(self._get(), action, result=result,
                          recompute=recompute, **payload)
        _call.__name__ = name
        _call.__doc__ = (
            f"Nereus operation {action!r}. `result=` is a fit_* result: the "
            "block that fit wrote is returned when it has one (pass "
            "`recompute=True` to evaluate again from its chains); everything "
            "else is the operation's own keywords.")
        return _call


def run_op(session, action: str, *, result=None, recompute: bool = False,
           **payload):
    """Call operation `action`, optionally on a fit's `result`."""
    if result is not None:
        if not recompute:
            hit = _from_result(action, result)
            if hit is not None:
                return hit
        payload = {**result_payload(action, result), **payload}
    return session.raw(action, payload)


DETECT = {"transits": "detect.transits", "rv_planets": "detect.rv_planets",
          "rotation": "detect.rotation", "segments": "detect.segments",
          "rv_periodogram": "detect.rv_periodogram",
          "lc_periodogram": "detect.lc_periodogram"}
DETREND = {"savgol": "detrend.savgol", "gp": "detrend.gp",
           "notch": "detrend.notch", "locor": "detrend.locor"}
TOMOGRAM = {"matched_filter": "tomogram.matched_filter",
            "null_distribution": "tomogram.null_distribution",
            "injection_test": "tomogram.injection_test",
            "shadow_track": "tomogram.shadow_track",
            "residuals": "tomogram.residuals",
            "ccf_profile": "tomogram.ccf_profile"}
DIAGNOSTICS = {"ess_rhat": "diagnostics.ess_rhat", "loo": "diagnostics.loo",
               "ppc": "diagnostics.ppc",
               "label_switching": "diagnostics.label_switching",
               "fit_health": "diagnostics.fit_health"}


# --- Session methods ----------------------------------------------------------

def evidence(session, *, result=None, recompute: bool = False, **payload):
    """Log evidence of a fit: `{"log_z", "evidence": {...}}`.

    Given `result=` it returns the evidence block the fit wrote (mode-Laplace,
    bridge, tempered variants and which one is `reported`); `recompute=True`
    evaluates again from the fit's chains (mode-Laplace and bridge sampling, the
    fit's own selection rule), which needs `output_dir` on the fit."""
    return run_op(session, "evidence", result=result, recompute=recompute, **payload)


def detection_limits(session, *, result=None, recompute: bool = False, **payload):
    """RV detection limits (the K that would have been detected, per period bin)
    from a fit's chains. `planet`, `detection_limits_n_bins` and
    `detection_limits_confidence` are the job config's `output` options. The
    arrays are written to `<output_dir>/detection_limits.nc`."""
    return run_op(session, "detection_limits", result=result,
                  recompute=recompute, **payload)


def pre_white(session, *, t=None, y=None, y_err=None, rv=None, rv_err=None,
              max_signals: int = 5, fap: float = 0.01, **payload):
    """Iterative prewhitening: the highest GLS peak is fitted as a sinusoid and
    removed, repeatedly, until `max_signals` or the best remaining peak's false-
    alarm probability exceeds `fap`. Returns the signals (period, amplitude,
    phase, FAP) and the residuals.

    Give `t`, `y`, `y_err`, or `rv` as the instrument map an `RV` channel
    carries (`{instrument: {t, rv, rv_err}}`; each instrument's mean is removed
    first), or an `RV` channel itself."""
    data = getattr(rv, "data", rv)
    if isinstance(data, Mapping) and data and all(isinstance(v, Mapping) for v in data.values()):
        ts, ys, es = [], [], []
        for blob in data.values():
            tt = [float(x) for x in blob["t"]]
            vv = [float(x) for x in blob["rv"]]
            mean = sum(vv) / len(vv)
            ts += tt
            ys += [x - mean for x in vv]
            es += [float(x) for x in blob["rv_err"]]
        order = sorted(range(len(ts)), key=ts.__getitem__)
        t, y, y_err = ([ts[i] for i in order], [ys[i] for i in order],
                       [es[i] for i in order])
    elif rv is not None:
        y, y_err = rv, rv_err
    if t is None or y is None or y_err is None:
        raise ValueError("pre_white needs t, y and y_err (or rv=<instrument map>)")
    return session.raw("pre_white", {"t": [float(x) for x in t],
                                     "y": [float(x) for x in y],
                                     "y_err": [float(x) for x in y_err],
                                     "max_signals": int(max_signals),
                                     "fap": float(fap), **payload})


def _fit_channels(session, channels, **kw):
    """The dedicated fit_* for a channel set: one RV, Transit or Astrometry
    channel goes to its own entry point, several to `fit_joint`."""
    from . import _fit
    from .channels import RV, Astrometry, Transit
    if len(channels) == 1:
        ch = channels[0]
        if isinstance(ch, RV):
            return _fit.fit_rv(ch, session=session, **kw)
        if isinstance(ch, Transit):
            return _fit.fit_transit(ch, session=session, **kw)
        if isinstance(ch, Astrometry):
            ak = {f: getattr(ch, f) for f in ("iad", "hgca", "gost", "relast",
                                              "hip_scans")}
            return _fit.fit_astrometry(session=session, **ak, **kw)
        raise TypeError(f"no single-channel fit for {type(ch).__name__}")
    return _fit.fit_joint(*channels, session=session, **kw)


def _table(rows: list[dict], key: str) -> dict[str, Any]:
    ok = [r for r in rows if isinstance(r["log_z"], (int, float)) and math.isfinite(r["log_z"])]
    best = max(ok, key=lambda r: r["log_z"]) if ok else None
    for r in rows:
        r["delta_log_z"] = (r["log_z"] - best["log_z"]
                            if best is not None and r in ok else None)
    return {"table": rows, "best": best[key] if best is not None else None}


def _sub(output_dir, name):
    return None if output_dir is None else str(Path(output_dir) / name)


def select_planets(session, *channels, candidates: Sequence[int] = (1, 2),
                   result=None, engine=None, output_dir=None, **fit_kw):
    """How many planets: the evidence of each candidate count, tabulated.

    With channels, a fixed-dimension fit is run for every count in `candidates`
    (any `fit_*` keyword is passed on: `engine`, `priors`, `stopping`, ...;
    `output_dir` gets one sub-directory per count) and the reported log Z of each
    is compared. Returns `{"table": [{"n_planets", "log_z", "delta_log_z",
    "reported", "status", "output_dir"}, ...], "best": n, "results": {n: result}}`.

    With `result=` a TRANS-DIM fit (`planets=range(...)`) instead: P(N_p = k)
    and the Bayes factor of each k against the best, from its chains."""
    if result is not None:
        if channels:
            raise TypeError("select_planets takes channels or result=, not both")
        return run_op(session, "select_planets", result=result)
    if not channels:
        raise TypeError("select_planets needs channels to fit, or result=")
    rows, results = [], {}
    for n in candidates:
        res = _fit_channels(session, channels, planets=int(n), engine=engine,
                            output_dir=_sub(output_dir, f"planets_{int(n)}"),
                            **fit_kw)
        results[int(n)] = res
        ev = res.raw.get("evidence")
        rows.append({"n_planets": int(n), "log_z": res.raw.get("log_z"),
                     "reported": ev.get("reported") if isinstance(ev, Mapping) else None,
                     "status": res.status, "output_dir": res.raw.get("output_dir")})
    out = _table(rows, "n_planets")
    out["results"] = results
    return out


def select_noise(session, *channels, candidates: Mapping[str, Any] | None = None,
                 result=None, engine=None, output_dir=None, planets=1, **fit_kw):
    """Which noise model: the evidence of each candidate, tabulated.

    `candidates` maps a name to what that model is: None (white noise only), a
    `noise_models` list (a job config's block), or a dict with `noise_menu` or
    `noise_models`, e.g. `{"white": None, "rotation": [{"kind":
    "CeleriteRotation", "channel": "rv", "instruments": ["HARPS"]}]}`. One fit
    per candidate, over `channels` (which must include RV), compared on log Z;
    returns `{"table": [{"model", "log_z", "delta_log_z", ...}], "best": name,
    "results": {...}}`.

    With `result=` a trans-dim-noise fit (`noise_menu=` with `transdim=`): the
    fraction of draws each noise model is active in, from its chains."""
    if result is not None:
        if channels:
            raise TypeError("select_noise takes channels or result=, not both")
        return run_op(session, "select_noise", result=result)
    if not channels or not candidates:
        raise TypeError("select_noise needs channels and candidates, or result=")
    rows, results = [], {}
    for name, spec in candidates.items():
        kw = dict(fit_kw)
        if isinstance(spec, Mapping):
            unknown = set(spec) - {"noise_models", "noise_menu"}
            if unknown:
                raise ValueError(f"select_noise candidate {name!r}: unknown {sorted(unknown)}")
            kw.update(spec)
        elif spec is not None:
            kw["noise_models"] = list(spec)
        res = _fit_channels(session, channels, planets=planets, engine=engine,
                            output_dir=_sub(output_dir, f"noise_{name}"), **kw)
        results[name] = res
        ev = res.raw.get("evidence")
        rows.append({"model": name, "log_z": res.raw.get("log_z"),
                     "reported": ev.get("reported") if isinstance(ev, Mapping) else None,
                     "status": res.status, "output_dir": res.raw.get("output_dir")})
    out = _table(rows, "model")
    out["results"] = results
    return out
