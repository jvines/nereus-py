"""Inference entry points — one per technique, plus one for joint fits.

Deliberately NOT a single `fit(rv=..., phot=..., iad=..., rm=..., ...)`. Ten
optional data arguments on one function is a god-function whichever way the
arguments are spelled: most are None on any given call, the signature is
unreadable, and you cannot tell from the call site what will run.

Each function below takes only the parameters that apply to its technique.
Multi-technique work is its own operation, `fit_joint`, which composes explicit
channel objects that each carry their own settings.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from . import engines as _eng
from .channels import (RV, Astrometry, Channel, Night, RM, SB2, Transit,
                       TTV)
from .stopping import Stopping

_DEFAULT_STOP = None


# Keywords every fit_* forwards to Julia. These are NOT transport envelope --
# the daemon passes anything outside the envelope straight through as a keyword
# to the Julia fit_* function. Before that passthrough existed the daemon
# hardcoded planets/engine/output_dir/priors and silently dropped the rest.
_MODEL_KW = (
    "transdim",          # True | int | config block -> trans-dim birth/death
    "external_priors",   # priors on DERIVED quantities: ecc, rho_s
    "plots",             # Nereus's own figures, e.g. ["corner", "rv_phasefold"]
    "plot_kwargs",       # per-plot options
    "save_pdf",          # emit a .pdf next to every .png
    # Nereus >= 0.5.1. `_MODEL_KW` is a WHITELIST -- anything absent is
    # rejected here before it ever reaches Julia -- so these were added to
    # fit_* on the Julia side and then silently unreachable from Python,
    # including the `science=False` escape hatch that the release notes told
    # people to use.
    "science",           # run the post-fit block (ppc/loo/tables/...); default True
    "output",            # the JOB_CONFIG `output` block: ppc_n_draws, loo, ...
    "parametrization",   # "a_driven" | "K_driven" | "M_sec_driven"
    "time_anchor",       # "Mo" | "Tp" | "Tc" | "phic" (transit phase)
    "stability",         # "none" | "amd" | "gladman"
    "R_s",               # stellar radius (R_sun)
    "M_s",               # stellar mass (M_sun)
    "phot_trend_order",
    "as_names",          # names for relative-astrometry imagers
    "sharing",           # instrument parameter sharing groups
    "transdim_noise",
    "ttv_n_transits",
    "ttv_backend",       # "ttvfaster" | "nbody"
)


# The stellar keywords that feed only the derived table (Nereus
# `_STAR_DERIVED_KEYS`; `M_s` and `R_s` are model keywords and are in
# `_MODEL_KW`): T_eff (K), J_mag, K_mag, Ab (Bond albedo) and the 1-sigma of M*,
# R* and T_eff. Julia's fit_rv, fit_transit, fit_astrometry and fit_joint read
# them; fit_rm, fit_ttv and fit_binary do not (fit_rm and fit_binary take a
# `star` dict instead), so those reject them. tests/test_star_keywords.py checks
# this tuple against Nereus.
_STAR_KW = ("T_eff", "J_mag", "K_mag", "Ab",
            "sigma_M_s", "sigma_R_s", "sigma_T_eff")


def _gaia_extra(gaia_id, name, extra: dict | None = None) -> dict | None:
    """`gaia_id` / `name` for the wire, only when given: a runtime that predates
    them has no such keyword on Julia's fit_astrometry / fit_joint."""
    out = dict(extra or {})
    if gaia_id is not None:
        if isinstance(gaia_id, bool) or not isinstance(gaia_id, (int, str)):
            raise TypeError("gaia_id must be a Gaia DR3 source_id: an int, a "
                            f"string of digits, or 'Gaia DR3 <id>'; got {gaia_id!r}")
        out["gaia_id"] = gaia_id
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"name must be a non-empty string; got {name!r}")
        out["name"] = name
    return out or None


def _run(session, op: str, channels: Sequence[Channel], *, planets,
         engine, stopping, output_dir, priors=None, stop_file=None,
         extra: dict[str, Any] | None = None, star: bool = False, **model_kw):
    known = _MODEL_KW + (_STAR_KW if star else ())
    bad = set(model_kw) - set(known)
    if bad:
        raise TypeError(
            f"unknown keyword(s) {sorted(bad)}. Known: {', '.join(known)}")
    payload = {
        "priors": priors or {},
        "op": op,
        "channels": [c.to_wire() for c in channels],
        "planets": planets,
        # None, not `engines.DEFAULT`: naming an engine here makes api.jl
        # attach `td` to that sampler's options, and `run_engine` rejects
        # options the sampler does not declare -- so a client that always
        # named pt_emcee turned every `transdim=` fit into "engine pt_emcee
        # does not accept td". Omitted, Julia defaults BY SHAPE: pt_emcee
        # fixed-dim, transdim_pt_emcee trans-dim, at that release's budgets.
        "engine": engine.to_wire() if engine is not None else None,
        "stopping": (stopping.to_wire() if stopping else None),
        "output_dir": str(output_dir) if output_dir else None,
    }
    if stop_file is not None:
        # A path whose appearance ends the run at the last recorded step (see
        # `Stopping`). With an output_dir Julia watches <output_dir>/STOP anyway.
        payload["stop_file"] = str(stop_file)
    if extra:
        payload.update(extra)
    payload.update({k: v for k, v in model_kw.items() if v is not None})
    from ._result import JobResult
    return JobResult(session.raw(op, payload))


def replot(*channels: Channel, output_dir, plots, priors: dict | None = None,
           planets: int | Sequence[int] = 1, plot_kwargs: dict | None = None,
           save_pdf: bool = False, session=None, **model_kw) -> Any:
    """Render figures from a fit you already ran. No sampling.

        r = fit_astrometry(iad=..., priors=..., output_dir="out/g4")
        replot(Astrometry(iad=...), priors=..., output_dir="out/g4",
               plots=["corner", "orbit_skyplane"])

    Reads `<output_dir>/chains.nc`. The channels and priors are given again
    because that file stores the posterior and the RV points, not the
    astrometric abscissae or the model configuration — rebuilding the target
    takes seconds against minutes of sampling.
    """
    bad = set(model_kw) - set(_MODEL_KW)
    if bad:
        raise TypeError(
            f"unknown keyword(s) {sorted(bad)}. Known: {', '.join(_MODEL_KW)}")
    payload = {
        "op": "replot",
        "channels": [c.to_wire() for c in channels],
        "priors": priors or {},
        "planets": planets,
        "output_dir": str(output_dir),
        "plots": list(plots) if not isinstance(plots, str) else [plots],
        "plot_kwargs": plot_kwargs or {},
        "save_pdf": save_pdf,
    }
    payload.update({k: v for k, v in model_kw.items() if v is not None})
    from ._result import JobResult
    return JobResult(_sess(session).raw("replot", payload))


# --- single technique --------------------------------------------------------

def fit_rv(rv: RV | dict, *, planets: int | Sequence[int] = 1,
           jitter: Any = "default", trend_order: int = 0,
           noise: Any = None, engine: _eng.Engine | None = None,
           stopping: Stopping | None = None, stop_file=None, output_dir=None,
           priors: dict | None = None,
           session=None, **model_kw) -> Any:
    """Fit radial velocities alone.

    `planets` may be an int (fixed) or a range (trans-dimensional — the planet
    count is then inferred, see also `select_planets`).
    """
    ch = rv if isinstance(rv, RV) else RV(data=rv, jitter=jitter,
                                          trend_order=trend_order)
    return _run(_sess(session), "fit_rv", [ch], planets=planets,
                engine=engine, stopping=stopping, stop_file=stop_file, output_dir=output_dir,
                priors=priors, star=True, **model_kw, extra={"noise": noise})


def fit_transit(phot: Transit | dict, *, planets: int | Sequence[int] = 1,
                use_rho_s: bool = False,
                limb_darkening: str = "quadratic", rho_star: Any = None,
                gravity_darkening: bool = False, engine: _eng.Engine | None = None,
                stopping: Stopping | None = None, stop_file=None, output_dir=None,
                priors: dict | None = None,
                session=None, **model_kw) -> Any:
    """Fit transit photometry alone.

    `use_rho_s=True` samples the stellar density (`rho_s`, solar units) and
    derives a/R★ from it and the period, as for `fit_joint`; the prior goes in
    `priors["rho_s"]`. `rho_star` was never read and now raises: use
    `use_rho_s` and that prior.

    The other photometry options (`exposure_time`, `dilution`, `jitter`) are
    those of the `Transit` channel; pass a `Transit` for them. `limb_darkening`
    must be "quadratic" or "kipping" (one model), and `gravity_darkening=True`
    raises: the gravity-darkened transit is a job-config option.
    """
    if rho_star is not None:
        raise TypeError(
            "fit_transit: `rho_star` was never read. Pass use_rho_s=True and "
            "the density prior (solar units) as priors={\"rho_s\": {...}}")
    ch = phot if isinstance(phot, Transit) else Transit(
        data=phot, limb_darkening=limb_darkening,
        gravity_darkening=gravity_darkening)
    extra = {"use_rho_s": True} if use_rho_s else None
    return _run(_sess(session), "fit_transit", [ch], planets=planets,
                engine=engine, stopping=stopping, stop_file=stop_file, output_dir=output_dir,
                priors=priors, star=True, extra=extra, **model_kw)


def fit_astrometry(*, iad=None, hgca=None, gost=None, relast=None,
                   hip_scans=None,
                   planets: int = 1, gaia_id: int | str | None = None,
                   name: str | None = None,
                   engine: _eng.Engine | None = None,
                   stopping: Stopping | None = None, stop_file=None, output_dir=None,
                   priors: dict | None = None,
                   session=None, **model_kw) -> Any:
    """Fit absolute or relative astrometry alone (no RV).

    `priors["plx"]` must be informative: the abscissae constrain a0 ~ M_sec * plx,
    so mass and parallax are degenerate without it. `priors["M_pri"]` sets the
    primary mass. Both are priors; there is no keyword for either.

    `gaia_id` (the star's Gaia DR3 source_id: an int, a string of digits or
    "Gaia DR3 <id>") or `name` (any Simbad-resolvable name, looked up to its
    DR3 source_id; `gaia_id` wins if both are given) make Nereus query the Gaia
    DR3 parallax at run time and use it as the default `plx` prior, a Gaussian
    truncated at +-10 sigma on the raw catalogue value (no zero-point
    correction). It ranks ahead of the HGCA parallax and behind your own
    `priors["plx"]`, in which case nothing is queried. If the query fails the
    fit stops rather than falling back. Where the value came from is in
    `summary["star"]["gaia"]`. Needs a runtime with the Gaia lookup (Nereus
    newer than 0.8.10).

    `hip_scans`: the Hipparcos scans the HGCA Hipparcos epoch is fitted
    through; see `Astrometry`. None lets Nereus take the IAD's or fetch them
    by the HGCA row's `hip_id`; False opts out.
    """
    ch = Astrometry(iad=iad, hgca=hgca, gost=gost, relast=relast,
                    hip_scans=hip_scans)
    return _run(_sess(session), "fit_astrometry", [ch], planets=planets,
                engine=engine, stopping=stopping, stop_file=stop_file, output_dir=output_dir, priors=priors,
                extra=_gaia_extra(gaia_id, name), star=True, **model_kw)


def fit_rm(rv_in_transit, *, phot,
           flavour: Literal["rm", "reloaded", "arome"] = "reloaded",
           vsini=None, beta=None, lambda_prior=None,
           engine: _eng.Engine | None = None, stopping: Stopping | None = None,
           priors: dict | None = None,
           output_dir=None, session=None, **model_kw) -> Any:
    """Fit the Rossiter-McLaughlin effect in radial velocity.

    `phot` is REQUIRED, and not as a convenience: Julia's `fit_rm` declares it
    a mandatory keyword (Nereus.jl src/api.jl:668). The RM amplitude scales
    with transit depth and impact parameter, so without the photometry the
    projected obliquity is degenerate with the geometry. Pass a `Transit`
    channel, or the same instrument map `fit_transit` takes.
    """
    ch = RM(data=rv_in_transit, flavour=flavour, vsini=vsini, beta=beta,
            lambda_prior=lambda_prior)
    ph = phot if isinstance(phot, Transit) else Transit(data=phot)
    return _run(_sess(session), "fit_rm", [ch, ph], planets=1,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


def fit_tomography(nights: Sequence[Night | Mapping[str, Any]], *,
                   P: float, a_Rs: float, inc: float, vsini: float, T14: float,
                   vsys: float = 0.0, n_null: int = 300, n_lambda: int = 721,
                   output_dir=None, session=None) -> Any:
    """Recover the sky-projected obliquity from the planet's shadow in the
    stellar line profile (Doppler tomography).

    Takes NO engine and no stopping criterion, unlike every other fit_*: this
    is not a sampler. The estimator is a matched filter over lambda against
    the shadow track predicted by the transit geometry, and its significance
    comes from a null built by scrambling the in-transit frames. There is no
    tomographic likelihood to sample.

    Use it instead of `fit_rm` when the star rotates fast enough, or is
    variable enough, that the RM signal in RV is swamped: the line profile
    keeps the spatial information that collapsing to one velocity throws away.

    `nights` is one `Night` per transit, each carrying its OWN measured `Tc` --
    pooling several transits on a single ephemeris smears the shadow away --
    and, for nights weeks apart, its barycentric velocities.

        fit_tomography([Night(profiles=P1, vgrid=v, times=t1, Tc=t01, bervs=b1),
                        Night(profiles=P2, vgrid=v, times=t2, Tc=t02, bervs=b2)],
                       P=2.828, a_Rs=6.81, inc=math.radians(83.6),
                       vsini=25.9, T14=2.62/24, vsys=18.93)

    Returns lambda (radians and degrees), the matched-filter score, the
    p-value from the null, and the full lambda scan so the landscape can be
    plotted rather than a single number trusted.
    """
    if not nights:
        raise ValueError("fit_tomography needs at least one night")
    wire = [n.to_wire() if isinstance(n, Night) else dict(n) for n in nights]
    for i, n in enumerate(wire):
        missing = {"profiles", "vgrid", "times", "Tc"} - set(n)
        if missing:
            raise ValueError(f"night {i} is missing {sorted(missing)} -- every "
                             f"night needs its own measured Tc")
    payload = {"op": "fit_tomography", "nights": wire,
               "orbit": {"P": P, "a_Rs": a_Rs, "inc": inc, "vsini": vsini,
                         "T14": T14, "vsys": vsys, "n_null": n_null,
                         "n_lambda": n_lambda},
               "output_dir": str(output_dir) if output_dir else None}
    from ._result import JobResult
    return JobResult(_sess(session).raw("fit_tomography", payload))


def fit_ttv(transit_times, *, planets: int = 2, nbody: bool = False,
            engine: _eng.Engine | None = None, stopping: Stopping | None = None,
            output_dir=None, priors: dict | None = None,
            session=None, **model_kw) -> Any:
    """Fit transit timing variations."""
    ch = TTV(transit_times=transit_times, nbody=nbody)
    return _run(_sess(session), "fit_ttv", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


def fit_binary(primary, secondary=None, *, engine: _eng.Engine | None = None,
               stopping: Stopping | None = None, output_dir=None,
               priors: dict | None = None,
               session=None, **model_kw) -> Any:
    """Fit a spectroscopic binary (SB1 if `secondary` is None, else SB2)."""
    ch = SB2(primary=primary, secondary=secondary)
    return _run(_sess(session), "fit_binary", [ch], planets=1,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


# --- several techniques together ---------------------------------------------

def fit_joint(*channels: Channel, planets: int | Sequence[int] = 1,
              use_rho_s: bool = False,
              gaia_id: int | str | None = None, name: str | None = None,
              engine: _eng.Engine | None = None, stopping: Stopping | None = None, stop_file=None,
              output_dir=None, priors: dict | None = None,
              session=None, **model_kw) -> Any:
    """Fit several techniques simultaneously.

        fit_joint(RV(rv_data, jitter=...),
                  Transit(phot_data, limb_darkening="kipping"),
                  Astrometry(iad=iad),
                  planets=2, engine=engines.PT(n_rounds=12))

    One operation for "fit these together", rather than a named function per
    combination — the Julia model composes source flags for exactly this reason.

    `use_rho_s=True` samples the stellar density (`rho_s`, solar units) and
    derives the transit's a/R★ from it and the period, as a run_job config's
    `parametrization.use_rho_s` does. It needs a `Transit` channel and raises
    without one. Put the prior on it in `priors`, e.g.
    `priors={"rho_s": {"type": "normal", "mu": 1.79, "sigma": 0.11,
    "lo": 0.1, "hi": 10.0}}`, with `M_s` / `R_s` for the derived table. This
    keyword is the switch; `Transit` takes neither `use_rho_s` nor `rho_star`
    (they were never read) and raises if given them.

    `gaia_id` / `name` (with an `Astrometry` channel) query the Gaia DR3
    parallax at run time for the default `plx` prior; see `fit_astrometry`.
    Without astrometry, or with your own `priors["plx"]`, nothing is queried.
    """
    if not channels:
        raise ValueError("fit_joint needs at least one channel; for a single "
                         "technique use the dedicated fit_* function")
    if len(channels) == 1:
        raise ValueError(
            f"fit_joint with one channel — use the dedicated entry point for "
            f"{type(channels[0]).__name__} instead, it has a clearer signature")
    if use_rho_s and not any(isinstance(c, Transit) for c in channels):
        raise ValueError(
            "fit_joint: use_rho_s samples the stellar density through the "
            "transit a/R★, so it needs a Transit channel among the channels")
    # Sent only when asked for: a runtime that predates the keyword has no
    # `use_rho_s` on Julia's fit_joint, and `False` is its default anyway.
    extra = _gaia_extra(gaia_id, name, {"use_rho_s": True} if use_rho_s else None)
    return _run(_sess(session), "fit_joint", list(channels), planets=planets,
                engine=engine, stopping=stopping, stop_file=stop_file, output_dir=output_dir, priors=priors,
                extra=extra, star=True, **model_kw)


def _sess(session):
    if session is not None:
        return session
    from ._api import _shared_session
    return _shared_session()


__all__ = ["fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint"]
