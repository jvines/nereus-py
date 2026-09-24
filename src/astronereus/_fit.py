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
    "time_anchor",       # "Mo" | "Tc"
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


def _run(session, op: str, channels: Sequence[Channel], *, planets,
         engine, stopping, output_dir, priors=None,
         extra: dict[str, Any] | None = None, **model_kw):
    bad = set(model_kw) - set(_MODEL_KW)
    if bad:
        raise TypeError(
            f"unknown keyword(s) {sorted(bad)}. Known: {', '.join(_MODEL_KW)}")
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
           stopping: Stopping | None = None, output_dir=None,
           priors: dict | None = None,
           session=None, **model_kw) -> Any:
    """Fit radial velocities alone.

    `planets` may be an int (fixed) or a range (trans-dimensional — the planet
    count is then inferred, see also `select_planets`).
    """
    ch = rv if isinstance(rv, RV) else RV(data=rv, jitter=jitter,
                                          trend_order=trend_order)
    return _run(_sess(session), "fit_rv", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir,
                priors=priors, **model_kw, extra={"noise": noise})


def fit_transit(phot: Transit | dict, *, planets: int | Sequence[int] = 1,
                limb_darkening: str = "quadratic", rho_star: Any = None,
                gravity_darkening: bool = False, engine: _eng.Engine | None = None,
                stopping: Stopping | None = None, output_dir=None,
                priors: dict | None = None,
                session=None, **model_kw) -> Any:
    """Fit transit photometry alone.

    `gravity_darkening=True` uses the oblate von Zeipel/Barnes model, which
    yields stellar inclination i* separately from lambda.
    """
    ch = phot if isinstance(phot, Transit) else Transit(
        data=phot, limb_darkening=limb_darkening, rho_star=rho_star,
        gravity_darkening=gravity_darkening)
    return _run(_sess(session), "fit_transit", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


def fit_astrometry(*, iad=None, hgca=None, gost=None, relast=None,
                   planets: int = 1,
                   engine: _eng.Engine | None = None,
                   stopping: Stopping | None = None, output_dir=None,
                   priors: dict | None = None,
                   session=None, **model_kw) -> Any:
    """Fit absolute or relative astrometry alone (no RV).

    `priors["plx"]` must be informative: the abscissae constrain a0 ~ M_sec * plx,
    so mass and parallax are degenerate without it. `priors["M_pri"]` sets the
    primary mass. Both are priors; there is no keyword for either.
    """
    ch = Astrometry(iad=iad, hgca=hgca, gost=gost, relast=relast)
    return _run(_sess(session), "fit_astrometry", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


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
              engine: _eng.Engine | None = None, stopping: Stopping | None = None,
              output_dir=None, priors: dict | None = None,
              session=None, **model_kw) -> Any:
    """Fit several techniques simultaneously.

        fit_joint(RV(rv_data, jitter=...),
                  Transit(phot_data, limb_darkening="kipping"),
                  Astrometry(iad=iad),
                  planets=2, engine=engines.PT(n_rounds=12))

    One operation for "fit these together", rather than a named function per
    combination — the Julia model composes source flags for exactly this reason.
    """
    if not channels:
        raise ValueError("fit_joint needs at least one channel; for a single "
                         "technique use the dedicated fit_* function")
    if len(channels) == 1:
        raise ValueError(
            f"fit_joint with one channel — use the dedicated entry point for "
            f"{type(channels[0]).__name__} instead, it has a clearer signature")
    return _run(_sess(session), "fit_joint", list(channels), planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir, priors=priors, **model_kw)


def _sess(session):
    if session is not None:
        return session
    from ._api import _shared_session
    return _shared_session()


__all__ = ["fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint"]
