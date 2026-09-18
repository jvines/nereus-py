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


def _run(session, op: str, channels: Sequence[Channel], *, planets,
         engine, stopping, output_dir, extra: dict[str, Any] | None = None):
    payload = {
        "op": op,
        "channels": [c.to_wire() for c in channels],
        "planets": planets,
        "engine": (engine or _eng.DEFAULT).to_wire(),
        "stopping": (stopping.to_wire() if stopping else None),
        "output_dir": str(output_dir) if output_dir else None,
    }
    if extra:
        payload.update(extra)
    from ._result import JobResult
    return JobResult(session.raw(op, payload))


# --- single technique --------------------------------------------------------

def fit_rv(rv: RV | dict, *, planets: int | Sequence[int] = 1,
           jitter: Any = "default", trend_order: int = 0,
           noise: Any = None, engine: _eng.Engine | None = None,
           stopping: Stopping | None = None, output_dir=None,
           session=None) -> Any:
    """Fit radial velocities alone.

    `planets` may be an int (fixed) or a range (trans-dimensional — the planet
    count is then inferred, see also `select_planets`).
    """
    ch = rv if isinstance(rv, RV) else RV(data=rv, jitter=jitter,
                                          trend_order=trend_order)
    return _run(_sess(session), "fit_rv", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir,
                extra={"noise": noise})


def fit_transit(phot: Transit | dict, *, planets: int | Sequence[int] = 1,
                limb_darkening: str = "quadratic", rho_star: Any = None,
                gravity_darkening: bool = False, engine: _eng.Engine | None = None,
                stopping: Stopping | None = None, output_dir=None,
                session=None) -> Any:
    """Fit transit photometry alone.

    `gravity_darkening=True` uses the oblate von Zeipel/Barnes model, which
    yields stellar inclination i* separately from lambda.
    """
    ch = phot if isinstance(phot, Transit) else Transit(
        data=phot, limb_darkening=limb_darkening, rho_star=rho_star,
        gravity_darkening=gravity_darkening)
    return _run(_sess(session), "fit_transit", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir)


def fit_astrometry(*, iad=None, hgca=None, gost=None, relast=None,
                   parallax=None, m_pri=None, planets: int = 1,
                   engine: _eng.Engine | None = None,
                   stopping: Stopping | None = None, output_dir=None,
                   session=None) -> Any:
    """Fit absolute or relative astrometry alone (no RV).

    `parallax` should be informative: the abscissae constrain a0 ~ M_sec * plx,
    degenerate without it.
    """
    ch = Astrometry(iad=iad, hgca=hgca, gost=gost, relast=relast,
                    parallax=parallax, m_pri=m_pri)
    return _run(_sess(session), "fit_astrometry", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir)


def fit_rm(rv_in_transit, *, phot,
           flavour: Literal["rm", "reloaded", "arome"] = "reloaded",
           vsini=None, beta=None, lambda_prior=None,
           engine: _eng.Engine | None = None, stopping: Stopping | None = None,
           output_dir=None, session=None) -> Any:
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
                engine=engine, stopping=stopping, output_dir=output_dir)


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
            output_dir=None, session=None) -> Any:
    """Fit transit timing variations."""
    ch = TTV(transit_times=transit_times, nbody=nbody)
    return _run(_sess(session), "fit_ttv", [ch], planets=planets,
                engine=engine, stopping=stopping, output_dir=output_dir)


def fit_binary(primary, secondary=None, *, engine: _eng.Engine | None = None,
               stopping: Stopping | None = None, output_dir=None,
               session=None) -> Any:
    """Fit a spectroscopic binary (SB1 if `secondary` is None, else SB2)."""
    ch = SB2(primary=primary, secondary=secondary)
    return _run(_sess(session), "fit_binary", [ch], planets=1,
                engine=engine, stopping=stopping, output_dir=output_dir)


# --- several techniques together ---------------------------------------------

def fit_joint(*channels: Channel, planets: int | Sequence[int] = 1,
              engine: _eng.Engine | None = None, stopping: Stopping | None = None,
              output_dir=None, session=None) -> Any:
    """Fit several techniques simultaneously.

        fit_joint(RV(rv_data, jitter=...),
                  Transit(phot_data, limb_darkening="kipping"),
                  Astrometry(iad=iad, parallax=plx),
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
                engine=engine, stopping=stopping, output_dir=output_dir)


def _sess(session):
    if session is not None:
        return session
    from ._api import _shared_session
    return _shared_session()


__all__ = ["fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint"]
