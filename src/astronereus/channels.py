"""Data channels — one class per observable, each carrying its own settings.

These exist so `fit_joint` can compose techniques without turning into a
function with ten optional keyword arguments. Each channel holds the data AND
the options that only make sense for that observable: limb darkening belongs to
Transit, jitter and systemic offsets to RV, parallax to Astrometry.

They mirror the Julia source flags in `src/model.jl` — RV_SOURCE, PM_SOURCE,
AS_SOURCE, RM_SOURCE, RM_R_SOURCE, RM_A_SOURCE, GD_SOURCE, TTV_SOURCE,
TTV_NB_SOURCE, SB_SOURCE — which compose into a `PlanetDataSources` set rather
than an enum of named combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


class Channel:
    source: str = ""

    def to_wire(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = {k: v for k, v in asdict(self).items() if v is not None}
        d["source"] = self.source
        return d


@dataclass
class RV(Channel):
    """Radial velocities, per instrument.

    `data` maps instrument name -> {t, rv, rv_err}. Each instrument gets its own
    systemic offset and jitter unless shared via `share`.
    """
    source = "RV"
    data: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    jitter: Any = "default"
    trend_order: int = 0
    gamma_per_instrument: bool = True
    share: Mapping[str, Sequence[Sequence[str]]] | None = None
    marginalize_gamma: bool = False
    indicators: Mapping[str, Sequence[float]] | None = None


@dataclass
class Transit(Channel):
    """Transit photometry.

    Set `gravity_darkening=True` for the oblate/von-Zeipel model, which gives
    stellar inclination i* separately from lambda — that is GD_SOURCE, a
    distinct source flag, not a cosmetic option.
    """
    source = "PM"
    data: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    limb_darkening: Literal["quadratic", "kipping", "nonlinear", "linear"] = "quadratic"
    rho_star: Any = None
    use_rho_s: bool = True
    gravity_darkening: bool = False
    supersample: int | None = None
    exposure_time: float | None = None
    jitter: Any = "default"
    dilution: Mapping[str, float] | None = None


@dataclass
class Astrometry(Channel):
    """Absolute astrometry: Hipparcos IAD, Gaia epoch abscissae, HGCA, GOST.

    `parallax` should be an informative prior — the abscissae constrain
    a0 proportional to M_sec times parallax, which is degenerate without it.
    """
    source = "AS"
    iad: Any = None
    hgca: Any = None
    gost: Any = None
    relast: Any = None
    parallax: Any = None
    m_pri: float | None = None


@dataclass
class RM(Channel):
    """Rossiter-McLaughlin in radial velocity.

    `flavour` picks the kernel: "reloaded" (RM_R_SOURCE) or "arome"
    (RM_A_SOURCE). Plain "rm" (RM_SOURCE) is the legacy flux-weighted-mean
    model, which carries known amplitude and shape error at high vsini/beta.
    """
    source = "RM"
    data: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    flavour: Literal["rm", "reloaded", "arome"] = "reloaded"
    vsini: Any = None
    beta: Any = None
    lambda_prior: Any = None

    def __post_init__(self):
        self.source = {"rm": "RM", "reloaded": "RM_R", "arome": "RM_A"}[self.flavour]


@dataclass
class Night:
    """One transit night of line profiles, for `fit_tomography`.

    NOT a Channel: tomography has no source flag and cannot enter `fit_joint`.
    There is no tomographic likelihood to add to the others -- the estimator is
    a matched filter, so it composes with nothing.

    `Tc` is this night's own measured mid-transit time. Pooling several
    transits folded on a single ephemeris smears the shadow away, which is why
    it is per night and required.

    `bervs` is required in practice for anything pooled across weeks: nights a
    month apart differ by ~13 km/s in barycentric velocity, a sizeable fraction
    of v sin i, so without it the tracks do not line up and the pooled peak is
    spurious.

    Every night's profiles must share a sign convention. A DRS CCF is a dip
    (`ccf/continuum`) while a mask-CCF built by accumulating line depth peaks;
    mixing the two makes one night subtract from the others.
    """
    profiles: Any = None            # n_time x n_velocity
    vgrid: Sequence[float] | None = None
    times: Sequence[float] | None = None
    Tc: float | None = None
    bervs: Sequence[float] | None = None

    def to_wire(self) -> dict[str, Any]:
        from dataclasses import asdict
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TTV(Channel):
    """Transit timing variations. `nbody=True` selects the N-body model."""
    source = "TTV"
    transit_times: Any = None
    nbody: bool = False

    def __post_init__(self):
        self.source = "TTV_NB" if self.nbody else "TTV"


@dataclass
class SB2(Channel):
    """Double-lined spectroscopic binary: both components' velocities."""
    source = "SB"
    primary: Any = None
    secondary: Any = None


__all__ = ["Channel", "RV", "Transit", "Astrometry", "RM", "Night",
           "TTV", "SB2"]
