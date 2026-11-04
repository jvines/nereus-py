"""Data channels — one class per observable, each carrying its own settings.

These exist so `fit_joint` can compose techniques without turning into a
function with ten optional keyword arguments. Each channel holds the data AND
the options that only make sense for that observable: limb darkening belongs to
Transit, jitter and systemic offsets to RV. Priors (plx, M_pri, per-planet)
are NOT channel fields -- they go in the fit's `priors` dict.

They mirror the Julia source flags in `src/model.jl` — RV_SOURCE, PM_SOURCE,
AS_SOURCE, RM_SOURCE, RM_R_SOURCE, RM_A_SOURCE, GD_SOURCE, TTV_SOURCE,
TTV_NB_SOURCE, SB_SOURCE — which compose into a `PlanetDataSources` set rather
than an enum of named combinations.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, Mapping, Sequence


class Channel:
    source: str = ""

    def to_wire(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = {k: v for k, v in asdict(self).items() if v is not None}
        d["source"] = self.source
        return d


def _load(action, payload, session=None):
    """Call a daemon loader op, on `session` or the process-wide one."""
    if session is None:
        from ._api import _shared_session
        session = _shared_session()
    return session.raw(action, payload)


def read_rv(path, *, session=None, format="auto", time_offset=2_400_000.5,
            rename=None, instrument_names=None,
            t_col="bjd", rv_col="rv", err_col="rv_err", inst_col="inst"):
    """Read an RV table into `{instrument: {t, rv, rv_err}}` -- the shape
    `RV(data=...)` wants -- using NEREUS'S OWN READER.

        rv = read_rv(".../gaia4_rv.dat")
        fit_joint(RV(data=rv), Astrometry(...), ...)

    This delegates to the exported Julia loaders (`load_vizier_rv`,
    `load_orvara_rv`); it does not parse anything in Python. That matters
    because Nereus already had these readers and no way to reach them, so every
    notebook grew its own parse loop.

    `format`:
      - `"auto"`    -- `.csv` -> vizier, anything else -> labeled
      - `"vizier"`  -- comma-delimited with a header; `t_col`/`rv_col`/
                       `err_col`/`inst_col` name the columns. Pass
                       `inst_col=None` for a single-instrument file.
      - `"orvara"`  -- whitespace, INTEGER instrument ids in column 4
      - `"labeled"` -- whitespace, STRING instrument labels in column 4

    `time_offset` is SUBTRACTED from the time column and applied identically
    for every format, so the readers cannot disagree about the time scale. The
    default converts BJD to MJD, which is what Nereus models in; pass 0 for
    times already in MJD.

    `rename` maps raw instrument labels to the names your priors use, e.g.
    `{"j": "HIRES", "lick": "Lick"}`. `instrument_names` replaces them
    positionally, which is how an orvara file's integer ids get real names.

    Needs a running session (it is the Julia reader doing the work); one is
    started lazily if you do not pass `session=`.
    """
    payload = {"path": str(path), "format": format, "time_offset": time_offset,
               "t_col": t_col, "rv_col": rv_col, "err_col": err_col,
               "inst_col": inst_col}
    if rename:
        payload["rename"] = dict(rename)
    if instrument_names:
        payload["instrument_names"] = list(instrument_names)
    return _load("load_rv", payload, session)["data"]


def read_photometry(path, *, session=None, instrument="TESS",
                    trim_window=None, time_offset=0.0):
    """Read a light curve into `{instrument: {t, flux, flux_err}}` for
    `Transit(data=...)`, via Nereus's exported `load_tess_lc`.

    `trim_window=(t_lo, t_hi)` keeps only points inside that window -- use it
    to pick a contiguous slab when a sector spans a downlink gap.
    """
    payload = {"path": str(path), "instrument": instrument,
               "time_offset": time_offset}
    if trim_window is not None:
        payload["trim_window"] = [float(trim_window[0]), float(trim_window[1])]
    return _load("load_photometry", payload, session)["data"]


def read_relastrom(path, *, session=None):
    """Read an orvara-format relative-astrometry file via Nereus's exported
    `load_orvara_relast`, returning the `values` block `Astrometry(relast=...)`
    takes: t (MJD), ra_off/dec_off (mas), ra_err/dec_err, corr, planet_idx.

    The (sep, PA) -> (dRA, ddec) transform and its full error propagation
    happen in Julia; nothing is recomputed here.
    """
    d = _load("load_relastrom", {"path": str(path)}, session)
    return {k: v for k, v in d.items() if k not in ("path", "n")}


@dataclass
class RV(Channel):
    """Radial velocities, per instrument.

    `data` maps instrument name -> {t, rv, rv_err}. Each instrument gets its own
    systemic offset and jitter unless shared via `share`.

    `RV.from_file(path, **kw)` builds that mapping from a table; see `read_rv`.
    """
    source = "RV"
    data: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    jitter: Any = "default"
    trend_order: int = 0
    gamma_per_instrument: bool = True
    share: Mapping[str, Sequence[Sequence[str]]] | None = None
    marginalize_gamma: bool = False
    indicators: Mapping[str, Sequence[float]] | None = None

    @classmethod
    def from_file(cls, path, *, jitter="default", trend_order=0,
                  gamma_per_instrument=True, share=None,
                  marginalize_gamma=False, indicators=None, **read_kw):
        """`RV` straight from a table. Channel options pass through; the rest
        goes to `read_rv` (`format=`, `rename=`, `time_offset=`, `session=`)."""
        return cls(data=read_rv(path, **read_kw), jitter=jitter,
                   trend_order=trend_order,
                   gamma_per_instrument=gamma_per_instrument, share=share,
                   marginalize_gamma=marginalize_gamma, indicators=indicators)

    @classmethod
    def from_table(cls, table, *, jitter="default", trend_order=0,
                   gamma_per_instrument=True, share=None,
                   marginalize_gamma=False, indicators=None, **kw):
        """`RV` from a table you already have in Python -- a pandas
        DataFrame, an astropy Table, a numpy structured array, or a dict of
        columns.

            RV.from_table(df, time="bjd", rv="rv", rv_err="rv_error",
                          instrument="inst")

        Column names are guessed from the usual spellings when not given.
        `time_offset="auto"` (the default) converts full Julian Dates to MJD
        and leaves MJD alone; BTJD/BKJD need an explicit offset.
        """
        from ._ingest import rv_from_table
        return cls(data=rv_from_table(table, **kw), jitter=jitter,
                   trend_order=trend_order,
                   gamma_per_instrument=gamma_per_instrument, share=share,
                   marginalize_gamma=marginalize_gamma, indicators=indicators)

    @classmethod
    def from_arrays(cls, t, rv, rv_err, *, instrument="INST",
                    time_offset="auto", rename=None, jitter="default",
                    trend_order=0, gamma_per_instrument=True, share=None,
                    marginalize_gamma=False, indicators=None):
        """`RV` from three bare arrays. `instrument` is one label for every
        row, or a per-row sequence of labels."""
        from ._ingest import group_by_instrument, resolve_time
        data = group_by_instrument(resolve_time(t, time_offset), rv, rv_err,
                                   instrument, "rv", "rv_err", rename=rename)
        return cls(data=data, jitter=jitter, trend_order=trend_order,
                   gamma_per_instrument=gamma_per_instrument, share=share,
                   marginalize_gamma=marginalize_gamma, indicators=indicators)

    @classmethod
    def from_vizier(cls, catalogue, *, jitter="default", trend_order=0,
                    gamma_per_instrument=True, share=None,
                    marginalize_gamma=False, indicators=None,
                    row_limit=-1, columns=None, constraints=None, **kw):
        """`RV` straight from a VizieR catalogue (needs `astroquery`).

            RV.from_vizier("J/AJ/169/107/table5",
                           time="BJD", rv="RV", rv_err="e_RV",
                           instrument="Inst")

        All rows and all columns are fetched -- VizieR's own defaults truncate
        both. `constraints` is a dict of column filters.
        """
        from ._ingest import vizier_table, rv_from_table
        tab = vizier_table(catalogue, row_limit=row_limit, columns=columns,
                           **(constraints or {}))
        return cls(data=rv_from_table(tab, **kw), jitter=jitter,
                   trend_order=trend_order,
                   gamma_per_instrument=gamma_per_instrument, share=share,
                   marginalize_gamma=marginalize_gamma, indicators=indicators)


@dataclass
class Transit(Channel):
    """Transit photometry.

    Every option below reaches the model (`fit_transit`, `fit_joint`, and
    `fit_rm` for `exposure_time`), or raises: nothing is accepted and ignored.

    - `exposure_time` (seconds; one value, or `{instrument: seconds}`) turns on
      finite-exposure supersampling, for cadences longer than 3 minutes (the
      sub-sample count follows from it, one per minute up to 30). Left unset the
      model is instantaneous, which is right for 2-minute data and wrong for
      30-minute data.
    - `dilution` (`{instrument: D}`, D the third-light fraction; a number is a
      fixed value, a prior dict a sampled one) and `jitter` (a number or prior
      dict for every instrument, or `{instrument: ...}`) set the priors on
      `dilution_<instrument>` and `jitter_<instrument>`. Left at the defaults
      (no third light, a modified-Jeffreys jitter) they change nothing; giving
      the same parameter in the fit's `priors` as well is an error.
    - `limb_darkening` is "quadratic" or "kipping": one model, sampled in
      Kipping's (q1, q2). The other laws do not exist in Nereus and raise.
    - `gravity_darkening=True` and `supersample` (an explicit sub-sample count)
      are not implemented on the typed route and raise; use a job config for
      gravity darkening, and `exposure_time` for supersampling.
    """
    source = "PM"
    data: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    limb_darkening: Literal["quadratic", "kipping"] = "quadratic"
    gravity_darkening: bool = False
    supersample: int | None = None
    exposure_time: float | Mapping[str, float] | None = None
    jitter: Any = "default"
    dilution: Mapping[str, Any] | None = None
    # REMOVED, kept only so passing them is an error rather than a silent no-op.
    # `use_rho_s` (default True here) and `rho_star` were carried to the daemon
    # and never read by Nereus, so they changed nothing. The stellar density is a
    # property of the fit, not of one light curve: `fit_joint(..., use_rho_s=True,
    # priors={"rho_s": ...})` or `fit_transit(...)` with the same keywords. An
    # InitVar is not a field, so it never reaches `to_wire`.
    rho_star: InitVar[Any] = None
    use_rho_s: InitVar[Any] = None

    def __post_init__(self, rho_star, use_rho_s):
        if rho_star is not None or use_rho_s is not None:
            raise TypeError(
                "Transit no longer takes `use_rho_s` or `rho_star`: they were "
                "accepted and never read. Sample the stellar density with "
                "`fit_joint(..., use_rho_s=True)` or `fit_transit(..., "
                "use_rho_s=True)` and put its prior (solar units) in "
                "`priors={\"rho_s\": {\"type\": \"normal\", \"mu\": ..., "
                "\"sigma\": ..., \"lo\": ..., \"hi\": ...}}`.")
        if self.limb_darkening not in ("quadratic", "kipping"):
            raise ValueError(
                f"Transit(limb_darkening={self.limb_darkening!r}): the transit "
                "model is quadratic, sampled in Kipping's (q1, q2); \"quadratic\" "
                "and \"kipping\" are the same model. Other laws are not "
                "implemented.")
        if self.gravity_darkening:
            raise ValueError(
                "Transit(gravity_darkening=True): the gravity-darkened transit "
                "(planet mode PM_GD) is a job-config option, not built by the "
                "typed fit_* functions.")
        if self.supersample is not None:
            raise ValueError(
                "Transit(supersample=...): there is no explicit sub-sample "
                "count. Set exposure_time (seconds) and the count follows from "
                "it.")
        ex = self.exposure_time
        if ex is not None:
            vals = ex.values() if isinstance(ex, Mapping) else [ex]
            if not all(isinstance(v, (int, float)) and v > 0 for v in vals):
                raise ValueError("Transit(exposure_time=...) must be positive "
                                 "seconds, one number or {instrument: seconds}")

    @classmethod
    def from_file(cls, path, *, limb_darkening="quadratic", rho_star=None,
                  use_rho_s=None, gravity_darkening=False, supersample=None,
                  exposure_time=None, jitter="default", dilution=None,
                  **read_kw):
        """`Transit` straight from a light curve; see `read_photometry`."""
        return cls(data=read_photometry(path, **read_kw),
                   limb_darkening=limb_darkening, rho_star=rho_star,
                   use_rho_s=use_rho_s, gravity_darkening=gravity_darkening,
                   supersample=supersample, exposure_time=exposure_time,
                   jitter=jitter, dilution=dilution)

    _CH = ("limb_darkening", "rho_star", "use_rho_s", "gravity_darkening",
           "supersample", "exposure_time", "jitter", "dilution")

    @classmethod
    def _with(cls, data, ch):
        return cls(data=data, **{k: ch[k] for k in cls._CH if k in ch})

    @classmethod
    def from_table(cls, table, *, limb_darkening="quadratic", rho_star=None,
                   use_rho_s=None, gravity_darkening=False, supersample=None,
                   exposure_time=None, jitter="default", dilution=None, **kw):
        """`Transit` from an in-memory table; see `RV.from_table`. Pass
        `normalize=True` for a light curve still in electrons per second."""
        from ._ingest import phot_from_table
        return cls._with(phot_from_table(table, **kw), locals())

    @classmethod
    def from_arrays(cls, t, flux, flux_err, *, instrument="TESS",
                    time_offset="auto", rename=None, limb_darkening="quadratic",
                    rho_star=None, use_rho_s=None, gravity_darkening=False,
                    supersample=None, exposure_time=None, jitter="default",
                    dilution=None):
        """`Transit` from three bare arrays."""
        from ._ingest import group_by_instrument, resolve_time
        data = group_by_instrument(resolve_time(t, time_offset), flux, flux_err,
                                   instrument, "flux", "flux_err", rename=rename)
        return cls._with(data, locals())

    @classmethod
    def from_lightkurve(cls, lc, *, normalize=True, per_sector=True,
                        instrument=None, limb_darkening="quadratic",
                        rho_star=None, use_rho_s=None, gravity_darkening=False,
                        supersample=None, exposure_time=None, jitter="default",
                        dilution=None):
        """`Transit` from a lightkurve `LightCurve` or `LightCurveCollection`.

            lc = lk.search_lightcurve("TOI-700", mission="TESS").download_all()
            Transit.from_lightkurve(lc)

        Times come off `lc.time.jd`, so TESS's BTJD and Kepler's BKJD epoch
        offsets are resolved by astropy instead of guessed. NaN cadences are
        dropped. `per_sector=True` gives each sector its own instrument name,
        hence its own jitter and dilution -- sectors differ in crowding, and
        pooling them under one label hides that.
        """
        from ._ingest import phot_from_lightkurve
        data = phot_from_lightkurve(lc, normalize=normalize,
                                    per_sector=per_sector, instrument=instrument)
        return cls._with(data, locals())


@dataclass
class Astrometry(Channel):
    """Absolute astrometry: Hipparcos IAD, Gaia epoch abscissae, HGCA, GOST.

    `plx` and `M_pri` are PRIORS and go in the fit's `priors` dict, not here.
    `priors["plx"]` must be informative: the abscissae constrain a0 ∝ M_sec·ϖ,
    so mass and parallax are degenerate without it.

    `hgca` and `gost` take the same blocks as a run_job config, or what the
    session's loaders return (Nereus >= 0.8.8):

        hgca={"hip_id": 85653}                          # fetch the Brandt eDR3 row
        hgca={"fits": "HGCA_vEDR3.fits", "hip_id": 85653}
        hgca=s.load_hgca(path="HGCA_vEDR3.fits", hip=85653)
        hgca={"values": {"t": ..., "pmra": ..., "pmdec": ..., "cov_ep": ...,
                         "plx": ..., "plx_err": ..., "hip_id": ...}}

        gost={"ra_deg": 263.09, "dec_deg": 8.91}        # fetch the scan plan
        gost=s.load_gost(path="gost.csv")
        gost={"values": {"t": ..., "psi": ..., "parallax_factor": ...}}

    RV + HGCA is `fit_joint(RV(...), Astrometry(hgca=..., gost=...), ...)`.
    """
    source = "AS"
    iad: Any = None
    hgca: Any = None
    gost: Any = None
    relast: Any = None

    @classmethod
    def from_relast_file(cls, path, *, iad=None, hgca=None, gost=None,
                         **read_kw):
        """`Astrometry` carrying relative astrometry read from an
        orvara-format file; see `read_relastrom`."""
        return cls(iad=iad, hgca=hgca, gost=gost,
                   relast={"values": read_relastrom(path, **read_kw)})


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
           "TTV", "SB2", "read_rv", "read_photometry", "read_relastrom"]
