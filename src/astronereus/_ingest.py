"""Turning things Python already holds into channel data.

The interfaces that matter for day-to-day work are: a table (DataFrame,
astropy Table, dict of columns, structured array), bare arrays, a VizieR
query, and a lightkurve object. Another code's on-disk format is not one of
them -- that belongs to whichever Julia loader was written for it.

Nothing here is a parser. It reshapes columns already in memory into
`{instrument: {t, rv, rv_err}}` / `{instrument: {t, flux, flux_err}}`; reading
a FILE is Julia's job (see `channels.read_rv`).

lightkurve, astroquery, pandas and astropy are all OPTIONAL and duck-typed.
The package keeps its single `zstandard` dependency: a workshop install must
not pull half of astropy.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["as_columns", "group_by_instrument", "resolve_time"]

#: Above this, a time column is a full Julian Date rather than an MJD.
#: BJD is ~2.46e6 and MJD ~6e4, so there is no real overlap to get wrong.
_JD_FLOOR = 2_000_000.0
_JD_TO_MJD = 2_400_000.5


def _seq(v) -> list:
    """Any column-ish thing -> a plain list of Python floats/strings.

    Handles astropy Columns and Quantities (`.value`), masked arrays
    (`.filled`), numpy arrays, and plain sequences, without importing numpy.
    """
    v = getattr(v, "value", v)          # Quantity / Column -> ndarray
    if hasattr(v, "filled"):            # masked array -> fill with NaN
        try:
            v = v.filled(float("nan"))
        except Exception:
            v = v.data
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v)


def as_columns(obj: Any) -> dict[str, list]:
    """Any in-memory table -> `{column name: list}`.

    Accepts a pandas DataFrame, an astropy Table/QTable, a numpy structured
    array, or a mapping of column name -> sequence. A file path is NOT
    accepted: reading files goes through Nereus's own loaders.
    """
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        raise TypeError(
            "as_columns takes an in-memory table, not a path. Read the file "
            "with read_rv()/read_photometry() -- those call Nereus's own "
            "loaders -- or load it yourself and pass the table.")

    names = (getattr(obj, "colnames", None)                    # astropy Table
             or getattr(getattr(obj, "dtype", None), "names", None)  # structured
             or (list(obj.columns) if hasattr(obj, "columns") else None))  # DataFrame
    if names is not None:
        return {str(n): _seq(obj[n]) for n in names}

    if isinstance(obj, Mapping):
        return {str(k): _seq(v) for k, v in obj.items()}

    raise TypeError(
        f"don't know how to read columns off {type(obj).__name__}; pass a "
        f"DataFrame, an astropy Table, a structured array, or a dict of "
        f"columns")


def _pick(cols: dict[str, list], want: str | None, defaults: Sequence[str],
          what: str) -> list | None:
    """Resolve one column, by explicit name or by trying the usual spellings."""
    if want is not None:
        if want not in cols:
            raise KeyError(
                f"no column {want!r} for {what}; the table has "
                f"{list(cols)}")
        return cols[want]
    lower = {k.lower(): k for k in cols}
    for d in defaults:
        if d in lower:
            return cols[lower[d]]
    return None


def resolve_time(t: Sequence[float], time_offset: Any = "auto") -> list[float]:
    """Put a time column on Nereus's scale (MJD).

    `time_offset="auto"` subtracts 2400000.5 when the values are full Julian
    Dates and nothing when they are already MJD. That distinction is safe --
    BJD is ~2.46e6, MJD ~6e4.

    It is NOT safe for mission-relative scales: TESS BTJD (BJD-2457000) and
    Kepler BKJD (BJD-2454833) both look like small numbers and would be left
    alone. Pass the offset yourself for those, or use
    `Transit.from_lightkurve`, which reads the scale off the astropy `Time`
    object instead of guessing.
    """
    t = [float(x) for x in t]
    if time_offset == "auto":
        mid = sorted(t)[len(t) // 2] if t else 0.0
        off = _JD_TO_MJD if mid > _JD_FLOOR else 0.0
    else:
        off = float(time_offset)
    return [x - off for x in t]


def group_by_instrument(t, y, e, inst, ykey: str, ekey: str, *,
                        rename: Mapping[str, str] | None = None,
                        drop_nan: bool = True) -> dict[str, dict[str, list]]:
    """Columns -> `{instrument: {t, <ykey>, <ekey>}}`.

    `inst` is a per-row sequence of labels or a single label for every row.
    Rows with a non-finite time, value or error are dropped by default --
    lightkurve light curves carry NaNs by construction and a NaN reaching the
    likelihood poisons the whole fit rather than one point.
    """
    n = len(t)
    if not (len(y) == len(e) == n):
        raise ValueError(f"column length mismatch: t={n}, value={len(y)}, "
                         f"err={len(e)}")
    if isinstance(inst, str) or inst is None:
        labels = [inst or "INST"] * n
    else:
        labels = [str(x) for x in _seq(inst)]
        if len(labels) != n:
            raise ValueError(f"instrument column has {len(labels)} rows, "
                             f"data has {n}")
    ren = dict(rename or {})

    out: dict[str, dict[str, list]] = {}
    for i in range(n):
        ti, yi, ei = float(t[i]), float(y[i]), float(e[i])
        if drop_nan and not (ti == ti and yi == yi and ei == ei):
            continue
        nm = ren.get(labels[i], labels[i])
        d = out.setdefault(nm, {"t": [], ykey: [], ekey: []})
        d["t"].append(ti)
        d[ykey].append(yi)
        d[ekey].append(ei)
    if not out:
        raise ValueError("no finite rows left after dropping NaNs")
    return out


# --- the two shapes -----------------------------------------------------------
_RV_DEFAULTS = {
    "time": ("bjd", "jd", "time", "t", "epoch", "bjd_tdb", "rjd", "mjd"),
    "rv": ("rv", "vrad", "radial_velocity", "velocity"),
    "rv_err": ("rv_err", "rv_error", "e_rv", "err", "svrad", "rv_unc",
               "sigma_rv"),
    "instrument": ("inst", "instrument", "tel", "telescope", "source"),
}

_PHOT_DEFAULTS = {
    "time": ("time", "bjd", "jd", "t", "bjd_tdb", "btjd", "bkjd"),
    "flux": ("flux", "pdcsap_flux", "sap_flux", "norm_flux"),
    "flux_err": ("flux_err", "flux_error", "e_flux", "pdcsap_flux_err",
                 "sap_flux_err", "err"),
    "instrument": ("inst", "instrument", "mission", "sector", "tel"),
}


def rv_from_table(table, *, time=None, rv=None, rv_err=None, instrument=None,
                  time_offset="auto", rename=None) -> dict:
    """Table -> `{instrument: {t, rv, rv_err}}`.

    Column names are guessed from the usual spellings (`bjd`/`jd`/`time`,
    `rv`, `rv_err`/`rv_error`/`e_rv`, `inst`/`instrument`) and can each be
    named explicitly. `instrument` may also be a literal label when the table
    has no instrument column -- if the name is not a column, it is used as the
    label.
    """
    cols = as_columns(table)
    t = _pick(cols, time, _RV_DEFAULTS["time"], "time")
    v = _pick(cols, rv, _RV_DEFAULTS["rv"], "rv")
    e = _pick(cols, rv_err, _RV_DEFAULTS["rv_err"], "rv_err")
    for name, got in (("time", t), ("rv", v), ("rv_err", e)):
        if got is None:
            raise KeyError(
                f"could not find a {name} column in {list(cols)}; name it "
                f"explicitly, e.g. {name}=\"<column>\"")
    if instrument is not None and instrument not in cols:
        inst = instrument                      # a literal label, not a column
    else:
        inst = _pick(cols, instrument, _RV_DEFAULTS["instrument"], "instrument")
        if inst is None:
            inst = "INST"
    return group_by_instrument(resolve_time(t, time_offset), v, e, inst,
                               "rv", "rv_err", rename=rename)


def phot_from_table(table, *, time=None, flux=None, flux_err=None,
                    instrument=None, time_offset="auto", rename=None,
                    normalize=False) -> dict:
    """Table -> `{instrument: {t, flux, flux_err}}`.

    `normalize=True` divides flux and its error by the median flux, for a
    light curve still in electrons per second.
    """
    cols = as_columns(table)
    t = _pick(cols, time, _PHOT_DEFAULTS["time"], "time")
    f = _pick(cols, flux, _PHOT_DEFAULTS["flux"], "flux")
    e = _pick(cols, flux_err, _PHOT_DEFAULTS["flux_err"], "flux_err")
    for name, got in (("time", t), ("flux", f), ("flux_err", e)):
        if got is None:
            raise KeyError(
                f"could not find a {name} column in {list(cols)}; name it "
                f"explicitly, e.g. {name}=\"<column>\"")
    if normalize:
        f, e = _normalize(f, e)
    if instrument is not None and instrument not in cols:
        inst = instrument
    else:
        inst = _pick(cols, instrument, _PHOT_DEFAULTS["instrument"], "instrument")
        if inst is None:
            inst = "TESS"
    return group_by_instrument(resolve_time(t, time_offset), f, e, inst,
                               "flux", "flux_err", rename=rename)


def _normalize(f, e):
    vals = sorted(x for x in (float(y) for y in f) if x == x)
    if not vals:
        raise ValueError("flux column is all NaN")
    med = vals[len(vals) // 2]
    if med == 0:
        raise ValueError("median flux is zero; cannot normalize")
    return [float(x) / med for x in f], [float(x) / med for x in e]


# --- lightkurve ---------------------------------------------------------------
def _one_lightkurve(lc, *, normalize: bool, label: str | None):
    """One lightkurve.LightCurve -> (t_mjd, flux, flux_err, label).

    Time comes off `lc.time.jd`, so the mission's epoch offset (BTJD for TESS,
    BKJD for Kepler) is resolved by astropy rather than guessed here. That is
    the whole reason this path exists instead of `from_table(lc.to_table())`.
    """
    time = getattr(lc, "time", None)
    if time is None or not hasattr(time, "jd"):
        raise TypeError(
            "expected a lightkurve LightCurve whose .time is an astropy Time; "
            f"got {type(lc).__name__}. If you already have plain arrays, use "
            "Transit.from_arrays.")
    t = [float(x) - _JD_TO_MJD for x in _seq(time.jd)]
    f = _seq(getattr(lc, "flux"))
    e_attr = getattr(lc, "flux_err", None)
    if e_attr is None:
        raise ValueError("light curve has no flux_err; Nereus needs "
                         "per-cadence uncertainties")
    e = _seq(e_attr)
    if normalize:
        f, e = _normalize(f, e)
    if label is None:
        meta = getattr(lc, "meta", {}) or {}
        mission = str(meta.get("MISSION") or meta.get("TELESCOP") or "TESS")
        sector = meta.get("SECTOR") or meta.get("QUARTER") or meta.get("CAMPAIGN")
        label = f"{mission}_s{int(sector)}" if sector is not None else mission
    return t, f, e, label


def phot_from_lightkurve(lc, *, normalize: bool = True, per_sector: bool = True,
                         instrument: str | None = None) -> dict:
    """A lightkurve `LightCurve` or `LightCurveCollection` -> channel data.

    `per_sector=True` (the default) gives each sector/quarter its own
    instrument name, so it gets its own jitter and dilution terms -- sectors
    differ in crowding and systematics, and pooling them under one label hides
    that. Pass `instrument="TESS"` to pool them deliberately.

    `normalize=True` divides by the median, which is what you want for a
    PDCSAP light curve in electrons per second.
    """
    parts = list(lc) if _is_collection(lc) else [lc]
    if not parts:
        raise ValueError("empty LightCurveCollection")
    out: dict[str, dict[str, list]] = {}
    for i, one in enumerate(parts):
        label = instrument if instrument is not None else (
            None if per_sector else "TESS")
        t, f, e, name = _one_lightkurve(one, normalize=normalize, label=label)
        if name in out and not per_sector:
            pass
        elif name in out:
            name = f"{name}_{i}"
        block = group_by_instrument(t, f, e, name, "flux", "flux_err")
        for k, v in block.items():
            if k in out:
                for col in ("t", "flux", "flux_err"):
                    out[k][col].extend(v[col])
            else:
                out[k] = v
    for v in out.values():                       # keep each block time-ordered
        order = sorted(range(len(v["t"])), key=lambda i: v["t"][i])
        for col in ("t", "flux", "flux_err"):
            v[col] = [v[col][i] for i in order]
    return out


def _is_collection(obj) -> bool:
    if isinstance(obj, (str, bytes, Mapping)):
        return False
    if type(obj).__name__.endswith("Collection"):
        return True
    # A LightCurve is itself iterable (over rows), so iterability alone is not
    # the test -- a collection's elements have a .time of their own.
    return (isinstance(obj, (list, tuple))
            and bool(obj) and hasattr(obj[0], "time"))


# --- VizieR -------------------------------------------------------------------
def vizier_table(catalogue: str, *, row_limit: int = -1,
                 columns: Sequence[str] | None = None, **constraints):
    """Fetch one VizieR table as an astropy Table (needs `astroquery`).

    `catalogue` is a VizieR identifier, e.g. "J/AJ/169/107/table5". Extra
    keywords become column constraints. All columns and all rows by default --
    VizieR's own defaults truncate both, which silently loses data.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:                  # pragma: no cover
        raise ImportError(
            "reading from VizieR needs astroquery: pip install astroquery"
        ) from exc
    v = Vizier(columns=list(columns) if columns else ["**"],
               row_limit=row_limit, column_filters={
                   k: str(val) for k, val in constraints.items()})
    got = v.get_catalogs(catalogue)
    if len(got) == 0:
        raise ValueError(f"VizieR returned nothing for {catalogue!r}")
    return got[0]
