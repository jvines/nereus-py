"""The data Nereus ships, by name.

Nereus carries the tables its examples use. Until now reaching them meant
building a path into the runtime depot and hand-parsing the file, which is how
every notebook ended up with its own copy of the same parse loop:

    rvfile = os.path.join(astronereus.runtime_dir(), "depot", "dev", "Nereus",
                          "test", "data", "gaia4_rv.dat")
    for line in open(rvfile):
        ...

There is no reason for a user to know any of that. The registry lives in the
daemon and knows both where each table is and which of Nereus's readers it
needs -- including the column-name overrides the shipped CSVs require, since
they predate `load_vizier_rv`'s defaults.

    from astronereus import dataset

    ds = dataset("gaia4")
    ds.describe()                      # a table of what is in it
    fit_rv(ds.rv, planets=1, ...)      # ready channels, no paths
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping

from .channels import RV, Astrometry, Transit

__all__ = ["Dataset", "dataset", "list_datasets", "rv_table"]


def _session(session=None):
    if session is not None:
        return session
    from ._api import _shared_session
    return _shared_session()


def list_datasets(*, session=None) -> list[dict[str, Any]]:
    """Every shipped dataset: name, target, reference, channels available."""
    return _session(session).raw("datasets", {})["datasets"]


def rv_table(data: Mapping[str, Mapping[str, list]], title: str = "Radial velocities") -> str:
    """One row per instrument: coverage, scatter and quoted precision.

    `rms` is the raw scatter about that instrument's OWN mean, so it is signal
    plus offset-free jitter, not a fit residual. The comparison that matters is
    rms against the median error.
    """
    rows = []
    for inst, d in sorted(data.items(), key=lambda kv: min(kv[1]["t"])):
        t, v, e = d["t"], d["rv"], d["rv_err"]
        rows.append((inst, len(t), min(t), max(t), max(t) - min(t),
                     statistics.median(v),
                     statistics.pstdev(v) if len(v) > 1 else float("nan"),
                     statistics.median(e)))
    if not rows:
        return f"{title} — no data"
    t0, t1 = min(r[2] for r in rows), max(r[3] for r in rows)
    n = sum(r[1] for r in rows)
    w = max(10, max(len(r[0]) for r in rows))

    head = (f"{'instrument':<{w}}  {'N':>4}  {'MJD start':>10}  {'MJD end':>10}  "
            f"{'span (d)':>9}  {'median RV (m/s)':>15}  {'rms (m/s)':>10}  "
            f"{'median err (m/s)':>16}")
    out = [f"{title} — {n} points, {len(rows)} instruments, {t1 - t0:.1f} d baseline",
           head, "─" * len(head)]
    for inst, ni, a, b, span, med, rms, err in rows:
        out.append(f"{inst:<{w}}  {ni:>4d}  {a:>10.3f}  {b:>10.3f}  {span:>9.1f}  "
                   f"{med:>15.2f}  {rms:>10.2f}  {err:>16.2f}")
    out.append("─" * len(head))
    out.append(f"{'all':<{w}}  {n:>4d}  {t0:>10.3f}  {t1:>10.3f}  {t1 - t0:>9.1f}")
    return "\n".join(out)


def _phot_table(data, title="Photometry"):
    rows = []
    for inst, d in sorted(data.items(), key=lambda kv: min(kv[1]["t"])):
        t, f, e = d["t"], d["flux"], d["flux_err"]
        rows.append((inst, len(t), min(t), max(t),
                     statistics.pstdev(f) * 1e6 if len(f) > 1 else float("nan"),
                     statistics.median(e) * 1e6))
    w = max(10, max(len(r[0]) for r in rows))
    head = (f"{'instrument':<{w}}  {'N':>6}  {'MJD start':>10}  {'MJD end':>10}  "
            f"{'rms (ppm)':>10}  {'median err (ppm)':>16}")
    out = [title, head, "─" * len(head)]
    for inst, n, a, b, rms, err in rows:
        out.append(f"{inst:<{w}}  {n:>6d}  {a:>10.3f}  {b:>10.3f}  "
                   f"{rms:>10.1f}  {err:>16.1f}")
    return "\n".join(out)


def _relast_table(rel, title="Relative astrometry"):
    t = rel["t"]
    rho = [ (ra ** 2 + de ** 2) ** 0.5
            for ra, de in zip(rel["ra_off"], rel["dec_off"]) ]
    head = (f"{'N':>4}  {'MJD start':>10}  {'MJD end':>10}  "
            f"{'sep min (mas)':>13}  {'sep max (mas)':>13}")
    return "\n".join([
        title, head, "─" * len(head),
        f"{len(t):>4d}  {min(t):>10.3f}  {max(t):>10.3f}  "
        f"{min(rho):>13.1f}  {max(rho):>13.1f}"])


@dataclass
class Dataset:
    """A shipped dataset with its channels already built.

    `rv`, `transit` and `astrometry` are Nereus channels, ready to hand to a
    `fit_*` call. `raw` keeps the loader output if you want the arrays.
    """
    name: str
    target: str
    ref: str
    rv: RV | None = None
    transit: Transit | None = None
    astrometry: Astrometry | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def describe(self, *, show: bool = True) -> str:
        """A table of what is in this dataset, per instrument."""
        parts = [f"{self.name} — {self.target}", f"reference: {self.ref}", ""]
        if self.rv is not None:
            parts.append(rv_table(self.rv.data, "Radial velocities"))
            parts.append("")
        if self.transit is not None:
            parts.append(_phot_table(self.transit.data))
            parts.append("")
        if self.astrometry is not None and self.astrometry.relast:
            parts.append(_relast_table(self.astrometry.relast["values"]))
            parts.append("")
        text = "\n".join(parts).rstrip()
        if show:
            print(text)
        return text

    def __repr__(self) -> str:
        chans = [n for n in ("rv", "transit", "astrometry")
                 if getattr(self, n) is not None]
        return (f"Dataset({self.name!r}, target={self.target!r}, "
                f"channels={chans})")


def dataset(name: str, *, session=None, jitter="default", trend_order=0,
            **rv_kw) -> Dataset:
    """Load a shipped dataset by name -- no paths, no parsing.

        ds = dataset("gaia4")
        ds.describe()
        fit_rv(ds.rv, planets=1, priors={...}, engine=..., output_dir=...)

    `jitter`, `trend_order` and any other keyword go to the `RV` channel.
    `list_datasets()` names what is available.
    """
    d = _session(session).raw("dataset", {"name": name})
    out = Dataset(name=d["name"], target=d["target"], ref=d["ref"], raw=d)
    if "rv" in d:
        out.rv = RV(data=d["rv"]["data"], jitter=jitter,
                    trend_order=trend_order, **rv_kw)
    if "photometry" in d:
        out.transit = Transit(data=d["photometry"]["data"])
    if "relast" in d:
        vals = {k: v for k, v in d["relast"].items() if k not in ("path", "n")}
        out.astrometry = Astrometry(relast={"values": vals})
    return out
