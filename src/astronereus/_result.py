"""Typed view over what `run_job` returns.

Nereus returns a 17-key nested dict (status, log_z, params, derived, figures,
plots, fit_health, loo, ppc, detection_limits, n_planets_posterior, sampler,
elapsed_sec, n_evals, config_path, error, traceback). Handing that to a user
raw means they must memorise key names and nesting, `print()` gives a wall of
JSON, and a failed run looks identical to a successful one until you check
`summary["status"]` yourself.

This wraps it. The underlying dict stays reachable as `.raw` — nothing is
hidden, and a key added by a newer Nereus is still accessible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping


class JobFailed(RuntimeError):
    """A job that Nereus itself reported as failed (status != "ok")."""

    def __init__(self, error: str, traceback: str = "", config_path: str = ""):
        self.error, self.traceback, self.config_path = error, traceback, config_path
        super().__init__(f"{error}\n{traceback}".rstrip())


class Figures(Mapping):
    """Logical figure name -> path on disk.

    Names come from the rendered `plots/` tree, e.g. "models/RV_phasefold_P1",
    "transdim/occupancy". Both mapping and attribute access work, the latter
    with '/' and '.' folded to '_' so tab-completion is usable.
    """

    def __init__(self, mapping: dict[str, str] | None):
        self._m = dict(mapping or {})
        self._alias = {k.replace("/", "_").replace(".", "_").replace("-", "_"): k
                       for k in self._m}

    def __getitem__(self, k: str) -> Path:
        if k in self._m:
            return Path(self._m[k])
        if k in self._alias:
            return Path(self._m[self._alias[k]])
        raise KeyError(f"no figure {k!r}. Have: {', '.join(sorted(self._m)) or '(none)'}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._m)

    def __len__(self) -> int:
        return len(self._m)

    def __dir__(self):
        return sorted(self._alias)

    def __getattr__(self, name: str) -> Path:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __repr__(self) -> str:
        return f"<Figures: {len(self._m)} — {', '.join(sorted(self._m)[:4])}{' …' if len(self._m) > 4 else ''}>"


class JobResult:
    """Result of `run_job`. Wraps the summary dict; `.raw` is the original."""

    def __init__(self, summary: dict[str, Any]):
        self.raw = summary or {}

    # -- status ------------------------------------------------------------
    @property
    def status(self) -> str:
        return self.raw.get("status", "unknown")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def check(self) -> "JobResult":
        """Raise `JobFailed` unless the run succeeded. Returns self, so chainable."""
        if not self.ok:
            raise JobFailed(self.raw.get("error", f"status={self.status}"),
                            self.raw.get("traceback", ""),
                            self.raw.get("config_path", ""))
        return self

    # -- the things people actually reach for ------------------------------
    @property
    def figures(self) -> Figures:
        return Figures(self.raw.get("figures"))

    @property
    def params(self) -> dict[str, Any]:
        """Per-parameter posterior summary."""
        return self.raw.get("params", {}) or {}

    @property
    def derived(self) -> dict[str, Any]:
        """Physical quantities per planet (M_p, a, T_eq, ρ_p, TSM, ESM, …)."""
        return self.raw.get("derived", {}) or {}

    @property
    def log_z(self) -> float | None:
        v = self.raw.get("log_z")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def n_planets_posterior(self):
        """Occupancy over planet count for a trans-dim run; None if fixed-dim."""
        return self.raw.get("n_planets_posterior")

    @property
    def fit_health(self):
        return self.raw.get("fit_health")

    @property
    def loo(self):
        return self.raw.get("loo")

    @property
    def ppc(self):
        return self.raw.get("ppc")

    @property
    def detection_limits(self):
        return self.raw.get("detection_limits")

    @property
    def sampler(self):
        return self.raw.get("sampler")

    @property
    def elapsed_sec(self) -> float | None:
        v = self.raw.get("elapsed_sec")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def n_evals(self):
        return self.raw.get("n_evals")

    @property
    def config_path(self):
        p = self.raw.get("config_path")
        return Path(p) if p else None

    # -- ergonomics --------------------------------------------------------
    def __contains__(self, k: str) -> bool:
        return k in self.raw

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def keys(self):
        return self.raw.keys()

    def __repr__(self) -> str:
        if not self.ok:
            return (f"<JobResult FAILED status={self.status!r} "
                    f"error={str(self.raw.get('error'))[:60]!r}>")
        bits = [f"status={self.status}"]
        if self.log_z is not None:
            bits.append(f"log_z={self.log_z:.2f}")
        if self.elapsed_sec is not None:
            bits.append(f"elapsed={self.elapsed_sec:.0f}s")
        bits.append(f"{len(self.params)} params")
        bits.append(f"{len(self.figures)} figures")
        return f"<JobResult {' '.join(bits)}>"

    def summary_lines(self) -> str:
        """A human-readable digest — what you want when you `print(result)`."""
        if not self.ok:
            return (f"Nereus job FAILED ({self.status})\n"
                    f"  error: {self.raw.get('error')}\n"
                    f"  {self.raw.get('traceback', '')[:800]}")
        out = [repr(self)]
        if self.sampler:
            out.append(f"  sampler : {self.sampler}")
        if self.n_planets_posterior is not None:
            out.append(f"  N_planets posterior: {self.n_planets_posterior}")
        if self.params:
            out.append("  params  : " + ", ".join(sorted(self.params)[:8])
                       + (" …" if len(self.params) > 8 else ""))
        if len(self.figures):
            out.append("  figures : " + ", ".join(sorted(self.figures)[:5])
                       + (" …" if len(self.figures) > 5 else ""))
        return "\n".join(out)
