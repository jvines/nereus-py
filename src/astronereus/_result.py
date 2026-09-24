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


class Figure(type(Path())):          # type(Path()) so this works pre-3.12 too
    """A path that draws itself in a notebook.

    Still a `Path` in every other respect — `open()`, `.name`, `str()` — so
    nothing that took a path before stops working.
    """

    def _repr_png_(self):
        return self.read_bytes()


class Figures(Mapping):
    """Logical figure name -> path on disk.

    Names come from the rendered `plots/` tree, e.g. "models/RV_phasefold_K1",
    "transdim/occupancy". Both mapping and attribute access work, the latter
    with '/' and '.' folded to '_' so tab-completion is usable.
    """

    def __init__(self, mapping: dict[str, str] | None):
        self._m = dict(mapping or {})
        self._alias = {k.replace("/", "_").replace(".", "_").replace("-", "_"): k
                       for k in self._m}

    def __getitem__(self, k: str) -> "Figure":
        if k in self._m:
            return Figure(self._m[k])
        if k in self._alias:
            return Figure(self._m[self._alias[k]])
        raise KeyError(f"no figure {k!r}. Have: {', '.join(sorted(self._m)) or '(none)'}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._m)

    def __len__(self) -> int:
        return len(self._m)

    def __dir__(self):
        return sorted(self._alias)

    def __getattr__(self, name: str) -> "Figure":
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __repr__(self) -> str:
        return f"<Figures: {len(self._m)} — {', '.join(sorted(self._m)[:4])}{' …' if len(self._m) > 4 else ''}>"

    def _repr_html_(self) -> str:
        """Put `result.figures` in a notebook cell and see all of them.

        Images are inlined as data URIs rather than file paths, because a
        notebook served from elsewhere (marimo, JupyterHub, a saved .html)
        cannot read the kernel's filesystem.
        """
        import base64
        if not self._m:
            return "<em>no figures — pass plots=[...] to the fit</em>"
        out = []
        for name in sorted(self._m):
            try:
                b64 = base64.b64encode(Path(self._m[name]).read_bytes()).decode()
            except OSError:
                out.append(f"<p><b>{name}</b> — missing: {self._m[name]}</p>")
                continue
            out.append(
                f'<figure style="margin:0 0 1.5em 0">'
                f'<figcaption style="font:600 12px/1.4 ui-monospace,monospace;'
                f'opacity:.7;margin-bottom:.3em">{name}</figcaption>'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;height:auto"></figure>')
        return "".join(out)


# --- parameter formatting -------------------------------------------------
# Nereus's fit_* summary reports RAW CHAIN VALUES: angles in radians, masses in
# solar units. Printing that dict straight out gives seven keys of unrounded
# float per parameter and an inclination of 2.1137, which is not a number
# anyone reads as 121 degrees.
#
# Only units VERIFIED against this package's own runs are listed. Anything not
# here is shown as plain numbers with no unit rather than a guessed one -- a
# wrong unit on a workshop slide is worse than no unit. `_SCI_UNITS` in
# Nereus.jl/src/science_tables.jl is the fuller registry, but it describes the
# science table's already-converted output, not the raw chain.
_UNITS = {
    "P": "d", "K": "m/s", "K_A": "m/s", "K_B": "m/s",
    "a": "AU", "M_sec": "M_sun", "M_pri": "M_sun", "M_s": "M_sun",
    "plx": "mas", "rho_s": "g/cm3",
    "Tp": "BJD", "Tc": "BJD", "T14": "d",
    "sesinw": "", "secosw": "", "esinw": "", "ecosw": "", "ecc": "",
    "b": "", "rr": "", "q1": "", "q2": "",
    # radians in the chain: sampled on [0, 2pi] or with a sine prior
    "inc": "rad", "Omega": "rad", "w": "rad", "omega": "rad",
    "Mo": "rad", "M0": "rad", "lambda": "rad",
}
_M_JUP_PER_M_SUN = 1047.5655        # IAU 2015 nominal
_M_EARTH_PER_M_SUN = 332946.0787

# Parameters whose posterior is routinely NOT unimodal, so a percentile
# interval across it describes the gap between two modes rather than an
# uncertainty.
#   _CIRC    live on [0, 2pi) and wrap: the 2- and 3-sigma bounds of a
#            posterior near either end come back from the far side, which is
#            how Mo_k1 reports lo3sig = 0.0035 sitting under a median of 6.00.
#   _SIGNED  flip sign together under omega -> omega+pi, the companion of the
#            Omega -> Omega+pi degeneracy that absolute astrometry alone
#            cannot break. Gaia-4 fitted on DR4 abscissae shows it plainly:
#            sesinw median +0.320 with lo16 -0.299.
_CIRC = {"Omega", "w", "omega", "Mo", "M0", "lambda",
         "Omega_deg", "w_deg", "omega_deg", "Mo_deg", "lambda_deg"}
_SIGNED = {"sesinw", "secosw", "esinw", "ecosw"}


def _base_name(name: str) -> str:
    """`a_k1` -> `a`, `sigma_HARPS` -> `sigma_HARPS`. Planet suffix only."""
    import re
    m = re.match(r"^(.*)_k\d+$", name)
    return m.group(1) if m else name


# Derived names embed their unit: `a_au_k1`, `omega_deg_k1`, `P_yr_k1`.
_UNIT_SUFFIX = {"au": "AU", "deg": "deg", "rad": "rad", "mjup": "M_jup",
                "mearth": "M_earth", "msun": "M_sun", "rjup": "R_jup",
                "rearth": "R_earth", "rsun": "R_sun", "yr": "yr", "d": "d",
                "mas": "mas", "kms": "km/s", "ms": "m/s", "k": "K"}


def _unit_of(name: str, stats=None) -> str:
    # Julia tags the science-contract entries with their unit; trust that over
    # anything inferred from the parameter's name.
    if isinstance(stats, Mapping):
        u = stats.get("unit")
        if isinstance(u, str) and u:
            return u
    b = _base_name(name)
    if b in _UNITS:
        return _UNITS[b]
    tail = b.rsplit("_", 1)[-1].lower()
    if "_" in b and tail in _UNIT_SUFFIX:
        return _UNIT_SUFFIX[tail]
    if b.startswith("gamma") or b.startswith("sigma") or b.startswith("trend"):
        return "m/s"                # RV offset/jitter; photometric ones are
    return ""                       # relative flux, so this is deliberately
                                    # not claimed for those -- see note above


def _also(name: str, v: float) -> str:
    """A second reading of the same number, where one is genuinely useful."""
    import math
    u = _unit_of(name)
    if u == "rad" and math.isfinite(v):
        return f"{math.degrees(v) % 360:.2f} deg"
    if u == "M_sun" and math.isfinite(v) and _base_name(name) == "M_sec":
        mj = v * _M_JUP_PER_M_SUN
        return (f"{mj:.3g} M_jup" if mj >= 0.1
                else f"{v * _M_EARTH_PER_M_SUN:.3g} M_earth")
    return ""


def _multimodal(name: str, d: Mapping) -> bool:
    """Is a `median +up -dn` summary misleading for this parameter?

    One rule plus two priors on which parameters are prone to it. The rule:
    if one side of the 68% interval is more than three times the other, the
    interval is spanning modes. The priors: a circular parameter whose
    interval is wider than pi has wrapped, and a sign-flipping one whose
    interval crosses zero while its median does not is showing both signs.
    """
    import math
    m = d.get("median")
    lo, hi = d.get("lo16"), d.get("hi84")
    if m is None or lo is None or hi is None:
        return False
    up, dn = hi - m, m - lo
    if min(up, dn) > 0 and max(up, dn) / min(up, dn) > 3:
        return True
    b = _base_name(name)
    if b in _CIRC and (up + dn) > math.pi:
        return True
    if b in _SIGNED and (m > 0) != (lo > 0):
        return True
    return False


def _err_scale(plus: float, minus: float) -> float:
    """Which error sets the rounding: the SMALLER one.

    Matches `_fmt3` in Nereus.jl src/science_tables.jl:379-386, which is what
    writes the LaTeX/ECSV tables. Rounding to the larger error destroys the
    precise side of an asymmetric interval -- `0.484 +0.082 -1.027` became
    `0.5 +0.1 -1.0`, a 20% rounding error on the upper bar -- and the two
    renderers then disagreed about the same numbers.
    """
    import math
    errs = [abs(e) for e in (plus, minus) if math.isfinite(e) and e > 0]
    return min(errs) if errs else float("nan")


def _sig(value: float, err: float) -> str:
    """Round to the uncertainty: two significant figures on the error.

    What every table in the field does, and it stops 1.179681790808942 from
    claiming sixteen digits of a quantity known to four.
    """
    import math
    if not math.isfinite(value):
        return str(value)
    if not (math.isfinite(err) and err > 0):
        return f"{value:.6g}"
    dec = max(0, -(math.floor(math.log10(abs(err))) - 1))
    return f"{value:.{dec}f}"


def _as_param_mapping(x) -> dict:
    """Normalise either parameter block into name -> stats.

    `fit_*` reports FITTED parameters as a mapping, but DERIVED ones come from
    `summarize_derived`, which returns a Julia `Vector{Pair{String,
    ParamStats}}` -- so over the wire it is a LIST of single-key objects,
    `[{"a_au_k1": {...}}, {"omega_deg_k1": {...}}]`, not a mapping at all.
    Assuming the two blocks looked alike is what made `result.table()` raise
    `'list' object has no attribute 'items'` on any fit with derived
    quantities.
    """
    if x is None:
        return {}
    if isinstance(x, Mapping):
        # The science contract wraps the parameters one level deeper:
        #   {"conditioning": {...}, "parameters": {name: {...}}}
        # `science_summary` sets summary["fitted"] and summary["derived"] to
        # that shape, so reading it as a flat name -> stats map produced two
        # rows called `conditioning` and `parameters`, both nan.
        inner = x.get("parameters")
        if isinstance(inner, Mapping):
            return dict(inner)
        return dict(x)
    out = {}
    for item in x:                      # list of pairs, either encoding
        if isinstance(item, Mapping):
            out.update(item)            # {"name": {...}}
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out[str(item[0])] = item[1]  # ["name", {...}]
    return out


class Estimate(float):
    """A posterior median that still behaves as a float.

    `params["a_k1"] * 2` works, `params["a_k1"]["median"]` still works, and
    printing it gives `1.1797 +0.0040 -0.0041 AU` instead of a seven-key dict.
    """

    def __new__(cls, d: Mapping, name: str = ""):
        d = dict(d)
        # ParamStats (derived) speaks best / unc_lo / unc_hi / ci1-3, where the
        # uncertainties are POSITIVE MAGNITUDES either side of `best`.
        # `_summarise_params` (fitted) speaks median / lo16 / hi84 / loNsig /
        # hiNsig. Normalise the former onto the latter so one renderer serves
        # both and `.plus` / `.minus` mean the same thing everywhere.
        # Science-contract entry: value / err_lo / err_hi / ci3, and a UNIT
        # supplied by Julia rather than inferred from the name.
        if "median" not in d and "value" in d:
            v = float(d["value"])
            d.setdefault("median", v)
            "err_lo" in d and d.setdefault("lo16", v - abs(float(d["err_lo"])))
            "err_hi" in d and d.setdefault("hi84", v + abs(float(d["err_hi"])))
            ci3 = d.get("ci3")
            if isinstance(ci3, (list, tuple)) and len(ci3) == 2:
                d.setdefault("lo3sig", ci3[0]); d.setdefault("hi3sig", ci3[1])
        if "median" not in d and "best" in d:
            best = float(d["best"])
            d.setdefault("median", best)
            if "unc_lo" in d:
                d.setdefault("lo16", best - abs(float(d["unc_lo"])))
            if "unc_hi" in d:
                d.setdefault("hi84", best + abs(float(d["unc_hi"])))
            for src, (lo, hi) in (("ci2", ("lo2sig", "hi2sig")),
                                  ("ci3", ("lo3sig", "hi3sig"))):
                v = d.get(src)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    d.setdefault(lo, v[0]); d.setdefault(hi, v[1])
        v = d.get("median", float("nan"))
        o = super().__new__(cls, v)
        o.raw, o.name = d, name
        return o

    median = property(lambda self: float(self))
    lo16 = property(lambda self: self.raw.get("lo16"))
    hi84 = property(lambda self: self.raw.get("hi84"))
    lo2sig = property(lambda self: self.raw.get("lo2sig"))
    hi2sig = property(lambda self: self.raw.get("hi2sig"))
    lo3sig = property(lambda self: self.raw.get("lo3sig"))
    hi3sig = property(lambda self: self.raw.get("hi3sig"))

    @property
    def plus(self) -> float:
        hi = self.raw.get("hi84")
        return float(hi) - float(self) if hi is not None else float("nan")

    @property
    def minus(self) -> float:
        lo = self.raw.get("lo16")
        return float(self) - float(lo) if lo is not None else float("nan")

    def __getitem__(self, k):           # the old dict access keeps working
        return self.raw[k]

    def keys(self):
        return self.raw.keys()

    def to_deg(self) -> float:
        import math
        return math.degrees(float(self))

    @property
    def multimodal(self) -> bool:
        """True when `median +up -dn` would misdescribe this posterior."""
        return _multimodal(self.name, self.raw)

    def text(self, unit: bool = True) -> str:
        # The interval is ALWAYS printed. `multimodal` is a flag on the number,
        # not a reason to withhold it: a reader who is told 0.484 and nothing
        # else cannot see that the posterior is wide, cannot tell it from a
        # tight measurement, and has no way to reach the quantiles short of
        # digging into `.raw`. Whether a percentile range is the right summary
        # for a multimodal posterior is a real question; hiding the numbers was
        # never the answer to it.
        u = _unit_of(self.name, self.raw) if unit else ""
        err = _err_scale(self.plus, self.minus)
        body = (f"{_sig(float(self), err)} "
                f"+{_sig(self.plus, err)} -{_sig(self.minus, err)}")
        alt = _also(self.name, float(self)) if unit else ""
        tail = f"  = {alt}" if alt else ""
        if self.multimodal:
            tail += "  [multimodal]"
        return f"{body}{' ' + u if u else ''}{tail}"

    def __repr__(self) -> str:
        return self.text()


class Params(Mapping):
    """Name -> `Estimate`, printing as a table rather than a wall of dicts."""

    def __init__(self, mapping, title: str = "parameters"):
        self._m = {k: Estimate(v, k) if isinstance(v, Mapping) else v
                   for k, v in _as_param_mapping(mapping).items()}
        self._title = title

    def __getitem__(self, k): return self._m[k]
    def __iter__(self): return iter(self._m)
    def __len__(self): return len(self._m)
    def __dir__(self): return sorted(self._m)

    def __getattr__(self, name):
        try:
            return self._m[name]
        except KeyError:
            raise AttributeError(name) from None

    def _rows(self):
        for k in sorted(self._m):
            v = self._m[k]
            if isinstance(v, Estimate):
                err = _err_scale(v.plus, v.minus)
                alt = _also(k, float(v))
                note = f"= {alt}" if alt else ""
                if v.multimodal:
                    note = f"{note}  multimodal" if note else "multimodal"
                yield (k, _sig(float(v), err), f"+{_sig(v.plus, err)}",
                       f"-{_sig(v.minus, err)}", _unit_of(k, v.raw), note)
            else:
                yield (k, str(v), "", "", "", "")

    def __repr__(self) -> str:
        rows = list(self._rows())
        if not rows:
            return f"<{self._title}: none>"
        w = [max(len(r[i]) for r in rows) for i in range(6)]
        out = [f"{self._title} ({len(rows)})"]
        for name, val, up, dn, unit, alt in rows:
            out.append(f"  {name:<{w[0]}}  {val:>{w[1]}} {up:>{w[2]}} {dn:>{w[3]}}"
                       f"  {unit:<{w[4]}}{('  ' + alt) if alt else ''}".rstrip())
        return "\n".join(out)

    def _repr_html_(self) -> str:
        import html as _h
        rows = list(self._rows())
        if not rows:
            return f"<em>no {_h.escape(self._title)}</em>"
        th = ('padding:.15rem .6rem;text-align:left;font:10px/1.4 ui-sans-serif,'
              'system-ui,sans-serif;text-transform:uppercase;letter-spacing:.07em;'
              'opacity:.55;border-bottom:1px solid color-mix(in srgb,currentColor 20%,transparent)')
        td = ('padding:.15rem .6rem;font-variant-numeric:tabular-nums;'
              'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px')
        body = "".join(
            f'<tr><td style="{td}">{_h.escape(n)}</td>'
            f'<td style="{td};text-align:right">{_h.escape(v)}</td>'
            f'<td style="{td};text-align:right;opacity:.7">{_h.escape(up)}</td>'
            f'<td style="{td};text-align:right;opacity:.7">{_h.escape(dn)}</td>'
            f'<td style="{td};opacity:.8">{_h.escape(u)}</td>'
            f'<td style="{td};opacity:.6">{_h.escape(a)}</td></tr>'
            for n, v, up, dn, u, a in rows)
        return (f'<table style="border-collapse:collapse;margin:.4rem 0">'
                f'<thead><tr><th style="{th}">parameter</th>'
                f'<th style="{th};text-align:right">median</th>'
                f'<th style="{th};text-align:right">+1σ</th>'
                f'<th style="{th};text-align:right">−1σ</th>'
                f'<th style="{th}">unit</th><th style="{th}"></th></tr></thead>'
                f'<tbody>{body}</tbody></table>')


class Diagnostics:
    """Sampler health for a finished fit: the ladder, and convergence.

    Two sources, because they live in different places. The ladder --
    within-rung acceptance, swap acceptance, the beta schedule -- comes back
    on the summary. R-hat and ESS do NOT: Nereus computes them every
    `diag_every` steps to drive the live progress bar and then discards them,
    so the numbers you would quote to justify a fit are not in the output.
    They are recomputed here from `<output_dir>/chains.nc`, which stores the
    full (chain x iter) cube precisely so a multi-chain R-hat is possible.

    That recompute needs netCDF4 and arviz, which astronereus does not depend
    on -- the package deliberately imports nothing, not even numpy. Install
    them with `pip install astronereus[diagnostics]`; everything else here
    works without.
    """

    def __init__(self, raw: Mapping, output_dir=None):
        self.raw = dict(raw or {})
        # Nereus >= 0.4.6 persists a per-engine diagnostics block. Prefer it:
        # it is what the sampler itself measured, including R-hat and ESS that
        # older runtimes computed for the progress bar and discarded. The
        # chains.nc recompute below stays as the fallback for a bundle that
        # predates it.
        self.block = dict(self.raw.get("diagnostics") or {})
        self._dir = str(output_dir or self.raw.get("output_dir", "") or "")

    @property
    def engine(self):
        return self.block.get("engine") or self.raw.get("sampler")

    # -- the ladder, always available --------------------------------------
    def _f(self, key):
        v = self.block.get(key, self.raw.get(key)) or []
        return [float(x) for x in v] if isinstance(v, (list, tuple)) else []

    def _s(self, key):
        v = self.block.get(key, self.raw.get(key))
        return v if isinstance(v, (int, float)) else None

    @property
    def acceptance_within(self): return self._f("acceptance_within")
    @property
    def acceptance_swap(self): return self._f("acceptance_swap")
    @property
    def betas(self): return self._f("betas")

    @property
    def min_swap(self):
        """Lowest swap acceptance between adjacent rungs.

        The number that matters, and the one the sampler's own docstring warns
        about: the ladder is the half of parallel tempering that fails
        SILENTLY. Once the rungs stop exchanging, the cold chain sits in one
        mode and R-hat cheerfully certifies convergence to it. Below ~0.01,
        do not trust the run.
        """
        v = self._s("min_swap")
        if v is not None:
            return float(v)
        a = self.acceptance_swap
        return min(a) if a else None

    @property
    def cold_acceptance(self):
        """Acceptance of the beta=1 rung -- the chain your posterior is.

        Distinct from the pooled figure, which averages in the hot rungs:
        they sample a flattened posterior, move further and are rejected more,
        so pooling drags the number below what the cold chain is doing.
        """
        v = self._s("cold_acceptance")
        if v is not None:
            return float(v)
        a = self.acceptance_within
        return a[0] if a else None

    @property
    def ok(self):
        """None when it cannot be judged without the convergence numbers."""
        ms = self.min_swap
        if ms is not None and ms < 0.01:
            return False
        c = self.convergence()
        if c is None or c.get("worst_rhat") is None:
            return None
        return c["worst_rhat"] <= 1.01

    # -- convergence, lazily ------------------------------------------------
    def convergence(self, force: bool = False):
        """R-hat / ESS per parameter, or None if it cannot be computed."""
        if not force and hasattr(self, "_conv"):
            return self._conv
        self._conv = self._compute_convergence()
        return self._conv

    def _compute_convergence(self):
        import os
        # Persisted by the runtime: no chains.nc read, no optional deps, and
        # it reflects what the sampler saw rather than a re-derivation.
        c = self.block.get("convergence")
        if isinstance(c, Mapping) and c.get("per_parameter"):
            return {"per_parameter": dict(c["per_parameter"]),
                    "n_chains": c.get("n_chains"), "n_iter": c.get("n_iter"),
                    "worst_rhat": c.get("worst_rhat"),
                    "worst_ess_bulk": c.get("worst_ess_bulk"),
                    "worst_ess_tail": c.get("worst_ess_tail"),
                    "note": c.get("rhat_note"), "table": None}
        if self.block.get("convergence_note"):
            return {"per_parameter": {}, "note": self.block["convergence_note"],
                    "table": None, "worst_rhat": None, "worst_ess_tail": None,
                    "n_chains": None, "n_iter": None}
        path = os.path.join(self._dir, "chains.nc") if self._dir else ""
        if not path or not os.path.exists(path):
            return None
        try:
            import netCDF4
            import arviz as az
        except ImportError:
            return None
        d = netCDF4.Dataset(path)
        post = {v: d.variables[v][:] for v in d.variables
                if set(d.variables[v].dimensions) == {"iter", "chain"}}
        if not post:
            return None
        try:
            # arviz 1.x takes the group dict positionally; 0.x wants posterior=.
            try:
                idata = az.from_dict({"posterior": post})
            except TypeError:
                idata = az.from_dict(posterior=post)
            summ = az.summary(idata, kind="diagnostics")
        except Exception:
            return None
        shape = next(iter(post.values())).shape
        per = {str(i): {"ess_bulk": float(r["ess_bulk"]),
                        "ess_tail": float(r["ess_tail"]),
                        "rhat": float(r["r_hat"])}
               for i, r in summ.iterrows()}
        return {"per_parameter": per, "table": summ,
                "n_chains": shape[0], "n_iter": shape[1],
                "note": "recomputed from chains.nc; each walker is a chain, "
                        "so R-hat is strict — ensemble walkers are not independent",
                "worst_rhat": float(summ["r_hat"].max()),
                "worst_ess_bulk": float(summ["ess_bulk"].min()),
                "worst_ess_tail": float(summ["ess_tail"].min())}

    # -- rendering ----------------------------------------------------------
    def _lines(self):
        out = ["sampler"]
        for label, key, fmt in (("engine", "sampler", "{}"),
                                ("log Z", "log_z", "{:.2f}"),
                                ("likelihood evals", "n_evals", "{:,}"),
                                ("elapsed", "elapsed_sec", "{:.0f} s")):
            v = self.raw.get(key)
            if v is not None:
                try: out.append(f"  {label:<18s}{fmt.format(v)}")
                except (ValueError, TypeError): out.append(f"  {label:<18s}{v}")

        aw = self.acceptance_within
        if aw:
            out += ["", "within-rung acceptance   (beta=1 is the posterior; want ~0.2-0.5)",
                    f"  cold (beta=1)     {aw[0]:.3f}",
                    f"  hottest           {aw[-1]:.3f}",
                    f"  min / mean        {min(aw):.3f} / {sum(aw)/len(aw):.3f}"]

        asw = self.acceptance_swap
        if asw:
            i = asw.index(min(asw))
            out += ["", "swap acceptance between adjacent rungs   (min < 0.01 => ladder decoupled)",
                    f"  min               {min(asw):.3f}   at rung pair {i}-{i+1}",
                    f"  mean              {sum(asw)/len(asw):.3f}",
                    "  ladder            " + " ".join(f"{x:.2f}" for x in asw)]
            if min(asw) < 0.01:
                out.append("  *** the ladder is NOT exchanging. Raise n_temps and refit. ***")

        b = self.betas
        if b:
            out += ["", f"temperatures        {len(b)} rungs, beta {min(b):.2e} .. {max(b):.2f}"]

        c = self.convergence()
        if c is None:
            out += ["", "convergence         unavailable — this runtime does not persist it;",
                    "                    `pip install astronereus[diagnostics]` to recompute",
                    "                    it from chains.nc"]
            return out

        if not c.get("per_parameter"):
            out += ["", "convergence         not applicable",
                    f"                    {c.get('note', '')}"]
            return out

        hdr = f"\nconvergence   ({c['n_chains']} chains x {c['n_iter']} kept iterations)"
        out.append(hdr)
        if c.get("note"):
            out.append(f"              {c['note']}")
        rows = sorted(c["per_parameter"].items())
        has_r = any("rhat" in v for _, v in rows)
        w = max(len(k) for k, _ in rows)
        out.append(f"  {'':<{w}}  {'ess_bulk':>9} {'ess_tail':>9}"
                   + (f" {'r_hat':>7}" if has_r else ""))
        for k, v in rows:
            line = (f"  {k:<{w}}  {v.get('ess_bulk', float('nan')):>9.0f}"
                    f" {v.get('ess_tail', float('nan')):>9.0f}")
            if has_r:
                line += f" {v.get('rhat', float('nan')):>7.4f}"
            out.append(line)
        out.append("")
        if c.get("worst_rhat") is not None:
            out.append(f"  worst R-hat       {c['worst_rhat']:.4f}   (< 1.01 is comfortable)")
        if c.get("worst_ess_tail") is not None:
            out.append(f"  worst tail ESS    {c['worst_ess_tail']:.0f}      (> 400 is plenty for 1-sigma)")
        if c.get("worst_rhat") is not None and c["worst_rhat"] > 1.01:
            out.append("  *** not converged on at least one parameter ***")
        return out

    def __repr__(self) -> str:
        return "\n".join(self._lines())

    def _repr_html_(self) -> str:
        import html as _h
        return (f'<pre style="font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,'
                f'monospace;white-space:pre-wrap">{_h.escape(repr(self))}</pre>')


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
    def params(self) -> "Params":
        """Per-parameter posterior summary, as a printable table."""
        return Params(self.raw.get("params", {}) or {}, "fitted parameters")

    @property
    def derived(self) -> "Params":
        """Physical quantities per planet (M_p, a, T_eq, ρ_p, TSM, ESM, …)."""
        return Params(self.raw.get("derived", {}) or {}, "derived quantities")

    def table(self, *, fitted: bool = True, derived: bool = True,
              diagnostics: bool = False) -> "ResultTable":
        """Everything worth reading off a fit, in sections.

            print(result.table())        # fitted + derived
            result.table()               # same, rendered in a notebook
            result.table(diagnostics=True)

        Sections are kept apart because the two are different kinds of
        number: `fitted` are the sampled parameters, in the raw units the
        chain carries; `derived` are quantities the model computes FROM them.
        Printing them in one undifferentiated block invites quoting a derived
        mass as though it had been sampled.
        """
        secs = []
        if fitted:
            secs.append(self.params)
        if derived and len(self.derived):
            secs.append(self.derived)
        return ResultTable(self, secs, self.diagnostics if diagnostics else None)

    @property
    def diagnostics(self) -> "Diagnostics":
        """Sampler health: ladder acceptance, swaps, R-hat and ESS."""
        return Diagnostics(self.raw)

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


class ResultTable:
    """Sections of a result, rendered together. See `JobResult.table`."""

    def __init__(self, result, sections, diagnostics=None):
        self._r, self._secs, self._diag = result, list(sections), diagnostics

    def __repr__(self) -> str:
        out = [repr(self._r)]
        for sec in self._secs:
            out += ["", repr(sec)]
        if self._diag is not None:
            out += ["", repr(self._diag)]
        return "\n".join(out)

    def _repr_html_(self) -> str:
        import html as _h
        out = [f'<div style="font:13px/1.5 ui-sans-serif,system-ui,sans-serif">'
               f'<div style="opacity:.7;margin-bottom:.5rem">{_h.escape(repr(self._r))}</div>']
        for sec in self._secs:
            out.append(f'<div style="font:10px/1.4 ui-sans-serif,system-ui;'
                       f'text-transform:uppercase;letter-spacing:.07em;opacity:.55;'
                       f'margin:.9rem 0 .2rem">{_h.escape(getattr(sec, "_title", ""))}</div>')
            out.append(sec._repr_html_())
        if self._diag is not None:
            out.append('<div style="font:10px/1.4 ui-sans-serif,system-ui;'
                       'text-transform:uppercase;letter-spacing:.07em;opacity:.55;'
                       'margin:.9rem 0 .2rem">diagnostics</div>')
            out.append(self._diag._repr_html_())
        out.append("</div>")
        return "".join(out)
