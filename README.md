# astronereus

Python client for [Nereus](https://github.com/jvines/Nereus.jl), a Julia package for
Bayesian orbit inference: radial velocities, transits, absolute and relative
astrometry, Rossiter-McLaughlin, Doppler tomography, TTVs, spectroscopic binaries
and trans-dimensional planet counts, alone or jointly. This page documents
astronereus 0.4.36 against Nereus 0.8.8. Source:
[jvines/nereus-py](https://github.com/jvines/nereus-py); licence:
[LICENSE](https://github.com/jvines/nereus-py/blob/main/LICENSE).

You do not need Julia. astronereus fetches a prebuilt Nereus runtime on first use,
starts it as a separate process (a warm daemon) and sends it JSON jobs. Nothing is
compiled at install time, and the daemon stays up between calls, so Julia's start-up
cost is paid once per session, not once per fit.

## Contents

- [Install](#install)
- [Sessions](#sessions)
- [Getting data in](#getting-data-in)
- [Pre-fit tools](#pre-fit-tools)
- [Fit modes](#fit-modes)
  - [Radial velocities](#radial-velocities)
  - [Transits](#transits)
  - [RV and transit together](#rv-and-transit-together)
  - [Rossiter-McLaughlin](#rossiter-mclaughlin)
  - [Doppler tomography](#doppler-tomography)
  - [Transit timing variations](#transit-timing-variations)
  - [Spectroscopic binaries](#spectroscopic-binaries)
- [Astrometry](#astrometry)
  - [RV and HGCA](#rv-and-hgca)
  - [Adding GOST](#adding-gost)
  - [Hipparcos and Gaia epoch astrometry (`iad`)](#hipparcos-and-gaia-epoch-astrometry-iad)
  - [Relative astrometry (imaging)](#relative-astrometry-imaging)
  - [Getting the catalogue data](#getting-the-catalogue-data)
- [Priors](#priors)
- [Samplers](#samplers)
- [Trans-dimensional fits](#trans-dimensional-fits)
- [Results](#results)
- [Job configs (`run_job`)](#job-configs-run_job)

## Install

```sh
pip install astronereus
```

The first call that needs Julia fetches the runtime (about 480 MB on
`macos-arm64`), unpacks it and warms it once, and says so:

```
astronereus: no Julia runtime yet -- fetching it once now.
astronereus: fetching runtime (macos-arm64, julia 1.11.9) ...
astronereus: unpacking ...
astronereus: warming the runtime (one-time, a few minutes) ...
astronereus: ready (207s)
```

Wheels have no post-install hook, so pip cannot do this step. To do it ahead of a
workshop or a flight:

```sh
python -c "import astronereus; astronereus.install()"
```

`NEREUS_NO_AUTO_INSTALL=1` turns the automatic fetch off; the first call then fails
with instructions instead. Use it in CI and on air-gapped machines.

The distribution and the import are both `astronereus`. The name `nereus` on PyPI
belongs to an unrelated geophysics package.

### Optional extras

The base install has one dependency (`zstandard`, to unpack the runtime). Everything
else is imported lazily, so install only what you use:

```sh
pip install "astronereus[arrays,tables]"
```

| extra | installs | enables |
|---|---|---|
| `arrays` | numpy | numpy arrays in payloads. astronereus never imports numpy itself; lists, pandas columns and astropy columns also work |
| `tables` | pandas | tables as pandas DataFrames in `RV.from_table` / `Transit.from_table` |
| `vizier` | astroquery | `RV.from_vizier` |
| `lightcurves` | lightkurve | `Transit.from_lightkurve` |
| `diagnostics` | netCDF4, arviz | R-hat and ESS in `result.diagnostics`, recomputed from the saved chains |
| `zstd` | zstandard | unpacking the runtime bundle. The base install already pulls it in, except on Python 3.14+, which unpacks with the standard library |

### Runtime bundles

Prebuilt runtimes are attached to the
[Nereus.jl releases](https://github.com/jvines/Nereus.jl/releases) with `SHA256SUMS`
alongside. `install()` picks the one for your platform and checks it against the
published checksum.

| platform | size |
|---|---|
| `macos-arm64` | 466 MiB |
| `linux-x86_64` | 690 MiB |
| `linux-aarch64` | 650 MiB |

Every process after the first `install()` starts a session in about twenty seconds.
Windows is not supported.

Upgrading the package does not refresh a runtime you already have: the bundle is a
separate download, cached outside site-packages. When a release ships against a newer
Nereus, `session()` tells you at start-up. Fetch it with:

```sh
python -c "import astronereus; astronereus.install(force=True)"
```

For a classroom, fetch the bundle once and point everyone at the local copy:

```sh
export NEREUS_BUNDLE_URL=~/nereus-runtime-1.11.9-macos-arm64.tar.zst   # path or file:// URL
```

`NEREUS_HOME` moves the runtime cache out of the default platform cache directory
(`~/Library/Caches/nereus` on macOS, `$XDG_CACHE_HOME/nereus` or `~/.cache/nereus`
on Linux).

## Sessions

Every call goes to a Julia daemon. You can leave its lifetime implicit or hold it
yourself.

```python
import astronereus

# One-shot: a process-wide daemon starts on first use, is reused, and closes at exit.
ds = astronereus.dataset("51peg")
astronereus.ping()

# Explicit: one daemon for the block, closed when it exits.
with astronereus.session() as s:
    ds = s.dataset("51peg")
    print(s.ping())
```

```
{'julia': '1.11.9', 'threads': 16, 'cpu': 'generic', 'api': 3, 'nereus': '0.8.8', 'pong': True}
```

Everything the module-level functions do is also a method on the session
(`s.fit_rv`, `s.fit_joint`, `s.read_rv`, `s.dataset`, ...), and the methods bind
themselves to that session. Prefer `with astronereus.session() as s:` for anything
beyond one call: the lifetime is explicit, and `s.detect`, `s.detrend` and
`s.tomogram` exist only there.

**Cost.** Starting a session takes about 20 s once the runtime is installed (loading
Nereus into a fresh Julia process). After that, calls are cheap, but each Julia
function is compiled on its first use in a session: the first `s.detect.rv_periodogram`
took 19 s and the first `s.detrend.gp` 40 s in a fresh session. A session that stays
open keeps that work, so keep one open across a script or notebook.

`session()` accepts `threads` (default `"auto"`), `startup_timeout` (300 s) and
`idle_timeout` (1800 s). The daemon exits by itself if its parent process dies or if
nothing talks to it for `idle_timeout` seconds.

| member | use |
|---|---|
| `s.ping()` | liveness plus the Julia version, thread count, CPU target, API version and the Nereus version of the runtime |
| `s.log_path` | the daemon's stdout/stderr log. Read it when a job dies |
| `s.raw(action, payload)` | call a daemon action by name. An escape hatch for actions newer than the client |
| `s.close()` | stop the daemon. Called for you by the `with` block |

`run_job(config)` (module level and on the session) takes a Nereus job config dict;
see [Job configs](#job-configs-run_job). The typed `fit_*` entry points described in
the later sections are the usual route.

### Interrupting a fit

Ctrl-C (or the stop button in a notebook) reaches both Python and the Julia sampler.
The fit stops, `FitInterrupted` is raised, nothing is written to `output_dir`, and
the daemon stays up and warm:

```python
try:
    s.fit_rv(ds.rv, planets=1, output_dir="out")
except astronereus.FitInterrupted:
    pass
s.ping()          # still answers; the next fit reuses the same session
```

A `timeout=` (seconds) on `run_job` stops waiting, but does not stop the Julia job.
Call `s.close()` if you need it dead.

### Errors

| exception | when |
|---|---|
| `DaemonError` | Julia raised. The message carries the Julia error and backtrace. `s.log_path` has the daemon's full output |
| `FitInterrupted` | the fit was cancelled (above). A `RuntimeError` subclass |
| `RuntimeVersionError` | at session start: the cached runtime speaks a different API version than this client |
| `BundleError` | the runtime bundle could not be fetched, verified or unpacked |

```python
try:
    s.detect.transits(t=t)               # forgot flux, flux_err
except astronereus.DaemonError as e:
    print(str(e).splitlines()[0])        # missing required argument flux. Got: t
```

`RuntimeVersionError` names the fix: when the cached runtime is older than the client,
refresh it with `astronereus.install(force=True)`; when it is newer, upgrade the
package. `NEREUS_SKIP_API_CHECK=1` bypasses the check, and the mismatch then surfaces
as a Julia error in the middle of a fit.

A runtime that is older than the client but speaks the same API is not an error. The
session starts and prints to stderr that the cached runtime is Nereus X while the
client ships against Y, with `astronereus.install(force=True)` as the remedy. It is
printed once per session. Fixes in the newer runtime are not present until you
refresh it.

## Getting data in

Every channel takes its data as `{instrument: {t, <value>, <error>}}`, with times in
MJD (days) and one instrument label per independent data source. Each instrument gets
its own offset and jitter downstream. There are three ways in: a file, a table or
arrays you already hold in Python, and the datasets Nereus ships.

### Time conventions

Nereus models in MJD. `time_offset` is subtracted from the time column, and is applied
the same way by every reader and constructor below.

| call | default `time_offset` | effect |
|---|---|---|
| `read_rv`, `RV.from_file` | `2400000.5` | BJD in, MJD out. Pass `0` for times already in MJD |
| `read_photometry`, `Transit.from_file` | `0.0` | times are returned as read. For a BJD file pass `time_offset=2400000.5` |
| `from_arrays`, `from_table`, `from_vizier` | `"auto"` | see below |

`"auto"` subtracts 2400000.5 when the median time exceeds 2,000,000 (a full Julian
Date) and leaves the column alone otherwise (an MJD). It cannot recognise
mission-relative clocks: TESS BTJD (BJD - 2457000) and Kepler BKJD (BJD - 2454833)
look like small numbers and pass through unchanged. Give the offset yourself, with the
sign that makes `t - offset` an MJD:

```python
TESS_BTJD_TO_MJD = 2400000.5 - 2457000       # t_mjd = t_btjd - (2400000.5 - 2457000)
Transit.from_arrays(btjd, flux, err, time_offset=TESS_BTJD_TO_MJD)
```

`Transit.from_lightkurve` avoids the problem by reading the epoch off the astropy
`Time` object.

### Arrays and tables

```python
import numpy as np
from astronereus import RV, Transit

rv = RV.from_arrays(t_bjd, rv_ms, err_ms, instrument="HARPS")      # BJD -> MJD automatically
```

`instrument` is one label for every row, or a per-row sequence of labels. `rename`
maps labels to the names your priors use. Rows with a non-finite time, value or error
are dropped.

From a table, with the column names guessed or given:

```python
import pandas as pd
df = pd.read_csv("rv.csv")        # columns: bjd, rv, rv_error, inst

rv = RV.from_table(df)            # guesses the columns
rv = RV.from_table(df, time="bjd", rv="rv", rv_err="rv_error",
                   instrument="inst", rename={"A": "HARPS"})
rv = RV.from_table(df, instrument="CORALIE")   # not a column name: used as the label
```

`from_table` accepts a pandas DataFrame, an astropy `Table`, a numpy structured array
or a dict of columns. A file path is rejected; use `read_rv` for files. Without
explicit names it looks (case-insensitively) for `bjd`/`jd`/`time`/`t`/`mjd` and
similar for time, `rv`/`vrad` for velocity, `rv_err`/`rv_error`/`e_rv`/`svrad` for the
error and `inst`/`instrument`/`tel`/`telescope` for the instrument (label `INST` if
absent). For photometry it looks for `time`/`bjd`/`btjd`, `flux`/`pdcsap_flux`,
`flux_err`/`pdcsap_flux_err` and `inst`/`mission`/`sector` (label `TESS` if absent).
A name it cannot find raises `KeyError` listing the table's columns.

Photometry works the same way. `normalize=True` divides flux and error by the median,
for a light curve still in electrons per second. A column called `time` in a TESS
table is usually BTJD, which `"auto"` will not convert.

```python
fl = pd.DataFrame({"time": btjd, "pdcsap_flux": f, "pdcsap_flux_err": e})
tr = Transit.from_table(fl, time_offset=2400000.5 - 2457000,
                        normalize=True, instrument="TESS_s1")
tr = Transit.from_arrays(t, flux, flux_err, instrument="TESS")
```

### VizieR and lightkurve

```python
rv = RV.from_vizier("J/AJ/169/107/table5", time="BJD", rv="RVel",
                    rv_err="e_RVel", instrument="Inst")     # needs astroquery
```

This returns three instruments (HPF, NEID, FIES). All rows and columns are fetched;
VizieR's own defaults truncate both. `constraints={"col": value}` adds column filters,
and `columns=[...]` restricts the columns.

```python
import lightkurve as lk
lcs = lk.search_lightcurve("TOI-700", mission="TESS").download_all()
tr = Transit.from_lightkurve(lcs)                  # needs lightkurve
```

Accepts a `LightCurve` or a `LightCurveCollection`. NaN cadences are dropped, flux is
divided by its median (`normalize=True`), and each sector gets its own instrument name
(`TESS_s1`, `TESS_s2`, ...) so it has its own jitter. Pass
`per_sector=False, instrument="TESS"` to pool them deliberately.

### Reading files

The file readers call Nereus's own loaders, so they run in a session (a process-wide
one is started if you do not pass `session=`).

```python
with astronereus.session() as s:
    rv  = s.read_rv("rv.csv")                          # BJD file; MJD out
    rv  = s.read_rv("hd.dat", rename={"j": "HIRES", "lick": "Lick"})
    lc  = s.read_photometry("lc.csv", instrument="TESS_s1",
                            time_offset=2400000.5, trim_window=(2459001.0, 2459002.0))
    rel = s.read_relastrom("relast.txt")
```

`read_rv(path, format=..., ...)`:

| `format` | file |
|---|---|
| `"auto"` | `.csv` is read as `"vizier"`, anything else as `"labeled"` |
| `"vizier"` | comma-delimited with a header. `t_col`, `rv_col`, `err_col`, `inst_col` name the columns (defaults `bjd`, `rv`, `rv_err`, `inst`); `inst_col=None` for one instrument |
| `"labeled"` | whitespace-delimited, string instrument labels in column 4 |
| `"orvara"` | whitespace-delimited, integer instrument ids in column 4; give `instrument_names=["HIRES", ...]` to name them |

`rename` maps raw labels to your names. Reading `51peg.csv` (which uses `rv_error`
and `instrument`) takes `format="vizier", err_col="rv_error", inst_col="instrument"`
and returns five instruments.

`read_photometry` reads a CSV with the columns `bjd_tdb, pdcsap_flux_norm,
pdcsap_flux_err_norm` (median-normalised, NaNs removed) as one instrument.
`trim_window` is in the file's own time units (before `time_offset` is applied) and
keeps only points inside it, useful to take one contiguous slab of a sector.
`read_relastrom` reads an orvara-format relative-astrometry file and returns the
`values` block that `Astrometry(relast=...)` takes (`t` in MJD; `ra_off`, `dec_off`,
`ra_err`, `dec_err` in mas; `corr`, `planet_idx`, `inst`); the (separation, PA) to
offset conversion and its error propagation happen in Julia.

Each reader has a constructor that goes straight to a channel:

```python
rv  = RV.from_file("rv.csv", session=s)
tr  = Transit.from_file("lc.csv", session=s, time_offset=2400000.5)
ast = Astrometry.from_relast_file("relast.txt", session=s)   # relative astrometry
```

`RV.from_file` and `Transit.from_file` pass `format=`, `rename=`, `time_offset=` and
`session=` to the reader.

### Shipped datasets

Nereus ships the tables its examples use. They load as ready channels, with no paths
and no parsing.

```python
astronereus.list_datasets()          # [{'name', 'target', 'ref', 'channels', 'available'}, ...]
ds = astronereus.dataset("hd159062") # or s.dataset(...)
ds.describe()                        # per-instrument table: N, span, rms, median error
ds.rv                                # RV channel, or None
ds.astrometry                        # Astrometry channel with relative astrometry, or None
ds.transit                           # Transit channel, or None (no shipped set has one yet)
```

`ds.raw` keeps the loader output. Names: `51peg`, `eps_eri`, `gaia4`, `gj876`,
`hd114762`, `hd18599`, `hd33636`, `hd33636_bean`, `hd38529` (RV only), `hd159062` and
`hd4747` (RV and relative astrometry). `list_datasets()` gives the target and
reference of each. `astronereus.datasets.rv_table(ds.rv.data)` returns the same
per-instrument summary as a string, for any `{instrument: {t, rv, rv_err}}` dict.

### Astrometric catalogue loaders

Three session methods load astrometric inputs and return plain dicts. How they feed a
fit is covered in [Astrometry](#astrometry); where the files come from is in
[Getting the catalogue data](#getting-the-catalogue-data). Everything is in MJD, mas
and mas/yr.

| call | reads | returns |
|---|---|---|
| `s.load_hgca(path="HGCA_vEDR3.fits", hip=27321)` | one row of the Brandt eDR3 Hipparcos-Gaia Catalog of Accelerations | `hip_id`, `plx`, `plx_err`, `epochs` (3 MJD: Hipparcos, Hipparcos-Gaia, Gaia), `pmra`, `pmdec` (3 each), `cov_ep` (3 flattened 2x2 RA-Dec covariances) |
| `s.load_gost(path="gost.csv")` | a Gaia GOST scan-forecast CSV | `t` (MJD, from BJD TCB), `psi` (scan angle, rad), `parallax_factor`, `along_scan_pos` |
| `s.load_iad(path="iad.txt")` | a 6-column Hipparcos-style IAD table: epoch (decimal year), parallax factor, proper-motion factor, abscissa residual (mas), error (mas), scan angle psi (rad) | `t`, `abscissa`, `abscissa_err`, `psi`, `parallax_factor`, `pm_factor`, plus the model terms Nereus precomputes |

The van Leeuwen 2007 per-source Hipparcos files have a different layout (`IORB EPOCH
PARF CPSI SPSI RES SRES`) and are not read by `load_iad`; a fit reads them for you
when given a `hip_id`.

## Pre-fit tools

Standalone operations on arrays that run before, or instead of, a fit. Each is a
method on a session: `s.detect.<op>(**payload)` and `s.detrend.<op>(**payload)`. They
take keyword arguments, and return a dict of lists (convert with `np.asarray`).
A missing required argument raises `DaemonError` naming it. Times are in days and
should be ascending. Detection: `transits`, `rv_periodogram`, `rotation`, `segments`.
Detrending: `savgol`, `notch`, `locor`, `gp`.

The examples below use a synthetic 27-day, 2-minute-cadence light curve (3.7 d transit
of depth 0.4%, plus an 8.5 d sinusoidal modulation) and a 60-point RV series:

```python
import numpy as np, astronereus

rng = np.random.default_rng(3)
t = np.arange(0, 27, 2 / 1440)
ph = ((t - 1.3 + 3.7 / 2) % 3.7) - 3.7 / 2
flux = 1 + 0.01 * np.sin(2 * np.pi * t / 8.5) - 0.004 * (np.abs(ph) < 0.06) + rng.normal(0, 4e-4, t.size)
flux_err = np.full(t.size, 4e-4)

tr = np.sort(rng.uniform(0, 400, 60))
rv = 12 * np.sin(2 * np.pi * tr / 23.4) + rng.normal(0, 1.5, 60)
rv_err = np.full(60, 1.5)

s = astronereus.session()      # or inside `with astronereus.session() as s:`
```

### Detection

**`s.detect.transits(t, flux, flux_err, ...)`** detrends the light curve (Savitzky-Golay)
and runs a box-least-squares search.

```python
r = s.detect.transits(t=t, flux=flux, flux_err=flux_err,
                      period_min=1.0, period_max=10.0, max_candidates=3,
                      M_s=0.9, R_s=0.9)
```

On this light curve the top candidate is the injected transit: `periods[0]` is 3.696 d,
at an SNR of 107.

| key | content |
|---|---|
| `periods`, `t0s`, `durations` | per candidate, best first (days). `t0s` are real epochs in the time system of `t` |
| `depths`, `snr` | transit depth (fractional) and detection significance |
| `R_p_earth`, `M_p_earth`, `K_ms` | planet radius (R_earth), mass (M_earth) and RV semi-amplitude (m/s) from a mass-radius relation and `M_s`, `R_s` |
| `flux_detrended`, `flux_err_detrended` | the detrended light curve the search ran on |
| `kept` | boolean per input cadence: survived `trim_mask`/`sigma_clip` (`t[kept]` aligns with `flux_detrended`) |
| `detrend_result` | the full detrender output |

Options: `period_min`, `period_max` (days, default 0.5 and 30), `max_candidates` (5),
`score_threshold` (minimum SNR, 7), `M_s`, `R_s` (solar units, 1 and 1), `trim_mask`
(boolean per cadence, true = drop entirely), `sigma_clip` (a number of sigma, removes
point outliers) and `detrend_kwargs` (a dict passed to the detrender, e.g.
`{"window_length": 361}`; the default window is derived from the cadence, about 0.5 d).
`find_transits` in the
[preprocessing docs](https://github.com/jvines/Nereus.jl/blob/main/docs/src/preprocessing.md)
lists the remaining options.

**`s.detect.rv_periodogram(t, rv, rv_err, ...)`** (also `rv_planets`, the same
operation) computes a generalised Lomb-Scargle periodogram and the window function on
a shared grid.

```python
r = s.detect.rv_periodogram(t=tr, rv=rv, rv_err=rv_err, period_min=2, period_max=100)
r["rv"]["peaks"][0]         # {'period': 23.41, 'frequency': 0.0427, 'power': 0.977, 'fap': 0.0}
```

Returns `rv` and `window`, each with `periods`, `frequencies`, `power`, `peaks` (up to
`max_candidates`, each with `period`, `power`, false-alarm probability `fap`),
`fap_levels` and `fap_thresholds`; plus `method`, `n_obs`, `baseline_days`.
Options: `period_min` (days, 1), `period_max` (days, the baseline), `max_candidates`
(5).

**`s.detect.rotation(t, flux, flux_err, ...)`** estimates the rotation period from the
autocorrelation function.

```python
r = s.detect.rotation(t=t, flux=flux, flux_err=flux_err, min_period=1, max_period=12)
r["P_rot"], r["P_rot_err"]           # 8.4, 1.5  (days; the injected value is 8.5)
```

Returns `P_rot`, `P_rot_err` (half-width of the peak), `lags`, `acf`, `peaks`,
`threshold`, `bin_cadence`, `n_bins` and `pacf`. Options: `min_period` (0.1),
`max_period` (a quarter of the baseline) and `n_peaks` (5). `t` must be ascending.

**`s.detect.segments(t, gap_size=0.5)`** splits a time series at gaps larger than
`gap_size` (days).

```python
r = s.detect.segments(t=np.r_[t[:5000], t[7000:]])
len(r["segments"]), r["segments"][0][:2]     # 2, [1, 2]
```

Returns `{"segments": [[i, ...], ...]}`: one list of 1-based indices per contiguous
block.

### Detrending

All four take `t, flux, flux_err` and return `flux_detrended` (flux divided by `trend`,
median near 1, except `gp`, which is additive), `trend` and `segments`. Segmentation at
gaps is automatic. `savgol` also takes `transit_mask`, a boolean per cadence marking
known in-transit points that are interpolated across; `notch` needs no mask, as it
flags transit-shaped dips itself.

| call | method | extra output | key options |
|---|---|---|---|
| `s.detrend.savgol` | Savitzky-Golay smoothing | | `window_length` (odd, in cadences; derived from the cadence to span about 0.5 d if omitted), `polyorder` (3) |
| `s.detrend.notch` | robust sliding notch filter; flags transit-shaped dips | `delta_bic`, `depth`, `dur_best`, `outlier` per cadence | `window` (days, 1.0) |
| `s.detrend.locor` | rotation-cycle detrender for fast rotators | `P_rot_used`, `phase`, `cycle_id`, `used_notch`, `outlier` | `P_rot` (days; falls back to notch if absent or longer than a third of the baseline) |
| `s.detrend.gp` | celerite GP fitted by Pathfinder | `gp_params`, `offsets` | `kernel` (required), `joint_segments` (true: one shared set of hyperparameters) |

```python
in_transit = np.abs(ph) < 0.12                   # known ephemeris, widened
d = s.detrend.savgol(t=t, flux=flux, flux_err=flux_err, window_length=361,
                     transit_mask=in_transit.tolist())
f = np.asarray(d["flux_detrended"])

d = s.detrend.notch(t=t, flux=flux, flux_err=flux_err, window=1.0)
in_transit = np.asarray(d["delta_bic"]) > 0      # 2080 of 19440 cadences here

d = s.detrend.locor(t=t, flux=flux, flux_err=flux_err, P_rot=8.5)

d = s.detrend.gp(t=t[::10], flux=flux[::10], flux_err=flux_err[::10],
                 kernel="CeleriteSHO")
d = s.detrend.gp(t=t[::10], flux=flux[::10], flux_err=flux_err[::10],
                 kernel={"kind": "CeleriteRotation"})
```

`kernel` is `"CeleriteSHO"`, `"CeleriteRotation"` or `"CeleriteRotationFM17"`, either
as a string or as `{"kind": ..., "kwargs": {...}}`. Its `gp_params` is a list of
hyperparameter vectors (one when `joint_segments=True`). GP detrending is the slow one:
thin the light curve as above for a quick look.

The cleaned `flux_detrended` goes into `Transit.from_arrays(t, d["flux_detrended"], flux_err)`.
Keep the native cadence; Nereus does not expect binned photometry. Doppler-tomography
operations (`s.tomogram.*`) are described under [Doppler tomography](#doppler-tomography).

## Fit modes

Every mode takes data per instrument and returns a `JobResult` ([Results](#results)).
Two routes exist:

| mode | call | data |
|---|---|---|
| radial velocities | `fit_rv(rv, ...)` | `{instrument: {t, rv, rv_err}}` |
| transits | `fit_transit(lc, ...)` | `{instrument: {t, flux, flux_err}}` |
| Doppler tomography | `fit_tomography(nights, ...)` | one `Night` per transit |
| RV + transit, Rossiter-McLaughlin, TTV, spectroscopic binaries | `session.run_job(cfg)` | a job config |
| astrometry, and anything joined to it | see [Astrometry](#astrometry) | |

A `run_job` config is documented in
[JOB_CONFIG.md](https://github.com/jvines/Nereus.jl/blob/main/docs/JOB_CONFIG.md);
the examples below show the blocks each mode needs. Times are MJD, RVs and their
errors are in m/s, flux is relative. Parameters are named `<name>_k<n>` per planet
(`P_k1`, `K_k1`, `Mo_k1`) and `<name>_<instrument>` per instrument (`gamma_HARPS`,
`sigma_HARPS`, `jitter_TESS`). `fit_*` priors are written
`{"type": "uniform", "lo": 3.4, "hi": 3.6}` and `run_job` priors
`{"type": "UniformPrior", "args": [3.4, 3.6]}`; see [Priors](#priors). The examples
use the default engine ([Samplers](#samplers)).

Examples that need data of their own use small synthetic sets. Save this block as
`synth.py`; later blocks import from it.

```python
import numpy as np


def transit_flux(t, P, Tc, rp, a_Rs, b, u=(0.40, 0.25), n=600):
    """Circular orbit, quadratic limb darkening, planet disc integrated on a grid."""
    ph = 2 * np.pi * (np.asarray(t) - Tc) / P
    x, y = a_Rs * np.sin(ph), b * np.cos(ph)
    k = np.arange(n) + 0.5                                  # sunflower grid over the planet disc
    rr, th = rp * np.sqrt(k / n), k * 2.399963
    px, py = x[:, None] + rr * np.cos(th), y[:, None] + rr * np.sin(th)
    r2 = px**2 + py**2
    mu = np.sqrt(np.clip(1 - r2, 0, 1))
    I = 1 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2
    lost = np.where(r2 < 1, I, 0).sum(1) * np.pi * rp**2 / n / (np.pi * (1 - u[0] / 3 - u[1] / 6))
    return 1 - np.where(np.cos(ph) > 0, lost, 0)


def make_lc(P=3.5, Tc=59000.5, rp=0.1, a_Rs=8.0, b=0.3, n_tr=3, n=150, sig=4e-4, seed=1):
    """{instrument: {t, flux, flux_err}} with n_tr transits."""
    rng = np.random.default_rng(seed)
    t = np.sort(np.concatenate([rng.uniform(Tc + i * P - 0.3, Tc + i * P + 0.3, n)
                                for i in range(n_tr)]))
    f = transit_flux(t, P, Tc, rp, a_Rs, b) + rng.normal(0, sig, t.size)
    return {"TESS": {"t": t.tolist(), "flux": f.tolist(), "flux_err": [sig] * t.size}}


def make_rv(P=3.5, Tc=59000.5, K=60.0, sig=5.0, n=30, seed=2):
    """{instrument: {t, rv, rv_err}}, two instruments, offsets 0 and 25 m/s, no jitter."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, gamma in (("HARPS", 0.0), ("HIRES", 25.0)):
        t = np.sort(rng.uniform(58900, 59100, n))
        v = -K * np.sin(2 * np.pi * (t - Tc) / P) + gamma + rng.normal(0, sig, n)
        out[name] = {"t": t.tolist(), "rv": v.tolist(), "rv_err": [sig] * n}
    return out


def rv_values(rv):
    """{instrument: {t, rv, rv_err}} -> the `values` block of a run_job `data.rv`."""
    v = {"bjd": [], "rv": [], "rv_err": [], "instrument": []}
    for inst, d in rv.items():
        v["bjd"] += d["t"]; v["rv"] += d["rv"]; v["rv_err"] += d["rv_err"]
        v["instrument"] += [inst] * len(d["t"])
    return v


def phot_blocks(lc, exposure_time=120.0):
    """{instrument: {t, flux, flux_err}} -> `data.transit_photometry` (exposure_time in s)."""
    return [{"instrument": k, "exposure_time": exposure_time,
             "values": {"bjd": d["t"], "flux": d["flux"], "flux_err": d["flux_err"]}}
            for k, d in lc.items()]


def show(r):
    """Print the fitted-parameter table of a run_job result."""
    for k, v in sorted(r.raw["fitted"]["parameters"].items()):
        print(f"{k:16s}{v['value']:13.6g}  +{v['err_hi']:.3g} -{v['err_lo']:.3g}  {v['unit']}")


def make_sb2(P=12.3, T0=58950.0, K1=30e3, K2=45e3, gamma=5e3, sig=300.0, n=40, seed=3):
    """Double-lined binary, velocities in m/s: {component: {instrument: {t, rv, rv_err}}}."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(58900, 59000, n))
    ph = 2 * np.pi * (t - T0) / P
    mk = lambda v: {"SOPHIE": {"t": t.tolist(), "rv": (v + rng.normal(0, sig, n)).tolist(),
                               "rv_err": [sig] * n}}
    return {"primary": mk(gamma + K1 * np.cos(ph)), "secondary": mk(gamma - K2 * np.cos(ph))}
```

A `run_job` result keeps its fitted table under `r.raw["fitted"]["parameters"]`
(value, `err_hi`, `err_lo`, unit); `show` prints it. The same tables are written to
`<output_dir>/tables/` as CSV, JSON, ECSV and LaTeX.

### Radial velocities

`fit_rv` fits one or more Keplerians to RVs from any number of instruments. Each
instrument gets a systemic offset `gamma_<inst>` (m/s) and a jitter `sigma_<inst>`
(m/s). The default planet block is `P`, `K`, `sesinw`, `secosw` and `Mo`. Bracket
`P_k1` whenever you know roughly where it is.

```python
import astronereus
from astronereus import fit_rv

ds = astronereus.dataset("51peg")
rv = {k: ds.rv.data[k] for k in ("HIRES_POST", "APF")}    # {instrument: {t, rv, rv_err}}
bracket = {"P_k1": {"type": "uniform", "lo": 4.1, "hi": 4.3}}

r = fit_rv(rv, priors=bracket, M_s=1.06, output_dir="out/51peg")
print(r.params)
print(r.derived)

# jitter of one instrument held at zero, one systemic offset shared by both
r0 = fit_rv(rv, planets=1, output_dir="out/51peg_shared",
            priors={**bracket, "sigma_APF": {"type": "fixed", "value": 0.0}},
            sharing={"gamma": [["HIRES_POST", "APF"]]})
print(list(r0.params))

# two Keplerians: parameters come back as P_k1, K_k1, ... and P_k2, K_k2, ...
r2 = fit_rv(rv, planets=2, priors=bracket)
print(list(r2.params))
```

Keywords beyond the data:

| keyword | effect |
|---|---|
| `planets=N` | N Keplerians, `P_k1` ... `P_kN`. A range of counts with `transdim=True` makes the number of planets a parameter ([Trans-dimensional fits](#trans-dimensional-fits)) |
| `priors` | per-parameter priors, keyed by name (`P_k1`, `gamma_APF`, `sigma_APF`, ...) |
| `sharing` | `{"gamma": [[inst_a, inst_b]]}` ties offsets, `{"sigma": ...}` ties jitters. The shared parameter is named `gamma_inst_a+inst_b` |
| `M_s`, `R_s` | stellar mass and radius (M_sun, R_sun); `M_s` adds `a_au` and the minimum masses to the derived table. `T_eff`, `J_mag`, `K_mag` and `sigma_M_s`, `sigma_R_s`, `sigma_T_eff` feed the derived table |
| `external_priors` | priors on derived quantities, e.g. eccentricity |
| `plots`, `science`, `output_dir` | figures, the post-fit block and where files go ([Results](#results)) |

Jitter can be zero. Its default prior is a normal on (0, 50) m/s truncated at 0, so a
posterior piled against zero is a valid answer. To remove the parameter instead, fix
it: `"sigma_APF": {"type": "fixed", "value": 0.0}`.

The data argument is the instrument map above, or an `RV` channel built with
`RV.from_arrays`, `RV.from_table` or `RV.from_file` ([Getting data in](#getting-data-in)).

#### Trends and noise models

A linear trend and the noise models are model-block options of a `run_job` config.
`"trend_order": 1` adds `dvdt` (m/s/day). `noise_models` is a list of
`{kind, channel, instruments, kwargs}`; the example uses `ARModel`, and the full list
of kinds (celerite GPs, activity decorrelation, ...) is in
[noise_models.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/noise_models.md).

```python
import astronereus
from synth import make_rv, rv_values, show

rv = make_rv()
cfg = {
    "output_dir": "out/rv_noise",
    "data": {"rv": {"values": rv_values(rv)}},
    "model": {"max_kplanet": 1, "planet_modes": ["RV_ONLY"],
              "trend_order": 1},                       # adds dvdt (m/s/day)
    "priors": {"P_k1": {"type": "UniformPrior", "args": [3.4, 3.6]}},
    "noise_models": [{"kind": "ARModel", "channel": "rv", "kwargs": {"order": 1}}],
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)
```

### Transits

`fit_transit` fits a limb-darkened transit per planet. Limb darkening is quadratic,
sampled as Kipping's `q1_<inst>`, `q2_<inst>` for each instrument; fix them with
`fixed` priors to hold the limb darkening. Each light curve also has a flux offset
`offset_<inst>` and a jitter `jitter_<inst>` (relative flux). The dilution
`dilution_<inst>` is fixed at 0 until you give it a prior. `phot_trend_order=1` adds a
per-instrument baseline slope, `phot_c1_<inst>`. `planets`, `M_s`, `R_s` and
`external_priors` work as for `fit_rv`.

```python
import astronereus
from astronereus import fit_transit
from synth import make_lc

lc = make_lc(n_tr=3)                       # {"TESS": {t, flux, flux_err}}
priors = {
    "P_k1": {"type": "uniform", "lo": 3.4, "hi": 3.6},
    "rr_k1": {"type": "uniform", "lo": 0.02, "hi": 0.3},
    "sesinw_k1": {"type": "fixed", "value": 0.0},          # circular
    "secosw_k1": {"type": "fixed", "value": 0.0},
    "dilution_TESS": {"type": "uniform", "lo": 0.0, "hi": 0.3},   # third light, fixed at 0 by default
}
r = fit_transit(lc, priors=priors, phot_trend_order=1, M_s=1.0, R_s=1.0,
                output_dir="out/transit")
print(r.params)
print(r.derived)
```

A single transit does not fix `P_k1`; bracket it, or fit several transits.
Finite-exposure supersampling is set in a job config, from the `exposure_time` field
(seconds) of a `transit_photometry` block (default 120 s; see `phot_blocks` in
`synth.py`). Gravity-darkened transits are also a job-config option; see
[obliquity.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/obliquity.md).

The orbital phase is the mean anomaly `Mo_k1` unless you pick another anchor. For a
transiting planet pass `time_anchor="phic"`. The sampler then takes the transit phase
`phic_k1 = 2 pi (Tc - t_ref_c) / P` (rad, uniform on `[-pi, pi]` by default), which the
transit pins whatever e and omega are, and reports the mid-transit time as the derived
`Tc_k1` (BJD), computed from each draw's own period. `t_ref_c` is the median
photometric time, so `Tc_k1` is the transit nearest the middle of the data. A
`Tc_k1` prior over the whole baseline (`time_anchor="Tc"`) instead holds one copy of
the orbit per period, which the sampler does not mix between; with the period longer
than the baseline it cuts off part of the phase. In a job config the same choice is
`model.parametrization.time = "phic"`.

### RV and transit together

In this release, run this mode through a job config.

Both datasets share one orbit: `P_k1`, `Mo_k1` and the eccentricity vector, with
`K_k1` for the RV amplitude and `rr_k1`, `b_k1` for the transit. Each instrument keeps
its own offsets and jitters. Use the `RVPM` planet mode; `trend_order`,
`phot_trend_order` and `noise_models` go in the `model` block. Adding astrometry is
covered in [Astrometry](#astrometry).

```python
import astronereus
from synth import make_lc, make_rv, rv_values, phot_blocks, show

cfg = {
    "output_dir": "out/rv_transit",
    "star": {"M_s": 1.0, "R_s": 1.0},
    "data": {"rv": {"values": rv_values(make_rv())},
             "transit_photometry": phot_blocks(make_lc(n_tr=3))},
    "model": {"max_kplanet": 1, "planet_modes": ["RVPM"]},
    "priors": {"P_k1": {"type": "UniformPrior", "args": [3.4, 3.6]},
               "rr_k1": {"type": "UniformPrior", "args": [0.02, 0.3]}},
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)
```

### Rossiter-McLaughlin

In this release, run this mode through a job config.

RM needs RV through the transit and photometry of the same transit: the kernel takes
the geometry from the light curve. The kernel is chosen by planet mode.

| mode | kernel | parameters it adds |
|---|---|---|
| `RVPM_RM_A` | ARoME (Boue et al. 2013), what a Gaussian fit to a CCF returns | `v_sin_i_star` (m/s), `lambda_k1`, `beta_p`, `sigma_ccf` (m/s) |
| `RVPM_RM_R` | Reloaded RM (Cegla et al. 2016) | `v_sin_i_star`, `lambda_k1` |
| `RVPM_RM` | flux-weighted mean, legacy; biased at high v sin i | `v_sin_i_star`, `lambda_k1` |

The projected obliquity `lambda_k1` has a prior in radians (default uniform on
(-pi, pi)) and is reported in degrees. `v_sin_i_star` defaults to a log-uniform prior
on (500, 100000) m/s; give a tighter one when you have a measurement. RM nights fitted
together with tomographic maps, with per-night offsets, use the `model.obliquity`
block of
[obliquity_jobs.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/obliquity_jobs.md).

```python
import numpy as np
import astronereus
from synth import make_lc, phot_blocks, rv_values, transit_flux, show

P, Tc, rp, a_Rs, b, vsini = 3.5, 59000.5, 0.1, 8.0, 0.3, 10_000.0      # vsini in m/s
lam = np.deg2rad(30.0)                                                  # true obliquity
rng = np.random.default_rng(4)
t = np.sort(np.concatenate([rng.uniform(Tc - 0.2, Tc + 0.2, 40),        # in transit
                            rng.uniform(Tc - 1.2, Tc - 0.4, 8),         # baseline
                            rng.uniform(Tc + 0.4, Tc + 1.2, 8)]))
ph = 2 * np.pi * (t - Tc) / P
x, y = a_Rs * np.sin(ph), b * np.cos(ph)
dF = 1 - transit_flux(t, P, Tc, rp, a_Rs, b)
rv = -50 * np.sin(ph) - dF * vsini * (x * np.cos(lam) - y * np.sin(lam)) + rng.normal(0, 3, t.size)
rv = {"HARPSN": {"t": t.tolist(), "rv": rv.tolist(), "rv_err": [3.0] * t.size}}

cfg = {
    "output_dir": "out/rm",
    "star": {"M_s": 1.0, "R_s": 1.0},
    "data": {"rv": {"values": rv_values(rv)},
             "transit_photometry": phot_blocks(make_lc(n_tr=1, n=300))},
    "model": {"max_kplanet": 1, "planet_modes": ["RVPM_RM_A"]},     # or RVPM_RM_R, RVPM_RM
    "priors": {"P_k1": {"type": "UniformPrior", "args": [3.4, 3.6]},
               "v_sin_i_star": {"type": "UniformPrior", "args": [5_000, 20_000]}},
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)                                   # lambda_k1 is reported in degrees
```

### Doppler tomography

`fit_tomography` recovers the sky-projected obliquity from the planet's shadow in the
line profiles. It is a matched filter over lambda, with significance from a null built
by scrambling the in-transit frames, so it takes no engine or priors and returns no
chain.

Each `Night` holds one transit: `profiles` (n_time x n_velocity), `vgrid` (km/s),
`times`, that night's own measured mid-transit `Tc`, and `bervs` (km/s, barycentric
velocity per frame). Three things to get right:

- `Tc` is per night. Folding several transits on one ephemeris smears the shadow.
- Pass `bervs` when nights are weeks apart; the shift is a large fraction of v sin i.
- All nights must share a sign convention. A DRS CCF is a dip, a mask CCF built from
  line depths peaks; do not mix them.

The geometry is `P` (days), `a_Rs`, `inc` (radians), `vsini` (km/s), `T14` (days, sets
which frames are in transit) and `vsys` (km/s). `n_null` sets the size of the null
(default 300) and `n_lambda` the lambda grid (default 721). The result holds
`lambda_deg`, `lambda_rad`, `score`, `p_value`, `n_nights`, `filter_pulsations` (on by
default) and the full scan, `lambda_scan_deg` and `lambda_scan_score`, so you can plot
the landscape instead of trusting one number.

The steps are also available one at a time as `session.tomogram.*`. They take keyword
arguments and return dictionaries; `lambda` is a Python keyword, so pass it as
`**{"lambda": value}`.

| tool | returns |
|---|---|
| `ccf_profile(lambda_obs, flux, mask_lambda, mask_weight, vgrid)` | `profile` on `vgrid` from a spectrum and a mask |
| `shadow_track(t, Tc, P, a_Rs, inc, vsini, lambda)` | `v_shadow` (km/s) and `in_transit` per frame |
| `residuals(profiles, vgrid, in_transit, vsys, bervs)` | the out-of-transit-subtracted map, `residuals` on `grid` |
| `matched_filter(residuals, grid, t, Tc, P, a_Rs, inc, vsini)` | best `lambda_deg`, `score` and the scan |
| `null_distribution(residuals, grid, t, Tc, P, a_Rs, inc, vsini, n, seed)` | `null`, `observed`, `null_p99`, `p_value` |
| `injection_test(profiles, vgrid, in_transit, rr, lambda_true, t, Tc, ...)` | recovered `lambda_deg` and `error_deg` for a known injected shadow |

```python
import numpy as np
import astronereus
from astronereus import Night

orbit = dict(P=2.83, a_Rs=5.6, inc=1.48, vsini=95.0)       # days, -, rad, km/s
vgrid = np.linspace(-130, 130, 131)                        # km/s


def simulate_night(s, Tc, lam_deg, berv, seed):
    """Mask-CCF-style profiles (they peak) with the planet's shadow subtracted."""
    rng = np.random.default_rng(seed)
    t = Tc + np.linspace(-0.12, 0.12, 40)
    x = (vgrid - berv) / orbit["vsini"]
    prof = np.tile(0.5 * np.sqrt(np.clip(1 - x**2, 0, None)), (t.size, 1))
    track = s.tomogram.shadow_track(t=t.tolist(), Tc=Tc, **orbit, **{"lambda": np.deg2rad(lam_deg)})
    for i, v in enumerate(track["v_shadow"]):
        if v is not None and np.isfinite(v):
            prof[i] -= 0.012 * np.exp(-0.5 * ((vgrid - berv - v) / 6.0) ** 2)
    prof += rng.normal(0, 0.004, prof.shape)
    return Night(profiles=prof, vgrid=vgrid, times=t, Tc=Tc, bervs=[berv] * t.size)


with astronereus.session() as s:
    nights = [simulate_night(s, 59000.3, -47.0, 0.0, 1),
              simulate_night(s, 59028.3, -47.0, 5.0, 2)]
    r = s.fit_tomography(nights, T14=0.16, **orbit, n_null=300, output_dir="out/tomo")
    print(r.raw["lambda_deg"], r.raw["score"], r.raw["p_value"])

    # the same pipeline, one step at a time, on the first night
    n0 = nights[0]
    geom = dict(t=n0.times.tolist(), Tc=n0.Tc, **orbit)
    sh = s.tomogram.shadow_track(**geom, **{"lambda": np.deg2rad(-47.0)})
    res = s.tomogram.residuals(profiles=n0.profiles.tolist(), vgrid=vgrid.tolist(),
                               in_transit=sh["in_transit"], bervs=n0.bervs)
    mf = s.tomogram.matched_filter(residuals=res["residuals"], grid=res["grid"], **geom)
    nd = s.tomogram.null_distribution(residuals=res["residuals"], grid=res["grid"], **geom, n=300, seed=1)
    inj = s.tomogram.injection_test(profiles=n0.profiles.tolist(), vgrid=vgrid.tolist(),
                                    in_transit=sh["in_transit"], rr=0.11,
                                    **{"lambda_true": np.deg2rad(-47.0)}, **geom)
    print(mf["lambda_deg"], mf["score"])
    print(nd["observed"], nd["null_p99"], nd["p_value"])
    print(inj["lambda_deg"], inj["error_deg"])

    lam_obs = np.linspace(5000, 5010, 2000)                # Angstrom
    flux = 1 - 0.3 * np.exp(-0.5 * ((lam_obs - 5005) / 0.05) ** 2)
    ccf = s.tomogram.ccf_profile(lambda_obs=lam_obs.tolist(), flux=flux.tolist(),
                                 mask_lambda=[5005.0], mask_weight=[1.0],
                                 vgrid=np.linspace(-30, 30, 61).tolist())
    print(len(ccf["profile"]))
```

### Transit timing variations

In this release, run this mode through a job config.

Two models, chosen by planet mode. `PM_TTV` (`RVPM_TTV` with RV) frees one timing
offset per transit, `ttv_k<k>_t<i>` (days); `ttv_n_transits` gives the count per
planet slot, keyed by slot number as a string. The default prior on each offset is a
normal of width 1 day, so set a tighter one. `RVPM_TTV_NB` predicts the timings from
the planets' masses and orbits: it needs two or more planets, and RV to give the
masses. `ttv_backend` selects the predictor: `"ttvfaster"` (first order, the default)
or `"nbody"` (full integration, several times slower; give `rr`, `b`, `Mo` and the
eccentricity vector bounded priors).

```python
import numpy as np
import astronereus
from synth import transit_flux, phot_blocks, rv_values, show

rng = np.random.default_rng(5)
P, Tc, n_tr = 3.5, 59000.5, 8
oc = 0.01 * np.sin(1.3 * np.arange(n_tr))                    # injected O-C (days)
t = np.concatenate([np.sort(rng.uniform(Tc + i * P - 0.3, Tc + i * P + 0.3, 60))
                    for i in range(n_tr)])
shift = np.repeat(oc, 60)
f = transit_flux(t - shift, P, Tc, 0.1, 8.0, 0.3) + rng.normal(0, 4e-4, t.size)
lc = {"TESS": {"t": t.tolist(), "flux": f.tolist(), "flux_err": [4e-4] * t.size}}

cfg = {
    "output_dir": "out/ttv",
    "star": {"M_s": 1.0, "R_s": 1.0},
    "data": {"transit_photometry": phot_blocks(lc)},
    "model": {"max_kplanet": 1, "planet_modes": ["PM_TTV"],
              "ttv_n_transits": {"1": n_tr}},                 # one free offset per transit
    "priors": {"P_k1": {"type": "UniformPrior", "args": [3.45, 3.55]},
               **{f"ttv_k1_t{i + 1}": {"type": "NormalPrior", "args": [0.0, 0.02]}
                  for i in range(n_tr)}},
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)                                  # ttv_k1_t1 ... ttv_k1_t8, in days

# Two planets with N-body-predicted timing: RV constrains the masses, so use RVPM_TTV_NB.
Pb, Tb, Pc, Tc2 = 3.5, 59000.5, 7.05, 59001.0
t = np.sort(np.concatenate([rng.uniform(Tb + i * Pb - 0.3, Tb + i * Pb + 0.3, 50) for i in range(5)]
                           + [rng.uniform(Tc2 + i * Pc - 0.35, Tc2 + i * Pc + 0.35, 50) for i in range(3)]))
f = (transit_flux(t, Pb, Tb, 0.1, 8.0, 0.3) + transit_flux(t, Pc, Tc2, 0.08, 12.0, 0.2) - 1
     + rng.normal(0, 4e-4, t.size))
lc2 = {"TESS": {"t": t.tolist(), "flux": f.tolist(), "flux_err": [4e-4] * t.size}}
rv = {}
for name, g in (("HARPS", 0.0), ("HIRES", 10.0)):
    tr = np.sort(rng.uniform(58990, 59020, 25))
    v = (-60 * np.sin(2 * np.pi * (tr - Tb) / Pb) - 30 * np.sin(2 * np.pi * (tr - Tc2) / Pc)
         + g + rng.normal(0, 3, 25))
    rv[name] = {"t": tr.tolist(), "rv": v.tolist(), "rv_err": [3.0] * 25}

cfg = {
    "output_dir": "out/ttv_nb",
    "star": {"M_s": 1.0, "R_s": 1.0},
    "data": {"rv": {"values": rv_values(rv)}, "transit_photometry": phot_blocks(lc2)},
    "model": {"max_kplanet": 2, "planet_modes": ["RVPM_TTV_NB", "RVPM_TTV_NB"],
              "ttv_backend": "ttvfaster",                     # or "nbody"
              "ttv_n_transits": {"1": 5, "2": 3}},
    "priors": {"P_k1": {"type": "UniformPrior", "args": [3.45, 3.55]},
               "P_k2": {"type": "UniformPrior", "args": [7.0, 7.1]}},
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)
```

### Spectroscopic binaries

In this release, run this mode through a job config.

A double-lined binary is one orbit with two amplitudes, `K_A_k1` and `K_B_k1` (m/s).
Tag every RV row with a `component` (`"primary"` or `"secondary"`); the systemic
velocity `gamma_<inst>` is shared by both. The derived table carries
`mass_ratio_q_k1`, `M_A_sini3_k1` and `M_B_sini3_k1` (M_sun).

```python
import astronereus
from synth import make_sb2, rv_values, show

sb = make_sb2()                                       # amplitudes ~30 and ~45 km/s, in m/s
vals = {k: [] for k in ("bjd", "rv", "rv_err", "instrument", "component")}
for comp, data in sb.items():
    v = rv_values(data)
    for k in v:
        vals[k] += v[k]
    vals["component"] += [comp] * len(v["bjd"])       # "primary" / "secondary"

cfg = {
    "output_dir": "out/sb2",
    "data": {"rv": {"values": vals}},
    "model": {"max_kplanet": 1, "planet_modes": ["BINARY_RV"]},
    "priors": {"P_k1": {"type": "UniformPrior", "args": [12.0, 12.6]}},
    "sampler": {"name": "pt_emcee"},
}
with astronereus.session() as s:
    r = s.run_job(cfg)
show(r)
d = r.raw["derived"]["parameters"]
print(d["mass_ratio_q_k1"]["value"], d["M_A_sini3_k1"]["value"], d["M_B_sini3_k1"]["value"])
```

## Astrometry

`Astrometry` is the channel for everything that places the star on the sky: Hipparcos
or Gaia epoch abscissae (IAD), the Hipparcos-Gaia Catalogue of Accelerations (HGCA),
the Gaia scan geometry (GOST), and relative astrometry from imaging. It fits alone
with `fit_astrometry`, or together with radial velocities (and transits) with
`fit_joint`. Adding astrometry changes the planet block: the default parameters become
`a_k1` (AU) and `M_sec_k1` (M_sun) plus `inc_k1`, `Omega_k1`, so the fit returns a true
mass and orientation instead of `m sin i`.

```python
Astrometry(iad=None, hgca=None, gost=None, relast=None)
```

Any subset may be given. Two quantities are priors, not arguments, and go in the fit's
`priors` dict ([Priors](#priors)):

| prior | what | note |
|---|---|---|
| `plx` | parallax (mas) | Give an informative one. The abscissae constrain a0 ∝ M_sec · plx, so mass and parallax are degenerate otherwise. With an `hgca` supplied, the default is a normal prior built from the catalogue parallax |
| `M_pri` | primary mass (M_sun) | Fixed or normal. The mass of the companion scales with it. Without a star mass the default is log-uniform on 0.05-5 |

For astrometric fits two model options are worth setting: `parametrization="a_driven"`
(sample the semi-major axis and companion mass, as orvara does) and `time_anchor="Mo"`
(mean anomaly at the reference epoch, which behaves better for long periods). Both are
keyword arguments of `fit_astrometry` and `fit_joint`; the parametrizations are
described in
[parametrizations.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/parametrizations.md).

### RV and HGCA

The HGCA gives three calibrated proper motions per star (Hipparcos, Hipparcos-Gaia
mean, Gaia). The Hipparcos-Gaia anomaly pins the absolute mass of a long-period
companion that the RVs only bound from below. This is the first thing to try for a
star with a long RV trend or a known companion.

HD 159062 (HIP 85653) has Keck/HIRES velocities and NIRC2 imaging of its white-dwarf
companion shipped with the package. The later snippets in this section continue in
this session and reuse `ds`, `priors` and `engine`:

```python
import astronereus
from astronereus import Astrometry, engines

with astronereus.session() as s:
    ds = s.dataset("hd159062")          # ds.rv (HIRES), ds.astrometry (NIRC2 relative astrometry)
    ds.describe()

    priors = {"plx":   {"type": "normal", "mu": 46.19, "sigma": 0.015},
              "M_pri": {"type": "normal", "mu": 0.79,  "sigma": 0.05}}
    engine = engines.PTEmcee(n_temps=8, n_walkers=60, n_steps=3500, n_burnin=2000)

    res = s.fit_joint(
        ds.rv,
        Astrometry(hgca={"hip_id": 85653}),      # fetches and caches the HGCA row
        planets=1,
        priors=priors,
        parametrization="a_driven",
        time_anchor="Mo",
        engine=engine,
        plots=["hgca_pm_residuals", "rv_astrom_phasefold", "corner"],
        output_dir="out/hd159062_rv_hgca",
    )

    print(res.params)      # a_k1 (AU), M_sec_k1 (M_sun), inc_k1 (rad), Omega_k1 (rad), plx, M_pri, ...
    print(res.derived)     # ecc_k1, omega_deg_k1, a_au_k1
    print(res.figures)
```

What the result adds over an RV-only fit:

- `M_sec_k1` (M_sun) is the companion mass itself, `inc_k1` and `Omega_k1` (rad) are
  the inclination and node. `res.params` also carries `plx` and `M_pri` as fitted
  parameters, so their posteriors reflect what the data say against your priors.
- `hgca_pm_residuals_K1` (under `models/` in the figure map) plots the three
  proper-motion residuals, in mas/yr, against the model: Hipparcos, Hipparcos-Gaia and
  Gaia, each for RA and Dec, with the marginalised chi-squared in the panel.
- `rv_astrom_phasefold_K1` folds the RVs and the astrometric signal on the same orbit.

The RVs plus HGCA alone leave a long-period companion's orientation loosely
constrained, and with a modest sampler budget the inclination and node can come out
multimodal. Adding imaging or Hipparcos IAD (below) breaks that.

#### The `hgca` forms

| form | use |
|---|---|
| `{"hip_id": 85653}` | Fetch the Brandt EDR3 catalogue (`HGCA_vEDR3.fits`) into the data cache on first use and read the row |
| `{"fits": "HGCA_vEDR3.fits", "hip_id": 85653}` | Read the row from a local copy of the catalogue. No download |
| `s.load_hgca(path="HGCA_vEDR3.fits", hip=85653)` | Same read, done in the session; returns a dict you can inspect and pass straight to `hgca=` |
| `{"values": {...}}` | Inline three-epoch row, for a catalogue you reduced yourself |

The inline block needs exactly three epochs (Hipparcos, Hipparcos-Gaia, Gaia):

```python
hgca = {"values": {
    "t":    [48329.1, 52872.5, 57415.8],          # MJD
    "pmra": [174.32, 172.50, 169.81],             # mas/yr
    "pmdec": [75.60, 75.78, 77.13],               # mas/yr
    "cov_ep": [[[0.444, 0.109], [0.109, 0.375]],  # one 2x2 covariance per epoch, (mas/yr)^2
               [[3.7e-4, 4.2e-5], [4.2e-5, 3.8e-4]],
               [[6.9e-4, 1.7e-4], [1.7e-4, 8.3e-4]]],
    "plx": 46.19, "plx_err": 0.015, "hip_id": 85653}}
```

`s.load_hgca(...)` returns the same quantities under the key `epochs` instead of `t`;
both are accepted.

### Adding GOST

GOST is the predicted Gaia scan plan (epochs, scan angles, parallax factors). With an
`hgca` it lets the likelihood window-average the Gaia-epoch reflex motion over the real
scans. GOST is not a measurement: alone, or paired only with `iad`, it does nothing
(the constructor warns).

```python
ast = Astrometry(hgca={"hip_id": 85653},
                 gost={"ra_deg": 262.6, "dec_deg": 53.4})     # fetches the scan plan from ESA's GOST service

res = s.fit_joint(ds.rv, ast, planets=1, priors=priors,
                  parametrization="a_driven", time_anchor="Mo",
                  plots=["pm_anomaly", "hgca_pm_residuals"],
                  output_dir="out/hd159062_rv_hgca_gost")
```

`pm_anomaly_K1` shows the window-averaged Gaia-epoch reflex. `ra_deg` and `dec_deg` are
the target position (deg); `from` and `to` (ISO strings, default 2014-07-26 to
2025-01-15) bound the prediction window and must lie inside GOST's queryable interval.

| form | use |
|---|---|
| `{"ra_deg": ..., "dec_deg": ..., "from": ..., "to": ...}` | Query GOST; the CSV is cached per query |
| `s.load_gost(path="gost.csv")` | Read a GOST CSV you downloaded yourself; returns `t`, `psi`, `parallax_factor`, `along_scan_pos` |
| `{"values": {"t": ..., "psi": ..., "parallax_factor": ...}}` | Inline arrays (`along_scan_pos` optional) |

### Hipparcos and Gaia epoch astrometry (`iad`)

IAD is the along-scan abscissae. It constrains the orbit through the full time series
rather than through three proper motions, and it can be combined with the HGCA and RVs.

```python
ds = s.dataset("hd33636")                                  # six RV instruments
ast = Astrometry(iad={"catalogue": "hipparcos", "hip": 24205},
                 hgca={"hip_id": 24205})

res = s.fit_joint(ds.rv, ast, planets=1,
                  priors={"plx":   {"type": "normal", "mu": 34.9, "sigma": 0.5},
                          "M_pri": {"type": "normal", "mu": 1.0,  "sigma": 0.05}},
                  parametrization="a_driven",
                  plots=["iad_residuals", "hgca_pm_residuals"],
                  output_dir="out/hd33636")
```

| form | source |
|---|---|
| `{"catalogue": "hipparcos", "hip": 24205}` | van Leeuwen (2007) Hipparcos IAD, fetched and cached |
| `{"catalogue": "gaia_dr4", "source_id": 1457486023639239296}` | Gaia DR4 epoch astrometry, read from the pre-release epoch-astrometry table (fetched and cached on first use). Add `"path": "..."` to read a local copy |
| a list of the above | Several instruments, merged (Hipparcos + Gaia). Several `Astrometry` channels in one `fit_joint` accumulate the same way |

Gaia DR4 alone, no RVs (Gaia-4 b):

```python
res = s.fit_astrometry(
    iad={"catalogue": "gaia_dr4", "source_id": 1457486023639239296},
    planets=1,
    priors={"plx":   {"type": "normal", "mu": 13.628, "sigma": 0.021},
            "M_pri": {"type": "fixed",  "value": 0.644}},
    parametrization="a_driven",
    engine=engines.PTEmcee(n_temps=16, n_walkers=100, n_steps=3000,
                           n_burnin=1500, init_strategy="prior"),
    plots=["epoch_astrometry_orbit"],
    output_dir="out/gaia4",
)
```

### Relative astrometry (imaging)

Separations and position angles of a resolved companion. `read_relastrom` reads an
orvara-format file and does the (separation, PA) to (dRA, dDec) conversion with full
error propagation. The shipped HD 159062 NIRC2 epochs come with the dataset
(`ds.astrometry`); for your own file:

```python
rel = s.read_relastrom("hd159062_relast.txt")        # dict: t (MJD), ra_off, dec_off (mas), ra_err, dec_err, corr, planet_idx
ast = Astrometry(relast={"values": rel})
# or in one step, optionally with HGCA/IAD/GOST alongside:
ast = Astrometry.from_relast_file("hd159062_relast.txt", hgca={"hip_id": 85653})
```

`planet_idx` assigns each epoch to a planet when more than one companion is imaged.
Pass `as_names=[...]` to name several imagers.

RV + HGCA + relative astrometry, the full HD 159062 recipe (`ds` is the HD 159062
dataset, as above):

```python
ds = s.dataset("hd159062")
ast = Astrometry(hgca={"hip_id": 85653}, relast=ds.astrometry.relast)

res = s.fit_joint(
    ds.rv, ast, planets=1, priors=priors,
    parametrization="a_driven", time_anchor="Mo", engine=engine,
    plots=["hgca_pm_residuals", "orbit_skyplane", "rv_astrom_phasefold"],
    output_dir="out/hd159062_full",
)
```

`orbit_skyplane` draws posterior orbits on the sky with the imaged positions. Relative
astrometry alone is `s.fit_astrometry(relast={"values": rel}, ...)`; RV + relative
astrometry is `fit_joint(ds.rv, ds.astrometry, ...)`. A mirror ambiguity i vs 180 - i
exists for relative astrometry alone; bound `inc_k1` below 90 deg with a prior if you
want to remove it.

For short arcs a Markov chain may not mix. `engines.OFTI` (orbits for the impatient) is
a rejection sampler for relative astrometry alone with fixed masses: no chain
diagnostics and no evidence.

```python
res = s.fit_astrometry(
    relast={"values": rel}, planets=1, parametrization="a_driven",
    priors={"plx": {"type": "normal", "mu": 46.19, "sigma": 0.015},
            "M_pri": {"type": "fixed", "value": 0.79},
            "M_sec_k1": {"type": "fixed", "value": 0.608}},
    engine=engines.OFTI(n_attempts=200000, seed=1),
    output_dir="out/ofti",
)
```

With the companion mass left free, the default priors let OFTI reject every draw ("0
samples accepted"); pin or tighten the mass as above. Other engines are listed under
[Samplers](#samplers), the prior vocabulary under [Priors](#priors).

### Getting the catalogue data

| data | how it arrives |
|---|---|
| HGCA | `{"hip_id": ...}` downloads Brandt's `HGCA_vEDR3.fits` (about 20 MB) into the data cache once and reuses it. To work offline, download it yourself and pass `{"fits": path, "hip_id": ...}` or `s.load_hgca(path=..., hip=...)` |
| Hipparcos IAD | Downloaded automatically (the van Leeuwen 2007 bundle, about 350 MB, then the star's file is extracted from it). The extraction calls the system `unzip`, which must be installed. `s.load_iad` reads only the 6-column format described under [Getting data in](#astrometric-catalogue-loaders) |
| Gaia DR4 epoch astrometry | Downloaded automatically on first use from the pre-release table |
| GOST | Queried automatically from `ra_deg`/`dec_deg`. To use a file, export the scan forecast for your target from the [Gaia GOST](https://gaia.esac.esa.int/gost/) tool as CSV and read it with `s.load_gost` |
| Relative astrometry | Yours. orvara-format file with `read_relastrom`, or inline arrays |
| RVs | Yours; see [Getting data in](#getting-data-in) |

The cache location is `~/.nereus` unless the `NEREUS_DATA_CACHE` environment variable
points elsewhere.

For depth on each block:
[data.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/data.md) and
[JOB_CONFIG.md](https://github.com/jvines/Nereus.jl/blob/main/docs/JOB_CONFIG.md).

## Priors

Every `fit_*` takes `priors=`, a dict from parameter name to a prior spec. Any
parameter you leave out gets a data-driven default, so a fit with no `priors` runs;
you supply priors to bracket a known period, bound a mass, or fix a nuisance
parameter.

The examples in this and the following three sections share one synthetic data set:
two instruments, a 12 d and a 47 d planet.

```python
import numpy as np
import astronereus
from astronereus import RV, engines

rng = np.random.default_rng(1)

def instrument(n, t0, t1, err):
    t = np.sort(rng.uniform(t0, t1, n))
    rv = (30 * np.cos(2 * np.pi * t / 12.0 + 0.5)
          + 12 * np.cos(2 * np.pi * t / 47.0 + 2.0)
          + rng.normal(0, err, n))
    return {"t": t.tolist(), "rv": rv.tolist(), "rv_err": [err] * n}

rv = RV(data={"HARPS": instrument(40, 0, 400, 1.5),
              "HIRES": instrument(30, 100, 400, 3.0)})
```

Loading your own data is covered in [Getting data in](#getting-data-in).

### Prior specs

A spec is a dict with a `"type"` key, or a bare number.

| spec | fields | meaning |
|---|---|---|
| `{"type": "uniform", "lo": a, "hi": b}` | `lo`, `hi` | flat on `[lo, hi]` |
| `{"type": "loguniform", "lo": a, "hi": b}` | `lo > 0`, `hi` | flat in `log x` (density proportional to `1/x`); periods, amplitudes |
| `{"type": "normal", "mu": m, "sigma": s}` | `mu`, `sigma` | unbounded normal |
| `{"type": "normal", "mu": m, "sigma": s, "lo": a, "hi": b}` | adds `lo` and/or `hi` | normal truncated to `[lo, hi]`; give one bound and the other is infinite |
| `{"type": "fixed", "value": v}` or `v` | `value` | parameter held constant, not sampled |
| `{"type": "sine"}` | none | `p(i) = sin(i)/2` on `[0, pi]`, the isotropic-orbit prior for an inclination slot |

Any other `"type"` raises `unknown prior type ...; use normal|uniform|loguniform|fixed|sine`
before sampling starts. The Beta and modified-Jeffreys priors are reachable through a
[job config](#job-configs-run_job), where the full list is `UniformPrior`,
`LogUniformPrior`, `ModJeffreysPrior`, `NormalPrior`, `FixedPrior`, `SinePrior` and
`BetaPrior`.

```python
priors = {
    "P_k1": {"type": "loguniform", "lo": 8, "hi": 16},     # days
    "K_k1": {"type": "uniform", "lo": 0, "hi": 100},       # m/s
    "sesinw_k1": 0.0,                                      # fixed: circular
    "secosw_k1": 0.0,
    "gamma_HIRES": {"type": "normal", "mu": 0, "sigma": 10},
}
with astronereus.session() as s:
    r = s.fit_rv(rv, planets=1, priors=priors, output_dir="out/priors")
print(list(r.params))
# ['K_k1', 'gamma_HARPS', 'gamma_HIRES', 'P_k1', 'sigma_HARPS', 'sigma_HIRES', 'Mo_k1']
```

Fixed parameters do not appear in `r.params`.

### Parameter names

Names follow `<parameter>_k<n>` for planet `n` (`k1`, `k2`, ...) and
`<parameter>_<instrument>` for per-instrument terms, with the instrument label taken
from your data dict.

| name | meaning | default prior |
|---|---|---|
| `P_k1` | period (days) | log-uniform from 0.1 d to three baselines |
| `K_k1` | RV semi-amplitude (m/s) | modified Jeffreys, ceiling from the RV scatter |
| `sesinw_k1`, `secosw_k1` | sqrt(e) sin(omega), sqrt(e) cos(omega) | uniform on `[-1, 1]` |
| `Mo_k1` | mean anomaly at the reference epoch (rad) | uniform on `[0, 2 pi]` |
| `phic_k1` | transit phase `2 pi (Tc - t_ref_c) / P` (rad), replaces `Mo_k1` under `time_anchor="phic"` | uniform on `[-pi, pi]` |
| `gamma_<inst>` | RV zero point (m/s) | uniform, data mean plus or minus three times the scatter |
| `sigma_<inst>` | extra RV jitter (m/s) | normal(5, 5) truncated to `[0, 50]` |
| `inc_k1`, `Omega_k1` | inclination, node (rad), astrometric planets | sine, uniform on `[0, 2 pi]` |
| `M_sec_k1`, `a_k1` | companion mass (M_sun), semi-major axis (AU), by parametrization | log-uniform |
| `plx`, `M_pri` | parallax (mas), primary mass (M_sun) in astrometric fits | set these yourself, see [Astrometry](#astrometry) |
| `lambda_k1`, `v_sin_i_star` | spin-orbit angle (rad), projected rotation (m/s) in RM fits | uniform `[-pi, pi]`, log-uniform |
| `offset_<inst>`, `jitter_<inst>`, `q1_<inst>`, `q2_<inst>`, `rho_s` | photometry offset, jitter, limb darkening, stellar density | see [Transits](#transits) |

The sampled eccentricity coordinates depend on the model parametrization, so the table
is the default set. The complete list, with the defaults for noise models, GPs and TTV
offsets, is in
[priors.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/priors.md); the
parametrizations are in
[parametrizations.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/parametrizations.md).
Derived quantities such as `ecc_k1`, `omega_deg_k1` or `a_au_k1` are not sampled and
cannot be given a `priors` entry. Constrain eccentricity or stellar density with
`external_priors`.

### Priors on derived quantities

`external_priors=` takes a list of `{"quantity", "prior"}` dicts. Only `ecc` (per
planet) and `rho_s` exist.

```python
r = s.fit_rv(rv, planets=1,
             priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16}},
             external_priors=[{"quantity": "ecc",
                               "prior": {"type": "normal", "mu": 0.0, "sigma": 0.1}}],
             output_dir="out/ecc_prior")
```

## Samplers

Pass `engine=` an instance from `astronereus.engines`. Each class carries the options
of exactly one sampler; a field you leave unset is not sent, and Nereus supplies its
own default. An option the sampler does not take is a `TypeError` from Python, or an
error from Nereus listing the valid names.

With no `engine`, Nereus picks by shape: `PTEmcee` for a fixed number of planets,
`TransdimPTEmcee` when the fit is trans-dimensional. The call below is the default
engine at its own budget (16 temperatures, 100 walkers, 3000 steps, 2000 burn-in), on
the data defined under [Priors](#priors):

```python
with astronereus.session() as s:
    r = s.fit_rv(rv, planets=2,
                 priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16},
                         "P_k2": {"type": "loguniform", "lo": 30, "hi": 70}},
                 output_dir="out/default")
print(r)
# <JobResult status=ok log_z=-194.82 elapsed=62s 14 params 0 figures>
```

### Choosing

| engine | Nereus name | use it for | main options | log Z |
|---|---|---|---|---|
| `PTEmcee` | `pt_emcee` | default for a known number of planets; multimodal, weak or astrometric posteriors | `n_temps`, `n_walkers`, `n_steps`, `n_burnin`, `init_strategy`, `thin`, `beta_min`, `seed` | yes |
| `PT` | `pt` | slice-sampling parallel tempering; evidence by thermodynamic integration | `n_rounds` (scans = 2^n_rounds), `n_chains`, `within_model`, `init_strategy` | yes |
| `PTHMC` | `pt_hmc` | NUTS inside PT; smooth, differentiable fixed-dimension models | `n_temps`, `n_walkers_per_temp`, `n_sweeps`, `n_warmup`, `target_accept` | yes |
| `PTWhitening` | `pt_whitening` | PT with whitened proposals, for strongly correlated posteriors | `n_temps`, `n_walkers`, `n_steps`, `n_burnin`, `whiten_window` | yes |
| `Nested` | `nested` | a single-run evidence for fixed dimension | `n_live`, `dlogz`, `bounds`, `proposal`, `parallel` | yes |
| `NestedINS` | `nested_ins` | importance nested sampling | `n_live`, `dlogz`, `proposal`, `max_iter` | no |
| `NestedDynamic` | `nested_dynamic` | dynamic nested sampling, live points follow the posterior | `n_live_init`, `n_live_batch`, `dlogz_init`, `dlogz_batch` | no |
| `NUTS` | `nuts` | smooth unimodal posteriors | `n_samples`, `n_warmup`, `n_chains`, `target_accept`, `ad_backend` | no |
| `Ensemble` | `ensemble` | plain affine-invariant ensemble, one temperature | `n_walkers`, `n_steps`, `n_burnin`, `thinning` | no |
| `ESS` | `ess` | elliptical slice sampling, Gaussian-prior latent models | `n_samples`, `n_burnin`, `n_chains` | no |
| `PA` | `pa` | particle annealing; avoid above about 20 dimensions | `n_replicas`, `n_mcmc`, `max_steps`, `mutation_kernel` | yes |
| `SMC` | `smc` | sequential Monte Carlo over a temperature ladder | `betas` | yes |
| `MAP` | `map` | mode finder: multi-start optimisation that returns flags and a Laplace log Z | `n_starts`, `method`, `maxiter` | Laplace |
| `OFTI` | `ofti` | rejection sampling for short-arc relative astrometry with fixed masses ([Astrometry](#relative-astrometry-imaging)) | `n_attempts`, `n_calibrate`, `planet_idx` | no |
| `TransdimPTEmcee` | `transdim_pt_emcee` | unknown number of planets; the blind-search default | `n_temps`, `n_walkers`, `n_steps`, `n_burnin`, `informed_birth_fraction`, `beta_min` | occupancy |
| `MoMS` | `moms` | trans-dimensional without reversible jumps | `n_samples`, `n_warmup`, `n_chains`, `informed_birth_fraction` | no |
| `RJMCMC` | `rjmcmc` | classic reversible-jump baseline | `n_samples`, `n_warmup`, `n_chains` | no |
| `Daedalus` | `daedalus` | trans-dimensional posterior and nested evidence in one run | `n_live`, `dlogz`, `n_mcmc`, `informed_birth_fraction` | yes |

Every field of every class is a dataclass attribute, so `help(engines.PTEmcee)` lists
them all. The full option lists, with Nereus's defaults, are in
[samplers.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/samplers.md); how
the samplers were checked against injected orbits is in
[sampler_validation.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/sampler_validation.md).

- For an evidence take `Nested` or a parallel-tempering engine. On a curved, eccentric
  RV ridge prefer `PTEmcee` or `PT` to `Nested`: nested bounds cannot follow the ridge.
- `PTEmcee(init_strategy="prior")` is what recovers weak signals: walkers start spread
  over the prior rather than around a single optimum.

### Evidence with a nested sampler

```python
with astronereus.session() as s:
    r = s.fit_rv(rv, planets=1,
                 priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16}},
                 engine=engines.Nested(n_live=500, dlogz=0.3),
                 output_dir="out/nested")
print(r.log_z)      # log evidence
print(r.table())    # fitted and derived parameters, sampler summary
```

For a nested run the sampler block of `r.table(diagnostics=True)` reads
`convergence: not applicable`, because the draws are resampled and carry no chain to
compute R-hat on.

### MAP

`MAP` is a fail-loud mode finder, not a global search. It starts from `n_starts` points
drawn from the prior and reports whether the best one sits against a prior bound.
Give it priors tight enough that the starts land in one basin. It returns the flags and
a Laplace log Z.

```python
with astronereus.session() as s:
    m = s.fit_rv(rv, planets=1,
                 priors={"P_k1": {"type": "normal", "mu": 12, "sigma": 0.1, "lo": 11, "hi": 13},
                         "sigma_HARPS": 1.0, "sigma_HIRES": 1.0},
                 engine=engines.MAP(n_starts=8), output_dir="out/map")
print(m.log_z)                 # Laplace evidence
print(m.raw["map"])
# {'railed': False, 'railed_params': [], 'log_posterior': -697.77,
#  'converged': False, 'n_basins': 8}
```

A railed fit comes back with `status="railed"` rather than `"ok"`; see
[JobFailed](#jobfailed).

## Trans-dimensional fits

When the number of planets is unknown, give `planets` a range and set
`transdim=True`. The chain then moves between planet counts with birth and death
moves, and the result is a posterior over the count.

```python
with astronereus.session() as s:
    r = s.fit_rv(rv, planets=[0, 1, 2], transdim=True,
                 priors={"P_k1": {"type": "loguniform", "lo": 3, "hi": 100},
                         "P_k2": {"type": "loguniform", "lo": 3, "hi": 100}},
                 engine=engines.TransdimPTEmcee(n_temps=10, n_walkers=120,
                                                n_steps=4000,
                                                informed_birth_fraction=0.7),
                 plots=["transdim_occupancy"],
                 output_dir="out/transdim")
print(r.n_planets_posterior)      # {count: fraction of draws}, see below
print(r.figures["transdim/occupancy"])
```

How it works:

- The upper end of the range sets the maximum count (`max_kplanet`). `[0, 1, 2]` and
  `2` allocate the same two planet slots; the lower end is not used.
  **`planets=[0, 1, 2]` without `transdim=True` is a fixed two-planet fit**:
  `r.n_planets_posterior` is `None`.
- Per-planet priors are written for each slot (`P_k1`, `P_k2`, ...). Planet labels are
  interchangeable, so with no ordering constraint the slots can swap and individual
  `_k1` and `_k2` posteriors can be bimodal. Judge the count first, then refit the
  winning model with fixed `planets=n`.
- With no `engine`, `TransdimPTEmcee` is used. `informed_birth_fraction` (the share of
  birth proposals seeded from the residual periodogram) is worth setting to about 0.7
  to find weak planets on real data. Use at least 10 temperatures if you want a usable
  evidence.
- `transdim` also takes a config block instead of `True`, to set the birth strategies
  and the share of moves that change dimension:

  ```python
  transdim={"birth_strategies": ["PriorBirth", "InformedBirth"],
            "birth_weights": [0.3, 0.7],
            "transdim_fraction": 0.3}
  ```

The other trans-dimensional engines are `MoMS`, `RJMCMC` and `Daedalus` (table above);
all four are passed the same way, with `transdim=True`. Only `TransdimPTEmcee` and
`Daedalus` give a log Z.

### Reading `n_planets_posterior`

`r.n_planets_posterior` maps the planet count, as a string, to the fraction of
posterior draws with that many planets. Counts with no draws are absent, so
`{'1': 0.26, '2': 0.74}` means no draw had zero planets. Under a uniform prior on the
count these fractions are the posterior model probabilities. They are only as good as
the chain: check `r.diagnostics` and `r.fit_health` ([Results](#diagnostics)), rerun
with more steps, and compare seeds before quoting a probability. The `r.params` table
of a trans-dimensional fit describes the draws in which each parameter exists.
Noise-model selection runs through the same machinery with a `transdim` block that
sets `noise`; the options are in
[transdim.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/transdim.md).

## Results

Every `fit_*` and `run_job` returns a `JobResult`. `r.raw` is the full summary dict
from Nereus, so anything newer than this wrapper is still reachable.

`print(r)` gives a one-line summary: status, log Z, wall time, and the number of
parameters and figures.

| attribute | what it is |
|---|---|
| `r.status`, `r.ok` | `"ok"` or a failure status such as `"failed"` or `"railed"` |
| `r.params` | fitted parameters, a `Params` table |
| `r.derived` | derived physical quantities, a `Params` table |
| `r.table(fitted=True, derived=True, diagnostics=False)` | both tables, plus sampler health with `diagnostics=True` |
| `r.log_z` | log evidence, or `None` (NaN for samplers without one) |
| `r.n_planets_posterior` | planet-count posterior, trans-dimensional fits only |
| `r.diagnostics` | `Diagnostics`: ladder acceptance, R-hat and ESS |
| `r.fit_health` | pass/fail checks on the run |
| `r.loo`, `r.ppc`, `r.detection_limits` | leave-one-out, posterior predictive and detection-limit summaries |
| `r.figures` | the figures written, as a mapping |
| `r.elapsed_sec`, `r.n_evals` | wall time and likelihood evaluations |
| `r.config_path` | the job config path, when there is one |
| `r.raw` | the summary dict |

### Params and Estimate

`r.params` prints as a table of median, +1 sigma and -1 sigma, rounded to the
uncertainty, with units where they are known. Each entry is an `Estimate`: a `float`
subclass holding the median, so it works in arithmetic.

```python
print(r.params)
# fitted parameters (9)
#   Mo_k1         3.54 +0.83 -2.28  rad  = 202.66 deg
#   P_k1         12.00 +1.38 -0.77  d
#   ...
p = r.params["P_k1"]        # also r.params.P_k1
float(p)                    # the median
p.plus, p.minus             # upper and lower 1-sigma distances
p.lo16, p.hi84              # the 16th and 84th percentiles
p.lo2sig, p.hi2sig          # 95 percent interval ends
p.lo3sig, p.hi3sig          # 99.7 percent interval ends
p.multimodal                # True when "median +up -dn" misdescribes the posterior
p.text()                    # '12.00 +1.38 -0.77 d'
```

Angles are in radians in the chain; `Estimate.to_deg()` converts, and the printed table
adds the degree value. The `multimodal` flag fires on a lopsided interval (one side
more than three times the other), on a wrapped circular parameter, or on a
sign-flipping `sesinw` or `secosw`. The interval is still printed; the flag says a
percentile range may be the wrong summary.

`r.derived` holds quantities computed from the draws, such as `P_yr_k1`, `ecc_k1`,
`omega_deg_k1`, and, once the star is known (`M_s` and the other star keywords), 
`a_au_k1`, `msini_jup_k1` and `msini_earth_k1`.

### ResultTable

`r.table()` returns a `ResultTable`: `print()` it for text, or leave it as the last
expression of a notebook cell for an HTML rendering. Fitted and derived parameters sit
in separate sections on purpose, so a derived mass is not quoted as a sampled one.

```python
print(r.table(derived=False, diagnostics=True))
```

### Diagnostics

`r.diagnostics` reports the temperature ladder and convergence of the run.

```python
d = r.diagnostics
d.min_swap            # lowest swap acceptance between adjacent rungs
d.cold_acceptance     # acceptance of the beta = 1 chain
d.acceptance_swap     # per adjacent pair
d.betas               # the temperature ladder
d.convergence()       # {'worst_rhat': ..., 'worst_ess_tail': ..., 'per_parameter': {...}, ...}
d.ok                  # False if min_swap < 0.01 or worst R-hat > 1.01; None if undecidable
```

The default budget of `PTEmcee` gave, for the two-planet fit under
[Samplers](#samplers), a worst R-hat of 1.019 and a worst tail ESS of 2779 on 100
walkers. The R-hat here is strict: each walker is counted as a chain, and walkers of an
ensemble are not independent, so the default budget lands near 1.01 to 1.02, and 1.2
means the run is too short. A swap acceptance below 0.01 means the ladder has decoupled
and the cold chain may be stuck in one mode, whatever R-hat says: raise `n_temps`.

R-hat and ESS are recomputed from `<output_dir>/chains.nc` when the runtime did not
persist them; that needs the `diagnostics` extra ([Optional extras](#optional-extras)).
Everything else on this object works without it.

`r.fit_health` is a dict of checks, each with a `status` and a `message`, and an
`overall` verdict:

```python
r.fit_health["overall"]            # 'fail'
list(r.fit_health["checks"])       # ['convergence', 'multimodality', 'prior_rail', 'logpost']
r.fit_health["checks"]["convergence"]["message"]
# 'NOT converged: 7/7 params fail R-hat > 1.05 or ESS < 200 (worst R-hat 2.09, ...)'
```

`prior_rail` flags a parameter pinned against a prior bound. A jitter at zero is not
counted as a rail.

`r.loo` (PSIS leave-one-out: `elpd_loo`, `se_elpd_loo`, `p_loo`, `pareto_k_max`, ...)
and `r.ppc` (reduced chi-square per instrument, residual periodogram peak and FAP,
3-sigma outliers) come with every fit unless you pass `science=False`.

### Figures

Request figures with `plots=[...]` on the fit. Without it nothing is drawn.

```python
r = s.fit_rv(rv, planets=1, output_dir="out/res",
             priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16}},
             plots=["corner", "rv_phasefold"])
list(r.figures)               # ['corner', 'models/RV_phasefold_K1']
r.figures["corner"]           # a Figure
r.figures.corner              # the same, by attribute
```

`r.figures` is a mapping from a logical name to a `Figure`, which is a `pathlib.Path`
that also renders itself in a notebook. In an attribute name `/`, `.` and `-` become
`_`: `r.figures.models_RV_phasefold_K1`. Putting `r.figures` in a notebook cell shows
all of them. Plot names, `plot_kwargs=` and `save_pdf=True` are described in
[plotting.md](https://github.com/jvines/Nereus.jl/blob/main/docs/src/plotting.md).

### Output files

With `output_dir="out/res"` the run writes:

```
out/res/
  summary.json            the machine-readable summary (r.raw)
  chains.nc               the posterior, NetCDF
  ppc.json, ppc.png       posterior predictive check
  detection_limits.nc/.png
  tables/                 fitted.* and derived.* as json, csv, ecsv, dat, tex
  plots/                  the figures you requested
```

### replot

`replot` draws figures from a finished fit without sampling again. It reads
`<output_dir>/chains.nc`, so it needs the same channels and priors again.

```python
fig = astronereus.replot(rv, output_dir="out/res", planets=1,
                         priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16}},
                         plots=["rv_phasefold"], session=s)
fig.figures        # <Figures: 1 - models/RV_phasefold_K1>
s.replot(rv, output_dir="out/res", planets=1, plots=["corner"],      # session form
         priors={"P_k1": {"type": "loguniform", "lo": 8, "hi": 16}})
```

### JobFailed

A `fit_*` call that Nereus rejects after running, such as a railed `MAP`, returns a
`JobResult` with `ok == False` and prints as
`<JobResult FAILED status='railed' error=...>`. It does not raise. Call `r.check()` to
turn that into an exception; it returns the result when the run succeeded, so it
chains. `s.run_job` calls it for you unless you pass `check=False`.

```python
from astronereus import JobFailed

m = s.fit_rv(rv, planets=1, output_dir="out/bad",
             priors={"P_k1": {"type": "loguniform", "lo": 5, "hi": 30}},
             engine=engines.MAP(n_starts=8))
try:
    m.check()
except JobFailed as e:
    print(e.error)
# MAP railed against a prior bound on: P_k1, K_k1 - treat this as a failed fit, not a result
```

`JobFailed` carries `error`, `traceback` and `config_path`. Errors raised before a fit
starts (an unknown prior type, an unknown sampler name in a config) are
`astronereus.DaemonError` instead, with the Julia message.

## Job configs (`run_job`)

`fit_*` builds a Nereus job from Python objects. `run_job` takes the job itself: one
dict with the data, model, priors, sampler and outputs. Use a config for the modes
that [Fit modes](#fit-modes) routes through one, for a block the `fit_*` signatures do
not expose, or when you already hold a job description as JSON (read it with
`json.load` and pass the dict). The schema, with every block and sampler option, is
[JOB_CONFIG.md](https://github.com/jvines/Nereus.jl/blob/main/docs/JOB_CONFIG.md).

```python
d = rv.data["HARPS"]
cfg = {
    "seed": 1,
    "output_dir": "out/job",                      # required
    "star": {"M_s": 1.0},
    "data": {"rv": {"values": {"bjd": d["t"], "rv": d["rv"], "rv_err": d["rv_err"],
                               "instrument": ["HARPS"] * len(d["t"])}}},
    "model": {"max_kplanet": 1, "planet_modes": ["RV_ONLY"]},
    "priors": {"P_k1": {"type": "LogUniformPrior", "args": [8, 16]}},
    "sampler": {"name": "pt_emcee",
                "kwargs": {"n_temps": 16, "n_walkers": 100,
                           "n_steps": 3000, "n_burnin": 2000}},
    "output": {"plots": ["corner"]},
}
r = astronereus.run_job(cfg)           # or s.run_job(cfg) inside a session
print(r)
# <JobResult status=ok log_z=-164.61 elapsed=20s 0 params 1 figures>
P = r.raw["fitted"]["parameters"]["P_k1"]
print(P["value"], P["err_lo"], P["err_hi"], P["unit"])
# 11.989 0.0165 0.0157 d
```

Differences from `fit_*`:

- Priors are `{"type": "<Name>Prior", "args": [...]}` with the constructor arguments in
  order, not the `normal`/`uniform` dicts of `fit_*`:
  `{"type": "NormalPrior", "args": [mu, sigma, lo, hi]}`,
  `{"type": "ModJeffreysPrior", "args": [knee, upper]}`.
- Parameters come back under `r.raw["fitted"]["parameters"]` (each entry has `value`,
  `err_lo`, `err_hi`, `ci3`, `unit`; the `show` helper in [Fit modes](#fit-modes)
  prints them). `r.derived`, `r.log_z` and `r.figures` work as for `fit_*`.
- A bad config (an unknown `sampler.name`, a missing block) raises `DaemonError`
  listing the valid names, even with `check=False`.
- `run_job` raises `ValueError` before starting if `output_dir` is missing.

`astronereus.run_job` starts a process-wide daemon on first use and reuses it; `with
astronereus.session() as s: s.run_job(cfg)` makes the lifetime explicit, which is the
better form for several jobs in a row.
