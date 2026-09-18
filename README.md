# astronereus

Python frontend for [Nereus](https://github.com/jvines/Nereus.jl) — trans-dimensional
Bayesian inference for exoplanet orbits.

**You do not need Julia.** Nereus itself is Julia, but this package fetches a
prebuilt runtime on first use and runs it out-of-process in a warm daemon.
Nothing is compiled at install time.

```sh
pip install astronereus
```

That is the whole install. The first time you actually ask for a fit, the
runtime (~480 MB) is fetched and warmed once, and you are told it is happening:

```
astronereus: no Julia runtime yet -- fetching it once now.
astronereus: fetching runtime (macos-arm64, julia 1.11.9) ...
astronereus: unpacking ...
astronereus: warming the runtime (one-time, a few minutes) ...
astronereus: ready (207s)
```

`pip` cannot do that step itself -- wheels have no post-install hook -- so it
happens on first use instead. To get it over with ahead of a workshop or a
flight, run it deliberately:

```sh
python -c "import astronereus; astronereus.install()"
```

`NEREUS_NO_AUTO_INSTALL=1` turns the automatic fetch off and makes the first
call fail with instructions instead, for CI and air-gapped machines.

The distribution and the import are both `astronereus`. The name `nereus` on
PyPI belongs to an unrelated geophysics package, and a distribution whose import
name differs from its install name is a thing every user has to be told, so they
are deliberately the same string here.

## Use

```python
import astronereus

summary = astronereus.run_job(cfg)          # one-shot

with astronereus.session() as s:            # or reuse one warm daemon
    a  = s.run_job(cfg_a)
    pg = s.detect.rv_periodogram(t=t, rv=rv, rv_err=err)
```

Typed entry points per observable — `fit_rv`, `fit_transit`, `fit_astrometry`,
`fit_rm`, `fit_tomography`, `fit_ttv`, `fit_binary`, `fit_joint` — take channel
objects (`RV`, `Transit`, `Astrometry`, `RM`, `Night`, `TTV`, `SB2`) and return
a `JobResult`.

The first `install()` unpacks the runtime and warms it once (a few minutes);
every process after that starts in about twenty seconds. `NEREUS_BUNDLE_URL`
points the installer at a local file or `file://` URL — that is the intended
path for a workshop room, where thirty people pulling half a gigabyte over
conference wifi is not a plan.

## Runtime bundles

Prebuilt runtimes are attached to the
[Nereus.jl releases](https://github.com/jvines/Nereus.jl/releases), with
`SHA256SUMS` alongside:

| platform | size |
|---|---|
| `macos-arm64` | 465 MiB |
| `linux-x86_64` | 688 MiB |

`install()` picks the one for your platform and verifies it against the
published checksum. For a classroom, fetch it once and point everyone at the
local copy instead:

```sh
export NEREUS_BUNDLE_URL=/Volumes/NEREUS/nereus-runtime-1.11.9-macos-arm64.tar.zst
```

`NEREUS_HOME` relocates the runtime cache out of the default platform cache
directory.

## Licence

MIT — see `LICENSE`.
