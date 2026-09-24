"""Locate, fetch and unpack the Nereus runtime bundle.

The bundle is a self-contained tarball carrying BOTH a Julia runtime and a
fully-precompiled depot, so the user needs no Julia and no compilation.

Two facts make this work, both measured rather than assumed (2026-08-07/08):

1. **Path relocatability** comes from including Julia's OWN bundled depot in
   ``JULIA_DEPOT_PATH``. Julia ships relocatable stdlib pkgimages under
   ``<julia>/share/julia``. If you override the depot path and exclude it,
   stdlibs get recompiled into your private depot with the BUILD host's
   absolute paths baked in, and every cache is rejected on another machine
   ("Rejecting cache file ... because it is for file <buildpath>/Printf.jl").
   That cascades, since everything depends on Printf/Dates/TOML/Unicode.

2. **CPU portability** comes from multiversioned pkgimages, built with
   ``JULIA_CPU_TARGET``. On x86-64 this is REQUIRED: Julia reports the exact
   microarchitecture (``znver2`` on a Ryzen 4700G) and a single-target image is
   rejected on an older CPU with "Rejecting this target due to use of
   runtime-disabled features". On macOS/arm64 it is a no-op, because Julia
   normalises every Apple Silicon chip to ``apple-m1``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

JULIA_VERSION = "1.11.9"

# Guards the auto-install path in julia_env(). install() calls warm(), warm()
# calls julia_env(), and julia_env() auto-installs when nothing is there -- so a
# bundle that unpacks but is BROKEN (no julia binary, so is_installed() stays
# false) would otherwise re-download half a gigabyte forever.
_AUTO_INSTALLING = False

# Set at BUILD time. Recorded here so the build and the docs cannot drift.
CPU_TARGETS = {
    # These are the strings tools/build_bundle.sh actually passes as
    # JULIA_CPU_TARGET. They are recorded here so the build and the docs cannot
    # drift; build_bundle.sh is the source of truth, not this dict.
    "linux-x86_64": "generic;sandybridge,-xsaveopt,clone_all;"
                    "haswell,-rdrnd,base(1);znver2,base(1)",
    "linux-aarch64": "generic;cortex-a76,clone_all;neoverse-n1,clone_all",
    # macOS arm64 still needs multiversioning off a generic base, but Julia
    # normalises every Apple Silicon chip to "apple-m1" (M1 through M4), so the
    # second target is the only one that ever matches.
    "macos-arm64": "generic;apple-m1,clone_all",
}


class BundleError(RuntimeError):
    pass


def platform_tag() -> str:
    """Canonical (os, arch) tag. Windows is deliberately unsupported."""
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64",
            "arm64": "arm64", "aarch64": "aarch64"}.get(machine, machine)
    if sys.platform == "darwin":
        if arch != "arm64":
            raise BundleError(
                "Only Apple Silicon is supported on macOS; got " + machine)
        return "macos-arm64"
    if sys.platform.startswith("linux"):
        return f"linux-{'x86_64' if arch == 'x86_64' else 'aarch64'}"
    raise BundleError(
        f"Unsupported platform {sys.platform!r}. Nereus supports Linux and "
        "macOS only — Windows is not, and is not planned.")


def cache_root() -> Path:
    """Where the runtime lives. Honours NEREUS_HOME, else a platform cache dir.

    The location is arbitrary — relocatability is handled by the depot-path
    recipe, NOT by pinning an install prefix. No sudo, no /opt.
    """
    if env := os.environ.get("NEREUS_HOME"):
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "nereus"
    return Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "nereus"


def runtime_dir(version: str = JULIA_VERSION) -> Path:
    return cache_root() / f"runtime-{version}-{platform_tag()}"


def find_julia(root: Path) -> Path | None:
    """Locate the julia binary under a bundle root.

    Layout is NOT fixed: the official tarballs give ``julia/bin/julia`` on both
    Linux and macOS, but a juliaup/DMG install on macOS nests it inside
    ``Julia-<ver>.app/Contents/Resources/julia/bin/julia``. Search rather than
    assume, so a bundle built from either source works.
    """
    direct = root / "julia" / "bin" / "julia"
    if direct.exists():
        return direct
    for cand in (root / "julia").rglob("bin/julia"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def julia_home(root: Path) -> Path | None:
    """The Julia install prefix (the dir containing bin/ and share/julia)."""
    jb = find_julia(root)
    return jb.parent.parent if jb else None


def runtime_parts(version: str = JULIA_VERSION) -> dict[str, bool]:
    """Every piece `julia_env` requires, and whether it is present.

    Kept in step with julia_env deliberately: anything that passes
    `is_installed` must then WORK. The old check was a julia binary plus a
    `depot/` directory, which a half-unpacked tree satisfies -- and since
    `install()` returns early on anything that looks installed, a partial
    runtime was unrepairable short of deleting the cache by hand.
    """
    d = runtime_dir(version)
    jhome = julia_home(d)
    return {
        "julia binary": find_julia(d) is not None,
        "depot": (d / "depot").is_dir(),
        # relocatability depends on Julia's own stdlib depot being layered in
        "julia stdlib depot": bool(jhome) and (jhome / "share" / "julia").is_dir(),
        # a depot is a cache; this is the environment `using Nereus` resolves against
        "bundled project": (d / "depot" / "dev" / "Nereus" / "Project.toml").exists(),
    }


def is_installed(version: str = JULIA_VERSION) -> bool:
    return all(runtime_parts(version).values())


def show_progress() -> bool:
    """Whether to echo the fit's progress bar to stderr.

    ON BY DEFAULT. This used to try to recognise the caller's front-end and
    stayed silent when it could not, which got it wrong every single time a
    new one appeared: first everything that was not classic Jupyter
    (`ZMQInteractiveShell`), then marimo, which is not built on IPython at all
    so there is no shell object to find. Each miss looked the same to the
    person running the fit -- several minutes of total silence with no way to
    tell a running sampler from a hung one.

    A bar nobody needed is a cosmetic annoyance; a silent multi-minute fit is
    the thing people file bugs about. So the burden of proof is now the other
    way round: show it unless something positively says not to.

    NEREUS_QUIET=1 silences it. CI=1 (set by GitHub Actions, GitLab, Travis,
    CircleCI and friends) does too, because a build log is not watched by
    anyone. NEREUS_PROGRESS=1 overrides both.
    """
    if os.environ.get("NEREUS_PROGRESS"):
        return True
    if os.environ.get("NEREUS_QUIET"):
        return False
    if os.environ.get("CI"):
        return False
    return True


def progress_style() -> str:
    """How this host can show a progress bar: "inplace", "native" or "lines".

    - "inplace": the host redraws a `\r` line. A terminal does, and so does
      anything IPython-backed (Jupyter, Colab, VS Code, qtconsole).
    - "native": marimo. It has no `\r` handling anywhere in its stream layer,
      and it makes tqdm work by monkeypatching tqdm into its OWN progress bar
      rather than by rendering the escape -- which is the proof that console
      carriage returns do not work there. So drive that same native bar.
    - "lines": anything else. A `\r` that is not honoured runs every update
      into one unreadable line, so those hosts get whole lines instead,
      throttled by the pump.
    """
    if _in_marimo():
        return "native"
    try:
        if sys.stderr.isatty():
            return "inplace"
    except Exception:
        pass
    ipy = sys.modules.get("IPython")
    if ipy is not None:
        try:
            if ipy.get_ipython() is not None:
                return "inplace"
        except Exception:
            pass
    return "lines"


def _in_marimo() -> bool:
    """True only inside a running marimo notebook.

    `running_in_notebook()` is marimo's own test and returns False when the
    same file is executed as a plain script, which is what we want: a script
    has no cell to attach a native bar to.
    """
    if "marimo" not in sys.modules:
        return False
    try:
        from marimo._runtime.context.utils import running_in_notebook
        return bool(running_in_notebook())
    except Exception:
        return False


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _fmt_secs(s: float) -> str:
    s = int(max(s, 0))
    return f"{s//60}m{s%60:02d}s" if s >= 60 else f"{s}s"


def _bar(done: int, total: int, elapsed: float, width: int = 26) -> str:
    rate = done / elapsed if elapsed > 0 else 0.0
    if total > 0:
        frac = min(done / total, 1.0)
        filled = int(width * frac)
        eta = (total - done) / rate if rate > 0 else 0.0
        return (f"  [{'=' * filled}{' ' * (width - filled)}] {frac * 100:5.1f}%  "
                f"{_fmt_bytes(done)}/{_fmt_bytes(total)}  "
                f"{_fmt_bytes(rate)}/s  ETA {_fmt_secs(eta)}")
    # no Content-Length: report what we can rather than a fake percentage
    return f"  {_fmt_bytes(done)} downloaded  {_fmt_bytes(rate)}/s"


def _download(url: str, dest: Path, progress: bool = True) -> None:
    """Stream `url` to `dest`, drawing a progress bar on a TTY.

    Stdlib only, deliberately: this package's only required dependency is
    zstandard (and only below Python 3.14), and a progress bar is not worth
    adding another. Falls back to silence when stderr is not a TTY, so CI logs
    do not fill with carriage returns.

    The timeout matters as much as the bar. Before this, a stalled connection
    hung forever with no output at all, which is indistinguishable from a slow
    one -- and this is a ~480 MB download that people run on conference wifi.
    """
    import time
    # Static UA: the version lives in __init__.py and importing it here
    # would be circular. Not worth a second copy to drift.
    req = urllib.request.Request(url, headers={"User-Agent": "astronereus"})
    tty = progress and show_progress()
    with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = last = time.monotonic()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if tty and now - last >= 0.2:
                last = now
                print("\r" + _bar(done, total, now - t0), end="",
                      file=sys.stderr, flush=True)
    if tty:
        print("\r" + _bar(done, total, time.monotonic() - t0) + "\n",
              end="", file=sys.stderr, flush=True)
    if total and done < total:
        raise BundleError(
            f"download truncated: got {done} of {total} bytes from {url}")


def _verify(path: Path, sha256: str | None) -> None:
    if not sha256:
        return
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != sha256:
        path.unlink(missing_ok=True)
        raise BundleError(f"bundle checksum mismatch: expected {sha256}, got {got}")


# Published runtime bundles, per platform: (url, sha256).
#
# Built by tools/build_bundle.sh, which prints the exact line to paste here.
# ALL THREE bundles below were built from Nereus.jl commit de0df05 (v0.6.1),
# on three machines, each gated on a real fit_rv dispatch AND on
# Nereus.PY_API_VERSION matching this client before compression. Each carries
# BUILD_INFO.txt with its own commit, CPU target, py_api and build time, so
# provenance is checkable rather than asserted.
#
# Check it that way, too, and check the TREE rather than the commit. The
# v0.2.3 bundles recorded commit 841be43, which is not in this repository --
# that build checkout was a fresh `git init` with a single snapshot commit,
# so the same source content hashed to a different commit id with no shared
# ancestry. Their trees are identical (6276865e), so v0.2.3 was built from
# exactly the v0.2.3 source; only the commit id was unrecognisable.
#
# Compare with:  git rev-parse <tag>^{tree}
# The bundles below are built from a checkout of this repository, so their
# recorded commit is directly reachable from main.
#
# The URLs point at a GitHub release on the Nereus.jl repo. That repo is
# public, so the assets are anonymously downloadable — which is the thing that
# makes `pip install astronereus; astronereus.install()` work for a stranger.
# The tag must be created on the Forgejo repository that mirrors here, never
# on GitHub directly: a GitHub-only tag is pruned by the next mirror sync,
# which demotes the release to a draft and makes these URLs 404. There is no
# intermediate staging repository -- the mirror is direct. See DEPLOYMENT.md.
_REL = "https://github.com/jvines/Nereus.jl/releases/download/v0.6.1"

#: The Nereus version inside the bundles below. A cached runtime is NOT
#: refreshed by `pip install -U astronereus`, and `PY_API_VERSION` only catches
#: an INCOMPATIBLE skew -- v0.6.0 fixed pt_emcee's convergence without touching
#: the contract, so a user upgrading the client alone would have gone on running
#: v0.5.3 with no sign of it. `Session.start()` compares this against `ping`'s
#: `nereus` field and says what to do about it.
RUNTIME_VERSION = "0.6.1"

BUNDLES: dict[str, tuple[str, str]] = {
    "macos-arm64": (
        f"{_REL}/nereus-runtime-1.11.9-macos-arm64.tar.zst",
        "ba656025fbd2e53eea0479aa73cdfbaa669ff07ed3a131bddb061d315d710b00"),
    "linux-x86_64": (
        f"{_REL}/nereus-runtime-1.11.9-linux-x86_64.tar.zst",
        "5c56f2dc006668d111881d93ec999173068c96a8d97d257709f5607807451382"),
    "linux-aarch64": (
        f"{_REL}/nereus-runtime-1.11.9-linux-aarch64.tar.zst",
        "42ee98da50a7bb5634ee150fdad65a1a352fe9635c2d1ff8a6f68b4e43902b09"),
}


def default_bundle(version: str = JULIA_VERSION) -> tuple[str, str]:
    """(url, sha256) for this platform, or raise with what to do about it.

    NEREUS_BUNDLE_URL overrides everything and may be a local path or a file://
    URL. That is not a debugging hook — it is the workshop path. A room of
    thirty people each pulling half a gigabyte over conference wifi does not
    work, so the bundles go on a USB stick and everyone sets one variable:

        export NEREUS_BUNDLE_URL=/Volumes/NEREUS/nereus-runtime-1.11.9-macos-arm64.tar.zst

    The checksum is skipped for local overrides: you handed us the file, so
    there is nothing to verify it against that you did not also supply.
    """
    if env := os.environ.get("NEREUS_BUNDLE_URL"):
        return env, ""
    tag = platform_tag()
    if tag not in BUNDLES:
        raise BundleError(
            f"no published runtime bundle for {tag}. Build one with "
            f"tools/build_bundle.sh and register it in _runtime.py BUNDLES, "
            f"or pass an explicit url, or set NEREUS_BUNDLE_URL to a local "
            f"copy. Available: {sorted(BUNDLES) or 'none'}")
    return BUNDLES[tag]


def _precompile_tasks() -> int:
    """How many precompile workers this machine can survive.

    Julia defaults to `Sys.CPU_THREADS + 1` (base/precompilation.jl), capped at
    16. That is tuned for machines with RAM proportional to cores. A fanless
    8-core / 8 GB laptop therefore fans out to NINE workers, and this dependency
    tree contains Makie, which alone took 121 s and gigabytes in one worker.
    Memory runs out, the kernel kills a worker, and Julia reports it as
    "Failed to precompile <whatever package that worker held>" -- so the named
    package looks arbitrary and the real cause is invisible.

    One worker per 4 GB, never more than the core count, never fewer than one.
    Slower on a small machine; finishes, which the default does not.
    """
    cpus = os.cpu_count() or 1
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return max(1, min(cpus, 2))     # unknown RAM: be conservative
    return max(1, min(cpus, int(ram / (4 * 1024 ** 3))))


def warm(version: str = JULIA_VERSION, progress: bool = True) -> float:
    """Force the one-time post-relocation recompile now. Returns seconds spent.

    WHY THIS EXISTS. pkgimages record absolute paths, so a depot precompiled in
    the build directory is partially invalidated when it lands in the user's
    cache directory. Measured on macOS arm64: unpack 15 s, first `using Nereus`
    207 s, every process after that 21 s. The bundle removes the need for a
    Julia install and a full `Pkg.instantiate` — it does not remove that one
    recompile.

    Left alone, the 207 s lands on the user's FIRST FIT, which looks like the
    fit hanging. Doing it here moves it to `install()`, where waiting is what
    the user already expects. Same total time, honest placement.
    """
    import time, subprocess
    julia, env = julia_env(version)
    # Respect an explicit choice; otherwise cap by RAM, not by core count.
    env.setdefault("JULIA_NUM_PRECOMPILE_TASKS", str(_precompile_tasks()))
    if progress:
        print(f"astronereus: warming the runtime (one-time, several minutes; "
              f"{env['JULIA_NUM_PRECOMPILE_TASKS']} compile workers) …",
              file=sys.stderr, flush=True)
    t0 = time.time()
    # Popen + poll rather than run(), purely so the wait is visible. This step
    # is minutes of a single-threaded recompile with no output of its own, and
    # silence that long is indistinguishable from a hang.
    proc = subprocess.Popen(
        [str(julia), "--startup-file=no", "-e", "using Nereus"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tty = progress and show_progress()
    while proc.poll() is None:
        time.sleep(1.0)
        if tty:
            print(f"\r  compiling … {_fmt_secs(time.time() - t0)} elapsed",
                  end="", file=sys.stderr, flush=True)
    if tty:
        print("\r" + " " * 46 + "\r", end="", file=sys.stderr, flush=True)
    _, stderr_text = proc.communicate()
    dt = time.time() - t0
    if proc.returncode != 0:
        # HEAD and tail, not just the tail. A Julia precompile failure puts the
        # actual cause on the FIRST lines ("Failed to precompile X", then the
        # error); everything after is stack frames through Base.require. Taking
        # only stderr[-1500:] reliably threw away the one thing needed to
        # diagnose it and kept the part nobody can act on.
        err = (stderr_text or "").strip()
        if len(err) > 3000:
            err = err[:1800] + "\n\n  … {} chars omitted …\n\n".format(
                len(err) - 2800) + err[-1000:]
        raise BundleError(
            "runtime unpacked but `using Nereus` failed.\n"
            f"  runtime: {runtime_dir(version)}\n"
            "  This is the post-relocation recompile, so the usual cause is a\n"
            "  package that cannot rebuild at the new path. Full log:\n\n"
            + err)
    if progress:
        print(f"astronereus: ready ({dt:.0f}s)", file=sys.stderr, flush=True)
    return dt


def install(version: str = JULIA_VERSION, url: str | None = None,
            sha256: str | None = None, progress: bool = True,
            warm_after: bool = True, force: bool = False) -> Path:
    """Fetch and unpack the runtime bundle. Idempotent.

    With no `url`, uses the published bundle for this platform.

    `force=True` re-fetches over an existing runtime. That is the repair path:
    without it a runtime that is present but BROKEN could not be fixed from
    Python at all, because install() returned early and every other entry point
    routes through it.

    `version` defaults to JULIA_VERSION because the advertised call — the one in
    the README and in this package's own docstring — is a bare
    `astronereus.install()`. It used to raise TypeError, which is the first
    thing a new user would have hit.
    """
    dest = runtime_dir(version)
    if is_installed(version) and not force:
        return dest
    if progress and not force:
        missing = [k for k, ok in runtime_parts(version).items() if not ok]
        if dest.exists() and missing:
            print(f"astronereus: runtime at {dest} is incomplete "
                  f"(missing: {', '.join(missing)}) — refetching.",
                  file=sys.stderr, flush=True)
    if url is None:
        url, sha256 = default_bundle(version)
        sha256 = sha256 or None

    dest.parent.mkdir(parents=True, exist_ok=True)
    # NOT dest.with_suffix(".partial"): with_suffix cuts at the LAST dot, so
    # "runtime-1.11.9-macos-arm64" became "runtime-1.11.partial" -- misleading,
    # and it would collide between two Julia versions staging at once.
    staging = dest.parent / (dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    archive = staging / "bundle.tar.zst"
    if progress:
        print(f"astronereus: fetching runtime ({platform_tag()}, julia {version}) …",
              file=sys.stderr, flush=True)

    if url.startswith("file://") or Path(url).exists():
        src = url[7:] if url.startswith("file://") else url
        shutil.copyfile(src, archive)
    else:
        _download(url, archive, progress=progress)

    _verify(archive, sha256)

    if progress:
        print("astronereus: unpacking …", file=sys.stderr, flush=True)
    _extract_zst(archive, staging)
    archive.unlink()

    # macOS marks downloaded binaries; Gatekeeper then refuses to exec them.
    if sys.platform == "darwin":
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(staging)],
                       check=False, capture_output=True)

    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    if warm_after:
        warm(version, progress=progress)
    return dest

def _no_appledouble(members):
    """Drop AppleDouble (``._*``) sidecars while extracting.

    A tarball built on macOS carries a ``._name`` companion for every file with
    extended attributes -- half the members of a macOS bundle. macOS's own tar
    silently folds them back into xattrs, so they are INVISIBLE to `tar -tf`
    and to any check run on a Mac. Python's tarfile has no such behaviour and
    writes them as real files.

    That is not cosmetic. Makie globs its icon directory and calls PNGFiles on
    whatever it finds:

        File .../icons/._icon-128.png is not a png file

    which kills Makie, and with it CairoMakie, PairPlots and every Makie
    extension -- reported as six unrelated precompile failures. The bundle is
    fine; the extractor was the difference.

    Fixed at the source too (build_bundle.sh sets COPYFILE_DISABLE), but this
    stays: it makes already-published bundles work, and costs one comparison
    per member.
    """
    for m in members:
        if not m.name.rsplit("/", 1)[-1].startswith("._"):
            yield m


def _extract_zst(archive: Path, dest: Path) -> None:
    """Extract .tar.zst. Prefers stdlib, falls back to the zstd binary."""
    try:
        from compression import zstd  # py3.14+
        with zstd.ZstdFile(archive, "rb") as fh, \
                tarfile.open(fileobj=fh, mode="r|") as tar:
            tar.extractall(dest, members=_no_appledouble(tar))
        return
    except ImportError:
        pass
    try:
        import zstandard  # optional dependency
        dctx = zstandard.ZstdDecompressor()
        with archive.open("rb") as fh, dctx.stream_reader(fh) as reader, \
                tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(dest, members=_no_appledouble(tar))
        return
    except ImportError:
        pass
    if shutil.which("zstd"):
        subprocess.run(["sh", "-c",
                        'zstd -dc "$1" | tar -x --exclude "._*" -C "$2"',
                        "sh", str(archive), str(dest)], check=True)
        return
    raise BundleError(
        "cannot decompress .tar.zst — install the 'zstandard' extra "
        "(pip install astronereus[zstd]) or the zstd binary")


def julia_env(version: str) -> tuple[Path, dict[str, str]]:
    """(julia binary, environment) wired for the prebuilt depot.

    The depot path MUST include <julia>/share/julia — that is what makes the
    bundle relocatable. See the module docstring.
    """
    # Development escape hatch: point NEREUS_JULIA at any julia binary and the
    # bundle is bypassed entirely, depot and all. Without this you cannot run
    # the client against a working copy of Nereus.jl without first building and
    # installing a ~550 MB bundle, which makes the Python side untestable while
    # the Julia side is being changed. Not a supported user path -- the whole
    # point of the bundle is that a user needs no Julia.
    if dev := os.environ.get("NEREUS_JULIA"):
        julia = Path(dev)
        if julia.is_dir():
            julia = find_julia(julia)
        if not julia or not julia.exists():
            raise BundleError(f"NEREUS_JULIA={dev} is not a julia binary")
        env = dict(os.environ)
        env["JULIA_LOAD_PATH"] = f"@{os.pathsep}@stdlib"
        env.pop("JULIA_PROJECT", None)
        return julia, env

    d = runtime_dir(version)
    if not is_installed(version):
        # Fetch it now rather than telling the user to run a second command.
        #
        # `pip install astronereus` CANNOT do this: wheels have no post-install
        # hook (PEP 427), and setuptools' post-install cmdclass only fires when
        # installing from an sdist. So the runtime has to arrive on first use.
        # Doing it here, at the point someone actually asked for a fit, is the
        # only place it can happen without them running `install()` by hand --
        # and asking for a fit is consent enough to fetch what a fit needs.
        #
        # NEREUS_NO_AUTO_INSTALL=1 restores the old behaviour (raise and tell
        # the caller what to run) for CI and air-gapped machines, where a
        # surprise half-gigabyte download is worse than a clear failure.
        if os.environ.get("NEREUS_NO_AUTO_INSTALL"):
            raise BundleError(
                f"runtime not installed at {d}, and NEREUS_NO_AUTO_INSTALL is "
                "set. Run astronereus.install() explicitly, or set "
                "NEREUS_BUNDLE_URL to a local bundle.")
        global _AUTO_INSTALLING
        if _AUTO_INSTALLING:
            raise BundleError(
                f"runtime still missing at {d} after an install attempt -- the "
                "bundle unpacked but carries no julia binary, so it is broken. "
                "Remove that directory and retry, or report the bundle.")
        print("astronereus: no Julia runtime yet -- fetching it once now.",
              file=sys.stderr, flush=True)
        _AUTO_INSTALLING = True
        try:
            install(version)
        finally:
            _AUTO_INSTALLING = False
        if not is_installed(version):
            raise BundleError(f"automatic install did not produce a runtime at {d}")
    julia = find_julia(d)
    jshare = julia_home(d) / "share" / "julia"
    if not jshare.is_dir():
        raise BundleError(
            f"bundled Julia depot missing at {jshare} — the bundle is broken. "
            "Relocatability depends on it; see the module docstring.")
    env = dict(os.environ)
    env["JULIA_DEPOT_PATH"] = f"{d / 'depot'}{os.pathsep}{jshare}"

    # The load path must name the BUNDLED PROJECT, not "@". The depot carries a
    # precompiled Nereus, but a depot is a cache — it does not tell Julia which
    # environment to resolve `using Nereus` against. With "@" that is the user's
    # default v1.11 environment, which has never heard of Nereus, so the bundle
    # unpacks perfectly and then fails with "Package Nereus not found in current
    # path". The daemon hid this because it passes --project explicitly.
    proj = d / "depot" / "dev" / "Nereus"
    if not (proj / "Project.toml").exists():
        raise BundleError(
            f"bundle at {d} has no project at {proj} — it was built by an older "
            "tools/build_bundle.sh that did not copy the package into the "
            "bundle. Rebuild it.")
    env["JULIA_LOAD_PATH"] = f"{proj}{os.pathsep}@stdlib"
    env.pop("JULIA_PROJECT", None)
    return julia, env
