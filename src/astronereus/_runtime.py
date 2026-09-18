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


def is_installed(version: str = JULIA_VERSION) -> bool:
    d = runtime_dir(version)
    return find_julia(d) is not None and (d / "depot").is_dir()


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
# BOTH bundles below were built from Nereus.jl commit 9f74632 — the same
# commit, which the previous pair were not: they came from two different
# commits about forty minutes apart and neither matched the comment that
# claimed to describe them. Each bundle now carries BUILD_INFO.txt with its
# own commit, CPU target and build time, so this can be checked rather than
# trusted, and the build refuses a dirty tree.
#
# The URLs point at a GitHub release on the Nereus.jl repo. That repo is
# public, so the assets are anonymously downloadable — which is the thing that
# makes `pip install astronereus; astronereus.install()` work for a stranger.
# The tag must be created on the Forgejo MIRROR SOURCE (Nereus-public), never
# on GitHub: a GitHub-only tag is pruned by the next mirror sync, which demotes
# the release to a draft and makes these URLs 404. See DEPLOYMENT.md.
_REL = "https://github.com/jvines/Nereus.jl/releases/download/v0.2.1"

BUNDLES: dict[str, tuple[str, str]] = {
    "macos-arm64": (
        f"{_REL}/nereus-runtime-1.11.9-macos-arm64.tar.zst",
        "0bfcc9428bea78b76d77ae4becb6b1c67cc9baba73c8ed482bdbe5c0cd7cbc09"),
    "linux-x86_64": (
        f"{_REL}/nereus-runtime-1.11.9-linux-x86_64.tar.zst",
        "d6bf0e45588c06d10b815a67836ef4f2dd7868c3fcc1786dd585b6710fc4cff7"),
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
    if progress:
        print("astronereus: warming the runtime (one-time, a few minutes) …",
              file=sys.stderr, flush=True)
    t0 = time.time()
    r = subprocess.run([str(julia), "--startup-file=no", "-e", "using Nereus"],
                       env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise BundleError(
            "runtime unpacked but `using Nereus` failed — the bundle is broken.\n"
            + (r.stderr or "")[-1500:])
    if progress:
        print(f"astronereus: ready ({dt:.0f}s)", file=sys.stderr, flush=True)
    return dt


def install(version: str = JULIA_VERSION, url: str | None = None,
            sha256: str | None = None, progress: bool = True,
            warm_after: bool = True) -> Path:
    """Fetch and unpack the runtime bundle. Idempotent.

    With no `url`, uses the published bundle for this platform.

    `version` defaults to JULIA_VERSION because the advertised call — the one in
    the README and in this package's own docstring — is a bare
    `astronereus.install()`. It used to raise TypeError, which is the first
    thing a new user would have hit.
    """
    dest = runtime_dir(version)
    if is_installed(version):
        return dest
    _fresh = True
    if url is None:
        url, sha256 = default_bundle(version)
        sha256 = sha256 or None

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix(".partial")
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
        with urllib.request.urlopen(url) as r, archive.open("wb") as out:
            shutil.copyfileobj(r, out)

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

def _extract_zst(archive: Path, dest: Path) -> None:
    """Extract .tar.zst. Prefers stdlib, falls back to the zstd binary."""
    try:
        from compression import zstd  # py3.14+
        with zstd.ZstdFile(archive, "rb") as fh, \
                tarfile.open(fileobj=fh, mode="r|") as tar:
            tar.extractall(dest)
        return
    except Exception:
        pass
    try:
        import zstandard  # optional dependency
        dctx = zstandard.ZstdDecompressor()
        with archive.open("rb") as fh, dctx.stream_reader(fh) as reader, \
                tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(dest)
        return
    except ImportError:
        pass
    if shutil.which("zstd"):
        subprocess.run(f"zstd -dc {archive} | tar -x -C {dest}",
                       shell=True, check=True)
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
        raise BundleError(f"runtime not installed at {d}; call astronereus.install() "
                          "(or set NEREUS_JULIA to a julia binary for dev)")
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
