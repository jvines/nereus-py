#!/usr/bin/env bash
# Build a self-contained Nereus runtime bundle for THIS platform.
#
# Produces  nereus-runtime-<juliaver>-<platform>.tar.zst  containing:
#
#   julia/   an official Julia tarball, unpacked
#   depot/   a depot with Nereus and every dependency PRECOMPILED
#
# The user then needs no Julia, no package manager, and no compilation: the
# first `import astronereus` unpacks this and runs.
#
# WHY IT IS BUILT PER PLATFORM. pkgimages are native code. They are portable
# ACROSS CPUs of the same architecture (measured: an M4-built depot runs on an
# M1 with zero recompilation, and a znver2-built depot runs on znver1) but not
# across architectures or operating systems. So: run this once on macOS arm64,
# once on Linux x86_64, once on macOS x86_64 if you care about Intel Macs.
# Windows is deliberately unsupported.
#
# WHY THE DEPOT PATH MATTERS. Relocatability comes from layering the bundled
# depot over Julia's own stdlib depot at runtime:
#     JULIA_DEPOT_PATH="<bundle>/depot:<bundle>/julia/share/julia"
# Omit the second entry and the stdlib pkgimages are rebuilt on first use,
# which is exactly the 10-minute stall the bundle exists to avoid. _runtime.py
# does this; do not "simplify" it.
set -euo pipefail

# The work dir holds an unpacked Julia plus a fully-precompiled depot: ~4 GB.
# Without this it is stranded on every successful build. The leftover dir also
# made "does the work dir exist" look like a build-is-running signal, which
# cost real time. KEEP_WORK=1 retains it for debugging.
cleanup() { [ -n "${KEEP_WORK:-}" ] || rm -rf "${WORK:-}"; }
trap cleanup EXIT

JULIA_VER="${JULIA_VER:-1.11.9}"
NEREUS_JL="${NEREUS_JL:-$(cd "$(dirname "$0")/../../Nereus.jl" && pwd)}"
OUT_DIR="${OUT_DIR:-$PWD/dist}"
WORK="${WORK:-$(mktemp -d)}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  PLAT=macos-arm64;   JTAR="julia-${JULIA_VER}-macaarch64.tar.gz";  JOS=mac/aarch64 ;;
  Darwin-x86_64) PLAT=macos-x86_64;  JTAR="julia-${JULIA_VER}-mac64.tar.gz";       JOS=mac/x64 ;;
  Linux-x86_64)  PLAT=linux-x86_64;  JTAR="julia-${JULIA_VER}-linux-x86_64.tar.gz";JOS=linux/x64 ;;
  Linux-aarch64) PLAT=linux-aarch64; JTAR="julia-${JULIA_VER}-linux-aarch64.tar.gz";JOS=linux/aarch64 ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
MINOR="${JULIA_VER%.*}"
URL="https://julialang-s3.julialang.org/bin/${JOS}/${MINOR}/${JTAR}"

echo "platform : $PLAT"
echo "julia    : $JULIA_VER"
echo "Nereus.jl: $NEREUS_JL"
echo "work     : $WORK"
[ -f "$NEREUS_JL/Project.toml" ] || { echo "no Project.toml at $NEREUS_JL" >&2; exit 1; }

BUNDLE="$WORK/bundle"
mkdir -p "$BUNDLE/julia" "$BUNDLE/depot"

echo "==> fetching Julia"
curl -fsSL "$URL" -o "$WORK/julia.tar.gz"
tar -xzf "$WORK/julia.tar.gz" -C "$BUNDLE/julia" --strip-components=1
JULIA="$BUNDLE/julia/bin/julia"
[ -x "$JULIA" ] || { echo "julia not at $JULIA after unpack" >&2; exit 1; }

# Copy the package IN so the depot records a path inside the bundle. A depot
# pointing at a source tree the user does not have triggers recompilation.
#
# `git archive` rather than rsync with an exclude list: the exclude list named
# results/, logs/ and paper/, and the repo has since grown studies/, ltt9779/,
# reproduction/ and two figure directories. rsync ships whatever nobody
# remembered to exclude — 5.7 GB of run outputs and 76 MB of untracked corner
# plots, in the tree as it stands. Tracked files are exactly the right set and
# stay right as the repo changes.
echo "==> staging Nereus.jl"
DIRTY=$(git -C "$NEREUS_JL" status --porcelain --untracked-files=no)
if [ -n "$DIRTY" ] && [ -z "${ALLOW_DIRTY:-}" ]; then
  echo "refusing to build from a dirty tree — the bundle would not match any commit:" >&2
  echo "$DIRTY" >&2
  echo "commit the changes, or set ALLOW_DIRTY=1 to override." >&2
  exit 1
fi
mkdir -p "$BUNDLE/depot/dev/Nereus"
git -C "$NEREUS_JL" archive HEAD | tar -x -C "$BUNDLE/depot/dev/Nereus"
[ -f "$BUNDLE/depot/dev/Nereus/Project.toml" ] || {
  echo "git archive produced no Project.toml — is $NEREUS_JL a git repo?" >&2; exit 1; }
echo "    staged $(find "$BUNDLE/depot/dev/Nereus" -type f | wc -l | tr -d ' ') tracked files"

echo "==> instantiating + precompiling (this is the slow part)"
# JULIA_CPU_TARGET: multiversioned native code so the depot runs on older CPUs
# of the same architecture, not just the build machine's. Without it a bundle
# built on a newer chip SIGILLs elsewhere.
case "$PLAT" in
  macos-arm64)   CPUT="generic;apple-m1,clone_all" ;;
  linux-aarch64) CPUT="generic;cortex-a76,clone_all;neoverse-n1,clone_all" ;;
  *) CPUT="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);znver2,base(1)" ;;
esac
env JULIA_DEPOT_PATH="$BUNDLE/depot:$BUNDLE/julia/share/julia" \
    JULIA_CPU_TARGET="$CPUT" \
    "$JULIA" --project="$BUNDLE/depot/dev/Nereus" -e '
      using Pkg
      Pkg.instantiate()
      Pkg.precompile()
      using Nereus
      println("precompiled OK: ", pathof(Nereus))'

echo "==> smoke test: dispatch a real job through the bundled depot"
# `using Nereus` proves the runtime imports and NOTHING else. astronereus 0.2.0
# shipped a default engine naming a backend removed at the 0.1.0 rename, and it
# got through because this script stopped at the import. A bundle that cannot
# dispatch a fit must not ship.
env JULIA_DEPOT_PATH="$BUNDLE/depot:$BUNDLE/julia/share/julia" \
    "$JULIA" --project="$BUNDLE/depot/dev/Nereus" -e '
      using Nereus
      isempty(Nereus.ENGINES) && error("ENGINES is empty")
      t   = collect(range(0.0, 300.0; length = 60))
      rv  = 25.0 .* sin.(2π .* t ./ 11.7)
      err = fill(3.0, length(t))
      res = Nereus.fit_rv(Dict("SIM" => (t = t, rv = rv, rv_err = err));
                          planets = 1,
                          engine  = Dict("engine" => "pt",
                                         "options" => Dict("n_rounds" => 4,
                                                           "n_chains" => 4)),
                          output_dir = mktempdir())
      haskey(res.summary, "log_z") || error("fit_rv summary has no log_z")
      println("smoke OK: ", length(Nereus.ENGINES), " engines; fit_rv dispatched")'

# The bundle and the client in THIS tree must speak the same contract. Without
# this, a bundle built from a commit predating an API change sails through the
# fit_rv gate above (which uses `pt`, untouched by most renames) and only fails
# on a user's machine. Session.start() would then reject it -- correctly, but
# after a ~700 MB download.
echo "==> contract check"
BUNDLE_API=$(env JULIA_DEPOT_PATH="$BUNDLE/depot:$BUNDLE/julia/share/julia" \
    "$JULIA" --project="$BUNDLE/depot/dev/Nereus" \
    -e 'using Nereus; print(Nereus.PY_API_VERSION)')
# Same sibling-layout assumption as NEREUS_JL above; NEREUS_PY overrides it
# for a build that does not have the Python tree in the usual place.
CLIENT_API_FILE="${NEREUS_PY:-$(cd "$(dirname "$0")/.." && pwd)}/src/astronereus/_api.py"
if [ ! -f "$CLIENT_API_FILE" ]; then
  echo "no client at $CLIENT_API_FILE -- set NEREUS_PY to the nereus-py tree" >&2
  exit 1
fi
CLIENT_API=$(sed -n 's/^PY_API_VERSION = \([0-9]*\)$/\1/p' "$CLIENT_API_FILE")
if [ -z "$CLIENT_API" ]; then
  echo "could not read PY_API_VERSION from $CLIENT_API_FILE" >&2; exit 1
fi
if [ "$BUNDLE_API" != "$CLIENT_API" ]; then
  echo "contract mismatch: bundle Nereus.PY_API_VERSION=$BUNDLE_API but" >&2
  echo "astronereus PY_API_VERSION=$CLIENT_API. Bump both or rebuild from a" >&2
  echo "commit that matches the client." >&2
  exit 1
fi
echo "contract OK: api v$BUNDLE_API on both sides"

echo "==> recording provenance"
# Reconstructing which commit a bundle came from has already cost a day. The
# two shipped 0.2.0 bundles were built ~40 min apart from different commits and
# neither matched the comment in _runtime.py.
{ echo "commit:      $(git -C "$NEREUS_JL" rev-parse HEAD)"
  echo "describe:    $(git -C "$NEREUS_JL" describe --always --dirty 2>/dev/null)"
  echo "julia:       $JULIA_VER"
  echo "platform:    $PLAT"
  echo "cpu_target:  $CPUT"
  # The Python<->Julia contract version, from the contract check above. Lets a
  # bundle be checked for skew without booting a daemon; the live check is
  # `ping`'s "api" field.
  echo "py_api:      $BUNDLE_API"
  echo "built_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # --untracked-files=no to match the DIRTY guard above. Without it the two
  # disagree: the guard passes a build (untracked files never enter `git
  # archive`) and then BUILD_INFO reports the tree as dirty anyway. That fired
  # for real on the v0.4.0 container build, which listed `?? .claude/` because
  # a container has no access to the host user's global gitignore.
  echo "dirty_files:"; git -C "$NEREUS_JL" status --porcelain --untracked-files=no || true
} > "$BUNDLE/depot/dev/Nereus/BUILD_INFO.txt"
cat "$BUNDLE/depot/dev/Nereus/BUILD_INFO.txt"

echo "==> pruning"
# Registries and downloaded tarballs are build inputs, not runtime needs.
rm -rf "$BUNDLE/depot/registries" "$BUNDLE/depot/clones" "$BUNDLE/depot/scratchspaces"
find "$BUNDLE/depot/packages" -name 'test' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE/depot/packages" -name 'docs' -type d -prune -exec rm -rf {} + 2>/dev/null || true

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/nereus-runtime-${JULIA_VER}-${PLAT}.tar.zst"
echo "==> compressing to $TARBALL"
# COPYFILE_DISABLE: without it macOS tar writes an AppleDouble "._name"
# sidecar for every file carrying an extended attribute -- HALF the members of
# a macOS bundle. macOS tar folds them back into xattrs on extract, so they are
# invisible to `tar -tf` and to any check run on a Mac, but Python's tarfile
# writes them as real files. Makie then globs its icon dir, hands PNGFiles a
# "._icon-128.png", and dies -- taking CairoMakie, PairPlots and every Makie
# extension with it, as six unrelated-looking precompile failures.
# --no-mac-metadata is belt and braces on newer bsdtar.
COPYFILE_DISABLE=1 tar -C "$BUNDLE" --no-mac-metadata -cf - . 2>/dev/null \
  | zstd -19 -T0 -o "$TARBALL" -f \
  || COPYFILE_DISABLE=1 tar -C "$BUNDLE" -cf - . | zstd -19 -T0 -o "$TARBALL" -f

# shasum is macOS, sha256sum is GNU. The bundle has to be buildable on both —
# that is the whole point of building per platform.
echo "==> verifying no AppleDouble sidecars survived"
# Needs a reader that reports tar HEADERS.
#   - macOS bsdtar folds "._*" entries back into xattrs on read, so `tar -tf`
#     reports zero even when half the archive is AppleDouble. That false
#     negative is how a bundle with 29,767 of them shipped.
#   - grepping the decompressed stream matched file CONTENTS too, and that
#     false POSITIVE blocked a perfectly good linux-aarch64 build over 9 hits
#     that were not tar entries at all.
# python3's tarfile is header-accurate on every platform. Fall back to the
# grep only if there is no python3, and say so rather than pretending.
if command -v python3 >/dev/null; then
  AD=$(zstd -dc "$TARBALL" | python3 -c '
import sys, tarfile
n = 0
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as t:
    for m in t:
        if m.name.rsplit("/", 1)[-1].startswith("._"):
            n += 1
print(n)')
else
  echo "    (no python3: falling back to a stream grep, which can false-positive)" >&2
  AD=$(zstd -dc "$TARBALL" | LC_ALL=C grep -a -c '/\._' || true)
fi
if [ "${AD:-0}" -ne 0 ]; then
  echo "refusing to ship: $AD AppleDouble (._*) entries in the tarball." >&2
  echo "They break Python-side extraction; see _no_appledouble in _runtime.py." >&2
  exit 1
fi
echo "    clean"

if command -v sha256sum >/dev/null; then
  SHA=$(sha256sum "$TARBALL" | cut -d' ' -f1)
else
  SHA=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
fi
SIZE=$(du -h "$TARBALL" | cut -f1)
echo
echo "built : $TARBALL"
echo "size  : $SIZE"
echo "sha256: $SHA"
echo
echo "Add to nereus/_runtime.py BUNDLES:"
echo "    \"${PLAT}\": (\"<url>/$(basename "$TARBALL")\", \"$SHA\"),"
