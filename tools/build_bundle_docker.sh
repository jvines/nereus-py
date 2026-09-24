#!/usr/bin/env bash
# Build a Linux runtime bundle in a container, for a platform this host can run
# NATIVELY. On Apple Silicon that means linux/arm64: Docker runs it on the host
# CPU, so the pkgimages are real aarch64 code built at full speed.
#
# Do NOT use this for linux/amd64 on an arm64 host. That path goes through qemu,
# which is slow and -- more importantly -- tests qemu rather than the machine a
# user will run on. Build amd64 natively on an x86_64 box.
#
# Usage:
#   tools/build_bundle_docker.sh [platform]     # default: linux/arm64
#
# Env: OUT_DIR (default ~/nereus-bundles), NEREUS_JL, NEREUS_PY, DOCKER.
set -euo pipefail

PLATFORM="${1:-linux/arm64}"
DOCKER="${DOCKER:-$(command -v docker || echo /usr/local/bin/docker)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NEREUS_PY="${NEREUS_PY:-$(cd "$HERE/.." && pwd)}"
NEREUS_JL="${NEREUS_JL:-$(cd "$HERE/../../Nereus.jl" && pwd)}"
OUT_DIR="${OUT_DIR:-$HOME/nereus-bundles}"

[ -x "$DOCKER" ] || { echo "no docker at $DOCKER (set DOCKER=)" >&2; exit 1; }
[ -d "$NEREUS_JL/.git" ] || { echo "NEREUS_JL=$NEREUS_JL is not a git repo" >&2; exit 1; }
[ -f "$NEREUS_PY/tools/build_bundle.sh" ] || { echo "no build_bundle.sh under NEREUS_PY=$NEREUS_PY" >&2; exit 1; }
mkdir -p "$OUT_DIR"

echo "platform : $PLATFORM"
echo "Nereus.jl: $NEREUS_JL ($(git -C "$NEREUS_JL" describe --always --dirty))"
echo "nereus-py: $NEREUS_PY"
echo "out      : $OUT_DIR"

# WORK lives on the container's own filesystem. The precompile writes ~4 GB and
# doing that through a bind mount to macOS is dramatically slower; only the
# finished tarball crosses the mount.
exec "$DOCKER" run --rm --platform "$PLATFORM" \
  -v "$NEREUS_JL":/src/Nereus.jl:ro \
  -v "$NEREUS_PY":/src/nereus-py:ro \
  -v "$OUT_DIR":/out \
  -e OUT_DIR=/out -e NEREUS_JL=/src/Nereus.jl -e NEREUS_PY=/src/nereus-py \
  -e WORK=/work \
  debian:bookworm-slim bash -c '
    set -euo pipefail
    apt-get update -qq
    # build-essential is NOT optional. Expect.jl'"'"'s build step shells out to
    # `make`, and without it Pkg.instantiate dies with
    #   ERROR: Error building `Expect`: could not spawn `make` (ENOENT)
    # A normal Linux dev box has make already, so this only bites in a slim
    # container -- which is exactly why this recipe belongs in the repo.
    apt-get install -y -qq --no-install-recommends \
        git curl ca-certificates zstd python3 xz-utils build-essential >/dev/null
    # The bind-mounted repo is owned by another uid inside the container, so
    # git refuses to touch it without this.
    git config --global --add safe.directory /src/Nereus.jl
    mkdir -p /work
    exec bash /src/nereus-py/tools/build_bundle.sh
  '
