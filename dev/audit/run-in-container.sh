#!/usr/bin/env bash
# Runs the Linux-only audit tests of the `liquid-cache` core crate in Docker.
#
#   dev/audit/run-in-container.sh build          cold build of the test binary
#   dev/audit/run-in-container.sh mem-limit      finding 1, under --memory=1g
#   dev/audit/run-in-container.sh mem-control    finding 1, no memory limit
#   dev/audit/run-in-container.sh vol-tmpfs      finding 2, on a 16 MiB tmpfs
#   dev/audit/run-in-container.sh vol-loop       finding 2, on a 32 MiB loop ext4
#                                                (needs --privileged)
#   dev/audit/run-in-container.sh all            everything, in that order
#   dev/audit/run-in-container.sh shell          a shell in the same container
#
# t4 does its IO through io_uring on Linux, and Docker's default seccomp
# profile refuses io_uring_setup with EPERM, so a t4 mount fails before any
# write. The vol-* modes therefore run with seccomp unconfined. Set
# AUDIT_SECCOMP=default to keep Docker's profile and see that failure.
# AUDIT_DIRECT_IO=0 mounts the store without O_DIRECT.
#
# The cargo registry, git checkouts and target dir live on named volumes, so
# nothing Linux-built lands in the host's target dir and the cold build
# happens once.
set -euo pipefail

IMAGE="${AUDIT_IMAGE:-rust:1.96.0-bookworm}"
WORK="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="${AUDIT_VOLUME_PREFIX:-liquid-cache-audit}"

COMMON=(
  --rm
  -v "$WORK":/work
  -v "$PREFIX-registry":/usr/local/cargo/registry
  -v "$PREFIX-git":/usr/local/cargo/git
  -v "$PREFIX-target":/target
  -e CARGO_TARGET_DIR=/target
  -e RUST_BACKTRACE=1
  -w /work
)

SECCOMP=()
if [ "${AUDIT_SECCOMP:-unconfined}" != default ]; then
  SECCOMP=(--security-opt "seccomp=${AUDIT_SECCOMP:-unconfined}")
fi

# Extra system packages the build turned out to need, if any.
APT_PACKAGES="${AUDIT_APT_PACKAGES:-}"
PRELUDE='rustc --version; uname -r;'
if [ -n "$APT_PACKAGES" ]; then
  PRELUDE="apt-get update -qq && apt-get install -y -qq $APT_PACKAGES >/dev/null; $PRELUDE"
fi

TEST='cargo test -p liquid-cache --lib'
MEM_TEST="$TEST audit_default_memory_respects_cgroup_limit -- --nocapture"
VOL_TEST="$TEST audit_full_volume_returns_cache_full_not_panic -- --nocapture --test-threads=1"

run() {
  local name="$1"
  shift
  echo "=== $name ==="
  docker run --name "$PREFIX-$name-$$" "${COMMON[@]}" "$@"
}

case "${1:-all}" in
  build)
    run build "$IMAGE" bash -c "$PRELUDE time $TEST --no-run"
    ;;
  mem-limit)
    run mem-limit --memory=1g "$IMAGE" bash -c "$PRELUDE $MEM_TEST"
    ;;
  mem-control)
    run mem-control "$IMAGE" bash -c "$PRELUDE $MEM_TEST"
    ;;
  vol-tmpfs)
    run vol-tmpfs ${SECCOMP[@]+"${SECCOMP[@]}"} --tmpfs /small:size=16m -e AUDIT_SMALL_DIR=/small \
      -e AUDIT_DIRECT_IO="${AUDIT_DIRECT_IO:-}" \
      "$IMAGE" bash -c "$PRELUDE df -B1 /small; $VOL_TEST; rc=\$?; df -B1 /small; ls -l /small; exit \$rc"
    ;;
  vol-loop)
    run vol-loop ${SECCOMP[@]+"${SECCOMP[@]}"} --privileged -e AUDIT_SMALL_DIR=/small \
      -e AUDIT_DIRECT_IO="${AUDIT_DIRECT_IO:-}" \
      "$IMAGE" bash -c "$PRELUDE
        truncate -s 32M /tmp/small.img && mkfs.ext4 -q /tmp/small.img &&
        mkdir -p /small && mount -o loop /tmp/small.img /small &&
        df -B1 /small; $VOL_TEST; rc=\$?; df -B1 /small; ls -l /small; exit \$rc"
    ;;
  shell)
    docker run -it "${COMMON[@]}" "$IMAGE" bash
    ;;
  all)
    rc=0
    "$0" build
    "$0" mem-control || rc=1
    "$0" mem-limit || rc=1
    "$0" vol-tmpfs || rc=1
    "$0" vol-loop || rc=1
    exit $rc
    ;;
  *)
    echo "unknown mode: $1" >&2
    exit 2
    ;;
esac
