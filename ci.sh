#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

run_fmt() {
    echo "==> cargo fmt --check"
    cargo fmt --check
}

run_clippy() {
    echo "==> cargo clippy --all-targets --all-features -- -D warnings"
    cargo clippy --all-targets --all-features -- -D warnings
}

run_test() {
    echo "==> cargo test"
    cargo test
}

run_build() {
    echo "==> cargo build --release"
    cargo build --release
}

usage() {
    echo "usage: $0 [fmt|clippy|lint|test|build|all]" >&2
    exit 2
}

case "${1:-all}" in
fmt)
    run_fmt
    ;;
clippy)
    run_clippy
    ;;
lint)
    run_fmt
    run_clippy
    ;;
test)
    run_test
    ;;
build)
    run_build
    ;;
all)
    run_fmt
    run_clippy
    run_test
    run_build
    ;;
*)
    usage
    ;;
esac
