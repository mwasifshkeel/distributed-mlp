#!/usr/bin/env bash
set -euo pipefail

COMPRESS_GRADIENTS=false PULL_EVERY=1 ./run.sh "$@"
