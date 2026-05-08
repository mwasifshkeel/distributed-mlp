#!/usr/bin/env bash
set -euo pipefail

PULL_EVERY=${PULL_EVERY:-10}
COMPRESS_GRADIENTS=true PULL_EVERY=${PULL_EVERY} ./run.sh "$@"
