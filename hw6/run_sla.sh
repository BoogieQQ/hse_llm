#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for rpm in "$@"; do
  bash run_load.sh "$rpm" "sla_$rpm"
done
