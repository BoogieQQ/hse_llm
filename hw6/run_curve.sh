#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rpm="$1"
name="$2"

mkdir -p logs
docker run --rm --name "hw6-curve-$name" --network host \
  -v "$PWD:/homework" -e HF_HOME=/homework/bench_cache -e RPM="$rpm" -e COUNT=320 \
  hw6-benchmark bash /homework/benchmark.sh "results/$name" 2>&1 | tee "logs/$name.log"
