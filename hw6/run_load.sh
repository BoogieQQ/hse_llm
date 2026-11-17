#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rpm="$1"
name="$2"
duration="${3:-300}"

mkdir -p logs
docker run --rm --name "hw6-load-$name" --network host \
  -v "$PWD:/homework" -e HF_HOME=/homework/bench_cache -e DURATION="$duration" \
  hw6-benchmark bash /homework/load_test.sh "$rpm" "results/$name" 2>&1 | tee "logs/$name.log"
