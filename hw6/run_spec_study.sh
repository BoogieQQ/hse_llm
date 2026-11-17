#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bash run_config.sh plain_best "$@"
for rpm in 60 120 180 240; do
  bash run_curve.sh "$rpm" "curve_plain_$rpm"
done

bash run_config.sh spec_ngram_gpu "$@" \
  --speculative-config '{"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
for rpm in 60 120 180 240; do
  bash run_curve.sh "$rpm" "curve_spec_$rpm"
done
