#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bash run_config.sh chunk4096 --max-num-batched-tokens 4096
bash run_config.sh chunk8192 --max-num-batched-tokens 8192
bash run_config.sh seq32_chunk8192 --max-num-seqs 32 --max-num-batched-tokens 8192
bash run_config.sh eager --enforce-eager
