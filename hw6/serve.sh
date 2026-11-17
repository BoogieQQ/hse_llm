#!/usr/bin/env bash
set -euo pipefail

export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B "$@"
