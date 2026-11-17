#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

name="$1"
shift

docker rm -f hw6-vllm >/dev/null 2>&1 || true
docker run -d --name hw6-vllm --gpus device=0 --network host --ipc=host \
  -v "$PWD:/homework" -e HF_HOME=/homework/model_cache \
  --entrypoint bash vllm/vllm-openai:v0.22.0-cu129 /homework/serve.sh "$@" >/dev/null

for i in {1..120}; do
  if curl -fsS --max-time 2 http://localhost:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS http://localhost:8000/health >/dev/null

mkdir -p logs
docker run --rm --name "hw6-bench-$name" --network host \
  -v "$PWD:/homework" -e HF_HOME=/homework/bench_cache \
  hw6-benchmark bash /homework/benchmark.sh "results/$name" 2>&1 | tee "logs/$name.log"
