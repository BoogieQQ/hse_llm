set -e
cd "$(dirname "$0")"
mkdir -p logs

if [ ! -f results/base/metrics.json ]; then
    python evaluate.py Qwen/Qwen3.5-2B --output results/base > logs/base.log 2>&1
fi

for name in rl_exact rl_partial rl_drgrpo; do
    if [ ! -f "results/$name/summary.json" ]; then
        python train.py "$name" > "logs/$name.log" 2>&1
    fi
    if [ ! -f "results/$name/bfcl/metrics.json" ]; then
        python evaluate.py "checkpoints/$name" --output "results/$name/bfcl" > "logs/${name}_eval.log" 2>&1
    fi
done
