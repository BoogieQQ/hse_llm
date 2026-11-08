# HW5: RL для вызова функций

- Модель: `Qwen/Qwen3.5-2B`
- Данные: `Team-ACE/ToolACE`
- Обучение: GRPO и Dr.GRPO с LoRA на одной A100 80 GB, GPU 0
- Результаты и графики: [REPORT.md](REPORT.md)

## Подготовка

`Dockerfile`, `evaluate.py` и `bfcl.jsonl` взяты из задания

```bash
docker build -t hw5 .
docker run --rm -it --gpus device=0 --ipc=host -v "$PWD":/workspace -v hw5-cache:/cache hw5
mkdir -p data
hf download Team-ACE/ToolACE data.json --repo-type dataset --revision 6bda777c88d21e5a204703c1ee45597a8fa4f734 --local-dir data
python3 prepare.py
```

`prepare.py` создает `data/train.jsonl` и `data/validation.jsonl` сплиты с seed 42

## Запуск

```bash
bash run_experiments.sh
```

Скрипт по очереди запускает бейзлайн и три эксперимента

Для одного варианта: `python train.py rl_exact`, затем `python evaluate.py checkpoints/rl_exact --output results/rl_exact/bfcl`

