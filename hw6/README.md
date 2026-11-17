# HW6: инференс Qwen3-8B

- Модель: `Qwen/Qwen3-8B`, vLLM 0.22.0, одна A100 80 GB — GPU 0
- Замеры: AIPerf 0.13.0, запросы из [архива задания](https://drive.google.com/file/d/1yb_6uAY-ydYOMGM3dq_HpbHekgFJbXMD/view) без изменений
- Во всех запросах отключен reasoning, `temperature=0`, `seed=42`, `max_tokens=512`

## Запуск

```bash
docker build -t hw6-benchmark .
bash run_config.sh baseline
bash run_fixed.sh
bash run_spec_study.sh --max-num-batched-tokens 4096
bash run_curve.sh 30 curve_spec_30
bash run_config.sh plain_sla --max-num-batched-tokens 4096
bash run_curve.sh 30 curve_plain_30
bash run_sla.sh 160 180 200
bash run_load.sh 190 sla_190
python3 collect.py
```

`run_config.sh` перезапускает сервер с выбранными параметрами, затем запускает исходный замер при 120 RPM. Все результаты складываются в отдельных папках `results/<name>/`

`run_spec_study.sh` сравнивает ту же конфигурацию с `ngram_gpu` и без него на 60, 120, 180 и 240 RPM

`run_sla.sh` держит каждый RPM 300 секунд, затем записывает отдельный результат
