## 1. Скачать данные и токенизатор
Необходимо скачать 21 Parquet-файл конфигурации `'wikimedia/wikipedia/20231101.ru` и сложить в папку `./data`
Также предварительно необходимо скачать токенизатор `ai-forever/rugpt3small_based_on_gpt2` и слржить его в `data/hf`.

## 2. Подготовить окружение обучения

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Запуск скрипта

### Подготовка данных
```bash
python solution.py prepare --num-proc 8
```

### Запуск одного обучения
```bash
python solution.py train \
  --name compiled_repeat \
  --batch-size 16 \
  --accumulation 1 \
  --lr 3e-4 \
  --scheduler constant \
  --optim adamw_torch_fused \
  --dtype bf16 \
  --compile
```

### Запуск ВСЕХ экспериментов из experiments.py
```bash
python run_experiments.py
```