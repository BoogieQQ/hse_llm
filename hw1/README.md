## 1. Скачать данные и токенизатор
Необходимо скачать 21 Parquet-файл конфигурации `'wikimedia/wikipedia/20231101.ru` и сложить в папку `./data`
Также предварительно необходимо скачать токенизатор `ai-forever/rugpt3small_based_on_gpt2` и слржить его в `data/hf`.

## 2. Подготовить окружение обучения

Далее нужен Linux с NVIDIA GPU, Python 3.10 и совместимым драйвером CUDA. Использованные версии: PyTorch 2.6.0 с CUDA 12.4 и FlashAttention 2.7.3. Установка пакетов требует интернета либо заранее подготовленного зеркала/набора wheels.

```bash
python3.10 -m venv .venv
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