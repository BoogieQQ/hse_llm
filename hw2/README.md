# HW2: ускорение обучения Qwen3

## 1. Данные и окружение в

Перед запусками понадобятся `wikimedia/wikipedia`, `20231101.ru` и токенизатор `ai-forever/rugpt3small_based_on_gpt2`

Функция `prepare_data()` загружает их через Hugging Face

Команда для подготовки данных:
```bash
python -c 'import train; train.prepare_data()'
```

Команда создаст `data/dataset` с подготовленными train/validation и `data/tokenizer`

Для повторения окружения надо скачать [Dockerfile из задания](https://drive.google.com/file/d/1X6CIPgI8owvjXR2-DH-D40lD1jzEkflz/view?usp=sharing) в корень этой папки под именем `Dockerfile` и собрать окружение

## 2. Обучение

Для запуска экспериментов нужны только `train.py`/`run_experiments.py`.

`run_experiments.py` последовательно запускает baseline и все эксперименты из `experiments.py`:
```bash
python run_experiments.py
```

`train.py` запускает "лучший" сетап по итогам экспериментов
```bash
python train.py
```

## 3. Кольцевые операции

`ring_reduce_scatter`, `ring_all_gather` и `ring_all_reduce` реализованы и проверены на работоспособность в `communications.ipynb`
