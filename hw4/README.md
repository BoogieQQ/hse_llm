# HW4: Full FT и LoRA на SAMSum

## Данные

Используется [SAMSum](https://huggingface.co/datasets/knkarthick/samsum): диалоги и их краткие пересказы. Датасет загружается из Hugging Face 

Модель — `Qwen/Qwen2.5-1.5B-Instruct`.


## Подготовка среды и подготовка данных
```bash
python -m pip install torch transformers datasets peft accelerate rouge-score
python train.py prepare
```


## Запуск

Всех экспериментов:
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py
```

Для одного эксперимента:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py lora_base
```

