import json
import sys
import time
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parent

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_REVISION = "6b929ff10edec703164e3ddb2e94aae058c9ab5f"

PROMPT = "Summarize this conversation in one concise sentence. Use the speakers' names and only facts from the dialogue.\n\n"
SEED = 42
MAX_DIALOGUE_TOKENS = 360
MAX_SUMMARY_TOKENS = 80

BASE_LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
EXPERIMENTS = {
    "base": None,
    "full": None,
    "lora_base": BASE_LORA,
    "lora_r8": {**BASE_LORA, "r": 8},
    "lora_alpha16": {**BASE_LORA, "lora_alpha": 16},
    "lora_dropout0": {**BASE_LORA, "lora_dropout": 0.0},
    "lora_all_attention": {**BASE_LORA, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}


def tokenizer_for_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_for(dialogue, tokenizer):
    tokens = tokenizer(dialogue, add_special_tokens=False).input_ids[:MAX_DIALOGUE_TOKENS]
    text = PROMPT + tokenizer.decode(tokens)
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def encode(row, tokenizer):
    prompt = tokenizer(prompt_for(row["dialogue"], tokenizer), add_special_tokens=False).input_ids
    answer = tokenizer(row["summary"], add_special_tokens=False).input_ids[:MAX_SUMMARY_TOKENS]
    answer += [tokenizer.eos_token_id]
    return {
        "input_ids": prompt + answer,
        "attention_mask": [1] * (len(prompt) + len(answer)),
        "labels": [-100] * len(prompt) + answer,
    }


def prepare_data():
    data = load_dataset("knkarthick/samsum", revision=DATASET_REVISION)

    (ROOT / "data").mkdir(exist_ok=True)

    data.save_to_disk(ROOT / "data/samsum")

    tokenizer = tokenizer_for_model()
    encoded = DatasetDict({
        split: data[split].map(lambda row: encode(row, tokenizer), remove_columns=data[split].column_names)
        for split in ("train", "test")
    })

    encoded.save_to_disk(ROOT / "data/encoded")
    
    print({split: len(data[split]) for split in data}, flush=True)


def model_for(name):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if name.startswith("lora_"):
        config = LoraConfig(task_type="CAUSAL_LM", bias="none", **EXPERIMENTS[name])
        model = get_peft_model(model, config)
    return model.cuda()


def arguments(name, learning_rate):
    return TrainingArguments(
        output_dir=str(ROOT / "results" / name / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        disable_tqdm=True,
    )


def generate_summaries(model, tokenizer, test, output):
    tokenizer.padding_side = "left"
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rougeL": []}

    model.eval()
    with output.open("w") as file:
        for start in range(0, len(test), 8):
            rows = test.select(range(start, min(start + 8, len(test))))
            prompts = [prompt_for(row["dialogue"], tokenizer) for row in rows]
            batch = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                tokens = model.generate(
                    **batch, max_new_tokens=MAX_SUMMARY_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )

            predictions = tokenizer.batch_decode(tokens[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, prediction in zip(rows, predictions):
                prediction = prediction.strip()
                result = scorer.score(row["summary"], prediction)
                for key in scores:
                    scores[key].append(result[key].fmeasure)
                file.write(json.dumps({"id": row["id"], "reference": row["summary"], "prediction": prediction}, ensure_ascii=False) + "\n")
            print(f"Generated {min(start + 8, len(test))}/{len(test)}", flush=True)
        
    return {key: sum(values) / len(values) for key, values in scores.items()}


def run(name):
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    output = ROOT / "results" / name
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = tokenizer_for_model()
    data = load_from_disk(ROOT / "data/encoded")
    raw = load_from_disk(ROOT / "data/samsum")

    model = model_for(name)
    total = sum(parameter.numel() for parameter in model.parameters())
    
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) if name != "base" else 0
    
    learning_rate = 2e-5 if name == "full" else 1e-4
    
    trainer = Trainer(
        model=model,
        args=arguments(name, learning_rate),
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    
    if name != "base":
        trainer.train()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        model.save_pretrained(output / "checkpoint", safe_serialization=True)
    else:
        train_seconds = 0
        peak_allocated = 0
        peak_reserved = 0

    evaluation = trainer.evaluate()
    
    scores = generate_summaries(model, tokenizer, raw["test"], output / "predictions.jsonl")
    
    summary = {
        "name": name,
        "method": "base" if name == "base" else "full" if name == "full" else "lora",
        "lora": EXPERIMENTS[name],
        "train_examples": len(data["train"]),
        "test_examples": len(data["test"]),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_seconds": train_seconds,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "learning_rate": learning_rate if name != "base" else None,
        "optimizer_steps": trainer.state.global_step,
        "test_loss": evaluation["eval_loss"],
        **scores,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare_data()
    else:
        run(sys.argv[1])
