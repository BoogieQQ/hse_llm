import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from reward import make_reward


MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SEED = 42
RUNS = {
    "rl_exact": {"reward": "exact", "loss_type": "grpo", "scale_rewards": "group"},
    "rl_partial": {"reward": "partial", "loss_type": "grpo", "scale_rewards": "group"},
    "rl_drgrpo": {"reward": "partial", "loss_type": "dr_grpo", "scale_rewards": "none"},
}


def export_model(model, tokenizer, checkpoint):
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=RUNS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent
    name = "smoke" if args.smoke else args.name
    output = root / "results" / name
    checkpoint = root / "checkpoints" / name
    output.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, revision=MODEL_REVISION, dtype="bfloat16", device_map="cpu"
        )
        model = PeftModel.from_pretrained(model, output / "adapter").merge_and_unload()
        export_model(model, tokenizer, checkpoint)
        history = json.loads((output / "log_history.json").read_text())
        summary = {
            "name": args.name,
            "selected_step": history[-1]["step"],
            "training_seconds": history[-1]["train_runtime"],
            "peak_memory_gib": None,
            "train_metrics": history[-1],
            "validation_metrics": None,
            "checkpoint": str(checkpoint),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    data = load_dataset("json", data_files={
        "train": str(root / "data/train.jsonl"),
        "validation": str(root / "data/validation.jsonl"),
    })
    validation = data["validation"].shuffle(seed=SEED).select(range(128))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"

    run = RUNS[args.name]
    config = GRPOConfig(
        output_dir=str(output),
        max_steps=2 if args.smoke else 120,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_completion_length=192,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        beta=0.01,
        epsilon=0.2,
        temperature=1.3,
        loss_type=run["loss_type"],
        scale_rewards=run["scale_rewards"],
        bf16=True,
        gradient_checkpointing=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=2304,
        model_init_kwargs={"revision": MODEL_REVISION, "dtype": "bfloat16", "attn_implementation": "sdpa"},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=MODEL,
        args=config,
        processing_class=tokenizer,
        reward_funcs=make_reward(run["reward"]),
        train_dataset=data["train"],
        eval_dataset=validation,
        peft_config=lora,
    )

    (output / "config.json").write_text(json.dumps({
        "name": args.name,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "reward": run["reward"],
        "loss_type": run["loss_type"],
        "scale_rewards": run["scale_rewards"],
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "num_generations": config.num_generations,
        "max_prompt_length": 2048,
        "max_completion_length": config.max_completion_length,
        "batch_size": config.per_device_train_batch_size,
        "beta": config.beta,
        "temperature": config.temperature,
        "bf16": config.bf16,
        "use_vllm": config.use_vllm,
        "lora_r": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_target_modules": sorted(lora.target_modules),
        "seed": SEED,
    }, ensure_ascii=False, indent=2) + "\n")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_result = trainer.train()
    training_seconds = time.monotonic() - started
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n")

    if args.smoke:
        return

    trainer.model.save_pretrained(output / "adapter")
    validation_metrics = trainer.evaluate()
    model = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
    export_model(model, tokenizer, checkpoint)

    summary = {
        "name": args.name,
        "selected_step": trainer.state.global_step,
        "training_seconds": training_seconds,
        "peak_memory_gib": peak_memory_gib,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "checkpoint": str(checkpoint),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
