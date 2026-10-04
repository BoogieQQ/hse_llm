import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from datasets import Features, Sequence, Value, load_dataset
from transformers import (AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Trainer,
                          TrainingArguments, TrainerCallback, default_data_collator, set_seed)

MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
NUM_SHARDS = 32

TOKENIZER_NAME = 'ai-forever/rugpt3small_based_on_gpt2'
OUTPUT_DIR = './data/tokenized'

VALIDATION_SIZE = 5000
SEED = 42

MODEL_CONFIG = {
    'hidden_size': 2048, 'num_hidden_layers': 12,
    'num_attention_heads': 16, 'num_key_value_heads': 8,
    'intermediate_size': 8192, 'head_dim': 128, 'hidden_act': 'silu',
    'initializer_range': 0.02, 'scale_attn_weights': True, 'use_cache': True,
}

PROMPTS = ['Москва — это', 'Искусственный интеллект — это',
           'В 1961 году', 'Вода состоит из']

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer

def tokenize_function(examples, tokenizer):
    out = tokenizer(examples['text'], truncation=True, padding='max_length',
                    max_length=MAX_LENGTH, return_token_type_ids=False)

    out['labels'] = [[tid if mask else -100 for tid, mask in zip(ids, masks)]
                     for ids, masks in zip(out['input_ids'], out['attention_mask'])]
    return out


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_shards):

        ds.shard(num_shards=num_shards, index=i, contiguous=True).to_parquet(
            root / f'{i:05d}.parquet')

def prepare_dataset(num_proc=8):    
    tokenizer = prepare_tokenizer()
    dataset = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train')
    
    
    features = Features({'input_ids': Sequence(Value('int32'), length=MAX_LENGTH),
                         'attention_mask': Sequence(Value('int8'), length=MAX_LENGTH),
                         'labels': Sequence(Value('int32'), length=MAX_LENGTH)})
    
    tokenized = dataset.map(tokenize_function, batched=True, batch_size=256,
                            num_proc=num_proc, fn_kwargs={'tokenizer': tokenizer},
                            remove_columns=dataset.column_names, features=features,
                            desc='Tokenizing Russian Wikipedia')
    
    save_as_parquets(tokenized)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted([str(p) for p in Path(data_dir).glob('*.parquet')])
    return load_dataset('parquet', data_files=files, split='train')

def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    return dataset.select(range(validation_size, len(dataset))), dataset.select(range(validation_size))

def create_model(tokenizer):
    config = Qwen3Config(vocab_size=tokenizer.vocab_size,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, **MODEL_CONFIG)
    
    model = Qwen3ForCausalLM._from_config(config, attn_implementation='flash_attention_2',
                                        torch_dtype=torch.bfloat16)
    
    return model

class TimeoutCallback(TrainerCallback):
    def __init__(self, timeout_seconds=MAX_TRAINING_TIME_SECONDS):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.elapsed = None
        self.step_seconds = []
        self.last_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.start_time = time.monotonic()
        self.last_step = self.start_time

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.monotonic()
        self.step_seconds.append(now - self.last_step)
        self.last_step = now
        self.elapsed = now - self.start_time
        if self.elapsed >= self.timeout_seconds:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.start_time is not None:
            logs['elapsed_seconds'] = round(time.monotonic() - self.start_time, 3)


class JsonLogCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = dict(logs or {}, step=state.global_step)
        with self.path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')

def environment():
    import importlib.metadata as md
    return {'python': platform.python_version(),
            'versions': {p: md.version(p) for p in ('torch', 'transformers', 'datasets',
                                                  'accelerate', 'flash-attn')},
            'gpu': torch.cuda.get_device_name(0),
            'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'gpu_count': torch.cuda.device_count(), 'torch_cuda': torch.version.cuda,
            'seed': SEED, 'model_config': MODEL_CONFIG}


def make_batch(dataset, start, size, device='cuda'):
    return {k: torch.tensor(v, dtype=torch.long, device=device)
            for k, v in dataset[start:start+size].items()}

@torch.inference_mode()
def evaluate_token_weighted(model, dataset, batch_size=8, dtype=torch.bfloat16):
    model.eval()

    total_nll, total_tokens = 0., 0
    for i in range(0, len(dataset), batch_size):
        batch = make_batch(dataset, i, min(batch_size, len(dataset) - i))
        count = int((batch['labels'][:, 1:] != -100).sum())
        
        with torch.autocast('cuda', dtype=dtype):
            loss = model(**batch).loss

        total_nll += float(loss) * count
        total_tokens += count

    value = total_nll / total_tokens

    return {'token_weighted_eval_loss': value, 
            'token_weighted_perplexity': math.exp(value),
            'eval_predicted_tokens': total_tokens}


@torch.inference_mode()
def generate_examples(model, tokenizer, dtype=torch.bfloat16):
    model.eval()

    rows = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        for sampled in (False, True):
            set_seed(SEED)
            kwargs = {'temperature': 0.8, 'top_p': 0.9, 'top_k': 50} if sampled else {}

            with torch.autocast('cuda', dtype=dtype):
                outputs = model.generate(**inputs, 
                                         max_new_tokens=96, 
                                         do_sample=sampled,
                                         repetition_penalty=1.1,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id, 
                                         **kwargs)
                
            rows.append({'prompt': prompt, 'mode': 'sample' if sampled else 'greedy',
                         'text': tokenizer.decode(outputs[0], skip_special_tokens=True)})
    return rows

def train_model(args):
    set_seed(SEED)    

    train, evaluation = split_dataset(load_tokenized_dataset())

    tokenizer = prepare_tokenizer()

    model = create_model(tokenizer)

    if args.dtype == 'fp16':
        model = model.float()
    
    model = model.cuda()

    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    out = Path('results') / args.name

    out.mkdir(parents=True)
    timer = TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)

    config = dict(output_dir=str(out), optim=args.optim, num_train_epochs=1,
                  per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=8,
                  gradient_accumulation_steps=args.accumulation, learning_rate=args.lr,
                  weight_decay=.01, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
                  max_grad_norm=1., logging_steps=5, logging_first_step=True,
                  save_strategy='no', eval_strategy='no', load_best_model_at_end=False,
                  bf16=args.dtype == 'bf16', fp16=args.dtype == 'fp16',
                  tf32=args.tf32, gradient_checkpointing=False,
                  dataloader_num_workers=4, dataloader_pin_memory=True,
                  torch_compile=args.compile, report_to='none', seed=SEED, data_seed=SEED,
                  disable_tqdm=True, remove_unused_columns=False, logging_nan_inf_filter=False,
                  dataloader_drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01,
                                  fused=args.optim == 'adamw_torch_fused')
    def lr_factor():
        if timer.start_time is None:
            return 0.01

        elapsed = time.monotonic() - timer.start_time
        if elapsed < 30:
            return max(0.01, elapsed / 30)
        
        progress = min(1.0, (elapsed - 30.0) / (MAX_TRAINING_TIME_SECONDS - 30.0))

        if args.scheduler == 'linear':
            return 1.0 - 0.9 * progress
        if args.scheduler == 'cosine':
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    training_args = TrainingArguments(**config)

    trainer = Trainer(model=model, args=training_args, train_dataset=train,
                      eval_dataset=evaluation, data_collator=default_data_collator,
                      processing_class=tokenizer, optimizers=(optimizer, scheduler),
                      callbacks=[timer, JsonLogCallback(out / 'metrics.jsonl')])
    
    write_json(out / 'experiment.json', dict(vars(args), budget_seconds=900,
        schedule_description='wall-clock; 30s warmup; decay to 0.1 for cosine/linear',
        training_arguments=config,
        parameter_count=sum(p.numel() for p in model.parameters())))
    
    if args.initial:
        initial = trainer.evaluate(metric_key_prefix='initial_eval')
        write_json(out / 'initial_eval.json', initial)
        write_json(out / 'initial_generations.json', generate_examples(model, tokenizer, dtype=dtype))

    trainer.train()
    elapsed = timer.elapsed
    training_peak = torch.cuda.max_memory_allocated() / 2**30
    trainer.save_state()

    trainer.save_model(str(out / 'checkpoint-final'))
    tokenizer.save_pretrained(out / 'checkpoint-final')
    
    print('FINAL_EVALUATION', flush=True)
    metrics = trainer.evaluate()
    metrics.update(evaluate_token_weighted(model, evaluation, dtype=dtype))
    metrics.update(perplexity=math.exp(metrics['eval_loss']),
                   training_seconds=elapsed, global_step=trainer.state.global_step,
                   train_examples=trainer.state.global_step * args.batch_size * args.accumulation,
                   processed_token_positions=trainer.state.global_step * args.batch_size * args.accumulation * MAX_LENGTH,
                   peak_training_memory_gb=training_peak,
                   last_step_seconds=timer.step_seconds[-1],
                   overrun_seconds=elapsed-MAX_TRAINING_TIME_SECONDS)
    write_json(out / 'final_metrics.json', metrics)
    write_json(out / 'generations.json', generate_examples(model, tokenizer, dtype=dtype))
    trainer.save_state()
    print('RUN_COMPLETE ' + json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--num-proc', type=int, default=8)
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler', choices=['constant', 'cosine', 'linear'], default='cosine')
    parser.add_argument('--optim', choices=['adamw_torch', 'adamw_torch_fused'], default='adamw_torch_fused')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false')
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_dataset(args.num_proc)
    else:
        train_model(args)


if __name__ == '__main__':
    main()
