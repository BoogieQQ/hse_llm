import ast
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SEED = 42
MAX_PROMPT_LENGTH = 2048


def split_calls(text):
    parts = []
    start = 0
    depth = 0
    quote = ""
    escaped = False

    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1

    parts.append(text[start:].strip())
    return [part for part in parts if part]


def parse_answer(answer):
    answer = answer.strip()
    if not answer.startswith("["):
        return []
    if not answer.endswith("]"):
        return None

    calls = []
    for part in split_calls(answer[1:-1]):
        if "(" not in part or not part.endswith(")"):
            return None
        name, arguments = part.split("(", 1)
        try:
            expression = ast.parse("f(" + arguments, mode="eval").body
            if expression.args:
                return None
            values = {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
        except Exception:
            return None
        calls.append({"name": name.strip(), "arguments": values})

    return calls


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    result = {key: normalize_schema(item) for key, item in value.items()}
    if result.get("type") in ("dict", "float", "tuple", "any"):
        result["type"] = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}[result["type"]]
    if "required" in result and result["required"] is None:
        result["required"] = []
    return result


def read_tools(system):
    try:
        functions = json.JSONDecoder().raw_decode(system[system.index("["):])[0]
    except Exception:
        return None
    return [{"type": "function", "function": normalize_schema(item)} for item in functions]


def percentile(values, fraction):
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)]


def main():
    root = Path(__file__).parent
    rows = json.loads((root / "data/data.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    indices = list(range(len(rows)))
    random.Random(SEED).shuffle(indices)
    validation_ids = set(indices[:len(indices) // 10])

    prepared = {"train": [], "validation": []}
    skipped = Counter()
    prompt_lengths = []
    answer_lengths = []
    tool_counts = []

    for index, row in enumerate(rows):
        tools = read_tools(row["system"])
        if not tools:
            skipped["tools"] += 1
            continue

        answer = next(message["value"] for message in row["conversations"] if message["from"] == "assistant")
        calls = parse_answer(answer)
        if calls is None or any(call["name"] not in {tool["function"]["name"] for tool in tools} for call in calls):
            skipped["answer"] += 1
            continue

        question = row["conversations"][0]["value"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length > MAX_PROMPT_LENGTH:
            skipped["length"] += 1
            continue

        split = "validation" if index in validation_ids else "train"
        prepared[split].append({
            "dialogue_id": index,
            "prompt": prompt,
            "reference_calls": json.dumps(calls, ensure_ascii=False),
            "answer": answer,
            "is_call": bool(calls),
        })
        prompt_lengths.append(length)
        answer_lengths.append(len(tokenizer(answer, add_special_tokens=False)["input_ids"]))
        tool_counts.append(len(tools))

    for split, examples in prepared.items():
        with (root / "data" / f"{split}.jsonl").open("w") as file:
            for example in examples:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary = {
        "source": "Team-ACE/ToolACE",
        "dataset_revision": DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "original_dialogues": len(rows),
        "train": len(prepared["train"]),
        "validation": len(prepared["validation"]),
        "skipped": dict(skipped),
        "call_examples": sum(example["is_call"] for examples in prepared.values() for example in examples),
        "no_call_examples": sum(not example["is_call"] for examples in prepared.values() for example in examples),
        "prompt_tokens_p50_p90_p99": [percentile(prompt_lengths, value) for value in (0.5, 0.9, 0.99)],
        "answer_tokens_p50_p90_p99": [percentile(answer_lengths, value) for value in (0.5, 0.9, 0.99)],
        "tools_p50_p90_p99": [percentile(tool_counts, value) for value in (0.5, 0.9, 0.99)],
        "examples": {
            split: [
                {"question": rows[item["dialogue_id"]]["conversations"][0]["value"], "answer": item["answer"]}
                for item in examples[:3]
            ]
            for split, examples in prepared.items()
        },
    }
    (root / "data/eda.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
