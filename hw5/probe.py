import json
import random
from pathlib import Path

from vllm import LLM, SamplingParams

from reward import exact_score, parse_completion, partial_score


MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def main():
    root = Path(__file__).parent
    groups = {"none": [], "one": [], "many": []}
    for line in (root / "data/train.jsonl").open():
        row = json.loads(line)
        count = len(json.loads(row["reference_calls"]))
        key = "none" if count == 0 else "one" if count == 1 else "many"
        groups[key].append(row)

    randomizer = random.Random(42)
    rows = [item for group in groups.values() for item in randomizer.sample(group, 10)]
    model = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="bfloat16",
        max_model_len=2304,
        language_model_only=True,
        enforce_eager=True,
    )
    result = []
    for temperature in (1.3, 1.6):
        generations = model.generate(
            [row["prompt"] for row in rows],
            SamplingParams(n=8, temperature=temperature, max_tokens=192, seed=42),
        )
        for row, generation in zip(rows, generations):
            expected = json.loads(row["reference_calls"])
            predictions = [parse_completion(item.text) for item in generation.outputs]
            exact = [exact_score(item, expected) for item in predictions]
            partial = [partial_score(item, expected) for item in predictions]
            result.append({
                "temperature": temperature,
                "dialogue_id": row["dialogue_id"],
                "calls": len(expected),
                "exact": exact,
                "partial": partial,
            })

    (root / "results/probe.json").write_text(json.dumps(result, indent=2) + "\n")
    for temperature in (1.3, 1.6):
        for count in (0, 1, 2):
            subset = [row for row in result if row["temperature"] == temperature and (0 if row["calls"] == 0 else 1 if row["calls"] == 1 else 2) == count]
            print(temperature, count, "exact variation", sum(len(set(row["exact"])) > 1 for row in subset), "partial variation", sum(len(set(row["partial"])) > 1 for row in subset))


if __name__ == "__main__":
    main()
