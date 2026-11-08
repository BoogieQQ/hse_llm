import json
import re
from collections import Counter


CALL_PATTERN = re.compile(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_completion(text):
    if isinstance(text, list):
        text = text[-1]["content"]
    if not text.strip():
        return None

    matches = list(CALL_PATTERN.finditer(text))
    if not matches:
        return None if any(tag in text for tag in ("<tool", "<function", "<parameter")) else []

    calls = []
    for match in matches:
        body = match.group(2)
        parameters = list(PARAM_PATTERN.finditer(body))
        if PARAM_PATTERN.sub("", body).strip():
            return None
        arguments = {}
        for parameter in parameters:
            value = parameter.group(2).strip()
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            arguments[parameter.group(1).strip()] = value
        calls.append({"name": match.group(1).strip(), "arguments": arguments})

    if CALL_PATTERN.sub("", text).strip().startswith("<tool"):
        return None
    return calls


def normalize(value):
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def exact_score(predicted, expected):
    if predicted is None:
        return 0.0
    left = sorted(json.dumps(normalize(call), sort_keys=True) for call in predicted)
    right = sorted(json.dumps(normalize(call), sort_keys=True) for call in expected)
    return float(left == right)


def partial_score(predicted, expected):
    if predicted is None:
        return 0.0
    if not expected:
        return float(not predicted)
    if not predicted:
        return 0.0

    predicted_names = Counter(call["name"] for call in predicted)
    expected_names = Counter(call["name"] for call in expected)
    matching_names = sum((predicted_names & expected_names).values())
    name_score = 2 * matching_names / (len(predicted) + len(expected))

    argument_scores = []
    for gold in expected:
        candidates = [call for call in predicted if call["name"] == gold["name"]]
        if not candidates:
            argument_scores.append(0.0)
            continue
        scores = []
        for call in candidates:
            keys = set(call["arguments"]) | set(gold["arguments"])
            scores.append(sum(normalize(call["arguments"].get(key)) == normalize(gold["arguments"].get(key)) for key in keys) / max(1, len(keys)))
        argument_scores.append(max(scores))

    return 0.1 + 0.4 * name_score + 0.5 * sum(argument_scores) / len(expected)


def make_reward(mode):
    scorer = exact_score if mode == "exact" else partial_score

    def reward(completions, reference_calls, **kwargs):
        return [scorer(parse_completion(text), json.loads(gold)) for text, gold in zip(completions, reference_calls)]

    return reward
