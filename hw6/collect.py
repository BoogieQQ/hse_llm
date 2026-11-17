import csv
import json
from pathlib import Path


root = Path(__file__).parent
rows = []

for path in sorted((root / "results").glob("*/profile/profile_export_aiperf.json")):
    data = json.loads(path.read_text())
    latency = data["request_latency"]
    rows.append({
        "name": path.parents[1].name,
        "requests": int(data["request_count"]["avg"]),
        "duration_s": round(data["benchmark_duration"]["avg"], 1),
        "mean_s": round(latency["avg"] / 1000, 3),
        "p90_s": round(latency["p90"] / 1000, 3),
        "throughput_rpm": round(data["request_throughput"]["avg"] * 60, 2),
        "output_tokens": round(data["output_token_count"]["avg"], 1),
        "errors": sum(item.get("count", 0) for item in data.get("error_summary", [])),
    })

with (root / "metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(row)
