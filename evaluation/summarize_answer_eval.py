import csv
import math
from collections import defaultdict
from pathlib import Path

INPUT = Path("evaluation/answer_results.csv")
OUTPUT = Path("evaluation/answer_quality_summary.csv")


def percentile(values, p):
    values = sorted(float(v) for v in values if v not in ("", None))
    if not values:
        return None

    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return values[lower] + (
        values[upper] - values[lower]
    ) * (position - lower)


with INPUT.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

groups = defaultdict(list)

for row in records:
    groups[row["retrieval_setting"]].append(row)

fields = [
    "configuration",
    "requests",
    "successful",
    "correct",
    "wrong",
    "correct_pct",
    "wrong_pct",
    "validator_pass_pct",
    "avg_tokens_in",
    "avg_tokens_out",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "total_p50_ms",
    "total_p95_ms",
]

summary = []

for config, rows in groups.items():
    successful = [r for r in rows if r["status"] == "ok"]

    if not successful:
        continue

    correct = sum(r["answer_grade"] == "correct" for r in successful)
    wrong = sum(r["answer_grade"] == "wrong" for r in successful)
    passed = sum(r["validation_passed"].lower() == "true"
                 for r in successful)

    count = len(successful)

    def average(field):
        values = [
            float(r[field]) for r in successful
            if r[field] not in ("", None)
        ]
        return round(sum(values) / len(values), 2) if values else None

    row = {
        "configuration": config,
        "requests": len(rows),
        "successful": count,
        "correct": correct,
        "wrong": wrong,
        "correct_pct": round(correct / count * 100, 2),
        "wrong_pct": round(wrong / count * 100, 2),
        "validator_pass_pct": round(passed / count * 100, 2),
        "avg_tokens_in": average("tokens_in"),
        "avg_tokens_out": average("tokens_out"),
        "ttft_p50_ms": round(percentile(
            [r["first_token_ms"] for r in successful], 50), 2),
        "ttft_p95_ms": round(percentile(
            [r["first_token_ms"] for r in successful], 95), 2),
        "total_p50_ms": round(percentile(
            [r["total_ms"] for r in successful], 50), 2),
        "total_p95_ms": round(percentile(
            [r["total_ms"] for r in successful], 95), 2),
    }

    summary.append(row)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print("\nANSWER QUALITY COMPARISON")
print("=" * 110)

for row in summary:
    print(f"\n{row['configuration']}")
    for key, value in row.items():
        if key != "configuration":
            print(f"  {key}: {value}")

print("\nSummary saved to:", OUTPUT)
