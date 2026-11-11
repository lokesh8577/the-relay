import csv
import math
from collections import Counter, defaultdict
from statistics import mean

PATH = "results/results.csv"


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = max(0, math.ceil(p * len(values)) - 1)
    return round(values[index], 2)


with open(PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for row in rows:
    groups[row["scenario"]].append(row)


def number(row, field):
    value = row.get(field, "").strip()
    return float(value) if value else None


print("CONSOLIDATED PERFORMANCE METRICS")
print("=" * 90)

for scenario, items in sorted(groups.items()):
    successes = [r for r in items if r["status"] == "ok"]
    statuses = Counter(r["status"] for r in items)

    ttft = [
        number(r, "first_token_ms")
        for r in successes
        if number(r, "first_token_ms") is not None
    ]

    total = [
        number(r, "total_ms")
        for r in successes
        if number(r, "total_ms") is not None
    ]

    retries = [
        number(r, "retries")
        for r in items
        if number(r, "retries") is not None
    ]

    print(f"\nSCENARIO: {scenario}")
    print("Requests:", len(items))
    print("Successes:", len(successes))
    print("Success rate:", round(len(successes) / len(items) * 100, 2), "%")
    print("Outcomes:", dict(statuses))
    print("Retries/request:", round(mean(retries), 2) if retries else "N/A")

    print("TTFT p50/p95/p99:",
          percentile(ttft, .50),
          percentile(ttft, .95),
          percentile(ttft, .99))

    print("Total latency p50/p95/p99:",
          percentile(total, .50),
          percentile(total, .95),
          percentile(total, .99))

print("\nANSWER QUALITY")
graded = [r for r in rows if r["answer_grade"].strip()]
grades = Counter(r["answer_grade"] for r in graded)
print("Graded answers:", len(graded))
print("Grades:", dict(grades))
print("Correct percentage:",
      round(grades["correct"] / len(graded) * 100, 2)
      if graded else "N/A")

print("\nOVERALL")
print("Total requests:", len(rows))
print("Sources:", dict(Counter(r["source"] for r in rows)))
print("Statuses:", dict(Counter(r["status"] for r in rows)))
