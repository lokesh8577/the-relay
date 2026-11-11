import csv
import math
from collections import defaultdict
from pathlib import Path

SOURCE = Path("results/results.csv")
OUTPUT = Path("results/load_summary.csv")

SCENARIOS = {
    "steady",
    "burst_concurrency",
    "overload",
    "halved_rpm",
    "daily_budget",
}


def percentile(values, p):
    values = sorted(float(x) for x in values if x not in ("", None))
    if not values:
        return ""

    pos = (len(values) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)

    if lo == hi:
        return round(values[lo], 2)

    return round(
        values[lo] + (values[hi] - values[lo]) * (pos - lo), 2
    )


with SOURCE.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

groups = defaultdict(list)

for row in records:
    if row["source"] == "stand-in" and row["scenario"] in SCENARIOS:
        groups[row["scenario"]].append(row)

fields = [
    "scenario", "requests", "successful", "errors",
    "rate_limited", "success_pct", "ttft_p50_ms",
    "ttft_p95_ms", "total_p50_ms", "total_p95_ms",
    "failed_latency_p50_ms", "rpm_limit"
]

summary = []

for scenario in sorted(SCENARIOS):
    rows = groups[scenario]

    if not rows:
        continue

    successful = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "error"]
    limited = [r for r in rows if r["status"] == "rate_limited"]

    total = len(rows)

    summary.append({
        "scenario": scenario,
        "requests": total,
        "successful": len(successful),
        "errors": len(failed),
        "rate_limited": len(limited),
        "success_pct": round(len(successful) / total * 100, 2),
        "ttft_p50_ms": percentile(
            [r["first_token_ms"] for r in successful], 50),
        "ttft_p95_ms": percentile(
            [r["first_token_ms"] for r in successful], 95),
        "total_p50_ms": percentile(
            [r["total_ms"] for r in successful], 50),
        "total_p95_ms": percentile(
            [r["total_ms"] for r in successful], 95),
        "failed_latency_p50_ms": percentile(
            [r["total_ms"] for r in failed + limited], 50),
        "rpm_limit": ",".join(sorted({
            r["request_limit"] for r in rows if r["request_limit"]
        })),
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print("\nFINAL LOAD SUMMARY")
print("=" * 75)

for row in summary:
    print("\n" + row["scenario"])
    for key, value in row.items():
        if key != "scenario":
            print(f"  {key}: {value}")

print("\nSaved:", OUTPUT)
