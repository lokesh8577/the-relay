import argparse
import asyncio
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.groq_client import GroqClient, GroqRequestError
from app.reliability import generate_with_retries


RESULTS_FILE = Path("results/results.csv")
ATTEMPTS_FILE = Path("results/retry_attempts.csv")

RESULT_FIELDS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting",
    "answer_grade", "http_status", "model", "request_limit",
    "requests_remaining", "token_limit", "tokens_remaining"
]

ATTEMPT_FIELDS = [
    "request_id", "attempt", "status", "http_status",
    "elapsed_ms", "error", "headers"
]


def append_csv(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def classify_failure(error):
    message = str(error).lower()

    if "timed out" in message or "timeout" in message:
        return "timeout"

    if error.status_code == 429:
        return "rate_limited"

    return "error"


async def run_test(mode, max_retries):
    client = GroqClient()

    if client.backend != "standin":
        raise RuntimeError(
            "This experiment must use MODEL_BACKEND=standin."
        )

    request_id = f"retry-{mode}-{uuid.uuid4().hex[:8]}"
    start_time = datetime.now(timezone.utc).isoformat()

    messages = [
        {
            "role": "user",
            "content": "Explain multi-head attention briefly."
        }
    ]

    result = None
    error = None

    try:
        outcome = await generate_with_retries(
            client,
            messages,
            max_retries=max_retries,
            standin_mode=mode
        )
        result = outcome["result"]
        history = outcome["history"]
        retries = outcome["retries"]
        total_ms = outcome["total_elapsed_ms"]
        status = "ok" if result.get("status") == "ok" else "error"

    except GroqRequestError as caught:
        error = caught
        history = getattr(caught, "attempt_history", [])
        retries = getattr(caught, "retries", max(0, len(history) - 1))
        total_ms = getattr(caught, "total_elapsed_ms", None)
        status = classify_failure(caught)

    if total_ms is None:
        total_ms = sum(item["elapsed_ms"] for item in history)

    final_attempt = history[-1] if history else {}
    http_status = (
        result.get("http_status") if result else
        final_attempt.get("http_status")
    )

    headers = (
        result.get("rate_limit_headers", {}) if result else
        final_attempt.get("headers", {})
    )

    if not isinstance(headers, dict):
        headers = {}

    for attempt in history:
        append_csv(
            ATTEMPTS_FILE,
            ATTEMPT_FIELDS,
            {
                "request_id": request_id,
                "attempt": attempt.get("attempt"),
                "status": attempt.get("status"),
                "http_status": attempt.get("http_status"),
                "elapsed_ms": attempt.get("elapsed_ms"),
                "error": attempt.get("error") or "",
                "headers": json.dumps(
                    attempt.get("headers", {}),
                    ensure_ascii=False
                )
            }
        )

    row = {
        "request_id": request_id,
        "scenario": f"retry_{mode}",
        "source": "stand-in",
        "start_time": start_time,
        "first_token_ms": (
            result.get("first_token_ms") if result else ""
        ),
        "total_ms": round(total_ms, 2),
        "status": status,
        "retries": retries,
        "tokens_in": result.get("tokens_in") if result else "",
        "tokens_out": result.get("tokens_out") if result else "",
        "retrieval_setting": "not_applicable",
        "answer_grade": "",
        "http_status": http_status or "",
        "model": client.model,
        "request_limit": headers.get(
            "x-ratelimit-limit-requests", ""
        ),
        "requests_remaining": headers.get(
            "x-ratelimit-remaining-requests", ""
        ),
        "token_limit": headers.get(
            "x-ratelimit-limit-tokens", ""
        ),
        "tokens_remaining": headers.get(
            "x-ratelimit-remaining-tokens", ""
        )
    }

    append_csv(RESULTS_FILE, RESULT_FIELDS, row)

    print("\nSaved retry experiment:")
    print(json.dumps({
        "request_id": request_id,
        "scenario": f"retry_{mode}",
        "status": status,
        "http_status": http_status,
        "attempts": len(history),
        "retries": retries,
        "total_ms": round(total_ms, 2),
        "results_file": str(RESULTS_FILE),
        "attempts_file": str(ATTEMPTS_FILE)
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Record The Relay retry experiments."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["flaky", "server_error", "rate_limit", "timeout"]
    )
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    asyncio.run(run_test(args.mode, args.max_retries))


if __name__ == "__main__":
    main()
