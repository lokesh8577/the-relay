import argparse
import asyncio
import csv
import json
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "results" / "results.csv"

STANDIN_URL = "http://127.0.0.1:8001/chat/completions"

CSV_FIELDS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting",
    "answer_grade", "http_status", "model",
    "request_limit", "requests_remaining",
    "token_limit", "tokens_remaining"
]


async def send_request(client, scenario, request_number):
    request_id = f"{scenario}-{uuid.uuid4().hex[:8]}"
    start_time = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    first_token_ms = None
    tokens_in = None
    tokens_out = None
    status = "error"
    http_status = None
    headers = {}
    response_text = ""

    payload = {
        "model": "standin-model",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Load test request {request_number}. "
                    "Explain multi-head attention briefly."
                )
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    try:
        async with client.stream(
            "POST",
            STANDIN_URL,
            headers={
                "X-Standin-Mode": "normal",
                "Authorization": "Bearer standin-local-key"
            },
            json=payload
        ) as response:

            http_status = response.status_code
            headers = response.headers

            if http_status >= 400:
                await response.aread()
                status = (
                    "rate_limited"
                    if http_status == 429
                    else "error"
                )
            else:
                async for line in response.aiter_lines():

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]" or not data:
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if event.get("usage"):
                        tokens_in = event["usage"].get(
                            "prompt_tokens"
                        )
                        tokens_out = event["usage"].get(
                            "completion_tokens"
                        )

                    choices = event.get("choices", [])

                    if choices:
                        content = choices[0].get(
                            "delta", {}
                        ).get("content")

                        if content:
                            if first_token_ms is None:
                                first_token_ms = round(
                                    (time.perf_counter() - start)
                                    * 1000,
                                    2
                                )

                            response_text += content

                status = "ok" if response_text else "error"

    except httpx.TimeoutException:
        status = "timeout"

    except httpx.HTTPError:
        status = "error"

    total_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return {
        "request_id": request_id,
        "scenario": scenario,
        "source": "stand-in",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retrieval_setting": "not_applicable",
        "answer_grade": "",
        "http_status": http_status,
        "model": "standin-model",
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


async def run_test(args):
    scenario = args.scenario

    timeout = httpx.Timeout(
        connect=5,
        read=30,
        write=10,
        pool=10
    )

    limits = httpx.Limits(
        max_connections=max(100, args.requests + 5),
        max_keepalive_connections=20
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits
    ) as client:

        tasks = []

        print(f"\nScenario: {scenario}")
        print(f"Requests: {args.requests}")

        if scenario == "burst":
            print("Traffic: all requests launched together")

            tasks = [
                asyncio.create_task(
                    send_request(client, scenario, i)
                )
                for i in range(args.requests)
            ]

        elif scenario in ("steady", "overload"):
            interval = (
                args.interval
                if scenario == "steady"
                else min(args.interval, 0.05)
            )

            print(f"Arrival interval: {interval} seconds")

            for i in range(args.requests):
                tasks.append(
                    asyncio.create_task(
                        send_request(client, scenario, i)
                    )
                )

                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_header = (
        not RESULTS_FILE.exists()
        or RESULTS_FILE.stat().st_size == 0
    )

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore"
        )

        if write_header:
            writer.writeheader()

        writer.writerows(results)

    successful = sum(
        result["status"] == "ok"
        for result in results
    )

    failed = len(results) - successful

    latencies = sorted(
        result["total_ms"]
        for result in results
    )

    ttfts = sorted(
        result["first_token_ms"]
        for result in results
        if result["first_token_ms"] is not None
    )

    def percentile(values, p):
        if not values:
            return None

        index = round((len(values) - 1) * p)
        return values[index]

    print("\n========== LOAD TEST RESULTS ==========")

    print("Total requests:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
    print(
        "Success rate:",
        round(successful / len(results) * 100, 2),
        "%"
    )

    print("\nTotal latency:")
    print("p50:", percentile(latencies, 0.50), "ms")
    print("p95:", percentile(latencies, 0.95), "ms")
    print("p99:", percentile(latencies, 0.99), "ms")

    print("\nFirst-token latency:")
    print("p50:", percentile(ttfts, 0.50), "ms")
    print("p95:", percentile(ttfts, 0.95), "ms")

    print("\nHTTP status distribution:")

    for code in sorted({
        str(result["http_status"])
        for result in results
    }):
        count = sum(
            str(result["http_status"]) == code
            for result in results
        )
        print(f"HTTP {code}: {count}")

    print("\nCSV saved:", RESULTS_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="The Relay load-testing harness"
    )

    parser.add_argument(
        "--scenario",
        choices=["steady", "burst", "overload"],
        required=True
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=10
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be positive")

    if args.interval < 0:
        parser.error("--interval cannot be negative")

    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
