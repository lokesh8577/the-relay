import asyncio
import json
import os
import time

from collections import deque
from datetime import datetime, timezone

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI(
    title="The Relay Stand-in Model",
    version="3.0.0"
)

BASE_DELAY_SECONDS = float(
    os.getenv("STANDIN_BASE_DELAY_SECONDS", "0.78")
)

CHUNK_DELAY_SECONDS = float(
    os.getenv("STANDIN_CHUNK_DELAY_SECONDS", "0.025")
)

MAX_IN_FLIGHT = int(
    os.getenv("STANDIN_MAX_IN_FLIGHT", "5")
)

RPM_LIMIT = int(
    os.getenv("STANDIN_RPM_LIMIT", "10")
)

DAILY_LIMIT = int(
    os.getenv("STANDIN_DAILY_LIMIT", "1000")
)

active_requests = 0
flaky_requests = 0

request_timestamps = deque()
daily_requests = 0
daily_date = datetime.now(timezone.utc).date()


def error_response(status, message):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "standin_error"
            }
        }
    )


def reset_daily_counter_if_needed():
    global daily_requests, daily_date

    today = datetime.now(timezone.utc).date()

    if today != daily_date:
        daily_requests = 0
        daily_date = today


def check_quota():
    """
    Count requests in a rolling 60-second window
    and against a UTC daily budget.
    """
    global daily_requests

    now = time.monotonic()

    reset_daily_counter_if_needed()

    while (
        request_timestamps
        and now - request_timestamps[0] >= 60
    ):
        request_timestamps.popleft()

    # Count each incoming request.
    request_timestamps.append(now)
    daily_requests += 1

    minute_count = len(request_timestamps)

    if daily_requests > DAILY_LIMIT:
        return {
            "allowed": False,
            "status": 429,
            "message": "Stand-in daily request budget exhausted",
            "retry_after": 86400,
            "remaining": 0
        }

    if minute_count > RPM_LIMIT:
        oldest = request_timestamps[0]
        retry_after = max(
            1,
            int(60 - (now - oldest))
        )

        return {
            "allowed": False,
            "status": 429,
            "message": "Stand-in per-minute request limit reached",
            "retry_after": retry_after,
            "remaining": 0
        }

    return {
        "allowed": True,
        "status": 200,
        "remaining": max(0, RPM_LIMIT - minute_count)
    }


@app.get("/health")
async def health():
    reset_daily_counter_if_needed()

    now = time.monotonic()

    while (
        request_timestamps
        and now - request_timestamps[0] >= 60
    ):
        request_timestamps.popleft()

    return {
        "status": "healthy",
        "service": "The Relay Stand-in Model",
        "active_requests": active_requests,
        "max_in_flight": MAX_IN_FLIGHT,
        "base_delay_seconds": BASE_DELAY_SECONDS,
        "rpm_limit": RPM_LIMIT,
        "requests_in_current_minute": len(request_timestamps),
        "daily_limit": DAILY_LIMIT,
        "daily_requests_used": daily_requests
    }


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    x_standin_mode: str = Header(default="normal")
):
    global active_requests, flaky_requests

    payload = await request.json()

    messages = payload.get("messages", [])
    model = payload.get("model", "standin-model")

    question = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            question = message.get("content", "")
            break

    mode = x_standin_mode.lower().strip()

    # Explicit failure mode for retry testing.
    if mode == "rate_limit":
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Stand-in rate limit reached",
                    "type": "rate_limit_error"
                }
            },
            headers={
                "retry-after": "5",
                "x-ratelimit-limit-requests": str(RPM_LIMIT),
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "0"
            }
        )

    if mode == "server_error":
        return error_response(
            503,
            "Simulated stand-in server failure"
        )

    if mode == "flaky":
        flaky_requests += 1

        if flaky_requests <= 2:
            return error_response(
                503,
                "Temporary simulated failure"
            )

        mode = "normal"

    allowed_modes = {
        "normal", "slow", "timeout"
    }

    if mode not in allowed_modes:
        return error_response(
            400,
            f"Unknown stand-in mode: {mode}"
        )

    # Enforce configurable request quotas.
    quota = check_quota()

    if not quota["allowed"]:
        return JSONResponse(
            status_code=quota["status"],
            content={
                "error": {
                    "message": quota["message"],
                    "type": "rate_limit_error"
                }
            },
            headers={
                "retry-after": str(quota["retry_after"]),
                "x-ratelimit-limit-requests": str(RPM_LIMIT),
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-requests": "60s"
            }
        )

    # Reject requests when the service is overloaded.
    if active_requests >= MAX_IN_FLIGHT:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "Stand-in concurrency limit exceeded",
                    "type": "overloaded"
                }
            },
            headers={
                "retry-after": "1",
                "x-standin-active-requests": str(active_requests),
                "x-standin-max-in-flight": str(MAX_IN_FLIGHT)
            }
        )

    active_requests += 1
    current_load = active_requests

    response_text = (
        "This is a simulated response from The Relay stand-in model. "
        "It is not generated by a real language model. "
        f"The received question or context was: {question[:250]}. "
        "The stand-in allows controlled testing of streaming, latency, "
        "failures, rate limits, and system reliability."
    )

    words = response_text.split()
    input_tokens = max(1, len(question.split()))
    output_tokens = len(words)

    load_multiplier = (
        1 + max(0, current_load - 1) * 0.5
    )

    async def generate_stream():
        global active_requests

        try:
            if mode == "timeout":
                await asyncio.sleep(60)
                return

            initial_delay = BASE_DELAY_SECONDS

            if mode == "slow":
                initial_delay *= 3

            initial_delay *= load_multiplier

            await asyncio.sleep(initial_delay)

            for word in words:
                chunk = {
                    "id": "standin-response",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " "
                            },
                            "finish_reason": None
                        }
                    ]
                }

                yield f"data: {json.dumps(chunk)}\n\n"

                await asyncio.sleep(
                    CHUNK_DELAY_SECONDS * load_multiplier
                )

            usage_event = {
                "id": "standin-response",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }

            yield f"data: {json.dumps(usage_event)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            active_requests -= 1

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "x-ratelimit-limit-requests": str(RPM_LIMIT),
            "x-ratelimit-remaining-requests": str(
                quota["remaining"]
            ),
            "x-ratelimit-reset-requests": "1m",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "7000",
            "x-ratelimit-reset-tokens": "1m",
            "x-standin-active-requests": str(current_load),
            "x-standin-max-in-flight": str(MAX_IN_FLIGHT)
        }
    )
