import argparse
import asyncio
import json
import time

from app.groq_client import GroqClient, GroqRequestError


def get_retry_delay(headers, retry_number):
    """Use Retry-After when available; otherwise exponential backoff."""

    retry_after = headers.get("retry-after")

    if retry_after:
        value = retry_after.strip().lower().removesuffix("s")

        try:
            return max(0.0, float(value))
        except ValueError:
            pass

    return min(0.5 * (2 ** (retry_number - 1)), 8.0)


def is_retryable(error):
    """Decide whether another attempt is appropriate."""

    status = error.status_code
    message = str(error).lower()

    # A streaming read timeout can happen after HTTP 200 headers arrive.
    if "timed out" in message or "timeout" in message:
        return True

    # Network errors may not have an HTTP status.
    if status is None:
        return True

    # Rate limits and temporary server failures.
    return status == 429 or status in (500, 502, 503, 504)


async def generate_with_retries(
    client,
    messages,
    max_retries=2,
    standin_mode="normal",
    on_chunk=None
):
    """
    Generate a response with bounded retries.

    Returns the successful result and attempt history.
    Raises GroqRequestError if all attempts fail.
    """

    history = []
    start = time.perf_counter()

    for attempt in range(max_retries + 1):
        attempt_number = attempt + 1

        print(f"\nRequest attempt {attempt_number}/{max_retries + 1}")

        attempt_start = time.perf_counter()

        try:
            result = await client.generate(
                messages,
                on_chunk=on_chunk,
                standin_mode=standin_mode
            )

            history.append({
                "attempt": attempt_number,
                "status": "ok",
                "http_status": result["http_status"],
                "elapsed_ms": round(
                    (time.perf_counter() - attempt_start) * 1000, 2
                ),
                "error": None
            })

            return {
                "result": result,
                "history": history,
                "retries": len(history) - 1,
                "total_elapsed_ms": round(
                    (time.perf_counter() - start) * 1000, 2
                )
            }

        except GroqRequestError as error:
            history.append({
                "attempt": attempt_number,
                "status": "error",
                "http_status": error.status_code,
                "elapsed_ms": round(
                    (time.perf_counter() - attempt_start) * 1000, 2
                ),
                "error": str(error),
                "headers": error.headers
            })

            if not is_retryable(error):
                print("Non-retryable error. Stopping.")
                error.attempt_history = history
                error.retries = len(history) - 1
                error.total_elapsed_ms = round(
                    (time.perf_counter() - start) * 1000, 2
                )
                raise

            if attempt >= max_retries:
                print("Retry limit exhausted.")
                error.attempt_history = history
                error.retries = len(history) - 1
                error.total_elapsed_ms = round(
                    (time.perf_counter() - start) * 1000, 2
                )
                raise

            delay = get_retry_delay(
                error.headers,
                retry_number=attempt_number
            )

            print(
                f"Retryable failure. Waiting {delay:.2f} seconds..."
            )

            await asyncio.sleep(delay)

    raise RuntimeError("Unexpected retry-loop exit")


async def main():
    parser = argparse.ArgumentParser(
        description="The Relay reliability test"
    )

    parser.add_argument(
        "--mode",
        default="normal",
        choices=[
            "normal",
            "slow",
            "rate_limit",
            "server_error",
            "timeout",
            "flaky"
        ]
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2
    )

    args = parser.parse_args()

    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    client = GroqClient()

    messages = [
        {
            "role": "user",
            "content": "Explain multi-head attention briefly."
        }
    ]

    try:
        outcome = await generate_with_retries(
            client,
            messages,
            max_retries=args.max_retries,
            standin_mode=args.mode
        )

        print("\nSuccessful response:")
        print(outcome["result"]["response"])

        print("\nRetry summary:")
        print(json.dumps({
            "retries": outcome["retries"],
            "total_elapsed_ms": outcome["total_elapsed_ms"],
            "attempt_history": outcome["history"]
        }, indent=2))

    except GroqRequestError as error:
        print("\nAll attempts failed or request was not retryable.")
        print("Final HTTP status:", error.status_code)
        print("Final error:", error)

        # Show the error without an unnecessary traceback.
        return


if __name__ == "__main__":
    asyncio.run(main())
