import argparse
import asyncio
import json
import os
import time

from pathlib import Path
from typing import Optional, Callable

import httpx
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class GroqRequestError(Exception):
    """Represents a failed model-service request."""

    def __init__(self, message, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def get_rate_limit_headers(headers):
    relevant_headers = [
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ]

    return {
        name: headers[name]
        for name in relevant_headers
        if name in headers
    }


class GroqClient:

    def __init__(self):

        self.backend = os.getenv(
            "MODEL_BACKEND", "groq"
        ).lower()

        self.model = (
            "standin-model"
            if self.backend == "standin"
            else os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b"
            )
        )

        self.timeout_seconds = float(
            os.getenv("GROQ_TIMEOUT_SECONDS", "45")
        )

        self.max_completion_tokens = int(
            os.getenv("GROQ_MAX_COMPLETION_TOKENS", "800")
        )

        if self.backend == "standin":

            self.base_url = os.getenv(
                "STANDIN_BASE_URL",
                "http://127.0.0.1:8001"
            ).rstrip("/")

            self.api_key = "standin-local-key"

        elif self.backend == "groq":

            self.base_url = os.getenv(
                "GROQ_BASE_URL",
                "https://api.groq.com/openai/v1"
            ).rstrip("/")

            self.api_key = os.getenv("GROQ_API_KEY")

            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is missing. Configure it in .env."
                )

        else:
            raise ValueError(
                f"Unknown MODEL_BACKEND: {self.backend}"
            )

        self.endpoint = (
            f"{self.base_url}/chat/completions"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        self.timeout = httpx.Timeout(
            self.timeout_seconds,
            connect=10.0
        )

    async def generate(
        self,
        messages,
        on_chunk: Optional[Callable] = None,
        standin_mode="normal"
    ):
        """Generate a streamed response from Groq or the stand-in."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": self.max_completion_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True
            }
        }

        request_headers = dict(self.headers)

        if self.backend == "standin":
            request_headers["X-Standin-Mode"] = standin_mode

        start_time = time.perf_counter()

        first_token_time = None
        response_text = ""
        usage = None
        response_headers = {}
        status_code = None

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=request_headers,
                    json=payload
                ) as response:

                    status_code = response.status_code

                    response_headers = get_rate_limit_headers(
                        response.headers
                    )

                    if response.status_code >= 400:

                        error_body = await response.aread()

                        raise GroqRequestError(
                            message=error_body.decode(
                                "utf-8",
                                errors="replace"
                            ),
                            status_code=status_code,
                            headers=response_headers
                        )

                    async for line in response.aiter_lines():

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()

                        if not data:
                            continue

                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if event.get("usage") is not None:
                            usage = event["usage"]

                        choices = event.get("choices", [])

                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content")

                        if not content:
                            continue

                        if first_token_time is None:
                            first_token_time = (
                                time.perf_counter() - start_time
                            )

                        response_text += content

                        if on_chunk is not None:
                            callback_result = on_chunk(content)

                            if asyncio.iscoroutine(callback_result):
                                await callback_result

        except httpx.TimeoutException as error:
            raise GroqRequestError(
                f"Model request timed out: {error}",
                status_code=status_code,
                headers=response_headers
            ) from error

        except httpx.HTTPError as error:
            raise GroqRequestError(
                f"Model network error: {error}",
                status_code=status_code,
                headers=response_headers
            ) from error

        except GroqRequestError:
            raise

        except Exception as error:
            raise GroqRequestError(
                f"Unexpected model client error: {error}",
                status_code=status_code,
                headers=response_headers
            ) from error

        total_time = time.perf_counter() - start_time

        first_token_ms = (
            round(first_token_time * 1000, 2)
            if first_token_time is not None
            else None
        )

        usage = usage or {}

        return {
            "model": self.model,
            "backend": self.backend,
            "response": response_text,
            "status": "ok" if response_text else "error",
            "http_status": status_code,
            "first_token_ms": first_token_ms,
            "total_ms": round(total_time * 1000, 2),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "rate_limit_headers": response_headers,
        }


async def main():

    parser = argparse.ArgumentParser(
        description="The Relay model client test"
    )

    parser.add_argument(
        "--prompt",
        required=True
    )

    parser.add_argument(
        "--mode",
        default="normal",
        choices=[
            "normal",
            "slow",
            "rate_limit",
            "server_error",
            "timeout"
        ]
    )

    args = parser.parse_args()

    client = GroqClient()

    messages = [
        {
            "role": "user",
            "content": args.prompt
        }
    ]

    print("Backend:", client.backend)
    print("Model:", client.model)
    print("Mode:", args.mode)

    try:
        result = await client.generate(
            messages,
            standin_mode=args.mode
        )

        print("\nResponse:")
        print(result["response"])

        print("\nMetrics:")
        print(json.dumps(result, indent=2))

    except GroqRequestError as error:
        print("\nRequest failed:", error)
        print("HTTP status:", error.status_code)
        print("Headers:", json.dumps(error.headers, indent=2))
        raise


if __name__ == "__main__":
    asyncio.run(main())
