import argparse
import asyncio
import csv
import json

from datetime import datetime, timezone
from pathlib import Path

from app.answer_generator import build_messages
from app.groq_client import GroqClient, GroqRequestError
from app.retrieval import BM25Retriever
from app.validator import validate_answer


BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
OUTPUT_DIR = BASE_DIR / "evaluation" / "answer_outputs"
CSV_FILE = BASE_DIR / "evaluation" / "answer_results.csv"

CONFIGURATIONS = [
    ("small_300_30", "data/chunks_300_overlap_30.jsonl"),
    ("baseline_500_50", "data/chunks_500_overlap_50.jsonl"),
    ("large_700_70", "data/chunks_700_overlap_70.jsonl"),
]

SELECTED_QUESTIONS = {"Q02", "Q04", "Q08"}

FIELDS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting",
    "answer_grade", "http_status", "model",
    "validation_passed", "decision", "error", "answer_file"
]


def save_results(records):
    with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records.values())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--request-id",
        type=str,
        help="Run only one request, e.g. large_700_70-Q04"
    )
    args = parser.parse_args()

    client = GroqClient()

    print("Backend:", client.backend)
    print("Model:", client.model)

    if not args.run:
        print("Dry run only. No requests sent.")
        return

    if client.backend != "groq":
        raise RuntimeError("Set MODEL_BACKEND=groq for this evaluation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with QUESTIONS_FILE.open(encoding="utf-8") as file:
        questions = [
            q for q in json.load(file)
            if q["id"] in SELECTED_QUESTIONS
        ]

    if len(questions) != 3:
        raise ValueError("Expected Q02, Q04 and Q08.")

    records = {}

    if CSV_FILE.exists():
        with CSV_FILE.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                records[row["request_id"]] = {
                    field: row.get(field, "") for field in FIELDS
                }

    for config_name, relative_path in CONFIGURATIONS:
        chunk_file = BASE_DIR / relative_path

        print("\n" + "=" * 65)
        print("Configuration:", config_name)
        print("=" * 65)

        retriever = BM25Retriever(str(chunk_file))

        for question in questions:
            qid = question["id"]
            request_id = f"{config_name}-{qid}"

            if args.request_id and request_id != args.request_id:
                continue

            previous = records.get(request_id)

            if previous and previous["status"] == "ok":
                print("Skipping completed request:", request_id)
                continue

            print("\nRequest:", request_id)

            row = {
                "request_id": request_id,
                "scenario": "answer_quality",
                "source": "real",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "first_token_ms": "",
                "total_ms": "",
                "status": "error",
                "retries": 0,
                "tokens_in": "",
                "tokens_out": "",
                "retrieval_setting": config_name,
                "answer_grade": (
                    previous.get("answer_grade", "")
                    if previous else ""
                ),
                "http_status": "",
                "model": client.model,
                "validation_passed": "",
                "decision": "",
                "error": "",
                "answer_file": "",
            }

            chunks = retriever.search(question["question"], top_k=5)
            messages = build_messages(question["question"], chunks)

            try:
                result = await client.generate(messages)

                validation = validate_answer(
                    result["response"], chunks
                )

                decision = (
                    "accepted" if validation["passed"] else "refused"
                )

                answer_path = OUTPUT_DIR / f"{request_id}.json"

                output = {
                    "request_id": request_id,
                    "question_id": qid,
                    "question": question["question"],
                    "configuration": config_name,
                    "answer": result["response"],
                    "decision": decision,
                    "validation": validation,
                    "model_metrics": result,
                    "retrieved_sources": chunks,
                }

                answer_path.write_text(
                    json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                row.update({
                    "first_token_ms": result["first_token_ms"],
                    "total_ms": result["total_ms"],
                    "status": result["status"],
                    "tokens_in": result["tokens_in"],
                    "tokens_out": result["tokens_out"],
                    "http_status": result["http_status"],
                    "validation_passed": validation["passed"],
                    "decision": decision,
                    "answer_file": str(answer_path.relative_to(BASE_DIR)),
                })

                print("HTTP:", result["http_status"])
                print("TTFT:", result["first_token_ms"], "ms")
                print("Total:", result["total_ms"], "ms")
                print("Tokens:", result["tokens_in"], "in /",
                      result["tokens_out"], "out")
                print("Structural validation:", validation["passed"])

            except GroqRequestError as error:
                row["http_status"] = error.status_code
                row["error"] = str(error)

                if error.status_code == 429:
                    row["status"] = "rate_limited"
                elif error.status_code is None:
                    row["status"] = "timeout"
                else:
                    row["status"] = "error"

                print("Request failed:", error)
                print("HTTP status:", error.status_code)

            except Exception as error:
                row["error"] = str(error)
                print("Evaluation error:", error)

            records[request_id] = row
            save_results(records)

            if row["status"] == "rate_limited":
                print("Stopping. Wait for the TPM window to recover.")
                return

            await asyncio.sleep(2)

    print("\nEvaluation run finished.")
    print("Results:", CSV_FILE)
    print("Answers:", OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
