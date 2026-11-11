import argparse
import asyncio
import json

from pathlib import Path

from app.retrieval import BM25Retriever
from app.groq_client import GroqClient, GroqRequestError
from app.validator import validate_answer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_messages(question, retrieved_chunks, previous_issues=None):
    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            f"""
SOURCE [{index}]
Paper: {chunk['paper_title']}
Citation: {chunk['citation']}

Passage:
{chunk['text']}
"""
        )

    context = "\n".join(sources)

    system_prompt = """
You are The Relay, a research-paper assistant.

Answer using ONLY the supplied research-paper passages.

Rules:
1. Do not invent facts or citations.
2. Cite factual claims using exact supplied citation references.
3. If evidence is insufficient, clearly say so.
4. Do not cite sources that do not support your statements.
5. Give a complete answer.
6. Never pretend that unsupported claims are verified.
"""

    retry_instruction = ""

    if previous_issues:
        retry_instruction = f"""
Your previous answer failed validation for these reasons:
{previous_issues}

Review the evidence carefully and produce a corrected,
complete answer with citations.
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PASSAGES:
{context}

{retry_instruction}

Answer the question with supporting citations.
Use this citation format:
[arXiv:1706.03762v7, PDF page 4]
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def generate_answer(question, top_k=3, chunk_file=None):
    if chunk_file is None:
        chunk_file = DATA_DIR / "chunks_500_overlap_50.jsonl"

    print("\nLoading research papers...")

    retriever = BM25Retriever(chunk_file)
    client = GroqClient()

    print("\nRetrieving relevant passages...")

    retrieved_chunks = retriever.search(question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "status": "refused",
            "reason": "No relevant paper passages found."
        }

    all_metrics = []
    previous_issues = None

    # Initial generation + at most one corrective attempt.
    for attempt in range(2):
        print(f"\nGeneration attempt: {attempt + 1}")

        messages = build_messages(
            question,
            retrieved_chunks,
            previous_issues
        )

        result = await client.generate(messages)
        all_metrics.append(result)

        answer = result["response"]

        validation = validate_answer(
            answer,
            retrieved_chunks
        )

        print("\nValidation:")
        print(json.dumps(validation, indent=2))

        if validation["passed"]:
            decision = "accepted"
            break

        previous_issues = validation["issues"]

        if attempt == 0:
            print("\nValidation failed. Expanding retrieval...")

            expanded_chunks = retriever.search(
                question,
                top_k=8
            )

            # Keep expanded retrieval only if it found more chunks.
            if len(expanded_chunks) > len(retrieved_chunks):
                retrieved_chunks = expanded_chunks

    else:
        decision = "refused"

    if validation["passed"]:
        decision = "accepted"
    else:
        decision = "refused"

    print("\n" + "=" * 70)
    print("THE RELAY — RESEARCH ANSWER")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    if decision == "accepted":
        print("\n" + answer)
    else:
        print("\nI could not verify a complete, properly cited answer.")
        print("Please refine the question or consult the cited papers.")

    print("\n" + "-" * 70)
    print("DECISION")
    print("-" * 70)
    print("Final decision:", decision)
    print("Generation attempts:", len(all_metrics))

    print("\n" + "-" * 70)
    print("RETRIEVED SOURCES")
    print("-" * 70)

    for chunk in retrieved_chunks:
        print(
            f"{chunk['paper_title']} | {chunk['citation']} "
            f"| BM25: {chunk['bm25_score']}"
        )

    print("\n" + "-" * 70)
    print("REAL GROQ METRICS")
    print("-" * 70)

    for index, metric in enumerate(all_metrics, start=1):
        print(f"\nAttempt {index}")
        print("HTTP status:", metric["http_status"])
        print("First token:", metric["first_token_ms"], "ms")
        print("Total time:", metric["total_ms"], "ms")
        print("Input tokens:", metric["tokens_in"])
        print("Output tokens:", metric["tokens_out"])

    return {
        "question": question,
        "answer": answer if decision == "accepted" else None,
        "decision": decision,
        "validation": validation,
        "retrieved_sources": retrieved_chunks,
        "attempts": all_metrics
    }


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-file", default=None)

    args = parser.parse_args()

    try:
        await generate_answer(
            question=args.query,
            top_k=args.top_k,
            chunk_file=args.chunk_file
        )

    except GroqRequestError as error:
        print("\nGroq request failed:", error)
        print("HTTP status:", error.status_code)
        print("Rate-limit headers:", error.headers)
        raise


if __name__ == "__main__":
    asyncio.run(main())
