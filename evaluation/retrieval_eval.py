import csv
import json
import time

from pathlib import Path
from statistics import median

from app.retrieval import BM25Retriever


BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"

CONFIGURATIONS = [
    {
        "name": "small_300_30",
        "file": "data/chunks_300_overlap_30.jsonl"
    },
    {
        "name": "baseline_500_50",
        "file": "data/chunks_500_overlap_50.jsonl"
    },
    {
        "name": "large_700_70",
        "file": "data/chunks_700_overlap_70.jsonl"
    }
]

OUTPUT_FILE = BASE_DIR / "evaluation" / "retrieval_results.csv"


def percentile(values, p):
    ordered = sorted(values)

    if not ordered:
        return 0

    index = round((len(ordered) - 1) * p)
    return ordered[index]


def main():
    with QUESTIONS_FILE.open(encoding="utf-8") as file:
        questions = json.load(file)

    all_results = []
    summaries = []

    for config in CONFIGURATIONS:
        chunks_path = BASE_DIR / config["file"]

        if not chunks_path.exists():
            raise FileNotFoundError(chunks_path)

        print("\n" + "=" * 65)
        print("Configuration:", config["name"])
        print("=" * 65)

        retriever = BM25Retriever(str(chunks_path))

        chunk_count = sum(
            1 for _ in chunks_path.open(encoding="utf-8")
        )

        latencies = []
        paper_hits = 0
        page_hits = 0
        page_rank1_hits = 0
        retrieved_token_counts = []

        for item in questions:
            start = time.perf_counter()

            results = retriever.search(
                item["question"],
                top_k=5
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                3
            )

            latencies.append(latency_ms)

            expected_paper = item["expected_paper"]
            expected_pages = set(item["expected_pages"])

            retrieved_papers = [
                result["paper_id"] for result in results
            ]

            retrieved_pages = [
                result["page_number"] for result in results
            ]

            paper_found = expected_paper in retrieved_papers

            matching_pages = [
                page for page in retrieved_pages
                if page in expected_pages
            ]

            # Passage must belong to the expected paper
            # and match one of its manually reviewed pages.
            supporting_results = [
                result for result in results
                if (
                    result["paper_id"] == expected_paper
                    and result["page_number"] in expected_pages
                )
            ]

            page_found = bool(supporting_results)

            page_rank1 = (
                bool(results)
                and results[0]["paper_id"] == expected_paper
                and results[0]["page_number"] in expected_pages
            )

            paper_hits += int(paper_found)
            page_hits += int(page_found)
            page_rank1_hits += int(page_rank1)

            retrieved_tokens = sum(
                result.get("token_count", 0)
                for result in results
            )

            retrieved_token_counts.append(retrieved_tokens)

            all_results.append({
                "configuration": config["name"],
                "question_id": item["id"],
                "question": item["question"],
                "expected_paper": expected_paper,
                "expected_pages": "|".join(
                    map(str, sorted(expected_pages))
                ),
                "retrieved_papers": "|".join(retrieved_papers),
                "retrieved_pages": "|".join(
                    map(str, retrieved_pages)
                ),
                "paper_in_top5": paper_found,
                "supporting_page_in_top5": page_found,
                "supporting_page_at_rank1": page_rank1,
                "matching_pages": "|".join(
                    map(str, matching_pages)
                ),
                "retrieval_latency_ms": latency_ms,
                "retrieved_chunks": len(results),
                "retrieved_tokens": retrieved_tokens
            })

            print(
                f'{item["id"]}: '
                f'paper={paper_found}, '
                f'supporting_page={page_found}, '
                f'page_rank1={page_rank1}, '
                f'latency={latency_ms} ms'
            )

        total = len(questions)

        summary = {
            "configuration": config["name"],
            "total_chunks": chunk_count,
            "questions": total,
            "paper_accuracy_percent": round(
                paper_hits / total * 100, 2
            ),
            "supporting_page_accuracy_percent": round(
                page_hits / total * 100, 2
            ),
            "supporting_page_rank1_percent": round(
                page_rank1_hits / total * 100, 2
            ),
            "median_retrieved_tokens": median(
                retrieved_token_counts
            ),
            "latency_p50_ms": round(
                percentile(latencies, 0.50), 3
            ),
            "latency_p95_ms": round(
                percentile(latencies, 0.95), 3
            )
        }

        summaries.append(summary)

    fields = list(all_results[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 75)
    print("FINAL PASSAGE-LEVEL RETRIEVAL COMPARISON")
    print("=" * 75)

    for summary in summaries:
        print(f'\n{summary["configuration"]}')
        print("Chunks:", summary["total_chunks"])
        print("Questions:", summary["questions"])
        print("Paper accuracy:",
              summary["paper_accuracy_percent"], "%")
        print("Supporting-page accuracy:",
              summary["supporting_page_accuracy_percent"], "%")
        print("Supporting page at rank 1:",
              summary["supporting_page_rank1_percent"], "%")
        print("Median retrieved tokens:",
              summary["median_retrieved_tokens"])
        print("Latency p50:",
              summary["latency_p50_ms"], "ms")
        print("Latency p95:",
              summary["latency_p95_ms"], "ms")

    print("\nRaw results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
