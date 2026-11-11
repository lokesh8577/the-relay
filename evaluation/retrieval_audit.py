import json
from pathlib import Path

from app.retrieval import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent.parent

with open(
    BASE_DIR / "evaluation/questions.json",
    encoding="utf-8"
) as file:
    questions = json.load(file)

retriever = BM25Retriever(
    str(BASE_DIR / "data/chunks_500_overlap_50.jsonl")
)

for question in questions:
    print("\n" + "=" * 75)
    print(question["id"], question["question"])
    print("Expected paper:", question["expected_paper"])
    print("=" * 75)

    results = retriever.search(
        question["question"],
        top_k=3
    )

    for result in results:
        print(
            f'\nRank {result["rank"]} | '
            f'Paper: {result["paper_id"]} | '
            f'Page: {result["page_number"]} | '
            f'BM25: {result["bm25_score"]:.3f}'
        )
        print(result["text"][:650].replace("\n", " "))

