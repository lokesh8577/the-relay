import argparse
import json
import re

from pathlib import Path
from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def tokenize(text):
    """
    Normalize text and split it into searchable terms.

    Keeps technical terms such as:
    multi-head, end-to-end, self-attention.
    """

    text = text.lower()

    return re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        text
    )


class BM25Retriever:

    def __init__(self, chunks_file):

        self.chunks_file = Path(chunks_file)

        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {self.chunks_file}"
            )

        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None

        self.load_documents()
        self.build_index()

    def load_documents(self):
        """Load chunks from JSONL."""

        print(f"Loading chunks: {self.chunks_file}")

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    document = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from error

                if not document.get("text", "").strip():
                    continue

                self.documents.append(document)

        if not self.documents:
            raise ValueError("No searchable chunks found.")

        print(f"Loaded {len(self.documents)} chunks.")

    def build_index(self):
        """Build the BM25 search index."""

        print("Building BM25 index...")

        self.tokenized_documents = [
            tokenize(document["text"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        print("BM25 index ready.")

    def search(self, query, top_k=5):
        """Retrieve the top K matching passages."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Rank documents by descending BM25 score.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            score = float(scores[index])

            # Do not return passages with no lexical match.
            if score <= 0:
                continue

            document = self.documents[index]

            result = {
                "rank": len(results) + 1,

                "chunk_id": document["chunk_id"],

                "paper_id": document["paper_id"],
                "paper_title": document["paper_title"],
                "paper_version": document["paper_version"],

                "citation": (
                    f"arXiv:{document['paper_id']}"
                    f"v{document['paper_version']}, "
                    f"PDF page {document['page_number']}"
                ),

                "page_number": document["page_number"],

                "chunk_index": document["chunk_index"],

                "bm25_score": round(score, 6),

                "token_count": document["token_count"],

                "section_candidates": document.get(
                    "section_candidates", []
                ),

                "text": document["text"]
            }

            results.append(result)

            if len(results) >= top_k:
                break

        return results


def print_results(query, results):

    print("\n" + "=" * 65)
    print("RETRIEVAL RESULTS")
    print("=" * 65)

    print(f"\nQuestion: {query}")
    print(f"Results found: {len(results)}")

    if not results:
        print("\nNo matching passages found.")
        return

    for result in results:

        print("\n" + "-" * 65)

        print(
            f"Rank: {result['rank']} | "
            f"BM25 score: {result['bm25_score']}"
        )

        print(f"Paper: {result['paper_title']}")
        print(
            f"Version: {result['paper_version']}"
        )

        print(f"Citation: {result['citation']}")

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print("\nPassage:")
        print(result["text"])

    print("\n" + "=" * 65)


def main():

    parser = argparse.ArgumentParser(
        description="The Relay BM25 Retrieval Engine"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Research question"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve"
    )

    parser.add_argument(
        "--chunks",
        type=str,
        default="data/chunks_500_overlap_50.jsonl",
        help="Path to the chunk JSONL file"
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    chunks_path = Path(args.chunks)

    if not chunks_path.is_absolute():
        chunks_path = BASE_DIR / chunks_path

    retriever = BM25Retriever(chunks_path)

    results = retriever.search(
        query=args.query,
        top_k=args.top_k
    )

    print_results(
        args.query,
        results
    )


if __name__ == "__main__":
    main()
