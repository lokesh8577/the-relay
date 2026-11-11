import json
import hashlib
import argparse

from pathlib import Path
import tiktoken


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "extracted_papers.json"
OUTPUT_DIR = BASE_DIR / "data"

# Consistent tokenizer for chunk-size experiments
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Count tokens, treating paper text as ordinary text."""

    return len(
        ENCODING.encode(
            text,
            disallowed_special=()
        )
    )


def create_chunk_id(
    paper_id,
    version,
    page_number,
    chunk_index,
    chunk_size
):
    """Generate a deterministic unique chunk ID."""

    raw_id = (
        f"{paper_id}-v{version}-"
        f"p{page_number}-c{chunk_index}-s{chunk_size}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]


def split_into_chunks(text, chunk_size, overlap):
    """
    Split page text into token-based chunks.

    disallowed_special=() ensures strings such as
    <|endoftext|> in research papers are treated as
    ordinary text rather than reserved tokenizer tokens.
    """

    tokens = ENCODING.encode(
        text,
        disallowed_special=()
    )

    chunks = []

    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:

        end = min(
            start + chunk_size,
            total_tokens
        )

        chunk_tokens = tokens[start:end]

        chunk_text = ENCODING.decode(
            chunk_tokens
        )

        chunks.append({
            "text": chunk_text,
            "start_token": start,
            "end_token": end,
            "token_count": len(chunk_tokens)
        })

        if end == total_tokens:
            break

        # Retain overlap with the previous chunk
        start = end - overlap

    return chunks


def build_chunks(extracted_data, chunk_size, overlap):
    """Generate chunks for every page of every paper."""

    all_chunks = []

    for paper in extracted_data["papers"]:

        paper_id = paper["paper_id"]
        title = paper["title"]
        version = paper["version"]

        print(f"\nProcessing: {title}")

        for page in paper["pages"]:

            page_number = page["page_number"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            page_chunks = split_into_chunks(
                page_text,
                chunk_size,
                overlap
            )

            for index, chunk in enumerate(page_chunks):

                chunk_id = create_chunk_id(
                    paper_id,
                    version,
                    page_number,
                    index,
                    chunk_size
                )

                record = {
                    "chunk_id": chunk_id,

                    "paper_id": paper_id,
                    "paper_title": title,
                    "paper_version": version,

                    "page_number": page_number,

                    "chunk_index": index,

                    "chunk_size": chunk_size,
                    "overlap": overlap,

                    "start_token": chunk["start_token"],
                    "end_token": chunk["end_token"],
                    "token_count": chunk["token_count"],

                    "section_candidates": page["headings"],

                    "text": chunk["text"]
                }

                all_chunks.append(record)

            print(
                f"Page {page_number}: "
                f"{len(page_chunks)} chunks"
            )

    return all_chunks


def save_chunks(chunks, chunk_size, overlap):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        f"chunks_{chunk_size}_overlap_{overlap}.jsonl"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    return output_file


def main():

    parser = argparse.ArgumentParser(
        description="The Relay paper chunking engine"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Chunk size in approximate tokens"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size <= 0:
        parser.error("Chunk size must be positive.")

    if args.overlap < 0 or args.overlap >= args.size:
        parser.error(
            "Overlap must be non-negative and smaller than chunk size."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Extracted papers not found. Run pdf_processor first."
        )

    extracted_data = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    print("========== CHUNKING STARTED ==========")

    chunks = build_chunks(
        extracted_data,
        args.size,
        args.overlap
    )

    output_file = save_chunks(
        chunks,
        args.size,
        args.overlap
    )

    total_tokens = sum(
        chunk["token_count"]
        for chunk in chunks
    )

    print("\n========== CHUNKING SUMMARY ==========")

    print(f"Chunk size: {args.size}")
    print(f"Overlap: {args.overlap}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total chunk tokens: {total_tokens}")

    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
