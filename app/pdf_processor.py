import json
import re
from pathlib import Path

import fitz


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "extracted_papers.json"


def clean_text(text):
    """Normalize extracted PDF text without removing meaningful content."""

    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_page_content(page):
    """
    Extract page text and identify likely section headings.

    Heading detection is heuristic. Original page text is preserved.
    """

    page_data = page.get_text("dict", sort=True)

    text = clean_text(
        page.get_text("text", sort=True)
    )

    headings = []

    # Find the typical font size on this page.
    font_sizes = []

    for block in page_data["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    font_sizes.append(span["size"])

    if font_sizes:
        body_font_size = sorted(font_sizes)[len(font_sizes) // 2]
    else:
        body_font_size = 10

    # Identify larger or bold text as possible headings.
    for block in page_data["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])

            if not spans:
                continue

            line_text = "".join(
                span.get("text", "") for span in spans
            ).strip()

            if not line_text:
                continue

            max_size = max(span["size"] for span in spans)

            is_bold = any(
                "bold" in span.get("font", "").lower()
                for span in spans
            )

            if (
                len(line_text) < 150
                and (
                    max_size >= body_font_size * 1.15
                    or is_bold
                )
            ):
                headings.append(line_text)

    # Remove duplicate heading candidates.
    headings = list(dict.fromkeys(headings))

    return {
        "text": text,
        "headings": headings
    }


def process_paper(paper):
    """Extract every page from one research paper."""

    filename = paper["filename"]
    pdf_path = PAPERS_DIR / filename

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing PDF: {pdf_path}"
        )

    print(f"\nProcessing: {filename}")

    pages = []
    total_characters = 0

    with fitz.open(pdf_path) as document:

        for index, page in enumerate(document):

            content = extract_page_content(page)

            page_text = content["text"]

            total_characters += len(page_text)

            pages.append({
                "paper_id": paper["arxiv_id"],
                "title": paper["name"],
                "arxiv_version": paper["version"],
                "pdf_filename": filename,

                # PDF page number, starting from 1.
                "page_number": index + 1,

                "headings": content["headings"],
                "text": page_text
            })

            print(
                f"Page {index + 1}/{len(document)} "
                f"| Characters: {len(page_text)}"
            )

    return {
        "paper_id": paper["arxiv_id"],
        "title": paper["name"],
        "version": paper["version"],
        "filename": filename,
        "total_pages": len(pages),
        "total_characters": total_characters,
        "pages": pages
    }


def main():

    manifest_path = PAPERS_DIR / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "manifest.json not found. Run paper_downloader first."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["downloaded"] != 3:
        raise ValueError(
            "Expected all three papers to be downloaded."
        )

    all_papers = []

    for paper in manifest["papers"]:

        result = process_paper(paper)

        all_papers.append(result)

    output = {
        "total_papers": len(all_papers),
        "papers": all_papers
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    total_pages = sum(
        paper["total_pages"]
        for paper in all_papers
    )

    total_characters = sum(
        paper["total_characters"]
        for paper in all_papers
    )

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"Papers processed: {len(all_papers)}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_characters}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
