import hashlib
import json
import time
from pathlib import Path

import fitz
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"

PAPERS = [
    {
        "name": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7,
        "filename": "attention_is_all_you_need_v7.pdf",
    },
    {
        "name": "Moshi",
        "arxiv_id": "2410.00037",
        "version": 1,
        "filename": "moshi_v1.pdf",
    },
    {
        "name": "Dynamic Chunking",
        "arxiv_id": "2507.07955",
        "version": 2,
        "filename": "dynamic_chunking_v2.pdf",
    },
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_paper(client, paper):
    filename = paper["filename"]
    destination = PAPERS_DIR / filename
    temporary = PAPERS_DIR / (filename + ".part")

    url = (
        f"https://arxiv.org/pdf/"
        f"{paper['arxiv_id']}v{paper['version']}"
    )

    print(f"\nDownloading {paper['name']}...")

    if destination.exists():
        try:
            with fitz.open(destination) as doc:
                if doc.page_count > 0:
                    print("Already downloaded and valid.")
                    return {
                        **paper,
                        "url": url,
                        "pages": doc.page_count,
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
        except Exception:
            destination.unlink(missing_ok=True)

    try:
        response = client.get(url)
        response.raise_for_status()

        temporary.write_bytes(response.content)

        if not temporary.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Response is not a PDF")

        with fitz.open(temporary) as doc:
            page_count = doc.page_count

            if page_count == 0:
                raise ValueError("PDF has no pages")

        temporary.replace(destination)

        result = {
            **paper,
            "url": url,
            "pages": page_count,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

        print(f"Downloaded successfully: {page_count} pages")
        return result

    except Exception as error:
        temporary.unlink(missing_ok=True)
        print(f"ERROR: {error}")
        return None


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "TheRelay/0.1 research assistant"},
    ) as client:

        for paper in PAPERS:
            result = download_paper(client, paper)

            if result:
                results.append(result)

            time.sleep(1)

    manifest = {
        "total_papers": len(PAPERS),
        "downloaded": len(results),
        "papers": results,
    }

    (PAPERS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    print("\n========== DOWNLOAD SUMMARY ==========")
    print(f"Required: {len(PAPERS)}")
    print(f"Downloaded: {len(results)}")
    print(f"Failed: {len(PAPERS) - len(results)}")

    if len(results) != len(PAPERS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
