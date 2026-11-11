import re


def validate_answer(answer, retrieved_chunks, min_relevance=0.0):
    """
    Validate an answer against its retrieved sources.

    This is a deterministic baseline validator, not a
    semantic fact-checker.
    """

    issues = []

    if not answer or not answer.strip():
        issues.append("empty_answer")

    if not retrieved_chunks:
        issues.append("no_retrieved_sources")

    # Identify whether any retrieved passage has a positive score.
    relevant_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("bm25_score", 0) > min_relevance
    ]

    if not relevant_chunks:
        issues.append("no_relevant_passages")

    # Extract citations from the generated answer.
    answer_citations = set(
        re.findall(
            r"arXiv:\d{4}\.\d{4,5}v\d+,\s*PDF page\s*\d+",
            answer
        )
    )

    # Citations available from retrieved passages.
    source_citations = {
        chunk["citation"]
        for chunk in retrieved_chunks
    }

    if not answer_citations:
        issues.append("missing_citations")

    unsupported_citations = answer_citations - source_citations

    if unsupported_citations:
        issues.append("unsupported_citations")

    # Detect likely output truncation.
    stripped_answer = answer.rstrip()

    if stripped_answer.endswith((",", ":", "=", "(", "[", "\\", "and")):
        issues.append("possibly_truncated")

    passed = len(issues) == 0

    return {
        "passed": passed,
        "decision": "accept" if passed else "retry_or_refuse",
        "issues": issues,
        "citation_count": len(answer_citations),
        "retrieved_source_count": len(source_citations),
        "unsupported_citations": sorted(unsupported_citations)
    }


if __name__ == "__main__":
    example_sources = [
        {
            "citation": "arXiv:1706.03762v7, PDF page 4",
            "bm25_score": 12.5
        }
    ]

    example_answer = (
        "Multi-head attention runs several attention heads "
        "[arXiv:1706.03762v7, PDF page 4]"
    )

    print(validate_answer(example_answer, example_sources))
