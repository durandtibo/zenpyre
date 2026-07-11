"""Generate synthetic LangChain Documents for testing and demos."""

from __future__ import annotations

from zenpyre.documents.analysis import (
    compute_content_stats_exact,
    compute_metadata_stats,
    print_content_stats_report,
)
from zenpyre.documents.fake import generate_fake_documents
from zenpyre.records.analysis import print_metadata_stats_report
from zenpyre.utils.rich import print_document


def main() -> None:
    """Generate a sample batch of documents and print analysis reports.

    Generates 100 documents with a fixed seed (for reproducible demo
    output), prints the first 5 with compact metadata, then prints
    content-statistics and metadata-statistics reports over the full
    batch.
    """
    docs = generate_fake_documents(n=100, seed=42)
    for doc in docs[:5]:
        print_document(doc, compact_metadata=True)

    print_content_stats_report(compute_content_stats_exact(docs))
    print_metadata_stats_report(compute_metadata_stats(docs))


if __name__ == "__main__":
    main()
