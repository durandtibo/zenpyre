"""Generate synthetic LangChain Documents for testing and demos."""

from __future__ import annotations

from zenpyre.records.analysis import (
    compute_metadata_stats,
    print_metadata_stats_report,
)
from zenpyre.records.fake import generate_fake_records


def main() -> None:
    """Generate a sample batch of records and print analysis reports.

    Generates 100 records with a fixed seed (for reproducible demo
    output), prints the first 5 with compact metadata, then prints
    content-statistics and metadata-statistics reports over the full
    batch.
    """
    docs = generate_fake_records(n=100, seed=42)

    print_metadata_stats_report(compute_metadata_stats(docs))


if __name__ == "__main__":
    main()
