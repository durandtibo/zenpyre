"""Generate synthetic LangChain Documents for testing and demos."""

from __future__ import annotations

from faker import Faker
from langchain_core.documents import Document

from zenpyre.documents.analysis import (
    compute_content_stats_exact,
    compute_metadata_stats,
    print_content_stats_report,
)
from zenpyre.records.analysis import print_metadata_stats_report
from zenpyre.utils.rich import print_document


def generate_documents(
    n: int = 5,
    seed: int | None = None,
    nb_sentences: int = 5,
) -> list[Document]:
    """Generate synthetic LangChain Documents with Faker-generated
    content.

    Each document gets a unique ``id`` (``"doc-{i}"``), a Faker-generated
    paragraph as its content, and metadata containing a fake author name
    and a single-word topic.

    Note:
        Content is **not guaranteed to be unique**: ``Faker.paragraph()``
        has no built-in uniqueness constraint, so with a large enough
        ``n`` (or a small enough ``nb_sentences``), two documents could
        coincidentally receive identical text. If strict uniqueness
        matters for your use case, deduplicate the result or use
        Faker's ``unique`` proxy (e.g. ``fake.unique.paragraph(...)``,
        which raises once its internal pool is exhausted).

    Warning:
        ``Faker.seed()`` seeds Faker's shared, process-wide random
        generator, not just the local ``fake`` instance created here.
        Passing ``seed`` therefore affects the reproducibility of *any*
        other ``Faker()`` instance used elsewhere in the same process
        after this function runs, not only calls made through this
        function.

    Args:
        n: Number of documents to generate. Must be non-negative;
            ``n=0`` returns an empty list.
        seed: Optional seed for reproducible output. If ``None``,
            content differs on every call (see the Warning above about
            seeding's process-wide scope).
        nb_sentences: Number of sentences per document's generated
            paragraph, passed through to ``Faker.paragraph()``. Higher
            values reduce (but do not eliminate) the chance of two
            documents coincidentally sharing identical content.

    Returns:
        A list of ``n`` Document objects, each with a distinct ``id``
            and independently generated content and metadata.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        msg = f"'n' must be non-negative, got {n}."
        raise ValueError(msg)

    if seed is not None:
        Faker.seed(seed)
    fake = Faker()

    return [
        Document(
            id=f"doc-{i}",
            page_content=fake.paragraph(nb_sentences=nb_sentences),
            metadata={"author": fake.name(), "topic": fake.word()},
        )
        for i in range(n)
    ]


def main() -> None:
    """Generate a sample batch of documents and print analysis reports.

    Generates 100 documents with a fixed seed (for reproducible demo
    output), prints the first 5 with compact metadata, then prints
    content-statistics and metadata-statistics reports over the full
    batch.
    """
    docs = generate_documents(n=100, seed=42)
    for doc in docs[:5]:
        print_document(doc, compact_metadata=True)

    print_content_stats_report(compute_content_stats_exact(docs))
    print_metadata_stats_report(compute_metadata_stats(docs))


if __name__ == "__main__":
    main()
