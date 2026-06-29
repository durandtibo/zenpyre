r"""Provide filtering utilities for LangChain document collections."""

from __future__ import annotations

__all__ = ["filter_by_metadata"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document


def filter_by_metadata(
    docs: list[Document],
    metadata_key: str,
    value: Any,
) -> list[Document]:
    """Filter a list of documents by the value of a metadata key.

    Returns a new list containing only documents whose metadata
    contains ``metadata_key`` with a value equal to ``value``.
    Documents missing ``metadata_key`` are excluded.

    Args:
        docs: The list of :class:`~langchain_core.documents.Document`
            instances to filter.
        metadata_key: The metadata key to filter by.
        value: The value to match against. Documents whose
            ``metadata_key`` equals this value are kept.

    Returns:
        A new list of :class:`~langchain_core.documents.Document`
        instances whose metadata matches the filter.  The original
        list is not modified.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents.ops import filter_by_metadata
        >>> docs = [
        ...     Document(page_content="A", metadata={"category": "Science"}),
        ...     Document(page_content="B", metadata={"category": "Cooking"}),
        ...     Document(page_content="C", metadata={"category": "Science"}),
        ... ]
        >>> result = filter_by_metadata(docs, "category", "Science")
        >>> [doc.page_content for doc in result]
        ['A', 'C']

        ```
    """
    return [doc for doc in docs if doc.metadata.get(metadata_key) == value]
