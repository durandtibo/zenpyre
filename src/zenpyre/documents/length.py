r"""Document content length utilities."""

from __future__ import annotations

__all__ = ["compute_document_lengths", "get_document_length"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.documents import Document


def get_document_length(document: Document) -> int:
    r"""Compute the number of characters in a document's
    ``page_content``.

    Args:
        document: The ``langchain_core.documents.Document`` to
            measure.

    Returns:
        The length, in characters, of ``page_content``, or ``0`` if
        ``page_content`` is not a string (e.g. ``None``).

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import get_document_length
        >>> get_document_length(Document(page_content="hello"))
        5

        ```
    """
    content = document.page_content
    return len(content) if isinstance(content, str) else 0


def compute_document_lengths(
    documents: Iterable[Document], *, sort: bool = False
) -> list[tuple[Any, int]]:
    r"""Compute the number of characters in each document's
    ``page_content``.

    Args:
        documents: A list, generator, or other iterable of
            ``langchain_core.documents.Document`` objects. Consumed
            exactly once; if a generator/iterator is passed in, it will
            be exhausted by this call.
        sort: If ``True``, sort the output by character count, from
            shortest to longest. If ``False``, the output preserves the
            order of ``documents``.

    Returns:
        A list of ``(document_id, char_count)`` tuples, one per input
        document. A document whose ``page_content`` is not a string
        (e.g. ``None``) is treated as having a length of ``0``.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import compute_document_lengths
        >>> docs = [
        ...     Document(id="a", page_content="hello"),
        ...     Document(id="b", page_content="hello world"),
        ... ]
        >>> compute_document_lengths(docs)
        [('a', 5), ('b', 11)]
        >>> compute_document_lengths(docs, sort=True)
        [('a', 5), ('b', 11)]

        ```
    """
    result = [(doc.id, get_document_length(doc)) for doc in documents]
    if sort:
        result.sort(key=lambda item: item[1])
    return result
