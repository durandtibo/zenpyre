r"""Empty document detection."""

from __future__ import annotations

__all__ = ["is_document_empty"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def is_document_empty(document: Document, *, treat_whitespace_as_empty: bool = False) -> bool:
    r"""Determine if a document's ``page_content`` is empty.

    Args:
        document: The ``langchain_core.documents.Document`` to check.
        treat_whitespace_as_empty: If ``True``, a ``page_content`` that
            contains only whitespace is also considered empty.

    Returns:
        ``True`` if ``page_content`` is the empty string (or is not a
        string, e.g. ``None``), or, if ``treat_whitespace_as_empty`` is
        ``True``, if ``page_content`` is non-empty but contains only
        whitespace.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import is_document_empty
        >>> is_document_empty(Document(page_content=""))
        True
        >>> is_document_empty(Document(page_content="hello"))
        False
        >>> is_document_empty(Document(page_content="  "), treat_whitespace_as_empty=True)
        True

        ```
    """
    content = document.page_content
    if not isinstance(content, str) or content == "":
        return True
    return treat_whitespace_as_empty and content.strip() == ""
