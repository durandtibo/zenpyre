r"""Provide a hashing utility for LangChain documents."""

from __future__ import annotations

__all__ = ["hash_document"]

import json
from typing import TYPE_CHECKING

from coola.hashing import hash_string

if TYPE_CHECKING:
    from langchain_core.documents import Document


def hash_document(doc: Document, length: int = 64) -> str:
    """Compute a stable, reproducible hash of a LangChain document.

    Combines the document's ``page_content`` and ``metadata`` into a
    single canonical string and hashes it.  Metadata is serialised via
    :func:`json.dumps` with ``sort_keys=True`` to guarantee a
    consistent ordering regardless of the dict insertion order.

    Args:
        doc: The :class:`~langchain_core.documents.Document` to hash.
        length: The desired length of the returned hex string.  Must be
            an even number between 2 and 128 inclusive.  Defaults to
            ``64``.

    Returns:
        A lowercase hexadecimal string of exactly ``length`` characters
        that uniquely identifies the document's content and metadata.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import hash_document
        >>> doc = Document(page_content="Hello", metadata={"source": "cats.txt"})
        >>> len(hash_document(doc))
        64

        ```
    """
    content = doc.page_content + json.dumps(doc.metadata, sort_keys=True)
    return hash_string(content, length=length)
