"""Provide utilities for concatenating documents."""

from __future__ import annotations

__all__ = ["format_documents_as_markdown", "format_documents_as_xml"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def format_documents_as_markdown(documents: list[Document], include_metadata: bool = False) -> str:
    """Concatenate a list of LangChain documents into a single LLM-
    friendly Markdown string.

    Each document is rendered under its own level-2 heading (``## Document N``)
    so the LLM can distinguish document boundaries. When ``include_metadata``
    is ``True``, each document's metadata is rendered as a Markdown bullet
    list above its content, sorted alphabetically by key.

    Args:
        documents: The documents to concatenate.
        include_metadata: If ``True``, include each document's metadata
            (as a bullet list, sorted by key) above its content. Defaults
            to ``False``.

    Returns:
        A single string with one ``## Document N`` section per document,
        in the same order as ``documents``. Returns an empty string if
        ``documents`` is empty.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import format_documents_as_markdown
        >>> docs = [
        ...     Document(
        ...         page_content="The cat sat on the mat.",
        ...         metadata={"source": "story.txt", "author": "Alice"},
        ...     ),
        ...     Document(
        ...         page_content="The dog chased the ball.",
        ...         metadata={"source": "story.txt", "author": "Bob"},
        ...     ),
        ... ]
        >>> print(format_documents_as_markdown(docs))
        ## Document 1
        <BLANKLINE>
        The cat sat on the mat.
        <BLANKLINE>
        ## Document 2
        <BLANKLINE>
        The dog chased the ball.
        >>> print(format_documents_as_markdown(docs, include_metadata=True))
        ## Document 1
        <BLANKLINE>
        - author: Alice
        - source: story.txt
        <BLANKLINE>
        The cat sat on the mat.
        <BLANKLINE>
        ## Document 2
        <BLANKLINE>
        - author: Bob
        - source: story.txt
        <BLANKLINE>
        The dog chased the ball.

        ```
    """
    blocks = []
    for i, doc in enumerate(documents, start=1):
        parts = []
        if include_metadata and doc.metadata:
            metadata_lines = "\n".join(
                f"- {key}: {doc.metadata[key]}" for key in sorted(doc.metadata)
            )
            parts.append(metadata_lines)
        parts.append(doc.page_content)
        body = "\n\n".join(parts)
        blocks.append(f"## Document {i}\n\n{body}")
    return "\n\n".join(blocks)


def format_documents_as_xml(documents: list[Document], include_metadata: bool = False) -> str:
    """Concatenate a list of LangChain documents into a single LLM-
    friendly XML-tagged string.

    Each document is rendered as a clearly delimited ``<document>`` block so
    the LLM can distinguish document boundaries. When ``include_metadata``
    is ``True``, each document's metadata is rendered above its content as
    ``key: value`` lines, sorted alphabetically by key.

    Args:
        documents: The documents to concatenate.
        include_metadata: If ``True``, include each document's metadata
            (as ``key: value`` lines, sorted by key) above its content.
            Defaults to ``False``.

    Returns:
        A single string with one ``<document>`` block per document, in the
        same order as ``documents``. Returns an empty string if
        ``documents`` is empty.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.documents import format_documents_as_xml
        >>> docs = [
        ...     Document(
        ...         page_content="The cat sat on the mat.",
        ...         metadata={"source": "story.txt", "author": "Alice"},
        ...     ),
        ...     Document(
        ...         page_content="The dog chased the ball.",
        ...         metadata={"source": "story.txt", "author": "Bob"},
        ...     ),
        ... ]
        >>> print(format_documents_as_xml(docs))
        <document id="1">
        The cat sat on the mat.
        </document>
        <BLANKLINE>
        <document id="2">
        The dog chased the ball.
        </document>
        >>> print(format_documents_as_xml(docs, include_metadata=True))
        <document id="1">
        author: Alice
        source: story.txt
        <BLANKLINE>
        The cat sat on the mat.
        </document>
        <BLANKLINE>
        <document id="2">
        author: Bob
        source: story.txt
        <BLANKLINE>
        The dog chased the ball.
        </document>

        ```
    """
    blocks = []
    for i, doc in enumerate(documents, start=1):
        parts = []
        if include_metadata and doc.metadata:
            metadata_lines = "\n".join(
                f"{key}: {doc.metadata[key]}" for key in sorted(doc.metadata)
            )
            parts.append(metadata_lines)
        parts.append(doc.page_content)
        body = "\n\n".join(parts)
        blocks.append(f'<document id="{i}">\n{body}\n</document>')
    return "\n\n".join(blocks)
