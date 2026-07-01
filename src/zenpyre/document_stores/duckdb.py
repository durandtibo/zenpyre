r"""Provide a DuckDB-backed store for LangChain documents with JSON
metadata."""

from __future__ import annotations

__all__ = ["DuckDBDocumentStore"]

import json
import logging
from typing import TYPE_CHECKING, Any

import duckdb
from langchain_core.documents import Document

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS documents (
        id           VARCHAR PRIMARY KEY,
        page_content VARCHAR NOT NULL,
        metadata     JSON
    )
"""


class DuckDBDocumentStore:
    """A DuckDB-backed store for LangChain documents.

    Persists documents to a DuckDB database and supports adding,
    retrieving, filtering, and deleting documents.  All metadata is
    stored as a JSON column, which provides flexibility for arbitrary
    metadata fields without requiring a fixed schema.  For better
    query performance on known metadata fields, see
    :class:`TypedDuckDBDocumentStore`.

    Args:
        path: Path to the DuckDB file, or ``":memory:"`` for an
            in-memory database (useful for testing).

    Example:
        ```pycon
        >>> from zenpyre.document_stores import DuckDBDocumentStore
        >>> from langchain_core.documents import Document
        >>> store = DuckDBDocumentStore(":memory:")
        >>> docs = [
        ...     Document(
        ...         id="1",
        ...         page_content="Intro to Python",
        ...         metadata={"author": "Alice", "category": "Programming"},
        ...     ),
        ...     Document(
        ...         id="2",
        ...         page_content="Advanced Python",
        ...         metadata={"author": "Alice", "category": "Programming"},
        ...     ),
        ...     Document(
        ...         id="3",
        ...         page_content="History of Rome",
        ...         metadata={"author": "Bob", "category": "History"},
        ...     ),
        ... ]
        >>> store.add_documents(docs)
        >>> len(store.filter(author="Alice"))
        2
        >>> len(store.filter(author="Alice", category="Programming"))
        2
        >>> len(store.filter(category="History"))
        1

        ```
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._conn = duckdb.connect(str(path))
        self._conn.execute(_CREATE_TABLE)

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> None:
        """Add or replace documents in the store.

        Documents whose ``id`` already exists are replaced (upsert).

        Args:
            docs: The list of :class:`~langchain_core.documents.Document`
                instances to add.  Each document must have an ``id``.

        Raises:
            ValueError: If any document has no ``id``.
        """
        for doc in docs:
            if doc.id is None:
                msg = "All documents must have an id. Assign one before adding."
                raise ValueError(msg)

        self._conn.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)",
            [(doc.id, doc.page_content, json.dumps(doc.metadata)) for doc in docs],
        )
        logger.info("Added %s documents.", f"{len(docs):,}")

    def get(self, doc_id: str) -> Document | None:
        """Retrieve a single document by its ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The :class:`~langchain_core.documents.Document`, or
            ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT id, page_content, metadata FROM documents WHERE id = ?",
            [doc_id],
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_many(self, doc_ids: list[str]) -> list[Document | None]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: The document IDs to look up.

        Returns:
            A list the same length as ``doc_ids``, with the
            corresponding :class:`~langchain_core.documents.Document`
            for each ID that exists, or ``None`` for IDs not found.
        """
        placeholders = ", ".join("?" * len(doc_ids))
        rows = self._conn.execute(
            f"SELECT id, page_content, metadata FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            doc_ids,
        ).fetchall()
        by_id = {row[0]: self._row_to_doc(row) for row in rows}
        return [by_id.get(doc_id) for doc_id in doc_ids]

    def filter(self, **metadata_filters: Any) -> list[Document]:
        """Retrieve documents matching all provided metadata filters.

        All filters are combined with ``AND``.  Each keyword argument
        matches the corresponding metadata key exactly via JSON
        extraction.

        Args:
            **metadata_filters: Key-value pairs where each key is a
                metadata field name and the value is the exact value
                to match.

        Returns:
            A list of matching
            :class:`~langchain_core.documents.Document` instances.
        """
        if not metadata_filters:
            return self.all()

        conditions = [f"json_extract_string(metadata, '$.{key}') = ?" for key in metadata_filters]
        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT id, page_content, metadata FROM documents WHERE {where}",  # noqa: S608
            [str(v) for v in metadata_filters.values()],
        ).fetchall()
        return [self._row_to_doc(row) for row in rows]

    def delete(self, doc_id: str) -> None:
        """Delete a document by its ID.

        IDs that do not exist are silently ignored.

        Args:
            doc_id: The ID of the document to delete.
        """
        self._conn.execute("DELETE FROM documents WHERE id = ?", [doc_id])

    def delete_many(self, doc_ids: list[str]) -> None:
        """Delete multiple documents by their IDs.

        IDs that do not exist are silently ignored.

        Args:
            doc_ids: The IDs of the documents to delete.
        """
        if not doc_ids:
            return
        placeholders = ", ".join("?" * len(doc_ids))
        self._conn.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            doc_ids,
        )

    def all(self) -> list[Document]:
        """Return all documents in the store.

        Returns:
            A list of all :class:`~langchain_core.documents.Document`
            instances.
        """
        rows = self._conn.execute("SELECT id, page_content, metadata FROM documents").fetchall()
        return [self._row_to_doc(row) for row in rows]

    def count(self) -> int:
        """Return the total number of documents in the store.

        Returns:
            The number of documents currently stored.
        """
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _row_to_doc(row: tuple) -> Document:
        """Convert a raw database row to a Document."""
        doc_id, page_content, metadata_json = row
        return Document(
            id=doc_id,
            page_content=page_content,
            metadata=json.loads(metadata_json) if metadata_json else {},
        )
