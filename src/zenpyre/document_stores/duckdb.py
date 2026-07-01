r"""Provide a DuckDB-backed store for LangChain documents with JSON
metadata."""

from __future__ import annotations

__all__ = ["DuckDBDocumentStore"]

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from zenpyre.document_stores.base import BaseDocumentStore
from zenpyre.utils.imports import is_duckdb_available

if TYPE_CHECKING:
    from pathlib import Path

if is_duckdb_available():  # pragma: no cover
    import duckdb

logger: logging.Logger = logging.getLogger(__name__)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS documents (
        id           VARCHAR PRIMARY KEY,
        page_content VARCHAR NOT NULL,
        metadata     JSON
    )
"""


class DuckDBDocumentStore(BaseDocumentStore):
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

    def add_documents(self, docs: list[Document]) -> None:
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
        row = self._conn.execute(
            "SELECT id, page_content, metadata FROM documents WHERE id = ?",
            [doc_id],
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_many(self, doc_ids: list[str]) -> list[Document | None]:
        placeholders = ", ".join("?" * len(doc_ids))
        rows = self._conn.execute(
            f"SELECT id, page_content, metadata FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            doc_ids,
        ).fetchall()
        by_id = {row[0]: self._row_to_doc(row) for row in rows}
        return [by_id.get(doc_id) for doc_id in doc_ids]

    def filter(self, **metadata_filters: Any) -> list[Document]:
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
        self._conn.execute("DELETE FROM documents WHERE id = ?", [doc_id])

    def delete_many(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        placeholders = ", ".join("?" * len(doc_ids))
        self._conn.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            doc_ids,
        )

    def check_ids(self, doc_ids: list[str]) -> tuple[list[str], list[str]]:
        if not doc_ids:
            return [], []
        placeholders = ", ".join("?" * len(doc_ids))
        existing = {
            row[0]
            for row in self._conn.execute(
                f"SELECT id FROM documents WHERE id IN ({placeholders})",  # noqa: S608
                doc_ids,
            ).fetchall()
        }
        found = [i for i in doc_ids if i in existing]
        missing = [i for i in doc_ids if i not in existing]
        return found, missing

    def all(self) -> list[Document]:
        rows = self._conn.execute("SELECT id, page_content, metadata FROM documents").fetchall()
        return [self._row_to_doc(row) for row in rows]

    def count(self) -> int:
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
