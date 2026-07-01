r"""Provide a DuckDB-backed store for LangChain documents with optional
typed metadata schema."""

from __future__ import annotations

__all__ = ["TypedDuckDBDocumentStore"]

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from zenpyre.utils.imports import check_duckdb, is_duckdb_available

if TYPE_CHECKING:
    from pathlib import Path

if is_duckdb_available():  # pragma: no cover
    import duckdb

logger: logging.Logger = logging.getLogger(__name__)


class TypedDuckDBDocumentStore:
    """A DuckDB-backed store for LangChain documents with metadata
    filtering.

    Persists documents to a DuckDB database and supports adding,
    retrieving, and filtering by metadata fields.  An optional
    ``metadata_schema`` maps known metadata field names to DuckDB
    types.  Known fields are stored as typed columns for fast,
    index-friendly queries.  Any metadata fields not in the schema are
    stored in an ``extra`` JSON overflow column, so nothing is lost.

    Args:
        path: Path to the DuckDB file, or ``":memory:"`` for an
            in-memory database (useful for testing).
        metadata_schema: Optional mapping of metadata field names to
            DuckDB type strings (e.g. ``{"ticker": "VARCHAR", "cik":
            "INTEGER", "year": "INTEGER"}``).  Fields in the schema get
            native typed columns; all other metadata fields go into the
            ``extra`` JSON overflow column.  Defaults to ``None``,
            which stores all metadata as JSON only.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> schema = {"author": "VARCHAR", "year": "INTEGER", "category": "VARCHAR"}
        >>> store = TypedDuckDBDocumentStore(":memory:", metadata_schema=schema)
        >>> docs = [
        ...     Document(
        ...         id="1",
        ...         page_content="Introduction to Python",
        ...         metadata={"author": "Alice", "year": 2022, "category": "Programming"},
        ...     ),
        ...     Document(
        ...         id="2",
        ...         page_content="Advanced Python",
        ...         metadata={"author": "Alice", "year": 2023, "category": "Programming"},
        ...     ),
        ...     Document(
        ...         id="3",
        ...         page_content="History of Rome",
        ...         metadata={"author": "Bob", "year": 2021, "category": "History"},
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

    def __init__(
        self,
        path: Path | str = ":memory:",
        metadata_schema: dict[str, str] | None = None,
    ) -> None:
        check_duckdb()
        self._schema: dict[str, str] = metadata_schema or {}
        self._conn = duckdb.connect(str(path))
        self._conn.execute(self._build_create_table())

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
            self._build_insert(),
            [self._doc_to_row(doc) for doc in docs],
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
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", [doc_id]).fetchone()
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
            f"SELECT * FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            doc_ids,
        ).fetchall()
        by_id = {row[0]: self._row_to_doc(row) for row in rows}
        return [by_id.get(doc_id) for doc_id in doc_ids]

    def filter(self, **metadata_filters: Any) -> list[Document]:
        """Retrieve documents matching all provided metadata filters.

        Filters on known schema fields use native typed column
        comparisons.  Filters on fields not in the schema fall back to
        JSON extraction from the ``extra`` column.  All filters are
        combined with ``AND``.

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

        conditions, values = [], []
        for key, value in metadata_filters.items():
            if key in self._schema:
                conditions.append(f"{key} = ?")
            else:
                conditions.append(f"json_extract_string(extra, '$.{key}') = ?")
            values.append(str(value))

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM documents WHERE {where}",  # noqa: S608
            values,
        ).fetchall()
        return [self._row_to_doc(row) for row in rows]

    def delete(self, doc_id: str) -> None:
        """Delete a document by its ID.

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
        rows = self._conn.execute("SELECT * FROM documents").fetchall()
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

    def _build_create_table(self) -> str:
        """Build the CREATE TABLE statement from the schema."""
        typed_cols = "".join(f", {name} {dtype}" for name, dtype in self._schema.items())
        return (
            f"CREATE TABLE IF NOT EXISTS documents "
            f"(id VARCHAR PRIMARY KEY, page_content VARCHAR{typed_cols}, extra JSON)"
        )

    def _build_insert(self) -> str:
        """Build the INSERT OR REPLACE statement from the schema."""
        col_names = ["id", "page_content", *self._schema.keys(), "extra"]
        placeholders = ", ".join("?" * len(col_names))
        return f"INSERT OR REPLACE INTO documents ({', '.join(col_names)}) VALUES ({placeholders})"  # noqa: S608

    def _doc_to_row(self, doc: Document) -> tuple:
        """Convert a Document to an INSERT row tuple."""
        known = [doc.metadata.get(k) for k in self._schema]
        extra = {k: v for k, v in doc.metadata.items() if k not in self._schema}
        return (doc.id, doc.page_content, *known, json.dumps(extra) if extra else None)

    def _row_to_doc(self, row: tuple) -> Document:
        """Convert a raw database row back to a Document."""
        # row layout: id, page_content, [schema cols...], extra
        doc_id, page_content = row[0], row[1]
        schema_vals = dict(zip(self._schema.keys(), row[2 : 2 + len(self._schema)]))
        extra_json = row[2 + len(self._schema)]
        metadata = {k: v for k, v in schema_vals.items() if v is not None}
        if extra_json:
            metadata.update(json.loads(extra_json))
        return Document(id=doc_id, page_content=page_content, metadata=metadata)
