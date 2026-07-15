r"""Provide a PostgreSQL-backed store for Record objects with JSONB
metadata."""

from __future__ import annotations

__all__ = ["BasePostgreSQLRecordStore", "PostgreSQLRecordStore"]

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin

from zenpyre.record_stores.base import BaseRecordStore
from zenpyre.records import Record
from zenpyre.utils.imports import check_psycopg, is_psycopg_available

if TYPE_CHECKING:
    from collections.abc import Generator

    from typing_extensions import Self

if is_psycopg_available():  # pragma: no cover
    import psycopg

logger: logging.Logger = logging.getLogger(__name__)


class BasePostgreSQLRecordStore(BaseRecordStore, MultilineDisplayMixin):
    r"""Define a base class for PostgreSQL-backed store for records.

    The constructor mirrors :func:`psycopg.connect`: the first
    positional argument is the ``conninfo`` connection string (or any
    other value accepted by ``psycopg.connect``, e.g. keyword
    arguments such as ``host``, ``port``, ``dbname``, ``user``, and
    ``password`` may also be passed via ``**kwargs``), and any
    additional keyword arguments are forwarded as-is.

    Args:
        conninfo: The connection string (DSN) passed to
            ``psycopg.connect`` (e.g.
            ``"postgresql://user:password@host:port/dbname"``).
        **kwargs: Additional keyword arguments to pass to
            ``psycopg.connect`` (e.g. ``host``, ``port``, ``dbname``,
            ``user``, ``password``, ``autocommit``).
    """

    def __init__(self, conninfo: str = "", **kwargs: Any) -> None:
        check_psycopg()
        self._conninfo = conninfo
        self._kwargs = kwargs
        self._closed = False
        self._conn = psycopg.connect(conninfo, **kwargs)

    def _ensure_schema(self) -> None:
        """Recreate the store's table schema on a fresh connection.

        Called once from ``__init__`` and again each time the store is
        reopened via :meth:`__enter__` after being closed. The default
        implementation does nothing; subclasses override it.
        """

    def close(self) -> None:
        if self._closed:
            return
        logger.info("Closing PostgreSQL connection to %s", self._conninfo)
        self._conn.close()
        self._closed = True

    def delete(self, record_id: str) -> None:
        self._conn.execute("DELETE FROM records WHERE id = %s", (record_id,))
        self._conn.commit()

    def delete_many(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        self._conn.execute("DELETE FROM records WHERE id = ANY(%s)", (record_ids,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def get_columns_info(self) -> dict[str, str]:
        """Return the column names and types of the records table.

        Returns:
            A mapping of column name to PostgreSQL data type.
        """
        rows = self._conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'records'"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def show_columns_info(self) -> None:
        """Print the records table's column names and types to stdout.

        This is a convenience wrapper around :meth:`get_columns_info`
        for interactive/debugging use. For programmatic access, use
        :meth:`get_columns_info` instead.
        """
        for name, dtype in self.get_columns_info().items():
            logger.info(f"{name}\t{dtype}")

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"count": self.count(), "conninfo": self._conninfo} | self._kwargs

    def __enter__(self) -> Self:
        if self._closed:
            self._conn = psycopg.connect(self._conninfo, **self._kwargs)
            self._closed = False
            self._ensure_schema()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS records (
        id       TEXT PRIMARY KEY,
        metadata JSONB
    )
"""


class PostgreSQLRecordStore(BasePostgreSQLRecordStore):
    """A PostgreSQL-backed store for :class:`~zenpyre.records.Record`
    objects.

    Persists records to a PostgreSQL database and supports adding,
    retrieving, filtering, and deleting records. All metadata is
    stored as a ``JSONB`` column, which provides flexibility for
    arbitrary metadata fields without requiring a fixed schema, and
    supports efficient equality filtering via containment (``@>``).

    The constructor mirrors :func:`psycopg.connect` directly.

    Args:
        conninfo: The connection string (DSN) passed to
            ``psycopg.connect`` (e.g.
            ``"postgresql://user:password@host:port/dbname"``).
        **kwargs: Additional keyword arguments to pass to
            ``psycopg.connect``.

    Example:
        ```pycon
        >>> from zenpyre.record_stores import PostgreSQLRecordStore
        >>> from zenpyre.records import Record
        >>> store = PostgreSQLRecordStore(  # doctest: +SKIP
        ...     "postgresql://user:password@localhost:5432/mydb"
        ... )
        >>> records = [
        ...     Record(
        ...         id="1",
        ...         metadata={"author": "Alice", "category": "Programming"},
        ...     ),
        ...     Record(
        ...         id="2",
        ...         metadata={"author": "Alice", "category": "Programming"},
        ...     ),
        ...     Record(
        ...         id="3",
        ...         metadata={"author": "Bob", "category": "History"},
        ...     ),
        ... ]
        >>> store.add_records(records)  # doctest: +SKIP
        >>> len(store.filter(author="Alice"))  # doctest: +SKIP
        2

        ```
    """

    def __init__(self, conninfo: str = "", **kwargs: Any) -> None:
        super().__init__(conninfo, **kwargs)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
        except psycopg.errors.InsufficientPrivilege:
            # Connection lacks CREATE privileges (e.g. a read-only role);
            # assume the table already exists.
            self._conn.rollback()

    def add_records(self, records: list[Record]) -> None:
        if records:
            with self._conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO records (id, metadata) VALUES (%s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET metadata = EXCLUDED.metadata",
                    [(record.id, json.dumps(record.metadata)) for record in records],
                )
            self._conn.commit()
        logger.info("Added %s records", f"{len(records):,}")

    def get(self, record_id: str) -> Record | None:
        row = self._conn.execute(
            "SELECT id, metadata FROM records WHERE id = %s",
            (record_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_many(self, record_ids: list[str]) -> list[Record | None]:
        if not record_ids:
            return []
        rows = self._conn.execute(
            "SELECT id, metadata FROM records WHERE id = ANY(%s)",
            (record_ids,),
        ).fetchall()
        by_id = {row[0]: self._row_to_record(row) for row in rows}
        return [by_id.get(record_id) for record_id in record_ids]

    def filter(self, **metadata_filters: Any) -> list[Record]:
        if not metadata_filters:
            return self.all()

        rows = self._conn.execute(
            "SELECT id, metadata FROM records WHERE metadata @> %s::jsonb",
            (json.dumps(metadata_filters),),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def check_ids(self, record_ids: list[str]) -> tuple[list[str], list[str]]:
        if not record_ids:
            return [], []
        existing = {
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM records WHERE id = ANY(%s)",
                (record_ids,),
            ).fetchall()
        }
        found = [i for i in record_ids if i in existing]
        missing = [i for i in record_ids if i not in existing]
        return found, missing

    def all(self) -> list[Record]:
        rows = self._conn.execute("SELECT id, metadata FROM records").fetchall()
        return [self._row_to_record(row) for row in rows]

    def iter_batches(self, batch_size: int = 32) -> Generator[list[Record], None, None]:
        if batch_size < 1:
            msg = f"batch_size must be a positive integer, got {batch_size}"
            raise ValueError(msg)

        with self._conn.cursor(name=f"record_store_{uuid.uuid4().hex}") as cursor:
            cursor.execute("SELECT id, metadata FROM records")
            while rows := cursor.fetchmany(batch_size):
                yield [self._row_to_record(row) for row in rows]

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> Record:
        """Convert a raw database row to a Record."""
        record_id, metadata = row
        return Record(id=record_id, metadata=metadata or {})
