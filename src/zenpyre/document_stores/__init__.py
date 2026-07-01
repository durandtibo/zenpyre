r"""Contain document stores."""

from __future__ import annotations

__all__ = ["DuckDBDocumentStore", "TypedDuckDBDocumentStore"]

from zenpyre.document_stores.duckdb import DuckDBDocumentStore
from zenpyre.document_stores.typed_duckdb import TypedDuckDBDocumentStore
