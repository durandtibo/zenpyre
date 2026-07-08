r"""Contain record stores."""

from __future__ import annotations

__all__ = ["BaseRecordStore", "DuckDBRecordStore", "InMemoryRecordStore", "TypedDuckDBRecordStore"]

from zenpyre.record_stores.base import BaseRecordStore
from zenpyre.record_stores.duckdb import DuckDBRecordStore
from zenpyre.record_stores.in_memory import InMemoryRecordStore
from zenpyre.record_stores.typed_duckdb import TypedDuckDBRecordStore
