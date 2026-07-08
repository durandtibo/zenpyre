r"""Contain record stores."""

from __future__ import annotations

__all__ = ["BaseRecordStore", "InMemoryRecordStore"]

from zenpyre.record_stores.base import BaseRecordStore
from zenpyre.record_stores.in_memory import InMemoryRecordStore
