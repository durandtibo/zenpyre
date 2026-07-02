r"""Contain all ingestors."""

from __future__ import annotations

__all__ = [
    "BaseIngestor",
    "DataclassIngestor",
    "FirstNIngestor",
    "InMemoryIngestor",
    "LastNIngestor",
    "PickleIngestor",
    "TextIngestor",
]

from zenpyre.ingestors.base import BaseIngestor
from zenpyre.ingestors.dataclass import DataclassIngestor
from zenpyre.ingestors.first_n import FirstNIngestor
from zenpyre.ingestors.in_memory import InMemoryIngestor
from zenpyre.ingestors.last_n import LastNIngestor
from zenpyre.ingestors.pickle import PickleIngestor
from zenpyre.ingestors.text import TextIngestor
