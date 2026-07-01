r"""Contain all ingestors."""

from __future__ import annotations

__all__ = ["BaseIngestor", "PickleIngestor", "TextIngestor"]

from zenpyre.ingestors.base import BaseIngestor
from zenpyre.ingestors.pickle import PickleIngestor
from zenpyre.ingestors.text import TextIngestor
