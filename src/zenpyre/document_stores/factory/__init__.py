r"""Contain factories for document stores."""

from __future__ import annotations

__all__ = [
    "BaseDocumentStoreFactory",
    "ConfigurableDocumentStoreFactory",
    "DocumentStoreFactory",
    "DuckDBDocumentStoreFactory",
]

from zenpyre.document_stores.factory.base import BaseDocumentStoreFactory
from zenpyre.document_stores.factory.configurable import (
    ConfigurableDocumentStoreFactory,
)
from zenpyre.document_stores.factory.duckdb import DuckDBDocumentStoreFactory
from zenpyre.document_stores.factory.vanilla import DocumentStoreFactory
