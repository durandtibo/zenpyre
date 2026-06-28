r"""Contain document loaders."""

from __future__ import annotations

__all__ = ["DocumentListLoader", "resolve_document_loader"]

from zenpyre.document_loaders.resolve import resolve_document_loader
from zenpyre.document_loaders.sequence import DocumentListLoader
