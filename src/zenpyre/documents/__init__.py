r"""Contain utilities for documents."""

from __future__ import annotations

__all__ = [
    "assign_ids",
    "filter_by_metadata",
    "filter_by_metadata_range",
    "filter_by_metadata_values",
    "hash_document",
    "hash_document_uuid",
    "hash_documents",
    "sort_by_metadata",
]

from zenpyre.documents.hashing import hash_document, hash_document_uuid, hash_documents
from zenpyre.documents.ops import (
    assign_ids,
    filter_by_metadata,
    filter_by_metadata_range,
    filter_by_metadata_values,
    sort_by_metadata,
)
