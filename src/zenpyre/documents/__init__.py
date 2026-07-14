r"""Contain utilities for documents."""

from __future__ import annotations

__all__ = [
    "DocumentHasher",
    "assign_ids",
    "copy_ids_to_metadata",
    "filter_by_metadata",
    "filter_by_metadata_range",
    "filter_by_metadata_values",
    "format_documents",
    "format_documents_as_json",
    "format_documents_as_markdown",
    "format_documents_as_xml",
    "hash_document",
    "hash_document_uuid",
    "hash_documents",
    "is_document_empty",
    "is_document_whitespace_only",
    "sort_by_metadata",
]

from zenpyre.documents.concatenation import (
    format_documents,
    format_documents_as_json,
    format_documents_as_markdown,
    format_documents_as_xml,
)
from zenpyre.documents.empty import is_document_empty, is_document_whitespace_only
from zenpyre.documents.filter import (
    filter_by_metadata,
    filter_by_metadata_range,
    filter_by_metadata_values,
)
from zenpyre.documents.hashing import (
    DocumentHasher,
    hash_document,
    hash_document_uuid,
    hash_documents,
)
from zenpyre.documents.id import assign_ids, copy_ids_to_metadata
from zenpyre.documents.sort import sort_by_metadata
