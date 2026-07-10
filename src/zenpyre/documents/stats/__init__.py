r"""Contain utilities to compute document statistics."""

from __future__ import annotations

__all__ = [
    "ApproxDocContentStats",
    "DocMetadataStats",
    "ExactDocContentStats",
    "compute_doc_content_stats_approx",
    "compute_doc_content_stats_exact",
    "compute_doc_metadata_stats",
    "print_doc_content_stats",
]

from zenpyre.documents.stats.content_approx import (
    ApproxDocContentStats,
    compute_doc_content_stats_approx,
)
from zenpyre.documents.stats.content_exact import (
    ExactDocContentStats,
    compute_doc_content_stats_exact,
)
from zenpyre.documents.stats.content_print import print_doc_content_stats
from zenpyre.documents.stats.metadata import (
    DocMetadataStats,
    compute_doc_metadata_stats,
)
