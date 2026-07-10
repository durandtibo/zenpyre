r"""Contain utilities to compute document statistics."""

from __future__ import annotations

__all__ = [
    "ApproxDocContentStats",
    "ExactDocContentStats",
    "compute_doc_content_stats_approx",
    "compute_doc_content_stats_exact",
]

from zenpyre.documents.stats.content_approx import (
    ApproxDocContentStats,
    compute_doc_content_stats_approx,
)
from zenpyre.documents.stats.content_exact import (
    ExactDocContentStats,
    compute_doc_content_stats_exact,
)
