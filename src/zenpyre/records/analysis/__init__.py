r"""Contain code to analyze records."""

from __future__ import annotations

__all__ = ["RecordMetadataStats", "compute_record_metadata_stats", "print_metadata_stats_report"]

from zenpyre.records.analysis.metadata import (
    RecordMetadataStats,
    compute_record_metadata_stats,
)
from zenpyre.records.analysis.metadata_print import print_metadata_stats_report
