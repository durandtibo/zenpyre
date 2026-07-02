r"""Contain data processors."""

from __future__ import annotations

__all__ = [
    "BaseProcessor",
    "FilterDocumentsByMetadataProcessor",
    "FirstNProcessor",
    "LambdaProcessor",
    "LastNProcessor",
    "SequenceProcessor",
    "SequentialProcessor",
    "SortDocumentsByMetadataProcessor",
    "SortRecordsByMetadataProcessor",
]

from zenpyre.data_processors.base import BaseProcessor
from zenpyre.data_processors.filter_documents import FilterDocumentsByMetadataProcessor
from zenpyre.data_processors.first_n import FirstNProcessor
from zenpyre.data_processors.lambdaa import LambdaProcessor
from zenpyre.data_processors.last_n import LastNProcessor
from zenpyre.data_processors.sequence import SequenceProcessor
from zenpyre.data_processors.sequential import SequentialProcessor
from zenpyre.data_processors.sort_documents import SortDocumentsByMetadataProcessor
from zenpyre.data_processors.sort_records import SortRecordsByMetadataProcessor
