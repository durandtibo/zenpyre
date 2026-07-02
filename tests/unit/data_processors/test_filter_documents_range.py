from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from zenpyre.data_processors import FilterDocumentsByMetadataRangeProcessor

MODULE = "zenpyre.data_processors.filter_documents_range"


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 3}),
        Document(page_content="C", metadata={"page": 5}),
        Document(page_content="D", metadata={"page": 7}),
    ]


#######################################################
#   Tests for FilterDocumentsByMetadataRangeProcessor #
#######################################################


# --- Constructor ---


def test_filter_documents_by_metadata_range_processor_stores_metadata_key() -> None:
    assert FilterDocumentsByMetadataRangeProcessor(metadata_key="page")._metadata_key == "page"


def test_filter_documents_by_metadata_range_processor_lower_default_none() -> None:
    assert FilterDocumentsByMetadataRangeProcessor(metadata_key="page")._lower is None


def test_filter_documents_by_metadata_range_processor_upper_default_none() -> None:
    assert FilterDocumentsByMetadataRangeProcessor(metadata_key="page")._upper is None


def test_filter_documents_by_metadata_range_processor_stores_lower() -> None:
    p = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=2)
    assert p._lower == 2


def test_filter_documents_by_metadata_range_processor_stores_upper() -> None:
    p = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", upper=5)
    assert p._upper == 5


def test_filter_documents_by_metadata_range_processor_repr_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataRangeProcessor" in repr(
        FilterDocumentsByMetadataRangeProcessor(metadata_key="page")
    )


def test_filter_documents_by_metadata_range_processor_str_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataRangeProcessor" in str(
        FilterDocumentsByMetadataRangeProcessor(metadata_key="page")
    )


# --- process ---


def test_filter_documents_by_metadata_range_processor_process_returns_list(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=1).process(docs)
    assert isinstance(result, list)


def test_filter_documents_by_metadata_range_processor_process_both_bounds(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=2, upper=5).process(
        docs
    )
    assert result == [
        Document(page_content="B", metadata={"page": 3}),
        Document(page_content="C", metadata={"page": 5}),
    ]


def test_filter_documents_by_metadata_range_processor_process_lower_only(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=5).process(docs)
    assert result == [
        Document(page_content="C", metadata={"page": 5}),
        Document(page_content="D", metadata={"page": 7}),
    ]


def test_filter_documents_by_metadata_range_processor_process_upper_only(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", upper=3).process(docs)
    assert result == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 3}),
    ]


def test_filter_documents_by_metadata_range_processor_process_no_match_returns_empty(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataRangeProcessor(
        metadata_key="page", lower=10, upper=20
    ).process(docs)
    assert result == []


def test_filter_documents_by_metadata_range_processor_process_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"page": 3}),
        Document(page_content="B"),
    ]
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=1, upper=5).process(
        docs
    )
    assert result == [Document(page_content="A", metadata={"page": 3})]


def test_filter_documents_by_metadata_range_processor_process_does_not_mutate_input(
    docs: list[Document],
) -> None:
    original_len = len(docs)
    FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=1, upper=5).process(docs)
    assert len(docs) == original_len


def test_filter_documents_by_metadata_range_processor_process_calls_filter_by_metadata_range(
    docs: list[Document],
) -> None:
    with patch(f"{MODULE}.filter_by_metadata_range", return_value=docs) as mock_filter:
        FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=2, upper=6).process(docs)
    mock_filter.assert_called_once_with(docs, "page", 2, 6)


def test_filter_documents_by_metadata_range_processor_process_empty_list() -> None:
    result = FilterDocumentsByMetadataRangeProcessor(metadata_key="page", lower=1, upper=5).process(
        []
    )
    assert result == []
