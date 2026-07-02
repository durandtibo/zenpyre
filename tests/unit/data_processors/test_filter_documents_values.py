"""Unit tests for FilterDocumentsByMetadataValuesProcessor."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from zenpyre.data_processors import FilterDocumentsByMetadataValuesProcessor

MODULE = "zenpyre.data_processors.filter_documents_values"


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="B", metadata={"category": "Cooking", "page": 2}),
        Document(page_content="C", metadata={"category": "Technology", "page": 3}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


#########################################################
#   Tests for FilterDocumentsByMetadataValuesProcessor  #
#########################################################


# --- Constructor ---


def test_filter_documents_by_metadata_values_processor_stores_metadata_key() -> None:
    p = FilterDocumentsByMetadataValuesProcessor(metadata_key="category", values={"Science"})
    assert p._metadata_key == "category"


def test_filter_documents_by_metadata_values_processor_stores_values() -> None:
    p = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science", "Technology"}
    )
    assert p._values == {"Science", "Technology"}


def test_filter_documents_by_metadata_values_processor_repr_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataValuesProcessor" in repr(
        FilterDocumentsByMetadataValuesProcessor(metadata_key="category", values={"Science"})
    )


def test_filter_documents_by_metadata_values_processor_str_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataValuesProcessor" in str(
        FilterDocumentsByMetadataValuesProcessor(metadata_key="category", values={"Science"})
    )


# --- process ---


def test_filter_documents_by_metadata_values_processor_process_returns_list(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science"}
    ).process(docs)
    assert isinstance(result, list)


def test_filter_documents_by_metadata_values_processor_process_single_value(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science"}
    ).process(docs)
    assert result == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


def test_filter_documents_by_metadata_values_processor_process_multiple_values(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science", "Technology"}
    ).process(docs)
    assert result == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="C", metadata={"category": "Technology", "page": 3}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


def test_filter_documents_by_metadata_values_processor_process_no_match_returns_empty(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Sports"}
    ).process(docs)
    assert result == []


def test_filter_documents_by_metadata_values_processor_process_empty_set_returns_empty(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values=set()
    ).process(docs)
    assert result == []


def test_filter_documents_by_metadata_values_processor_process_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"category": "Science"}),
        Document(page_content="B"),
    ]
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science"}
    ).process(docs)
    assert result == [Document(page_content="A", metadata={"category": "Science"})]


def test_filter_documents_by_metadata_values_processor_process_does_not_mutate_input(
    docs: list[Document],
) -> None:
    original_len = len(docs)
    FilterDocumentsByMetadataValuesProcessor(metadata_key="category", values={"Science"}).process(
        docs
    )
    assert len(docs) == original_len


def test_filter_documents_by_metadata_values_processor_process_calls_filter_by_metadata_values(
    docs: list[Document],
) -> None:
    values = {"Science", "Technology"}
    with patch(f"{MODULE}.filter_by_metadata_values", return_value=docs) as mock_filter:
        FilterDocumentsByMetadataValuesProcessor(metadata_key="category", values=values).process(
            docs
        )
    mock_filter.assert_called_once_with(docs, "category", values)


def test_filter_documents_by_metadata_values_processor_process_empty_list() -> None:
    result = FilterDocumentsByMetadataValuesProcessor(
        metadata_key="category", values={"Science"}
    ).process([])
    assert result == []
