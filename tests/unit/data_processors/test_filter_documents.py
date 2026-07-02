from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from zenpyre.data_processors import FilterDocumentsByMetadataProcessor

MODULE = "zenpyre.data_processors.filter_documents"


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="B", metadata={"category": "Cooking", "page": 2}),
        Document(page_content="C", metadata={"category": "Science", "page": 3}),
        Document(page_content="D", metadata={"category": "Technology", "page": 4}),
    ]


###################################################
#   Tests for FilterDocumentsByMetadataProcessor  #
###################################################


# --- Constructor ---


def test_filter_documents_by_metadata_processor_stores_metadata_key() -> None:
    assert (
        FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science")._metadata_key
        == "category"
    )


def test_filter_documents_by_metadata_processor_stores_value() -> None:
    assert (
        FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science")._value
        == "Science"
    )


def test_filter_documents_by_metadata_processor_stores_integer_value() -> None:
    assert FilterDocumentsByMetadataProcessor(metadata_key="page", value=1)._value == 1


def test_filter_documents_by_metadata_processor_repr_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataProcessor" in repr(
        FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science")
    )


def test_filter_documents_by_metadata_processor_str_contains_class_name() -> None:
    assert "FilterDocumentsByMetadataProcessor" in str(
        FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science")
    )


# --- process ---


def test_filter_documents_by_metadata_processor_process_returns_list(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(
        docs
    )
    assert isinstance(result, list)


def test_filter_documents_by_metadata_processor_process_filters_matching(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(
        docs
    )
    assert result == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="C", metadata={"category": "Science", "page": 3}),
    ]


def test_filter_documents_by_metadata_processor_process_no_match_returns_empty(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataProcessor(metadata_key="category", value="Sports").process(
        docs
    )
    assert result == []


def test_filter_documents_by_metadata_processor_process_integer_value(
    docs: list[Document],
) -> None:
    result = FilterDocumentsByMetadataProcessor(metadata_key="page", value=1).process(docs)
    assert result == [Document(page_content="A", metadata={"category": "Science", "page": 1})]


def test_filter_documents_by_metadata_processor_process_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"category": "Science"}),
        Document(page_content="B"),
    ]
    result = FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(
        docs
    )
    assert result == [Document(page_content="A", metadata={"category": "Science"})]


def test_filter_documents_by_metadata_processor_process_does_not_mutate_input(
    docs: list[Document],
) -> None:
    original_len = len(docs)
    FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(docs)
    assert len(docs) == original_len


def test_filter_documents_by_metadata_processor_process_calls_filter_by_metadata(
    docs: list[Document],
) -> None:
    with patch(f"{MODULE}.filter_by_metadata", return_value=docs) as mock_filter:
        FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(docs)
    mock_filter.assert_called_once_with(docs, "category", "Science")


def test_filter_documents_by_metadata_processor_process_empty_list() -> None:
    result = FilterDocumentsByMetadataProcessor(metadata_key="category", value="Science").process(
        []
    )
    assert result == []
