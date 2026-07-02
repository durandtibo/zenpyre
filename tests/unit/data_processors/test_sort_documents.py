from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from zenpyre.data_processors import SortDocumentsByMetadataProcessor

MODULE = "zenpyre.data_processors.sort_documents"


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="B", metadata={"source": "b.txt", "page": 2}),
        Document(page_content="A", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="C", metadata={"source": "c.txt", "page": 3}),
    ]


###################################################
#   Tests for SortDocumentsByMetadataProcessor    #
###################################################


# --- Constructor ---


def test_sort_documents_by_metadata_processor_stores_metadata_key() -> None:
    assert SortDocumentsByMetadataProcessor(metadata_key="source")._metadata_key == "source"


def test_sort_documents_by_metadata_processor_keep_missing_default_true() -> None:
    assert SortDocumentsByMetadataProcessor(metadata_key="source")._keep_missing is True


def test_sort_documents_by_metadata_processor_stores_keep_missing() -> None:
    p = SortDocumentsByMetadataProcessor(metadata_key="source", keep_missing=False)
    assert p._keep_missing is False


def test_sort_documents_by_metadata_processor_reverse_default_false() -> None:
    assert SortDocumentsByMetadataProcessor(metadata_key="source")._reverse is False


def test_sort_documents_by_metadata_processor_stores_reverse() -> None:
    p = SortDocumentsByMetadataProcessor(metadata_key="source", reverse=True)
    assert p._reverse is True


def test_sort_documents_by_metadata_processor_repr_contains_class_name() -> None:
    assert "SortDocumentsByMetadataProcessor" in repr(
        SortDocumentsByMetadataProcessor(metadata_key="source")
    )


def test_sort_documents_by_metadata_processor_str_contains_class_name() -> None:
    assert "SortDocumentsByMetadataProcessor" in str(
        SortDocumentsByMetadataProcessor(metadata_key="source")
    )


# --- process ---


def test_sort_documents_by_metadata_processor_process_returns_list(
    docs: list[Document],
) -> None:
    result = SortDocumentsByMetadataProcessor(metadata_key="source").process(docs)
    assert isinstance(result, list)


def test_sort_documents_by_metadata_processor_process_sorts_ascending(
    docs: list[Document],
) -> None:
    result = SortDocumentsByMetadataProcessor(metadata_key="source").process(docs)
    assert [d.metadata["source"] for d in result] == ["a.txt", "b.txt", "c.txt"]


def test_sort_documents_by_metadata_processor_process_sorts_descending(
    docs: list[Document],
) -> None:
    result = SortDocumentsByMetadataProcessor(metadata_key="source", reverse=True).process(docs)
    assert [d.metadata["source"] for d in result] == ["c.txt", "b.txt", "a.txt"]


def test_sort_documents_by_metadata_processor_process_keep_missing_true(
    docs: list[Document],
) -> None:
    docs_with_missing = [*docs, Document(page_content="X")]
    result = SortDocumentsByMetadataProcessor(metadata_key="source").process(docs_with_missing)
    assert len(result) == 4
    assert result[-1].page_content == "X"


def test_sort_documents_by_metadata_processor_process_keep_missing_false(
    docs: list[Document],
) -> None:
    docs_with_missing = [*docs, Document(page_content="X")]
    result = SortDocumentsByMetadataProcessor(metadata_key="source", keep_missing=False).process(
        docs_with_missing
    )
    assert len(result) == 3
    assert all("source" in d.metadata for d in result)


def test_sort_documents_by_metadata_processor_process_does_not_mutate_input(
    docs: list[Document],
) -> None:
    original_order = [d.page_content for d in docs]
    SortDocumentsByMetadataProcessor(metadata_key="source").process(docs)
    assert [d.page_content for d in docs] == original_order


def test_sort_documents_by_metadata_processor_process_calls_sort_by_metadata(
    docs: list[Document],
) -> None:
    with patch(f"{MODULE}.sort_by_metadata", return_value=docs) as mock_sort:
        SortDocumentsByMetadataProcessor(
            metadata_key="source", keep_missing=False, reverse=True
        ).process(docs)
    mock_sort.assert_called_once_with(docs, "source", keep_missing=False, reverse=True)


def test_sort_documents_by_metadata_processor_process_empty_list() -> None:
    result = SortDocumentsByMetadataProcessor(metadata_key="source").process([])
    assert result == []
