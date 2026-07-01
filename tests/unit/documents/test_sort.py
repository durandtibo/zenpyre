"""Unit tests for sort_by_metadata."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import sort_by_metadata


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="B", metadata={"source": "b.txt", "category": "Science"}),
        Document(page_content="A", metadata={"source": "a.txt", "category": "Cooking"}),
        Document(page_content="C", metadata={"source": "c.txt", "category": "Technology"}),
    ]


##########################################
#     Tests for sort_by_metadata         #
##########################################


# --- Return type and non-mutation ---


def test_sort_by_metadata_returns_list(docs: list[Document]) -> None:
    assert isinstance(sort_by_metadata(docs, "source"), list)


def test_sort_by_metadata_does_not_mutate_input(docs: list[Document]) -> None:
    original_order = [doc.page_content for doc in docs]
    sort_by_metadata(docs, "source")
    assert [doc.page_content for doc in docs] == original_order


def test_sort_by_metadata_returns_new_list(docs: list[Document]) -> None:
    assert sort_by_metadata(docs, "source") is not docs


# --- Sorting ---


def test_sort_by_metadata_ascending(docs: list[Document]) -> None:
    result = sort_by_metadata(docs, "source")
    assert [doc.metadata["source"] for doc in result] == ["a.txt", "b.txt", "c.txt"]


def test_sort_by_metadata_reverse(docs: list[Document]) -> None:
    result = sort_by_metadata(docs, "source", reverse=True)
    assert [doc.metadata["source"] for doc in result] == ["c.txt", "b.txt", "a.txt"]


def test_sort_by_metadata_already_sorted() -> None:
    docs = [
        Document(page_content="A", metadata={"source": "a.txt"}),
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="C", metadata={"source": "c.txt"}),
    ]
    result = sort_by_metadata(docs, "source")
    assert [doc.metadata["source"] for doc in result] == ["a.txt", "b.txt", "c.txt"]


def test_sort_by_metadata_integer_values() -> None:
    docs = [
        Document(page_content="C", metadata={"page": 3}),
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
    ]
    result = sort_by_metadata(docs, "page")
    assert [doc.metadata["page"] for doc in result] == [1, 2, 3]


# --- Missing keys ---


def test_sort_by_metadata_missing_key_placed_last_by_default() -> None:
    docs = [
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="X"),
        Document(page_content="A", metadata={"source": "a.txt"}),
    ]
    result = sort_by_metadata(docs, "source")
    assert result[-1].page_content == "X"
    assert [doc.metadata.get("source") for doc in result] == ["a.txt", "b.txt", None]


def test_sort_by_metadata_keep_missing_true() -> None:
    docs = [
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="X"),
        Document(page_content="A", metadata={"source": "a.txt"}),
    ]
    result = sort_by_metadata(docs, "source", keep_missing=True)
    assert len(result) == 3


def test_sort_by_metadata_keep_missing_false() -> None:
    docs = [
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="X"),
        Document(page_content="A", metadata={"source": "a.txt"}),
    ]
    result = sort_by_metadata(docs, "source", keep_missing=False)
    assert len(result) == 2
    assert all("source" in doc.metadata for doc in result)


def test_sort_by_metadata_keep_missing_false_preserves_order() -> None:
    docs = [
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="X"),
        Document(page_content="A", metadata={"source": "a.txt"}),
    ]
    result = sort_by_metadata(docs, "source", keep_missing=False)
    assert [doc.metadata["source"] for doc in result] == ["a.txt", "b.txt"]


def test_sort_by_metadata_all_missing_keep_true() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert len(sort_by_metadata(docs, "source")) == 2


def test_sort_by_metadata_all_missing_keep_false() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert sort_by_metadata(docs, "source", keep_missing=False) == []


# --- Edge cases ---


def test_sort_by_metadata_empty_list() -> None:
    assert sort_by_metadata([], "source") == []


def test_sort_by_metadata_single_document() -> None:
    docs = [Document(page_content="A", metadata={"source": "a.txt"})]
    assert sort_by_metadata(docs, "source") == docs


def test_sort_by_metadata_reverse_missing_still_last() -> None:
    docs = [
        Document(page_content="B", metadata={"source": "b.txt"}),
        Document(page_content="X"),
        Document(page_content="A", metadata={"source": "a.txt"}),
    ]
    result = sort_by_metadata(docs, "source", reverse=True)
    assert result[-1].page_content == "X"
