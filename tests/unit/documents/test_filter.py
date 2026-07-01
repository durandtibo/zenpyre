from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import (
    filter_by_metadata,
    filter_by_metadata_range,
    filter_by_metadata_values,
)


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="B", metadata={"category": "Cooking", "page": 2}),
        Document(page_content="C", metadata={"category": "Science", "page": 3}),
        Document(page_content="D", metadata={"category": "Technology", "page": 4}),
    ]


@pytest.fixture
def page_docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 5}),
        Document(page_content="C", metadata={"page": 10}),
        Document(page_content="D", metadata={"page": 15}),
    ]


@pytest.fixture
def category_docs() -> list[Document]:
    return [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="B", metadata={"category": "Cooking", "page": 2}),
        Document(page_content="C", metadata={"category": "Technology", "page": 3}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


##########################################
#     Tests for filter_by_metadata       #
##########################################


# --- Return type and non-mutation ---


def test_filter_by_metadata_returns_list(docs: list[Document]) -> None:
    assert isinstance(filter_by_metadata(docs, "category", "Science"), list)


def test_filter_by_metadata_does_not_mutate_input(docs: list[Document]) -> None:
    original_len = len(docs)
    filter_by_metadata(docs, "category", "Science")
    assert len(docs) == original_len


def test_filter_by_metadata_returns_new_list(docs: list[Document]) -> None:
    assert filter_by_metadata(docs, "category", "Science") is not docs


# --- Filtering ---


def test_filter_by_metadata_matching_docs(docs: list[Document]) -> None:
    assert filter_by_metadata(docs, "category", "Science") == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="C", metadata={"category": "Science", "page": 3}),
    ]


def test_filter_by_metadata_no_match(docs: list[Document]) -> None:
    assert filter_by_metadata(docs, "category", "Sports") == []


def test_filter_by_metadata_all_match(docs: list[Document]) -> None:
    result = filter_by_metadata(docs, "category", "Science")
    assert all(doc.metadata["category"] == "Science" for doc in result)


def test_filter_by_metadata_integer_value() -> None:
    docs = [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="C", metadata={"page": 1}),
    ]
    assert filter_by_metadata(docs, "page", 1) == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="C", metadata={"page": 1}),
    ]


def test_filter_by_metadata_boolean_value() -> None:
    docs = [
        Document(page_content="A", metadata={"published": True}),
        Document(page_content="B", metadata={"published": False}),
        Document(page_content="C", metadata={"published": True}),
    ]
    assert filter_by_metadata(docs, "published", True) == [
        Document(page_content="A", metadata={"published": True}),
        Document(page_content="C", metadata={"published": True}),
    ]


# --- Missing keys ---


def test_filter_by_metadata_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"category": "Science"}),
        Document(page_content="B"),
    ]
    assert filter_by_metadata(docs, "category", "Science") == [
        Document(page_content="A", metadata={"category": "Science"}),
    ]


def test_filter_by_metadata_all_missing_key_returns_empty() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert filter_by_metadata(docs, "category", "Science") == []


# --- Edge cases ---


def test_filter_by_metadata_empty_list() -> None:
    assert filter_by_metadata([], "category", "Science") == []


def test_filter_by_metadata_single_match() -> None:
    assert filter_by_metadata(
        [Document(page_content="A", metadata={"category": "Science"})],
        "category",
        "Science",
    ) == [Document(page_content="A", metadata={"category": "Science"})]


def test_filter_by_metadata_single_no_match() -> None:
    assert (
        filter_by_metadata(
            [Document(page_content="A", metadata={"category": "Cooking"})],
            "category",
            "Science",
        )
        == []
    )


def test_filter_by_metadata_preserves_order(docs: list[Document]) -> None:
    assert filter_by_metadata(docs, "category", "Science") == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="C", metadata={"category": "Science", "page": 3}),
    ]


###############################################
#     Tests for filter_by_metadata_range      #
###############################################


# --- Return type and non-mutation ---


def test_filter_by_metadata_range_returns_list(page_docs: list[Document]) -> None:
    assert isinstance(filter_by_metadata_range(page_docs, "page", lower=1), list)


def test_filter_by_metadata_range_does_not_mutate_input(page_docs: list[Document]) -> None:
    original_len = len(page_docs)
    filter_by_metadata_range(page_docs, "page", lower=1, upper=10)
    assert len(page_docs) == original_len


def test_filter_by_metadata_range_returns_new_list(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=1) is not page_docs


# --- Both bounds ---


def test_filter_by_metadata_range_lower_and_upper(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=2, upper=8) == [
        Document(page_content="B", metadata={"page": 5}),
    ]


def test_filter_by_metadata_range_inclusive_lower_bound(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=5, upper=10) == [
        Document(page_content="B", metadata={"page": 5}),
        Document(page_content="C", metadata={"page": 10}),
    ]


def test_filter_by_metadata_range_inclusive_upper_bound(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=1, upper=5) == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 5}),
    ]


# --- Lower bound only ---


def test_filter_by_metadata_range_lower_only(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=5) == [
        Document(page_content="B", metadata={"page": 5}),
        Document(page_content="C", metadata={"page": 10}),
        Document(page_content="D", metadata={"page": 15}),
    ]


# --- Upper bound only ---


def test_filter_by_metadata_range_upper_only(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", upper=5) == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 5}),
    ]


# --- Both bounds None ---


def test_filter_by_metadata_range_no_bounds_returns_docs_with_key(
    page_docs: list[Document],
) -> None:
    docs_with_missing = [*page_docs, Document(page_content="X")]
    assert filter_by_metadata_range(docs_with_missing, "page") == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 5}),
        Document(page_content="C", metadata={"page": 10}),
        Document(page_content="D", metadata={"page": 15}),
    ]


# --- Missing keys ---


def test_filter_by_metadata_range_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"page": 5}),
        Document(page_content="B"),
    ]
    assert filter_by_metadata_range(docs, "page", lower=1, upper=10) == [
        Document(page_content="A", metadata={"page": 5}),
    ]


def test_filter_by_metadata_range_all_missing_key_returns_empty() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert filter_by_metadata_range(docs, "page", lower=1, upper=10) == []


# --- No match ---


def test_filter_by_metadata_range_no_match(page_docs: list[Document]) -> None:
    assert filter_by_metadata_range(page_docs, "page", lower=20, upper=30) == []


# --- Edge cases ---


def test_filter_by_metadata_range_empty_list() -> None:
    assert filter_by_metadata_range([], "page", lower=1, upper=10) == []


def test_filter_by_metadata_range_exact_match() -> None:
    assert filter_by_metadata_range(
        [Document(page_content="A", metadata={"page": 5})],
        "page",
        lower=5,
        upper=5,
    ) == [Document(page_content="A", metadata={"page": 5})]


###################################################
#     Tests for filter_by_metadata_values         #
###################################################


# --- Return type and non-mutation ---


def test_filter_by_metadata_values_returns_list(category_docs: list[Document]) -> None:
    assert isinstance(filter_by_metadata_values(category_docs, "category", {"Science"}), list)


def test_filter_by_metadata_values_does_not_mutate_input(category_docs: list[Document]) -> None:
    original_len = len(category_docs)
    filter_by_metadata_values(category_docs, "category", {"Science"})
    assert len(category_docs) == original_len


def test_filter_by_metadata_values_returns_new_list(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", {"Science"}) is not category_docs


# --- Filtering ---


def test_filter_by_metadata_values_single_value(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", {"Science"}) == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


def test_filter_by_metadata_values_multiple_values(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", {"Science", "Technology"}) == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="C", metadata={"category": "Technology", "page": 3}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]


def test_filter_by_metadata_values_all_values(category_docs: list[Document]) -> None:
    assert (
        filter_by_metadata_values(category_docs, "category", {"Science", "Cooking", "Technology"})
        == category_docs
    )


def test_filter_by_metadata_values_no_match(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", {"Sports"}) == []


def test_filter_by_metadata_values_integer_values() -> None:
    docs = [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="C", metadata={"page": 3}),
    ]
    assert filter_by_metadata_values(docs, "page", {1, 3}) == [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="C", metadata={"page": 3}),
    ]


# --- Missing keys ---


def test_filter_by_metadata_values_missing_key_excluded() -> None:
    docs = [
        Document(page_content="A", metadata={"category": "Science"}),
        Document(page_content="B"),
    ]
    assert filter_by_metadata_values(docs, "category", {"Science"}) == [
        Document(page_content="A", metadata={"category": "Science"}),
    ]


def test_filter_by_metadata_values_all_missing_key_returns_empty() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert filter_by_metadata_values(docs, "category", {"Science"}) == []


# --- Edge cases ---


def test_filter_by_metadata_values_empty_list() -> None:
    assert filter_by_metadata_values([], "category", {"Science"}) == []


def test_filter_by_metadata_values_empty_set(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", set()) == []


def test_filter_by_metadata_values_single_document_match() -> None:
    docs = [Document(page_content="A", metadata={"category": "Science"})]
    assert filter_by_metadata_values(docs, "category", {"Science"}) == [
        Document(page_content="A", metadata={"category": "Science"}),
    ]


def test_filter_by_metadata_values_single_document_no_match() -> None:
    docs = [Document(page_content="A", metadata={"category": "Cooking"})]
    assert filter_by_metadata_values(docs, "category", {"Science"}) == []


def test_filter_by_metadata_values_preserves_order(category_docs: list[Document]) -> None:
    assert filter_by_metadata_values(category_docs, "category", {"Science", "Cooking"}) == [
        Document(page_content="A", metadata={"category": "Science", "page": 1}),
        Document(page_content="B", metadata={"category": "Cooking", "page": 2}),
        Document(page_content="D", metadata={"category": "Science", "page": 4}),
    ]
