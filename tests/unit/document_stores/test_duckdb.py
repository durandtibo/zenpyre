"""Unit tests for DuckDBDocumentStore."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.testing.fixtures import duckdb_available

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> DuckDBDocumentStore:
    return DuckDBDocumentStore(":memory:")


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            id="1",
            page_content="Intro to Python",
            metadata={"author": "Alice", "year": 2022, "category": "Programming"},
        ),
        Document(
            id="2",
            page_content="Advanced Python",
            metadata={"author": "Alice", "year": 2023, "category": "Programming"},
        ),
        Document(
            id="3",
            page_content="History of Rome",
            metadata={"author": "Bob", "year": 2021, "category": "History"},
        ),
        Document(
            id="4",
            page_content="History of Greece",
            metadata={"author": "Bob", "year": 2020, "category": "History"},
        ),
    ]


##################################################
#     Tests for DuckDBDocumentStore              #
##################################################


# --- add_documents ---


@duckdb_available
def test_add_documents_increases_count(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


@duckdb_available
def test_add_documents_no_id_raises(store: DuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match=r"id"):
        store.add_documents([Document(page_content="No id")])


@duckdb_available
def test_add_documents_upsert_replaces_existing(store: DuckDBDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Original", metadata={})])
    store.add_documents([Document(id="1", page_content="Updated", metadata={})])
    assert store.count() == 1
    assert store.get("1").page_content == "Updated"


# --- count ---


@duckdb_available
def test_count_empty_store(store: DuckDBDocumentStore) -> None:
    assert store.count() == 0


@duckdb_available
def test_count_after_adding(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


# --- get ---


@duckdb_available
def test_get_existing_document(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.get("1") == docs[0]


@duckdb_available
def test_get_missing_document_returns_none(store: DuckDBDocumentStore) -> None:
    assert store.get("nonexistent") is None


@duckdb_available
def test_get_round_trips_metadata(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.get("1").metadata == docs[0].metadata


# --- get_many ---


@duckdb_available
def test_get_many_returns_correct_length(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert len(store.get_many(["1", "2", "99"])) == 3


@duckdb_available
def test_get_many_returns_none_for_missing(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


@duckdb_available
def test_get_many_preserves_order(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- all ---


@duckdb_available
def test_all_empty_store(store: DuckDBDocumentStore) -> None:
    assert store.all() == []


@duckdb_available
def test_all_returns_all_documents(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.all()
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


# --- filter ---


@duckdb_available
def test_filter_no_args_returns_all(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert len(store.filter()) == len(docs)


@duckdb_available
def test_filter_single_field(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


@duckdb_available
def test_filter_multiple_fields(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Alice", category="Programming")
    assert len(result) == 2


@duckdb_available
def test_filter_no_match_returns_empty(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.filter(author="Charlie") == []


@duckdb_available
def test_filter_preserves_full_document(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Bob", category="History")
    assert all(
        r.metadata == d.metadata
        for r, d in zip(
            sorted(result, key=lambda x: x.id),
            sorted([d for d in docs if d.metadata["author"] == "Bob"], key=lambda x: x.id),
        )
    )


@duckdb_available
def test_filter_empty_store_returns_empty(store: DuckDBDocumentStore) -> None:
    assert store.filter(author="Alice") == []


# --- delete ---


@duckdb_available
def test_delete_removes_document(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete("1")
    assert store.count() == len(docs) - 1
    assert store.get("1") is None


@duckdb_available
def test_delete_nonexistent_is_silent(store: DuckDBDocumentStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


@duckdb_available
def test_delete_many_removes_documents(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.count() == len(docs) - 2
    assert store.get("1") is None
    assert store.get("3") is None


@duckdb_available
def test_delete_many_preserves_other_documents(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


@duckdb_available
def test_delete_many_empty_list_is_no_op(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many([])
    assert store.count() == len(docs)


@duckdb_available
def test_delete_many_nonexistent_ids_are_silent(store: DuckDBDocumentStore) -> None:
    store.delete_many(["99", "100"])


@duckdb_available
def test_delete_many_single_id(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["2"])
    assert store.count() == len(docs) - 1
    assert store.get("2") is None


# --- check_ids ---


@duckdb_available
def test_check_ids_all_found(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


@duckdb_available
def test_check_ids_all_missing(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


@duckdb_available
def test_check_ids_mixed(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


@duckdb_available
def test_check_ids_preserves_order(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


@duckdb_available
def test_check_ids_empty_input_returns_empty_lists(store: DuckDBDocumentStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


@duckdb_available
def test_check_ids_empty_store_returns_all_missing(store: DuckDBDocumentStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


@duckdb_available
def test_check_ids_returns_tuple_of_two_lists(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
