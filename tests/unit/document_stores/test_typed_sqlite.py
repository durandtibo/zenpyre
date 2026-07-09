from __future__ import annotations

import sqlite3
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document

from zenpyre.document_stores import TypedSQLiteDocumentStore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("store")


@pytest.fixture
def store() -> Generator[TypedSQLiteDocumentStore, None, None]:
    """In-memory store with no schema."""
    with TypedSQLiteDocumentStore(":memory:") as store:
        yield store


@pytest.fixture
def typed_store() -> Generator[TypedSQLiteDocumentStore, None, None]:
    """In-memory store with a typed schema."""
    with TypedSQLiteDocumentStore(
        ":memory:",
        metadata_schema={"author": "TEXT", "year": "INTEGER", "category": "TEXT"},
    ) as store:
        yield store


@pytest.fixture(scope="module")
def store_read_only(
    store_path: Path, docs: list[Document]
) -> Generator[TypedSQLiteDocumentStore, None, None]:
    path = store_path / "data.sqlite"
    store = TypedSQLiteDocumentStore.from_path(path)
    store.add_documents(docs)
    store._conn.close()
    with TypedSQLiteDocumentStore.from_path(path, read_only=True) as store:
        yield store


@pytest.fixture(scope="module")
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
#     Tests for TypedSQLiteDocumentStore         #
##################################################

# --- constructor ---


def test_init_defaults_to_in_memory() -> None:
    with TypedSQLiteDocumentStore() as store:
        assert store.count() == 0


def test_init_accepts_sqlite_connect_kwargs() -> None:
    with TypedSQLiteDocumentStore(":memory:", timeout=5.0) as store:
        assert store.count() == 0


def test_init_no_schema_stores_everything_in_extra(store: TypedSQLiteDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="hello", metadata={"author": "Alice"})])
    assert set(store.get_columns_info().keys()) == {"id", "page_content", "extra"}


def test_init_with_schema_creates_typed_columns(typed_store: TypedSQLiteDocumentStore) -> None:
    columns = typed_store.get_columns_info()
    assert set(columns.keys()) == {"id", "page_content", "author", "year", "category", "extra"}


# --- from_path ---


def test_from_path_creates_file_backed_store(store_path: Path) -> None:
    path = store_path / "from_path.sqlite"
    with TypedSQLiteDocumentStore.from_path(path) as store:
        store.add_documents([Document(id="1", page_content="hello", metadata={})])
        assert store.count() == 1
        assert path.exists()


def test_from_path_memory_uses_shared_cache_uri() -> None:
    with TypedSQLiteDocumentStore.from_path(":memory:") as store:
        assert store.count() == 0


def test_from_path_read_only_can_read_existing_data(
    store_read_only: TypedSQLiteDocumentStore,
) -> None:
    assert store_read_only.count() == 4


def test_from_path_read_only_rejects_writes(store_read_only: TypedSQLiteDocumentStore) -> None:
    with pytest.raises(sqlite3.OperationalError, match=r"attempt to write a readonly database"):
        store_read_only.add_documents([Document(id="99", page_content="x", metadata={})])


def test_from_path_with_schema(store_path: Path) -> None:
    path = store_path / "with_schema.sqlite"
    schema = {"author": "TEXT", "year": "INTEGER"}
    with TypedSQLiteDocumentStore.from_path(path, metadata_schema=schema) as store:
        store.add_documents(
            [Document(id="1", page_content="hello", metadata={"author": "Alice", "year": 2022})]
        )
        assert store.get("1").metadata["year"] == 2022


# --- repr/str ---


def test_repr(store: TypedSQLiteDocumentStore) -> None:
    assert repr(store).startswith("TypedSQLiteDocumentStore(")


def test_str(store: TypedSQLiteDocumentStore) -> None:
    assert repr(store).startswith("TypedSQLiteDocumentStore(")


# --- add_documents ---


def test_add_documents_increases_count(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


def test_add_documents_no_id_raises(store: TypedSQLiteDocumentStore) -> None:
    with pytest.raises(ValueError, match=r"id"):
        store.add_documents([Document(page_content="No id")])


def test_add_documents_upsert_replaces_existing(store: TypedSQLiteDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Original", metadata={})])
    store.add_documents([Document(id="1", page_content="Updated", metadata={})])
    assert store.count() == 1
    assert store.get("1").page_content == "Updated"


def test_add_documents_empty(store: TypedSQLiteDocumentStore) -> None:
    store.add_documents([])


# --- count ---


def test_count_empty_store(store: TypedSQLiteDocumentStore) -> None:
    assert store.count() == 0


def test_count_after_adding(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


# --- get ---


def test_get_existing_document(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get("1")
    assert result == docs[0]


def test_get_missing_document_returns_none(store: TypedSQLiteDocumentStore) -> None:
    assert store.get("nonexistent") is None


def test_get_round_trips_metadata(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.get("1").metadata == docs[0].metadata


def test_get_round_trips_page_content(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.get("1").page_content == docs[0].page_content


def test_get_round_trips_extra_metadata(typed_store: TypedSQLiteDocumentStore) -> None:
    doc = Document(
        id="1", page_content="Test", metadata={"author": "Alice", "publisher": "O'Reilly"}
    )
    typed_store.add_documents([doc])
    assert typed_store.get("1").metadata["publisher"] == "O'Reilly"


# --- get_many ---


def test_get_many_returns_correct_length(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    assert len(store.get_many(["1", "2", "99"])) == 3


def test_get_many_returns_none_for_missing(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


def test_get_many_preserves_order(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- filter ---


def test_filter_no_args_returns_all(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert len(typed_store.filter()) == len(docs)


def test_filter_single_typed_field(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


def test_filter_multiple_typed_fields(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice", category="Programming")
    assert len(result) == 2


def test_filter_no_match_returns_empty(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.filter(author="Charlie") == []


def test_filter_extra_field(typed_store: TypedSQLiteDocumentStore) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"author": "Alice", "publisher": "O'Reilly"}),
        Document(id="2", page_content="B", metadata={"author": "Bob", "publisher": "Manning"}),
    ]
    typed_store.add_documents(docs)
    result = typed_store.filter(publisher="O'Reilly")
    assert len(result) == 1
    assert result[0].id == "1"


def test_filter_mixed_schema_and_extra_fields(typed_store: TypedSQLiteDocumentStore) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"author": "Alice", "publisher": "O'Reilly"}),
        Document(id="2", page_content="B", metadata={"author": "Alice", "publisher": "Manning"}),
    ]
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice", publisher="O'Reilly")
    assert len(result) == 1
    assert result[0].id == "1"


def test_filter_integer_typed_column(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(year=2022)
    assert len(result) == 1
    assert result[0].id == "1"


def test_filter_integer_typed_column_no_match(
    typed_store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.filter(year=9999) == []


# --- delete ---


def test_delete_removes_document(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete("1")
    assert store.count() == len(docs) - 1
    assert store.get("1") is None


def test_delete_nonexistent_is_silent(store: TypedSQLiteDocumentStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


def test_delete_many_removes_documents(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.count() == len(docs) - 2
    assert store.get("1") is None
    assert store.get("3") is None


def test_delete_many_preserves_other_documents(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


def test_delete_many_empty_list_is_no_op(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many([])
    assert store.count() == len(docs)


def test_delete_many_nonexistent_ids_are_silent(store: TypedSQLiteDocumentStore) -> None:
    store.delete_many(["99", "100"])


def test_delete_many_single_id(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["2"])
    assert store.count() == len(docs) - 1
    assert store.get("2") is None


# --- check_ids ---


def test_check_ids_all_found(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


def test_check_ids_all_missing(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


def test_check_ids_mixed(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


def test_check_ids_preserves_order(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


def test_check_ids_empty_input_returns_empty_lists(store: TypedSQLiteDocumentStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


def test_check_ids_empty_store_returns_all_missing(store: TypedSQLiteDocumentStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


def test_check_ids_returns_tuple_of_two_lists(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- columns_info ---


def test_get_columns_info_returns_dict(store: TypedSQLiteDocumentStore) -> None:
    result = store.get_columns_info()
    assert isinstance(result, dict)


def test_get_columns_info_typed_store_has_schema_columns(
    typed_store: TypedSQLiteDocumentStore,
) -> None:
    columns = typed_store.get_columns_info()
    assert "author" in columns
    assert "year" in columns
    assert "category" in columns


def test_get_columns_info_has_extra_column(typed_store: TypedSQLiteDocumentStore) -> None:
    assert "extra" in typed_store.get_columns_info()


def test_show_columns_info_does_not_raise(
    store: TypedSQLiteDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.show_columns_info()
    captured = capsys.readouterr()
    assert captured.out != ""


def test_show_columns_info_output_contains_column_names(
    typed_store: TypedSQLiteDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_columns = typed_store.get_columns_info().keys()
    typed_store.show_columns_info()
    captured = capsys.readouterr()
    for col in expected_columns:
        assert col in captured.out


def test_show_columns_info_returns_none(store: TypedSQLiteDocumentStore) -> None:
    assert store.show_columns_info() is None


# --- lazy_all ---


def test_lazy_all_empty_store_yields_nothing(store: TypedSQLiteDocumentStore) -> None:
    assert list(store.lazy_all()) == []


def test_lazy_all_returns_generator(store: TypedSQLiteDocumentStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


def test_lazy_all_yields_one_document_at_a_time(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Document)


def test_lazy_all_returns_all_documents(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


def test_lazy_all_matches_all(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert list(store.lazy_all()) == store.all()


def test_lazy_all_yields_document_instances(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert all(isinstance(doc, Document) for doc in result)


def test_lazy_all_preserves_metadata(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = {doc.id: doc for doc in store.lazy_all()}
    for doc in docs:
        assert result[doc.id].metadata == doc.metadata


def test_lazy_all_does_not_mutate_store(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.lazy_all())
    assert store.count() == len(docs)


def test_lazy_all_single_document(store: TypedSQLiteDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="solo", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


def test_lazy_all_accepts_batch_size_kwarg(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all(batch_size=2))
    assert len(result) == len(docs)


# --- iter_batches ---


def test_iter_batches_empty_store_yields_nothing(store: TypedSQLiteDocumentStore) -> None:
    assert list(store.iter_batches()) == []


def test_iter_batches_returns_generator(store: TypedSQLiteDocumentStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


def test_iter_batches_default_batch_size(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


def test_iter_batches_yields_correct_batch_sizes(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


def test_iter_batches_last_batch_may_be_smaller(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


def test_iter_batches_batch_size_larger_than_store(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


def test_iter_batches_batch_size_one(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


def test_iter_batches_returns_all_documents(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


def test_iter_batches_matches_all(store: TypedSQLiteDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    flattened = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert flattened == store.all()


def test_iter_batches_batches_contain_document_instances(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(doc, Document) for batch in batches for doc in batch)


def test_iter_batches_zero_batch_size_raises(store: TypedSQLiteDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


def test_iter_batches_negative_batch_size_raises(store: TypedSQLiteDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


def test_iter_batches_error_raised_before_any_query(store: TypedSQLiteDocumentStore) -> None:
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


def test_iter_batches_does_not_mutate_store(
    store: TypedSQLiteDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(docs)


# --- close ---


def test_close_closes_underlying_connection(store: TypedSQLiteDocumentStore) -> None:
    store.close()
    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_close_is_idempotent(store: TypedSQLiteDocumentStore) -> None:
    store.close()
    store.close()  # should not raise


def test_close_returns_none(store: TypedSQLiteDocumentStore) -> None:
    assert store.close() is None


# --- context manager ---


def test_context_manager_returns_self() -> None:
    with TypedSQLiteDocumentStore(":memory:") as store:
        assert isinstance(store, TypedSQLiteDocumentStore)


def test_context_manager_closes_on_normal_exit() -> None:
    with TypedSQLiteDocumentStore(":memory:") as store:
        store.add_documents([Document(id="1", page_content="hello", metadata={})])
        assert store.count() == 1

    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_context_manager_closes_on_exception() -> None:
    msg = "boom"
    with pytest.raises(ValueError, match="boom"), TypedSQLiteDocumentStore(":memory:") as store:
        raise ValueError(msg)

    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_context_manager_usable_for_reads_and_writes() -> None:
    with TypedSQLiteDocumentStore(":memory:") as store:
        store.add_documents(
            [
                Document(id="1", page_content="hello", metadata={"author": "Alice"}),
                Document(id="2", page_content="world", metadata={"author": "Bob"}),
            ]
        )
        assert store.count() == 2
        assert store.filter(author="Alice")[0].id == "1"
        store.delete("1")
        assert store.count() == 1
