from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document

from zenpyre.document_stores import TypedDuckDBDocumentStore
from zenpyre.testing.fixtures import duckdb_available
from zenpyre.utils.imports import is_duckdb_available

if TYPE_CHECKING:
    from pathlib import Path

if is_duckdb_available():
    import duckdb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("store")


@pytest.fixture
def store() -> Generator[TypedDuckDBDocumentStore, None, None]:
    """In-memory store with no schema."""
    with TypedDuckDBDocumentStore(":memory:") as store:
        yield store


@pytest.fixture
def typed_store() -> Generator[TypedDuckDBDocumentStore, None, None]:
    """In-memory store with a typed schema."""
    with TypedDuckDBDocumentStore(
        ":memory:",
        metadata_schema={"author": "VARCHAR", "year": "INTEGER", "category": "VARCHAR"},
    ) as store:
        yield store


@pytest.fixture(scope="module")
def store_read_only(
    store_path: Path, docs: list[Document]
) -> Generator[TypedDuckDBDocumentStore, None, None]:
    path = store_path / "data.duckdb"
    store = TypedDuckDBDocumentStore(path)
    store.add_documents(docs)
    store._conn.close()
    with TypedDuckDBDocumentStore(path, read_only=True) as store:
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
#     Tests for TypedDuckDBDocumentStore         #
##################################################

# --- repr/str ---


@duckdb_available
def test_repr(store: TypedDuckDBDocumentStore) -> None:
    assert repr(store).startswith("TypedDuckDBDocumentStore(")


@duckdb_available
def test_str(store: TypedDuckDBDocumentStore) -> None:
    assert repr(store).startswith("TypedDuckDBDocumentStore(")


# --- add_documents ---


@duckdb_available
def test_add_documents_increases_count(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


@duckdb_available
def test_add_documents_no_id_raises(store: TypedDuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match=r"All documents must have an id"):
        store.add_documents([Document(page_content="No id")])


@duckdb_available
def test_add_documents_upsert_replaces_existing(store: TypedDuckDBDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Original", metadata={})])
    store.add_documents([Document(id="1", page_content="Updated", metadata={})])
    assert store.count() == 1
    assert store.get("1").page_content == "Updated"


@duckdb_available
def test_add_documents_empty(store: TypedDuckDBDocumentStore) -> None:
    store.add_documents([])


@duckdb_available
def test_add_documents_when_read_only(store_read_only: TypedDuckDBDocumentStore) -> None:
    with pytest.raises(
        duckdb.InvalidInputException,
        match=r'Cannot execute statement of type "INSERT" on database "data" which is attached in read-only mode!',
    ):
        store_read_only.add_documents([Document(id="1", page_content="Original", metadata={})])


# --- count ---


@duckdb_available
def test_count_empty_store(store: TypedDuckDBDocumentStore) -> None:
    assert store.count() == 0


@duckdb_available
def test_count_after_adding(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


# --- get ---


@duckdb_available
def test_get_existing_document(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get("1")
    assert result == docs[0]


@duckdb_available
def test_get_missing_document_returns_none(store: TypedDuckDBDocumentStore) -> None:
    assert store.get("nonexistent") is None


@duckdb_available
def test_get_round_trips_metadata(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.get("1")
    assert result.metadata == docs[0].metadata


@duckdb_available
def test_get_round_trips_extra_metadata(typed_store: TypedDuckDBDocumentStore) -> None:
    doc = Document(
        id="1", page_content="Test", metadata={"author": "Alice", "publisher": "O'Reilly"}
    )
    typed_store.add_documents([doc])
    assert typed_store.get("1").metadata["publisher"] == "O'Reilly"


# --- get_many ---


@duckdb_available
def test_get_many_returns_correct_length(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.get_many(["1", "2", "99"])
    assert len(result) == 3


@duckdb_available
def test_get_many_returns_none_for_missing(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


@duckdb_available
def test_get_many_preserves_order(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- filter ---


@duckdb_available
def test_filter_no_args_returns_all(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert len(typed_store.filter()) == len(docs)


@duckdb_available
def test_filter_single_typed_field(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


@duckdb_available
def test_filter_multiple_typed_fields(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice", category="Programming")
    assert len(result) == 2


@duckdb_available
def test_filter_no_match_returns_empty(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.filter(author="Charlie") == []


@duckdb_available
def test_filter_extra_field(typed_store: TypedDuckDBDocumentStore) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"author": "Alice", "publisher": "O'Reilly"}),
        Document(id="2", page_content="B", metadata={"author": "Bob", "publisher": "Manning"}),
    ]
    typed_store.add_documents(docs)
    result = typed_store.filter(publisher="O'Reilly")
    assert len(result) == 1
    assert result[0].id == "1"


@duckdb_available
def test_filter_mixed_schema_and_extra_fields(typed_store: TypedDuckDBDocumentStore) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"author": "Alice", "publisher": "O'Reilly"}),
        Document(id="2", page_content="B", metadata={"author": "Alice", "publisher": "Manning"}),
    ]
    typed_store.add_documents(docs)
    result = typed_store.filter(author="Alice", publisher="O'Reilly")
    assert len(result) == 1
    assert result[0].id == "1"


@duckdb_available
def test_filter_integer_typed_column(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    result = typed_store.filter(year=2022)
    assert len(result) == 1
    assert result[0].id == "1"


@duckdb_available
def test_filter_integer_typed_column_no_match(
    typed_store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    typed_store.add_documents(docs)
    assert typed_store.filter(year=9999) == []


# --- delete ---


@duckdb_available
def test_delete_removes_document(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete("1")
    assert store.count() == len(docs) - 1
    assert store.get("1") is None


@duckdb_available
def test_delete_nonexistent_is_silent(store: TypedDuckDBDocumentStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


@duckdb_available
def test_delete_many_removes_documents(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.count() == len(docs) - 2
    assert store.get("1") is None
    assert store.get("3") is None


@duckdb_available
def test_delete_many_preserves_other_documents(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


@duckdb_available
def test_delete_many_empty_list_is_no_op(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many([])
    assert store.count() == len(docs)


@duckdb_available
def test_delete_many_nonexistent_ids_are_silent(store: TypedDuckDBDocumentStore) -> None:
    store.delete_many(["99", "100"])


@duckdb_available
def test_delete_many_single_id(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["2"])
    assert store.count() == len(docs) - 1
    assert store.get("2") is None


# --- check_ids ---


@duckdb_available
def test_check_ids_all_found(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


@duckdb_available
def test_check_ids_all_missing(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


@duckdb_available
def test_check_ids_mixed(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


@duckdb_available
def test_check_ids_preserves_order(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


@duckdb_available
def test_check_ids_empty_input_returns_empty_lists(store: TypedDuckDBDocumentStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


@duckdb_available
def test_check_ids_empty_store_returns_all_missing(store: TypedDuckDBDocumentStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


@duckdb_available
def test_check_ids_returns_tuple_of_two_lists(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- all ---


@duckdb_available
def test_all_empty_store(store: TypedDuckDBDocumentStore) -> None:
    assert store.all() == []


@duckdb_available
def test_all_returns_all_documents(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.all()
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


# --- lazy_all ---


@duckdb_available
def test_lazy_all_empty_store_yields_nothing(store: TypedDuckDBDocumentStore) -> None:
    assert list(store.lazy_all()) == []


@duckdb_available
def test_lazy_all_returns_generator(store: TypedDuckDBDocumentStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


@duckdb_available
def test_lazy_all_yields_one_document_at_a_time(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Document)


@duckdb_available
def test_lazy_all_returns_all_documents(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


@duckdb_available
def test_lazy_all_matches_all(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert list(store.lazy_all()) == store.all()


@duckdb_available
def test_lazy_all_yields_document_instances(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert all(isinstance(doc, Document) for doc in result)


@duckdb_available
def test_lazy_all_preserves_metadata(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = {doc.id: doc for doc in store.lazy_all()}
    for doc in docs:
        assert result[doc.id].metadata == doc.metadata


@duckdb_available
def test_lazy_all_does_not_mutate_store(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.lazy_all())
    assert store.count() == len(docs)


@duckdb_available
def test_lazy_all_single_document(store: TypedDuckDBDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Solo", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


@duckdb_available
def test_lazy_all_is_lazy_not_exhausted_on_creation(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    """Calling lazy_all() should not itself execute the query eagerly
    consuming results before iteration begins; adding docs after
    creating the generator but before the first next() call should still
    be reflected, confirming the query executes lazily."""
    gen = store.lazy_all()
    store.add_documents(docs)
    result = list(gen)
    assert len(result) == len(docs)


@duckdb_available
def test_lazy_all_independent_generators_do_not_interfere(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    """Two separate lazy_all() generators should each independently
    yield the full set of documents, since each uses its own cursor."""
    store.add_documents(docs)
    gen1 = store.lazy_all()
    gen2 = store.lazy_all()
    first_from_gen1 = next(gen1)
    result2 = list(gen2)
    remaining_from_gen1 = list(gen1)

    assert len(result2) == len(docs)
    assert len(remaining_from_gen1) + 1 == len(docs)
    assert first_from_gen1.id not in {d.id for d in remaining_from_gen1}


# --- iter_batches ---


@duckdb_available
def test_iter_batches_empty_store_yields_nothing(store: TypedDuckDBDocumentStore) -> None:
    assert list(store.iter_batches()) == []


@duckdb_available
def test_iter_batches_returns_generator(store: TypedDuckDBDocumentStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


@duckdb_available
def test_iter_batches_default_batch_size(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


@duckdb_available
def test_iter_batches_yields_correct_batch_sizes(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


@duckdb_available
def test_iter_batches_last_batch_may_be_smaller(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


@duckdb_available
def test_iter_batches_batch_size_larger_than_store(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


@duckdb_available
def test_iter_batches_batch_size_one(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


@duckdb_available
def test_iter_batches_returns_all_documents(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


@duckdb_available
def test_iter_batches_matches_all(store: TypedDuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    flattened = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert flattened == store.all()


@duckdb_available
def test_iter_batches_batches_contain_document_instances(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(doc, Document) for batch in batches for doc in batch)


@duckdb_available
def test_iter_batches_zero_batch_size_raises(store: TypedDuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


@duckdb_available
def test_iter_batches_negative_batch_size_raises(store: TypedDuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


@duckdb_available
def test_iter_batches_error_raised_before_any_query(store: TypedDuckDBDocumentStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


@duckdb_available
def test_iter_batches_does_not_mutate_store(
    store: TypedDuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(docs)


# --- columns_info ---


@duckdb_available
def test_get_columns_info_returns_dict(store: TypedDuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    assert isinstance(result, dict)


@duckdb_available
def test_get_columns_info_keys_are_column_names(store: TypedDuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    expected_columns = {row[0] for row in store._conn.sql("DESCRIBE documents").fetchall()}
    assert set(result.keys()) == expected_columns


@duckdb_available
def test_get_columns_info_values_are_strings(store: TypedDuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    assert all(isinstance(v, str) for v in result.values())


@duckdb_available
def test_get_columns_info_matches_describe_output(store: TypedDuckDBDocumentStore) -> None:
    """Cross-check against the raw DESCRIBE output as the source of
    truth."""
    rows = store._conn.sql("DESCRIBE documents").fetchall()
    expected = {row[0]: row[1] for row in rows}
    assert store.get_columns_info() == expected


@duckdb_available
def test_get_columns_info_non_empty_for_created_table(store: TypedDuckDBDocumentStore) -> None:
    """The documents table is created in __init__, so this should never
    be empty for a freshly constructed store."""
    result = store.get_columns_info()
    assert len(result) > 0


@duckdb_available
def test_get_columns_info_does_not_mutate_between_calls(store: TypedDuckDBDocumentStore) -> None:
    first = store.get_columns_info()
    second = store.get_columns_info()
    assert first == second
    assert first is not second  # each call builds a fresh dict


@duckdb_available
def test_show_columns_info_does_not_raise(
    store: TypedDuckDBDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.show_columns_info()  # should not raise
    captured = capsys.readouterr()
    assert captured.out != ""


@duckdb_available
def test_show_columns_info_output_contains_column_names(
    store: TypedDuckDBDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_columns = store.get_columns_info().keys()
    store.show_columns_info()
    captured = capsys.readouterr()
    for col in expected_columns:
        assert col in captured.out


@duckdb_available
def test_show_columns_info_returns_none(store: TypedDuckDBDocumentStore) -> None:
    assert store.show_columns_info() is None


# --- close ---


@duckdb_available
def test_close_closes_underlying_connection(store: TypedDuckDBDocumentStore) -> None:
    store.close()
    with pytest.raises(duckdb.ConnectionException, match=r"Connection already closed!"):
        store.count()


@duckdb_available
def test_close_is_idempotent(store: TypedDuckDBDocumentStore) -> None:
    store.close()
    store.close()  # should not raise


@duckdb_available
def test_close_returns_none(store: TypedDuckDBDocumentStore) -> None:
    assert store.close() is None


# --- context manager ---


@duckdb_available
def test_context_manager_returns_self() -> None:
    with TypedDuckDBDocumentStore(":memory:") as store:
        assert isinstance(store, TypedDuckDBDocumentStore)


@duckdb_available
def test_context_manager_closes_on_normal_exit() -> None:
    with TypedDuckDBDocumentStore(":memory:") as store:
        store.add_documents([Document(id="1", page_content="hello", metadata={})])
        assert store.count() == 1

    with pytest.raises(duckdb.ConnectionException, match=r"Connection already closed!"):
        store.count()


@duckdb_available
def test_context_manager_closes_on_exception() -> None:
    msg = "boom"
    with (
        pytest.raises(ValueError, match="boom"),
        TypedDuckDBDocumentStore(":memory:") as store,
    ):
        raise ValueError(msg)

    with pytest.raises(duckdb.ConnectionException, match=r"Connection already closed!"):
        store.count()


@duckdb_available
def test_context_manager_usable_for_reads_and_writes() -> None:
    with TypedDuckDBDocumentStore(":memory:") as store:
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
