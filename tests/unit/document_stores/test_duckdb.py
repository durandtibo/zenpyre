"""Unit tests for DuckDBDocumentStore."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document

from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.document_stores.duckdb import prepare_duckdb_path
from zenpyre.testing.fixtures import duckdb_available
from zenpyre.utils.imports import is_duckdb_available

if is_duckdb_available():
    from duckdb import InvalidInputException

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("store")


@pytest.fixture
def store() -> DuckDBDocumentStore:
    return DuckDBDocumentStore(":memory:")


@pytest.fixture(scope="module")
def store_read_only(store_path: Path, docs: list[Document]) -> DuckDBDocumentStore:
    path = store_path / "data.duckdb"
    store = DuckDBDocumentStore(path)
    store.add_documents(docs)
    store._conn.close()
    return DuckDBDocumentStore(path, read_only=True)


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
#     Tests for DuckDBDocumentStore              #
##################################################

# --- repr/str ---


@duckdb_available
def test_repr(store: DuckDBDocumentStore) -> None:
    assert repr(store).startswith("DuckDBDocumentStore(")


@duckdb_available
def test_str(store: DuckDBDocumentStore) -> None:
    assert repr(store).startswith("DuckDBDocumentStore(")


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


@duckdb_available
def test_add_documents_empty(store: DuckDBDocumentStore) -> None:
    store.add_documents([])


@duckdb_available
def test_add_documents_when_read_only(store_read_only: DuckDBDocumentStore) -> None:
    with pytest.raises(
        InvalidInputException,
        match=r'Cannot execute statement of type "INSERT" on database "data" which is attached in read-only mode!',
    ):
        store_read_only.add_documents([Document(id="1", page_content="Original", metadata={})])


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


# --- columns_info ---


@duckdb_available
def test_get_columns_info_returns_dict(store: DuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    assert isinstance(result, dict)


@duckdb_available
def test_get_columns_info_keys_are_column_names(store: DuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    expected_columns = {row[0] for row in store._conn.sql("DESCRIBE documents").fetchall()}
    assert set(result.keys()) == expected_columns


@duckdb_available
def test_get_columns_info_values_are_strings(store: DuckDBDocumentStore) -> None:
    result = store.get_columns_info()
    assert all(isinstance(v, str) for v in result.values())


@duckdb_available
def test_get_columns_info_matches_describe_output(store: DuckDBDocumentStore) -> None:
    """Cross-check against the raw DESCRIBE output as the source of
    truth."""
    rows = store._conn.sql("DESCRIBE documents").fetchall()
    expected = {row[0]: row[1] for row in rows}
    assert store.get_columns_info() == expected


@duckdb_available
def test_get_columns_info_non_empty_for_created_table(store: DuckDBDocumentStore) -> None:
    """The documents table is created in __init__, so this should never
    be empty for a freshly constructed store."""
    result = store.get_columns_info()
    assert len(result) > 0


@duckdb_available
def test_get_columns_info_does_not_mutate_between_calls(store: DuckDBDocumentStore) -> None:
    first = store.get_columns_info()
    second = store.get_columns_info()
    assert first == second
    assert first is not second  # each call builds a fresh dict


@duckdb_available
def test_show_columns_info_does_not_raise(
    store: DuckDBDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.show_columns_info()  # should not raise
    captured = capsys.readouterr()
    assert captured.out != ""


@duckdb_available
def test_show_columns_info_output_contains_column_names(
    store: DuckDBDocumentStore, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_columns = store.get_columns_info().keys()
    store.show_columns_info()
    captured = capsys.readouterr()
    for col in expected_columns:
        assert col in captured.out


@duckdb_available
def test_show_columns_info_returns_none(store: DuckDBDocumentStore) -> None:
    assert store.show_columns_info() is None


# --- lazy_all ---


@duckdb_available
def test_lazy_all_empty_store_yields_nothing(store: DuckDBDocumentStore) -> None:
    assert list(store.lazy_all()) == []


@duckdb_available
def test_lazy_all_returns_generator(store: DuckDBDocumentStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


@duckdb_available
def test_lazy_all_yields_one_document_at_a_time(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Document)


@duckdb_available
def test_lazy_all_returns_all_documents(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


@duckdb_available
def test_lazy_all_matches_all(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert list(store.lazy_all()) == store.all()


@duckdb_available
def test_lazy_all_yields_document_instances(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert all(isinstance(doc, Document) for doc in result)


@duckdb_available
def test_lazy_all_preserves_metadata(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = {doc.id: doc for doc in store.lazy_all()}
    for doc in docs:
        assert result[doc.id].metadata == doc.metadata


@duckdb_available
def test_lazy_all_does_not_mutate_store(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    list(store.lazy_all())
    assert store.count() == len(docs)


@duckdb_available
def test_lazy_all_single_document(store: DuckDBDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Solo", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


@duckdb_available
def test_lazy_all_is_lazy_not_exhausted_on_creation(
    store: DuckDBDocumentStore, docs: list[Document]
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
    store: DuckDBDocumentStore, docs: list[Document]
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
def test_iter_batches_empty_store_yields_nothing(store: DuckDBDocumentStore) -> None:
    assert list(store.iter_batches()) == []


@duckdb_available
def test_iter_batches_returns_generator(store: DuckDBDocumentStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


@duckdb_available
def test_iter_batches_default_batch_size(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


@duckdb_available
def test_iter_batches_yields_correct_batch_sizes(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


@duckdb_available
def test_iter_batches_last_batch_may_be_smaller(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


@duckdb_available
def test_iter_batches_batch_size_larger_than_store(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


@duckdb_available
def test_iter_batches_batch_size_one(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


@duckdb_available
def test_iter_batches_returns_all_documents(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


@duckdb_available
def test_iter_batches_matches_all(store: DuckDBDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    flattened = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert flattened == store.all()


@duckdb_available
def test_iter_batches_batches_contain_document_instances(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(doc, Document) for batch in batches for doc in batch)


@duckdb_available
def test_iter_batches_zero_batch_size_raises(store: DuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


@duckdb_available
def test_iter_batches_negative_batch_size_raises(store: DuckDBDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


@duckdb_available
def test_iter_batches_error_raised_before_any_query(store: DuckDBDocumentStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


@duckdb_available
def test_iter_batches_does_not_mutate_store(
    store: DuckDBDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(docs)


########################################################
#     Tests for prepare_duckdb_path                    #
########################################################


def test_prepare_duckdb_path_in_memory_returns_unchanged() -> None:
    assert prepare_duckdb_path(":memory:") == ":memory:"


def test_prepare_duckdb_path_creates_missing_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dirs" / "db.duckdb"
    assert not target.parent.exists()

    result = prepare_duckdb_path(target)

    assert target.parent.is_dir()
    assert result == target


def test_prepare_duckdb_path_creates_deeply_nested_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "d" / "db.duckdb"

    result = prepare_duckdb_path(target)

    assert target.parent.is_dir()
    assert result == target


def test_prepare_duckdb_path_existing_parent_directory_is_noop(tmp_path: Path) -> None:
    target = tmp_path / "db.duckdb"
    assert tmp_path.is_dir()  # tmp_path fixture already creates this

    result = prepare_duckdb_path(target)

    assert target.parent == tmp_path
    assert result == target


def test_prepare_duckdb_path_does_not_create_the_file_itself(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "db.duckdb"

    prepare_duckdb_path(target)

    assert target.parent.is_dir()
    assert not target.exists()


def test_prepare_duckdb_path_accepts_str_input(tmp_path: Path) -> None:
    target_str = str(tmp_path / "sub" / "db.duckdb")

    result = prepare_duckdb_path(target_str)

    assert Path(target_str).parent.is_dir()
    assert Path(result) == Path(target_str)


def test_prepare_duckdb_path_returns_path_instance_for_file_paths(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "db.duckdb"
    result = prepare_duckdb_path(target)

    assert isinstance(result, Path)


def test_prepare_duckdb_path_idempotent_when_called_twice(tmp_path: Path) -> None:
    """Calling twice on the same path should not raise even though the
    directory already exists after the first call."""
    target = tmp_path / "sub" / "db.duckdb"

    first = prepare_duckdb_path(target)
    second = prepare_duckdb_path(target)

    assert first == second
    assert target.parent.is_dir()
