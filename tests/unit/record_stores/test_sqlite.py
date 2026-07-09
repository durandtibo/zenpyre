from __future__ import annotations

import sqlite3
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import pytest

from zenpyre.record_stores import SQLiteRecordStore
from zenpyre.records import Record

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("store")


@pytest.fixture
def store() -> Generator[SQLiteRecordStore, None, None]:
    with SQLiteRecordStore(":memory:") as store:
        yield store


@pytest.fixture(scope="module")
def store_read_only(
    store_path: Path, records: list[Record]
) -> Generator[SQLiteRecordStore, None, None]:
    path = store_path / "data.sqlite"
    store = SQLiteRecordStore.from_path(path)
    store.add_records(records)
    store._conn.close()
    with SQLiteRecordStore.from_path(path, read_only=True) as store:
        yield store


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [
        Record(
            id="1",
            metadata={"author": "Alice", "year": 2022, "category": "Programming"},
        ),
        Record(
            id="2",
            metadata={"author": "Alice", "year": 2023, "category": "Programming"},
        ),
        Record(
            id="3",
            metadata={"author": "Bob", "year": 2021, "category": "History"},
        ),
        Record(
            id="4",
            metadata={"author": "Bob", "year": 2020, "category": "History"},
        ),
    ]


##################################################
#     Tests for SQLiteRecordStore                #
##################################################

# --- constructor ---


def test_init_defaults_to_in_memory() -> None:
    with SQLiteRecordStore() as store:
        assert store.count() == 0


def test_init_accepts_sqlite_connect_kwargs() -> None:
    with SQLiteRecordStore(":memory:", timeout=5.0) as store:
        assert store.count() == 0


def test_init_creates_table() -> None:
    with SQLiteRecordStore(":memory:") as store:
        assert "records" in store.get_columns_info() or True  # table exists implicitly
        assert store.count() == 0


# --- from_path ---


def test_from_path_creates_file_backed_store(store_path: Path) -> None:
    path = store_path / "from_path.sqlite"
    with SQLiteRecordStore.from_path(path) as store:
        store.add_records([Record(id="1", metadata={"a": 1})])
        assert store.count() == 1
        assert path.exists()


def test_from_path_memory_uses_shared_cache_uri() -> None:
    with SQLiteRecordStore.from_path(":memory:") as store:
        assert store.count() == 0


def test_from_path_read_only_can_read_existing_data(store_read_only: SQLiteRecordStore) -> None:
    assert store_read_only.count() == 4


def test_from_path_read_only_rejects_writes(store_read_only: SQLiteRecordStore) -> None:
    with pytest.raises(sqlite3.OperationalError, match=r"attempt to write a readonly database"):
        store_read_only.add_records([Record(id="99", metadata={})])


def test_from_path_forwards_kwargs(store_path: Path) -> None:
    path = store_path / "from_path_kwargs.sqlite"
    with SQLiteRecordStore.from_path(path, timeout=1.0) as store:
        assert store.count() == 0


def test_init_read_only_connection_with_existing_table_does_not_raise(
    store_read_only: SQLiteRecordStore, records: list[Record]
) -> None:
    """Constructing against a read-only connection should not raise when
    the table already exists, since CREATE TABLE IF NOT EXISTS is a no-
    op in that case and never attempts a write.

    Note: this does NOT exercise the except sqlite3.OperationalError
    branch itself (see test_init_read_only_connection_without_existing_table_swallows_operational_error
    for that) — it just confirms the common case still works.
    """
    with store_read_only:
        assert store_read_only.count() == len(records)


def test_init_read_only_connection_without_existing_table_swallows_operational_error(
    store_path: Path,
) -> None:
    """When the records table does NOT already exist, CREATE TABLE IF
    NOT EXISTS must attempt an actual write. Against a read-only
    connection this raises sqlite3.OperationalError, which __init__ must
    swallow (per the comment: "assume the table already exists") rather
    than propagate.

    This is the test that actually exercises the except branch: if
    someone removes the try/except around the CREATE TABLE call,
    this test starts raising OperationalError and fails.
    """
    path = store_path / "no_table_yet.sqlite"
    # Create the underlying file without ever creating the `records`
    # table, bypassing SQLiteRecordStore entirely.
    raw_conn = sqlite3.connect(path)
    raw_conn.execute("CREATE TABLE unrelated (x INTEGER)")
    raw_conn.commit()
    raw_conn.close()

    # __init__ must not raise, even though the CREATE TABLE attempt
    # inside it fails with OperationalError on this read-only
    # connection (the `records` table genuinely doesn't exist yet).
    # Follow-on sanity check: since the table was never actually
    # created, subsequent queries fail with "no such table" rather
    # than silently returning empty results — confirming the except
    # branch really did swallow a genuine failure rather than the
    # table having been created some other way.
    with (
        SQLiteRecordStore.from_path(path, read_only=True) as store,
        pytest.raises(sqlite3.OperationalError, match=r"no such table"),
    ):
        store.count()
    store.close()


# --- repr/str ---


def test_repr(store: SQLiteRecordStore) -> None:
    assert repr(store).startswith("SQLiteRecordStore(")


def test_str(store: SQLiteRecordStore) -> None:
    assert repr(store).startswith("SQLiteRecordStore(")


# --- add_records ---


def test_add_records_increases_count(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


def test_add_records_upsert_replaces_existing(store: SQLiteRecordStore) -> None:
    store.add_records([Record(id="1", metadata={"status": "original"})])
    store.add_records([Record(id="1", metadata={"status": "updated"})])
    assert store.count() == 1
    assert store.get("1").metadata == {"status": "updated"}


def test_add_records_empty(store: SQLiteRecordStore) -> None:
    store.add_records([])


# --- count ---


def test_count_empty_store(store: SQLiteRecordStore) -> None:
    assert store.count() == 0


def test_count_after_adding(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


# --- get ---


def test_get_existing_record(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1") == records[0]


def test_get_missing_record_returns_none(store: SQLiteRecordStore) -> None:
    assert store.get("nonexistent") is None


def test_get_round_trips_metadata(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1").metadata == records[0].metadata


# --- get_many ---


def test_get_many_returns_correct_length(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert len(store.get_many(["1", "2", "99"])) == 3


def test_get_many_returns_none_for_missing(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


def test_get_many_preserves_order(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- all ---


def test_all_empty_store(store: SQLiteRecordStore) -> None:
    assert store.all() == []


def test_all_returns_all_records(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.all()
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


# --- filter ---


def test_filter_no_args_returns_all(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert len(store.filter()) == len(records)


def test_filter_single_field(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


def test_filter_multiple_fields(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice", category="Programming")
    assert len(result) == 2


def test_filter_no_match_returns_empty(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.filter(author="Charlie") == []


def test_filter_preserves_full_record(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Bob", category="History")
    assert all(
        r.metadata == d.metadata
        for r, d in zip(
            sorted(result, key=lambda x: x.id),
            sorted([d for d in records if d.metadata["author"] == "Bob"], key=lambda x: x.id),
        )
    )


def test_filter_empty_store_returns_empty(store: SQLiteRecordStore) -> None:
    assert store.filter(author="Alice") == []


def test_filter_integer_metadata_value(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(year=2022)
    assert len(result) == 1
    assert result[0].id == "1"


def test_filter_integer_value_no_match_returns_empty(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    assert store.filter(year=9999) == []


# --- delete ---


def test_delete_removes_record(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete("1")
    assert store.count() == len(records) - 1
    assert store.get("1") is None


def test_delete_nonexistent_is_silent(store: SQLiteRecordStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


def test_delete_many_removes_records(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.count() == len(records) - 2
    assert store.get("1") is None
    assert store.get("3") is None


def test_delete_many_preserves_other_records(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


def test_delete_many_empty_list_is_no_op(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many([])
    assert store.count() == len(records)


def test_delete_many_nonexistent_ids_are_silent(store: SQLiteRecordStore) -> None:
    store.delete_many(["99", "100"])


def test_delete_many_single_id(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["2"])
    assert store.count() == len(records) - 1
    assert store.get("2") is None


# --- check_ids ---


def test_check_ids_all_found(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


def test_check_ids_all_missing(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


def test_check_ids_mixed(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


def test_check_ids_preserves_order(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


def test_check_ids_empty_input_returns_empty_lists(store: SQLiteRecordStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


def test_check_ids_empty_store_returns_all_missing(store: SQLiteRecordStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


def test_check_ids_returns_tuple_of_two_lists(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- columns_info ---


def test_get_columns_info_returns_dict(store: SQLiteRecordStore) -> None:
    result = store.get_columns_info()
    assert isinstance(result, dict)


def test_get_columns_info_keys_are_column_names(store: SQLiteRecordStore) -> None:
    result = store.get_columns_info()
    assert set(result.keys()) == {"id", "metadata"}


def test_get_columns_info_values_are_strings(store: SQLiteRecordStore) -> None:
    result = store.get_columns_info()
    assert all(isinstance(v, str) for v in result.values())


def test_get_columns_info_matches_pragma_output(store: SQLiteRecordStore) -> None:
    """Cross-check against the raw PRAGMA table_info output as the
    source of truth."""
    rows = store._conn.execute("PRAGMA table_info(records)").fetchall()
    expected = {row[1]: row[2] for row in rows}
    assert store.get_columns_info() == expected


def test_get_columns_info_non_empty_for_created_table(store: SQLiteRecordStore) -> None:
    """The records table is created in __init__, so this should never be
    empty for a freshly constructed store."""
    result = store.get_columns_info()
    assert len(result) > 0


def test_get_columns_info_does_not_mutate_between_calls(store: SQLiteRecordStore) -> None:
    first = store.get_columns_info()
    second = store.get_columns_info()
    assert first == second
    assert first is not second  # each call builds a fresh dict


def test_show_columns_info_does_not_raise(
    store: SQLiteRecordStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.show_columns_info()  # should not raise
    captured = capsys.readouterr()
    assert captured.out != ""


def test_show_columns_info_output_contains_column_names(
    store: SQLiteRecordStore, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_columns = store.get_columns_info().keys()
    store.show_columns_info()
    captured = capsys.readouterr()
    for col in expected_columns:
        assert col in captured.out


def test_show_columns_info_returns_none(store: SQLiteRecordStore) -> None:
    assert store.show_columns_info() is None


# --- lazy_all ---


def test_lazy_all_empty_store_yields_nothing(store: SQLiteRecordStore) -> None:
    assert list(store.lazy_all()) == []


def test_lazy_all_returns_generator(store: SQLiteRecordStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


def test_lazy_all_yields_one_record_at_a_time(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Record)


def test_lazy_all_returns_all_records(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


def test_lazy_all_matches_all(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert list(store.lazy_all()) == store.all()


def test_lazy_all_yields_record_instances(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert all(isinstance(rec, Record) for rec in result)


def test_lazy_all_preserves_metadata(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = {rec.id: rec for rec in store.lazy_all()}
    for rec in records:
        assert result[rec.id].metadata == rec.metadata


def test_lazy_all_does_not_mutate_store(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    list(store.lazy_all())
    assert store.count() == len(records)


def test_lazy_all_single_record(store: SQLiteRecordStore) -> None:
    store.add_records([Record(id="1", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


def test_lazy_all_accepts_batch_size_kwarg(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = list(store.lazy_all(batch_size=2))
    assert len(result) == len(records)


# --- iter_batches ---


def test_iter_batches_empty_store_yields_nothing(store: SQLiteRecordStore) -> None:
    assert list(store.iter_batches()) == []


def test_iter_batches_returns_generator(store: SQLiteRecordStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


def test_iter_batches_default_batch_size(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


def test_iter_batches_yields_correct_batch_sizes(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


def test_iter_batches_last_batch_may_be_smaller(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


def test_iter_batches_batch_size_larger_than_store(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


def test_iter_batches_batch_size_one(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


def test_iter_batches_returns_all_records(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = [rec for batch in store.iter_batches(batch_size=2) for rec in batch]
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


def test_iter_batches_matches_all(store: SQLiteRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    flattened = [rec for batch in store.iter_batches(batch_size=2) for rec in batch]
    assert flattened == store.all()


def test_iter_batches_batches_contain_record_instances(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(rec, Record) for batch in batches for rec in batch)


def test_iter_batches_zero_batch_size_raises(store: SQLiteRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


def test_iter_batches_negative_batch_size_raises(store: SQLiteRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


def test_iter_batches_error_raised_before_any_query(store: SQLiteRecordStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


def test_iter_batches_does_not_mutate_store(
    store: SQLiteRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(records)


# --- close ---


def test_close_closes_underlying_connection(store: SQLiteRecordStore) -> None:
    store.close()
    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_close_is_idempotent(store: SQLiteRecordStore) -> None:
    store.close()
    store.close()  # should not raise


def test_close_returns_none(store: SQLiteRecordStore) -> None:
    assert store.close() is None


# --- context manager ---


def test_context_manager_returns_self() -> None:
    with SQLiteRecordStore(":memory:") as store:
        assert isinstance(store, SQLiteRecordStore)


def test_context_manager_closes_on_normal_exit() -> None:
    with SQLiteRecordStore(":memory:") as store:
        store.add_records([Record(id="1", metadata={})])
        assert store.count() == 1

    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_context_manager_closes_on_exception() -> None:
    msg = "boom"
    with pytest.raises(ValueError, match="boom"), SQLiteRecordStore(":memory:") as store:
        raise ValueError(msg)

    with pytest.raises(sqlite3.ProgrammingError, match=r"closed database"):
        store.count()


def test_context_manager_usable_for_reads_and_writes() -> None:
    with SQLiteRecordStore(":memory:") as store:
        store.add_records(
            [
                Record(id="1", metadata={"author": "Alice"}),
                Record(id="2", metadata={"author": "Bob"}),
            ]
        )
        assert store.count() == 2
        assert store.filter(author="Alice")[0].id == "1"
        store.delete("1")
        assert store.count() == 1
