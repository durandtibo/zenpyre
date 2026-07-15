from __future__ import annotations

import contextlib
import os
from collections.abc import Generator, Iterator

import pytest

from zenpyre.testing.fixtures import psycopg_available
from zenpyre.utils.imports import is_psycopg_available

if is_psycopg_available():
    import psycopg

    from zenpyre.record_stores import PostgreSQLRecordStore
    from zenpyre.records import Record

# ---------------------------------------------------------------------------
# Connection to a live PostgreSQL server is required for these tests. Set
# ZENPYRE_TEST_POSTGRES_DSN to point at a reachable server/database; a
# reasonable local default is used otherwise. The whole module is skipped if
# no server can be reached, so this file is a no-op in environments without
# PostgreSQL available.
# ---------------------------------------------------------------------------

_DSN = os.environ.get("ZENPYRE_TEST_POSTGRES_DSN", "postgresql://localhost/zenpyre_test")


def _postgres_reachable() -> bool:
    if not is_psycopg_available():
        return False
    try:
        with psycopg.connect(_DSN, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
    except Exception:  # noqa: BLE001
        return False
    return True


if not is_psycopg_available() or not _postgres_reachable():
    pytest.skip(
        f"Requires a reachable PostgreSQL server at {_DSN!r} (set ZENPYRE_TEST_POSTGRES_DSN)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> Generator[PostgreSQLRecordStore, None, None]:
    with PostgreSQLRecordStore(_DSN) as store:
        store._conn.execute("TRUNCATE TABLE records")
        store._conn.commit()
        yield store


@pytest.fixture
def readonly_dsn() -> Generator[str, None, None]:
    """DSN for a role that lacks CREATE privileges on the schema.

    Used to exercise the ``InsufficientPrivilege`` fallback in
    ``PostgreSQLRecordStore._ensure_schema``. Skips if the connected
    role cannot create/drop roles (e.g. a restricted CI database).
    """
    role = "zenpyre_readonly_test_role"
    with psycopg.connect(_DSN, autocommit=True) as conn:
        with contextlib.suppress(psycopg.Error):
            conn.execute(f"DROP OWNED BY {role}")
        try:
            conn.execute(f"DROP ROLE IF EXISTS {role}")
            conn.execute(f"CREATE ROLE {role} LOGIN")
            conn.execute(f"REVOKE CREATE ON SCHEMA public FROM {role}")
            conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
            conn.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, metadata JSONB)")
            conn.execute("TRUNCATE TABLE records")
            conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON records TO {role}")
        except psycopg.Error as exc:
            pytest.skip(f"Cannot create a restricted role for this test: {exc}")

        try:
            yield psycopg.conninfo.make_conninfo(_DSN, user=role)
        finally:
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE IF EXISTS {role}")


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
#     Tests for PostgreSQLRecordStore           #
##################################################

# --- repr/str ---


@psycopg_available
def test_repr(store: PostgreSQLRecordStore) -> None:
    assert repr(store).startswith("PostgreSQLRecordStore(")


@psycopg_available
def test_str(store: PostgreSQLRecordStore) -> None:
    assert str(store).startswith("PostgreSQLRecordStore(")


# --- add_records ---


@psycopg_available
def test_add_records_increases_count(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


@psycopg_available
def test_add_records_upsert_replaces_existing(store: PostgreSQLRecordStore) -> None:
    store.add_records([Record(id="1", metadata={"status": "original"})])
    store.add_records([Record(id="1", metadata={"status": "updated"})])
    assert store.count() == 1
    assert store.get("1").metadata == {"status": "updated"}


@psycopg_available
def test_add_records_empty(store: PostgreSQLRecordStore) -> None:
    store.add_records([])


# --- _ensure_schema ---


@psycopg_available
def test_ensure_schema_insufficient_privilege_falls_back_to_existing_table(
    readonly_dsn: str,
) -> None:
    """When the connecting role lacks CREATE privileges,
    ``_ensure_schema`` should swallow the ``InsufficientPrivilege``
    error (assuming the table already exists) instead of raising."""
    with PostgreSQLRecordStore(readonly_dsn) as store:
        assert store.count() == 0


# --- count ---


@psycopg_available
def test_count_empty_store(store: PostgreSQLRecordStore) -> None:
    assert store.count() == 0


@psycopg_available
def test_count_after_adding(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


# --- get ---


@psycopg_available
def test_get_existing_record(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1") == records[0]


@psycopg_available
def test_get_missing_record_returns_none(store: PostgreSQLRecordStore) -> None:
    assert store.get("nonexistent") is None


@psycopg_available
def test_get_round_trips_metadata(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1").metadata == records[0].metadata


# --- get_many ---


@psycopg_available
def test_get_many_returns_correct_length(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    assert len(store.get_many(["1", "2", "99"])) == 3


@psycopg_available
def test_get_many_returns_none_for_missing(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


@psycopg_available
def test_get_many_preserves_order(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


@psycopg_available
def test_get_many_empty_list_returns_empty(store: PostgreSQLRecordStore) -> None:
    assert store.get_many([]) == []


# --- all ---


@psycopg_available
def test_all_empty_store(store: PostgreSQLRecordStore) -> None:
    assert store.all() == []


@psycopg_available
def test_all_returns_all_records(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.all()
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


# --- filter ---


@psycopg_available
def test_filter_no_args_returns_all(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert len(store.filter()) == len(records)


@psycopg_available
def test_filter_single_field(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


@psycopg_available
def test_filter_multiple_fields(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice", category="Programming")
    assert len(result) == 2


@psycopg_available
def test_filter_no_match_returns_empty(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.filter(author="Charlie") == []


@psycopg_available
def test_filter_preserves_full_record(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Bob", category="History")
    assert all(
        r.metadata == d.metadata
        for r, d in zip(
            sorted(result, key=lambda x: x.id),
            sorted([d for d in records if d.metadata["author"] == "Bob"], key=lambda x: x.id),
        )
    )


@psycopg_available
def test_filter_empty_store_returns_empty(store: PostgreSQLRecordStore) -> None:
    assert store.filter(author="Alice") == []


@psycopg_available
def test_filter_integer_metadata_value(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(year=2022)
    assert len(result) == 1
    assert result[0].id == "1"


@psycopg_available
def test_filter_integer_value_no_match_returns_empty(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    assert store.filter(year=9999) == []


# --- delete ---


@psycopg_available
def test_delete_removes_record(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete("1")
    assert store.count() == len(records) - 1
    assert store.get("1") is None


@psycopg_available
def test_delete_nonexistent_is_silent(store: PostgreSQLRecordStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


@psycopg_available
def test_delete_many_removes_records(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.count() == len(records) - 2
    assert store.get("1") is None
    assert store.get("3") is None


@psycopg_available
def test_delete_many_preserves_other_records(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


@psycopg_available
def test_delete_many_empty_list_is_no_op(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    store.delete_many([])
    assert store.count() == len(records)


@psycopg_available
def test_delete_many_nonexistent_ids_are_silent(store: PostgreSQLRecordStore) -> None:
    store.delete_many(["99", "100"])


@psycopg_available
def test_delete_many_single_id(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["2"])
    assert store.count() == len(records) - 1
    assert store.get("2") is None


# --- check_ids ---


@psycopg_available
def test_check_ids_all_found(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


@psycopg_available
def test_check_ids_all_missing(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


@psycopg_available
def test_check_ids_mixed(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


@psycopg_available
def test_check_ids_preserves_order(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


@psycopg_available
def test_check_ids_empty_input_returns_empty_lists(store: PostgreSQLRecordStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


@psycopg_available
def test_check_ids_empty_store_returns_all_missing(store: PostgreSQLRecordStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


@psycopg_available
def test_check_ids_returns_tuple_of_two_lists(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- columns_info ---


@psycopg_available
def test_get_columns_info_returns_dict(store: PostgreSQLRecordStore) -> None:
    result = store.get_columns_info()
    assert isinstance(result, dict)


@psycopg_available
def test_get_columns_info_keys_are_column_names(store: PostgreSQLRecordStore) -> None:
    result = store.get_columns_info()
    assert set(result.keys()) == {"id", "metadata"}


@psycopg_available
def test_get_columns_info_values_are_strings(store: PostgreSQLRecordStore) -> None:
    result = store.get_columns_info()
    assert all(isinstance(v, str) for v in result.values())


@psycopg_available
def test_get_columns_info_non_empty_for_created_table(store: PostgreSQLRecordStore) -> None:
    result = store.get_columns_info()
    assert len(result) > 0


@psycopg_available
def test_get_columns_info_does_not_mutate_between_calls(store: PostgreSQLRecordStore) -> None:
    first = store.get_columns_info()
    second = store.get_columns_info()
    assert first == second
    assert first is not second  # each call builds a fresh dict


@psycopg_available
def test_show_columns_info_does_not_raise(
    store: PostgreSQLRecordStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.show_columns_info()  # should not raise
    captured = capsys.readouterr()
    assert captured.out != ""


@psycopg_available
def test_show_columns_info_output_contains_column_names(
    store: PostgreSQLRecordStore, capsys: pytest.CaptureFixture[str]
) -> None:
    expected_columns = store.get_columns_info().keys()
    store.show_columns_info()
    captured = capsys.readouterr()
    for col in expected_columns:
        assert col in captured.out


@psycopg_available
def test_show_columns_info_returns_none(store: PostgreSQLRecordStore) -> None:
    assert store.show_columns_info() is None


# --- lazy_all ---


@psycopg_available
def test_lazy_all_empty_store_yields_nothing(store: PostgreSQLRecordStore) -> None:
    assert list(store.lazy_all()) == []


@psycopg_available
def test_lazy_all_returns_generator(store: PostgreSQLRecordStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


@psycopg_available
def test_lazy_all_yields_one_record_at_a_time(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Record)


@psycopg_available
def test_lazy_all_returns_all_records(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


@psycopg_available
def test_lazy_all_matches_all(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert sorted(store.lazy_all(), key=lambda r: r.id) == sorted(store.all(), key=lambda r: r.id)


@psycopg_available
def test_lazy_all_yields_record_instances(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert all(isinstance(rec, Record) for rec in result)


@psycopg_available
def test_lazy_all_preserves_metadata(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = {rec.id: rec for rec in store.lazy_all()}
    for rec in records:
        assert result[rec.id].metadata == rec.metadata


@psycopg_available
def test_lazy_all_does_not_mutate_store(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    list(store.lazy_all())
    assert store.count() == len(records)


@psycopg_available
def test_lazy_all_single_record(store: PostgreSQLRecordStore) -> None:
    store.add_records([Record(id="1", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


@psycopg_available
def test_lazy_all_independent_generators_do_not_interfere(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    """Two separate lazy_all() generators should each independently
    yield the full set of records, since each uses its own named
    cursor."""
    store.add_records(records)
    gen1 = store.lazy_all()
    gen2 = store.lazy_all()
    first_from_gen1 = next(gen1)
    result2 = list(gen2)
    remaining_from_gen1 = list(gen1)

    assert len(result2) == len(records)
    assert len(remaining_from_gen1) + 1 == len(records)
    assert first_from_gen1.id not in {d.id for d in remaining_from_gen1}


# --- iter_batches ---


@psycopg_available
def test_iter_batches_empty_store_yields_nothing(store: PostgreSQLRecordStore) -> None:
    assert list(store.iter_batches()) == []


@psycopg_available
def test_iter_batches_returns_generator(store: PostgreSQLRecordStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


@psycopg_available
def test_iter_batches_default_batch_size(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


@psycopg_available
def test_iter_batches_yields_correct_batch_sizes(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


@psycopg_available
def test_iter_batches_last_batch_may_be_smaller(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


@psycopg_available
def test_iter_batches_batch_size_larger_than_store(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


@psycopg_available
def test_iter_batches_batch_size_one(store: PostgreSQLRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


@psycopg_available
def test_iter_batches_returns_all_records(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = [rec for batch in store.iter_batches(batch_size=2) for rec in batch]
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


@psycopg_available
def test_iter_batches_batches_contain_record_instances(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(rec, Record) for batch in batches for rec in batch)


@psycopg_available
def test_iter_batches_zero_batch_size_raises(store: PostgreSQLRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


@psycopg_available
def test_iter_batches_negative_batch_size_raises(store: PostgreSQLRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


@psycopg_available
def test_iter_batches_error_raised_before_any_query(store: PostgreSQLRecordStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


@psycopg_available
def test_iter_batches_does_not_mutate_store(
    store: PostgreSQLRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(records)


# --- close ---


@psycopg_available
def test_close_closes_underlying_connection(store: PostgreSQLRecordStore) -> None:
    store.close()
    with pytest.raises(psycopg.OperationalError):
        store.count()


@psycopg_available
def test_close_is_idempotent(store: PostgreSQLRecordStore) -> None:
    store.close()
    store.close()  # should not raise


@psycopg_available
def test_close_returns_none(store: PostgreSQLRecordStore) -> None:
    assert store.close() is None


# --- context manager ---


@psycopg_available
def test_context_manager_returns_self() -> None:
    with PostgreSQLRecordStore(_DSN) as store:
        assert isinstance(store, PostgreSQLRecordStore)


@psycopg_available
def test_context_manager_closes_on_normal_exit() -> None:
    with PostgreSQLRecordStore(_DSN) as store:
        store._conn.execute("TRUNCATE TABLE records")
        store._conn.commit()
        store.add_records([Record(id="1", metadata={})])
        assert store.count() == 1

    with pytest.raises(psycopg.OperationalError):
        store.count()


@psycopg_available
def test_context_manager_closes_on_exception() -> None:
    msg = "boom"
    with pytest.raises(ValueError, match="boom"), PostgreSQLRecordStore(_DSN) as store:
        raise ValueError(msg)

    with pytest.raises(psycopg.OperationalError):
        store.count()


@psycopg_available
def test_context_manager_usable_for_reads_and_writes() -> None:
    with PostgreSQLRecordStore(_DSN) as store:
        store._conn.execute("TRUNCATE TABLE records")
        store._conn.commit()
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


@psycopg_available
def test_context_manager_multiple_open_close_reopens_connection() -> None:
    record_store = PostgreSQLRecordStore(_DSN)
    record_store._conn.execute("TRUNCATE TABLE records")
    record_store._conn.commit()
    record_store.close()
    with record_store as store:
        assert store.count() == 0
        store.add_records([Record(id="1", metadata={})])
        assert store.count() == 1
    store.close()
