from __future__ import annotations

from collections.abc import Iterator

import pytest

from zenpyre.record_stores import InMemoryRecordStore
from zenpyre.records import Record

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


@pytest.fixture
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
#     Tests for InMemoryRecordStore              #
##################################################


# --- repr/str ---


def test_repr(store: InMemoryRecordStore) -> None:
    assert repr(store).startswith("InMemoryRecordStore(")


def test_str(store: InMemoryRecordStore) -> None:
    assert repr(store).startswith("InMemoryRecordStore(")


# --- add_records ---


def test_add_records_increases_count(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


def test_add_records_empty_id_raises(store: InMemoryRecordStore) -> None:
    with pytest.raises(ValueError, match=r"id"):
        store.add_records([Record(id="", metadata={"note": "no id"})])


def test_add_records_upsert_replaces_existing(store: InMemoryRecordStore) -> None:
    store.add_records([Record(id="1", metadata={"version": "original"})])
    store.add_records([Record(id="1", metadata={"version": "updated"})])
    assert store.count() == 1
    assert store.get("1").metadata["version"] == "updated"


def test_add_records_empty(store: InMemoryRecordStore) -> None:
    store.add_records([])


# --- count ---


def test_count_empty_store(store: InMemoryRecordStore) -> None:
    assert store.count() == 0


def test_count_after_adding(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.count() == len(records)


# --- get ---


def test_get_existing_record(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1") == records[0]


def test_get_missing_record_returns_none(store: InMemoryRecordStore) -> None:
    assert store.get("nonexistent") is None


def test_get_round_trips_metadata(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.get("1").metadata == records[0].metadata


# --- get_many ---


def test_get_many_returns_correct_length(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert len(store.get_many(["1", "2", "99"])) == 3


def test_get_many_returns_none_for_missing(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


def test_get_many_preserves_order(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- filter ---


def test_filter_no_args_returns_all(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert len(store.filter()) == len(records)


def test_filter_single_field(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


def test_filter_multiple_fields(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Alice", category="Programming")
    assert len(result) == 2


def test_filter_no_match_returns_empty(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert store.filter(author="Charlie") == []


def test_filter_preserves_full_record(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.filter(author="Bob", category="History")
    assert all(
        r.metadata == d.metadata
        for r, d in zip(
            sorted(result, key=lambda x: x.id),
            sorted([d for d in records if d.metadata["author"] == "Bob"], key=lambda x: x.id),
        )
    )


def test_filter_empty_store_returns_empty(store: InMemoryRecordStore) -> None:
    assert store.filter(author="Alice") == []


# --- delete ---


def test_delete_removes_record(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete("1")
    assert store.count() == len(records) - 1
    assert store.get("1") is None


def test_delete_nonexistent_is_silent(store: InMemoryRecordStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


def test_delete_many_removes_records(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.count() == len(records) - 2
    assert store.get("1") is None
    assert store.get("3") is None


def test_delete_many_preserves_other_records(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


def test_delete_many_empty_list_is_no_op(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many([])
    assert store.count() == len(records)


def test_delete_many_nonexistent_ids_are_silent(store: InMemoryRecordStore) -> None:
    store.delete_many(["99", "100"])


def test_delete_many_single_id(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    store.delete_many(["2"])
    assert store.count() == len(records) - 1
    assert store.get("2") is None


# --- check_ids ---


def test_check_ids_all_found(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


def test_check_ids_all_missing(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


def test_check_ids_mixed(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


def test_check_ids_preserves_order(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


def test_check_ids_empty_input_returns_empty_lists(store: InMemoryRecordStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


def test_check_ids_empty_store_returns_all_missing(store: InMemoryRecordStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


def test_check_ids_returns_tuple_of_two_lists(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- all ---


def test_all_empty_store(store: InMemoryRecordStore) -> None:
    assert store.all() == []


def test_all_returns_all_records(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = store.all()
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


# --- lazy_all ---


def test_lazy_all_empty_store_yields_nothing(store: InMemoryRecordStore) -> None:
    assert list(store.lazy_all()) == []


def test_lazy_all_returns_generator(store: InMemoryRecordStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


def test_lazy_all_yields_one_record_at_a_time(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Record)


def test_lazy_all_returns_all_records(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


def test_lazy_all_matches_all(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    assert list(store.lazy_all()) == store.all()


def test_lazy_all_yields_record_instances(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = list(store.lazy_all())
    assert all(isinstance(record, Record) for record in result)


def test_lazy_all_preserves_metadata(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    result = {record.id: record for record in store.lazy_all()}
    for record in records:
        assert result[record.id].metadata == record.metadata


def test_lazy_all_does_not_mutate_store(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    list(store.lazy_all())
    assert store.count() == len(records)


def test_lazy_all_single_record(store: InMemoryRecordStore) -> None:
    store.add_records([Record(id="1", metadata={"note": "solo"})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


def test_lazy_all_is_lazy_not_exhausted_on_creation(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    """Calling lazy_all() should not itself execute the query eagerly
    consuming results before iteration begins; adding records after
    creating the generator but before the first next() call should still
    be reflected, confirming the query executes lazily."""
    gen = store.lazy_all()
    store.add_records(records)
    result = list(gen)
    assert len(result) == len(records)


def test_lazy_all_independent_generators_do_not_interfere(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    """Two separate lazy_all() generators should each independently
    yield the full set of records, since each uses its own cursor."""
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


def test_iter_batches_empty_store_yields_nothing(store: InMemoryRecordStore) -> None:
    assert list(store.iter_batches()) == []


def test_iter_batches_returns_generator(store: InMemoryRecordStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


def test_iter_batches_default_batch_size(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


def test_iter_batches_yields_correct_batch_sizes(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


def test_iter_batches_last_batch_may_be_smaller(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


def test_iter_batches_batch_size_larger_than_store(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(records)


def test_iter_batches_batch_size_one(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


def test_iter_batches_returns_all_records(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    result = [record for batch in store.iter_batches(batch_size=2) for record in batch]
    assert len(result) == len(records)
    assert {r.id for r in result} == {d.id for d in records}


def test_iter_batches_matches_all(store: InMemoryRecordStore, records: list[Record]) -> None:
    store.add_records(records)
    flattened = [record for batch in store.iter_batches(batch_size=2) for record in batch]
    assert flattened == store.all()


def test_iter_batches_batches_contain_record_instances(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(record, Record) for batch in batches for record in batch)


def test_iter_batches_zero_batch_size_raises(store: InMemoryRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


def test_iter_batches_negative_batch_size_raises(store: InMemoryRecordStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


def test_iter_batches_error_raised_before_any_query(store: InMemoryRecordStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


def test_iter_batches_does_not_mutate_store(
    store: InMemoryRecordStore, records: list[Record]
) -> None:
    store.add_records(records)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(records)
