from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.record_stores import BaseRecordStore, InMemoryRecordStore
from zenpyre.record_stores.factory import (
    BaseRecordStoreFactory,
    RecordStoreFactory,
)


def _make_record_store() -> InMemoryRecordStore:
    """Return an InMemoryRecordStore instance for testing."""
    return InMemoryRecordStore()


##################################################
#     Tests for RecordStoreFactory              #
##################################################


# --- Inheritance ---


def test_record_store_factory_is_base_record_store_factory() -> None:
    assert isinstance(RecordStoreFactory(_make_record_store()), BaseRecordStoreFactory)


# --- make_record_store ---


def test_record_store_factory_make_record_store_returns_base_record_store() -> None:
    factory = RecordStoreFactory(_make_record_store())
    assert isinstance(factory.make_record_store(), BaseRecordStore)


def test_record_store_factory_make_record_store_returns_same_instance() -> None:
    record_store = _make_record_store()
    factory = RecordStoreFactory(record_store)
    assert factory.make_record_store() is record_store


def test_record_store_factory_make_record_store_returns_same_instance_across_calls() -> None:
    record_store = _make_record_store()
    factory = RecordStoreFactory(record_store)
    assert factory.make_record_store() is factory.make_record_store()


# --- _get_repr_kwargs ---


def test_record_store_factory_get_repr_kwargs() -> None:
    record_store = _make_record_store()
    factory = RecordStoreFactory(record_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"record_store": record_store})


# --- __repr__ and __str__ ---


def test_record_store_factory_repr_starts_with_class_name() -> None:
    factory = RecordStoreFactory(_make_record_store())
    assert repr(factory).startswith("RecordStoreFactory(")


def test_record_store_factory_str_starts_with_class_name() -> None:
    factory = RecordStoreFactory(_make_record_store())
    assert str(factory).startswith("RecordStoreFactory(")


def test_record_store_factory_repr_contains_record_store() -> None:
    factory = RecordStoreFactory(_make_record_store())
    assert "record_store" in repr(factory)


def test_record_store_factory_str_contains_record_store() -> None:
    factory = RecordStoreFactory(_make_record_store())
    assert "record_store" in str(factory)
