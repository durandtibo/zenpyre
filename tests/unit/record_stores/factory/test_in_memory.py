from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.record_stores import BaseRecordStore, InMemoryRecordStore
from zenpyre.record_stores.factory import (
    BaseRecordStoreFactory,
    InMemoryRecordStoreFactory,
)

#################################################
#     Tests for InMemoryRecordStoreFactory     #
#################################################


# --- Inheritance ---


def test_in_memory_record_store_factory_is_base_record_store_factory() -> None:
    assert isinstance(InMemoryRecordStoreFactory(), BaseRecordStoreFactory)


# --- make_record_store ---


def test_in_memory_record_store_factory_make_record_store_returns_base_record_store() -> None:
    factory = InMemoryRecordStoreFactory()
    assert isinstance(factory.make_record_store(), BaseRecordStore)


def test_in_memory_record_store_factory_make_record_store_returns_in_memory_record_store() -> None:
    factory = InMemoryRecordStoreFactory()
    assert isinstance(factory.make_record_store(), InMemoryRecordStore)


def test_in_memory_record_store_factory_make_record_store_returns_new_instance_each_call() -> None:
    factory = InMemoryRecordStoreFactory()
    assert factory.make_record_store() is not factory.make_record_store()


def test_in_memory_record_store_factory_make_record_store_returns_empty_store() -> None:
    factory = InMemoryRecordStoreFactory()
    assert factory.make_record_store().count() == 0


# --- _get_repr_kwargs ---


def test_in_memory_record_store_factory_get_repr_kwargs() -> None:
    factory = InMemoryRecordStoreFactory()
    assert objects_are_equal(factory._get_repr_kwargs(), {})


# --- __repr__ and __str__ ---


def test_in_memory_record_store_factory_repr_starts_with_class_name() -> None:
    factory = InMemoryRecordStoreFactory()
    assert repr(factory).startswith("InMemoryRecordStoreFactory(")


def test_in_memory_record_store_factory_str_starts_with_class_name() -> None:
    factory = InMemoryRecordStoreFactory()
    assert str(factory).startswith("InMemoryRecordStoreFactory(")
