from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.record_stores import InMemoryRecordStore
from zenpyre.record_stores.base import BaseRecordStore
from zenpyre.record_stores.factory import (
    BaseRecordStoreFactory,
    ConfigurableRecordStoreFactory,
)

IN_MEMORY_RECORD_STORE_TARGET = "zenpyre.record_stores.InMemoryRecordStore"


def _make_record_store() -> InMemoryRecordStore:
    """Return an InMemoryRecordStore instance for testing."""
    return InMemoryRecordStore()


#####################################################
#     Tests for ConfigurableRecordStoreFactory     #
#####################################################


# --- Inheritance ---


def test_configurable_record_store_factory_is_base_record_store_factory() -> None:
    assert isinstance(ConfigurableRecordStoreFactory(_make_record_store()), BaseRecordStoreFactory)


# --- make_record_store from instance ---


def test_configurable_record_store_factory_make_record_store_returns_base_record_store() -> None:
    factory = ConfigurableRecordStoreFactory(_make_record_store())
    assert isinstance(factory.make_record_store(), BaseRecordStore)


def test_configurable_record_store_factory_make_record_store_returns_same_instance() -> None:
    record_store = _make_record_store()
    factory = ConfigurableRecordStoreFactory(record_store)
    assert factory.make_record_store() is record_store


# --- make_record_store from dict ---


def test_configurable_record_store_factory_make_record_store_from_dict_returns_base_record_store() -> (
    None
):
    factory = ConfigurableRecordStoreFactory({"_target_": IN_MEMORY_RECORD_STORE_TARGET})
    assert isinstance(factory.make_record_store(), BaseRecordStore)


def test_configurable_record_store_factory_make_record_store_from_dict_returns_correct_type() -> (
    None
):
    factory = ConfigurableRecordStoreFactory({"_target_": IN_MEMORY_RECORD_STORE_TARGET})
    assert isinstance(factory.make_record_store(), InMemoryRecordStore)


# --- _get_repr_kwargs ---


def test_configurable_record_store_factory_get_repr_kwargs_instance() -> None:
    record_store = _make_record_store()
    factory = ConfigurableRecordStoreFactory(record_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"record_store": record_store})


def test_configurable_record_store_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": IN_MEMORY_RECORD_STORE_TARGET}
    factory = ConfigurableRecordStoreFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"record_store": config})


# --- __repr__ and __str__ ---


def test_configurable_record_store_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableRecordStoreFactory(_make_record_store())
    assert repr(factory).startswith("ConfigurableRecordStoreFactory(")


def test_configurable_record_store_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableRecordStoreFactory(_make_record_store())
    assert str(factory).startswith("ConfigurableRecordStoreFactory(")


def test_configurable_record_store_factory_repr_contains_record_store() -> None:
    factory = ConfigurableRecordStoreFactory(_make_record_store())
    assert "record_store" in repr(factory)


def test_configurable_record_store_factory_str_contains_record_store() -> None:
    factory = ConfigurableRecordStoreFactory(_make_record_store())
    assert "record_store" in str(factory)
