from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.document_stores import InMemoryDocumentStore
from zenpyre.document_stores.base import BaseDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    ConfigurableDocumentStoreFactory,
)

IN_MEMORY_DOCUMENT_STORE_TARGET = "zenpyre.document_stores.InMemoryDocumentStore"


def _make_document_store() -> InMemoryDocumentStore:
    """Return an InMemoryDocumentStore instance for testing."""
    return InMemoryDocumentStore()


#######################################################
#     Tests for ConfigurableDocumentStoreFactory     #
#######################################################


# --- Inheritance ---


def test_configurable_document_store_factory_is_base_document_store_factory() -> None:
    assert isinstance(
        ConfigurableDocumentStoreFactory(_make_document_store()), BaseDocumentStoreFactory
    )


# --- make_document_store from instance ---


def test_configurable_document_store_factory_make_document_store_returns_base_document_store() -> (
    None
):
    factory = ConfigurableDocumentStoreFactory(_make_document_store())
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


def test_configurable_document_store_factory_make_document_store_returns_same_instance() -> None:
    document_store = _make_document_store()
    factory = ConfigurableDocumentStoreFactory(document_store)
    assert factory.make_document_store() is document_store


# --- make_document_store from dict ---


def test_configurable_document_store_factory_make_document_store_from_dict_returns_base_document_store() -> (
    None
):
    factory = ConfigurableDocumentStoreFactory({"_target_": IN_MEMORY_DOCUMENT_STORE_TARGET})
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


def test_configurable_document_store_factory_make_document_store_from_dict_returns_correct_type() -> (
    None
):
    factory = ConfigurableDocumentStoreFactory({"_target_": IN_MEMORY_DOCUMENT_STORE_TARGET})
    assert isinstance(factory.make_document_store(), InMemoryDocumentStore)


# --- _get_repr_kwargs ---


def test_configurable_document_store_factory_get_repr_kwargs_instance() -> None:
    document_store = _make_document_store()
    factory = ConfigurableDocumentStoreFactory(document_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"document_store": document_store})


def test_configurable_document_store_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": IN_MEMORY_DOCUMENT_STORE_TARGET}
    factory = ConfigurableDocumentStoreFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"document_store": config})


# --- __repr__ and __str__ ---


def test_configurable_document_store_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableDocumentStoreFactory(_make_document_store())
    assert repr(factory).startswith("ConfigurableDocumentStoreFactory(")


def test_configurable_document_store_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableDocumentStoreFactory(_make_document_store())
    assert str(factory).startswith("ConfigurableDocumentStoreFactory(")


def test_configurable_document_store_factory_repr_contains_document_store() -> None:
    factory = ConfigurableDocumentStoreFactory(_make_document_store())
    assert "document_store" in repr(factory)


def test_configurable_document_store_factory_str_contains_document_store() -> None:
    factory = ConfigurableDocumentStoreFactory(_make_document_store())
    assert "document_store" in str(factory)
