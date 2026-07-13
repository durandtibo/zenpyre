from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.document_stores import BaseDocumentStore, InMemoryDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    InMemoryDocumentStoreFactory,
)

#####################################################
#     Tests for InMemoryDocumentStoreFactory        #
#####################################################


# --- Inheritance ---


def test_in_memory_document_store_factory_is_base_document_store_factory() -> None:
    assert isinstance(InMemoryDocumentStoreFactory(), BaseDocumentStoreFactory)


# --- make_document_store ---


def test_in_memory_document_store_factory_make_document_store_returns_base_document_store() -> None:
    factory = InMemoryDocumentStoreFactory()
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


def test_in_memory_document_store_factory_make_document_store_returns_in_memory_document_store() -> (
    None
):
    factory = InMemoryDocumentStoreFactory()
    assert isinstance(factory.make_document_store(), InMemoryDocumentStore)


def test_in_memory_document_store_factory_make_document_store_returns_new_instance_each_call() -> (
    None
):
    factory = InMemoryDocumentStoreFactory()
    assert factory.make_document_store() is not factory.make_document_store()


def test_in_memory_document_store_factory_make_document_store_returns_empty_store() -> None:
    factory = InMemoryDocumentStoreFactory()
    assert factory.make_document_store().count() == 0


# --- _get_repr_kwargs ---


def test_in_memory_document_store_factory_get_repr_kwargs() -> None:
    factory = InMemoryDocumentStoreFactory()
    assert objects_are_equal(factory._get_repr_kwargs(), {})


# --- __repr__ and __str__ ---


def test_in_memory_document_store_factory_repr_starts_with_class_name() -> None:
    factory = InMemoryDocumentStoreFactory()
    assert repr(factory).startswith("InMemoryDocumentStoreFactory(")


def test_in_memory_document_store_factory_str_starts_with_class_name() -> None:
    factory = InMemoryDocumentStoreFactory()
    assert str(factory).startswith("InMemoryDocumentStoreFactory(")
