from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.document_stores import BaseDocumentStore, InMemoryDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    DocumentStoreFactory,
)


def _make_document_store() -> InMemoryDocumentStore:
    """Return an InMemoryDocumentStore instance for testing."""
    return InMemoryDocumentStore()


##################################################
#     Tests for DocumentStoreFactory              #
##################################################


# --- Inheritance ---


def test_document_store_factory_is_base_document_store_factory() -> None:
    assert isinstance(DocumentStoreFactory(_make_document_store()), BaseDocumentStoreFactory)


# --- make_document_store ---


def test_document_store_factory_make_document_store_returns_base_document_store() -> None:
    factory = DocumentStoreFactory(_make_document_store())
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


def test_document_store_factory_make_document_store_returns_same_instance() -> None:
    document_store = _make_document_store()
    factory = DocumentStoreFactory(document_store)
    assert factory.make_document_store() is document_store


def test_document_store_factory_make_document_store_returns_same_instance_across_calls() -> None:
    document_store = _make_document_store()
    factory = DocumentStoreFactory(document_store)
    assert factory.make_document_store() is factory.make_document_store()


# --- _get_repr_kwargs ---


def test_document_store_factory_get_repr_kwargs() -> None:
    document_store = _make_document_store()
    factory = DocumentStoreFactory(document_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"document_store": document_store})


# --- __repr__ and __str__ ---


def test_document_store_factory_repr_starts_with_class_name() -> None:
    factory = DocumentStoreFactory(_make_document_store())
    assert repr(factory).startswith("DocumentStoreFactory(")


def test_document_store_factory_str_starts_with_class_name() -> None:
    factory = DocumentStoreFactory(_make_document_store())
    assert str(factory).startswith("DocumentStoreFactory(")


def test_document_store_factory_repr_contains_document_store() -> None:
    factory = DocumentStoreFactory(_make_document_store())
    assert "document_store" in repr(factory)


def test_document_store_factory_str_contains_document_store() -> None:
    factory = DocumentStoreFactory(_make_document_store())
    assert "document_store" in str(factory)
