"""Unit tests for VectorStoreFactory."""

from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from zenpyre.vectorstores.factory import BaseVectorStoreFactory, VectorStoreFactory


def _make_vector_store() -> InMemoryVectorStore:
    """Return an InMemoryVectorStore instance for testing."""
    return InMemoryVectorStore(FakeEmbeddings(size=128))


#######################################
#     Tests for VectorStoreFactory    #
#######################################


# --- Inheritance ---


def test_vector_store_factory_is_base_vector_store_factory() -> None:
    assert isinstance(VectorStoreFactory(_make_vector_store()), BaseVectorStoreFactory)


# --- make_vector_store ---


def test_vector_store_factory_make_vector_store_returns_vector_store() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert isinstance(factory.make_vector_store(), VectorStore)


def test_vector_store_factory_make_vector_store_returns_same_instance() -> None:
    vector_store = _make_vector_store()
    factory = VectorStoreFactory(vector_store)
    assert factory.make_vector_store() is vector_store


def test_vector_store_factory_make_vector_store_returns_same_instance_on_repeated_calls() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert factory.make_vector_store() is factory.make_vector_store()


def test_vector_store_factory_different_instances_independent() -> None:
    factory_a = VectorStoreFactory(_make_vector_store())
    factory_b = VectorStoreFactory(_make_vector_store())
    assert factory_a.make_vector_store() is not factory_b.make_vector_store()


# --- _get_repr_kwargs ---


def test_vector_store_factory_get_repr_kwargs() -> None:
    vector_store = _make_vector_store()
    factory = VectorStoreFactory(vector_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"vector_store": vector_store})


# --- __repr__ and __str__ ---


def test_vector_store_factory_repr_starts_with_class_name() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert repr(factory).startswith("VectorStoreFactory(")


def test_vector_store_factory_str_starts_with_class_name() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert str(factory).startswith("VectorStoreFactory(")


def test_vector_store_factory_repr_contains_vector_store() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert "vector_store" in repr(factory)


def test_vector_store_factory_repr_equals_str() -> None:
    factory = VectorStoreFactory(_make_vector_store())
    assert repr(factory) == str(factory)
