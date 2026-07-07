from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from zenpyre.vectorstores.factory import (
    BaseVectorStoreFactory,
    ConfigurableVectorStoreFactory,
)

IN_MEMORY_VECTOR_STORE_TARGET = "langchain_core.vectorstores.InMemoryVectorStore"


def _make_vector_store() -> InMemoryVectorStore:
    """Return an InMemoryVectorStore instance for testing."""
    return InMemoryVectorStore(FakeEmbeddings(size=128))


##################################################
#     Tests for ConfigurableVectorStoreFactory   #
##################################################


# --- Inheritance ---


def test_configurable_vector_store_factory_is_base_vector_store_factory() -> None:
    assert isinstance(ConfigurableVectorStoreFactory(_make_vector_store()), BaseVectorStoreFactory)


# --- make_vector_store from instance ---


def test_configurable_vector_store_factory_make_vector_store_returns_vector_store() -> None:
    factory = ConfigurableVectorStoreFactory(_make_vector_store())
    assert isinstance(factory.make_vector_store(), VectorStore)


def test_configurable_vector_store_factory_make_vector_store_returns_same_instance() -> None:
    vector_store = _make_vector_store()
    factory = ConfigurableVectorStoreFactory(vector_store)
    assert factory.make_vector_store() is vector_store


# --- make_vector_store from dict ---


def test_configurable_vector_store_factory_make_vector_store_from_dict_returns_vector_store() -> (
    None
):
    factory = ConfigurableVectorStoreFactory(
        {
            "_target_": IN_MEMORY_VECTOR_STORE_TARGET,
            "embedding": FakeEmbeddings(size=128),
        }
    )
    assert isinstance(factory.make_vector_store(), VectorStore)


def test_configurable_vector_store_factory_make_vector_store_from_dict_returns_correct_type() -> (
    None
):
    factory = ConfigurableVectorStoreFactory(
        {
            "_target_": IN_MEMORY_VECTOR_STORE_TARGET,
            "embedding": FakeEmbeddings(size=128),
        }
    )
    assert isinstance(factory.make_vector_store(), InMemoryVectorStore)


# --- _get_repr_kwargs ---


def test_configurable_vector_store_factory_get_repr_kwargs_instance() -> None:
    vector_store = _make_vector_store()
    factory = ConfigurableVectorStoreFactory(vector_store)
    assert objects_are_equal(factory._get_repr_kwargs(), {"vector_store": vector_store})


def test_configurable_vector_store_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": IN_MEMORY_VECTOR_STORE_TARGET, "embedding": FakeEmbeddings(size=128)}
    factory = ConfigurableVectorStoreFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"vector_store": config})


# --- __repr__ and __str__ ---


def test_configurable_vector_store_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableVectorStoreFactory(_make_vector_store())
    assert repr(factory).startswith("ConfigurableVectorStoreFactory(")


def test_configurable_vector_store_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableVectorStoreFactory(_make_vector_store())
    assert str(factory).startswith("ConfigurableVectorStoreFactory(")


def test_configurable_vector_store_factory_repr_contains_vector_store() -> None:
    factory = ConfigurableVectorStoreFactory(_make_vector_store())
    assert "vector_store" in repr(factory)


def test_configurable_vector_store_factory_str_contains_vector_store() -> None:
    factory = ConfigurableVectorStoreFactory(_make_vector_store())
    assert "vector_store" in str(factory)
