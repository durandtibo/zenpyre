from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from zenpyre.embeddings.factory import (
    BaseEmbeddingsFactory,
    ConfigurableEmbeddingsFactory,
)

FAKE_EMBEDDINGS_TARGET = "langchain_core.embeddings.fake.FakeEmbeddings"


#######################################################
#     Tests for ConfigurableEmbeddingsFactory         #
#######################################################


# --- Inheritance ---


def test_configurable_embeddings_factory_is_base_embeddings_factory() -> None:
    assert isinstance(
        ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128)), BaseEmbeddingsFactory
    )


# --- make_embeddings from instance ---


def test_configurable_embeddings_factory_make_embeddings_returns_embeddings() -> None:
    factory = ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128))
    assert isinstance(factory.make_embeddings(), Embeddings)


def test_configurable_embeddings_factory_make_embeddings_returns_same_instance() -> None:
    embeddings = FakeEmbeddings(size=128)
    factory = ConfigurableEmbeddingsFactory(embeddings)
    assert factory.make_embeddings() is embeddings


# --- make_embeddings from dict ---


def test_configurable_embeddings_factory_make_embeddings_from_dict_returns_embeddings() -> None:
    factory = ConfigurableEmbeddingsFactory({"_target_": FAKE_EMBEDDINGS_TARGET, "size": 128})
    assert isinstance(factory.make_embeddings(), Embeddings)


def test_configurable_embeddings_factory_make_embeddings_from_dict_returns_correct_type() -> None:
    factory = ConfigurableEmbeddingsFactory({"_target_": FAKE_EMBEDDINGS_TARGET, "size": 128})
    assert isinstance(factory.make_embeddings(), FakeEmbeddings)


# --- __repr__ ---


def test_configurable_embeddings_factory_repr_contains_class_name() -> None:
    factory = ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128))
    assert repr(factory).startswith("ConfigurableEmbeddingsFactory(")


def test_configurable_embeddings_factory_repr_contains_embeddings() -> None:
    factory = ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128))
    assert "FakeEmbeddings" in repr(factory)


# --- __str__ ---


def test_configurable_embeddings_factory_str_contains_class_name() -> None:
    factory = ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128))
    assert str(factory).startswith("ConfigurableEmbeddingsFactory(")


def test_configurable_embeddings_factory_str_differs_from_repr() -> None:
    factory = ConfigurableEmbeddingsFactory(FakeEmbeddings(size=128))
    assert str(factory) != repr(factory)
