from __future__ import annotations

from langchain_core.embeddings.fake import FakeEmbeddings

from zenpyre.embeddings.factory import BaseEmbeddingsFactory, EmbeddingsFactory

###########################################
#     Tests for EmbeddingsFactory         #
###########################################


def test_embeddings_factory_is_base_embeddings_factory() -> None:
    assert isinstance(EmbeddingsFactory(FakeEmbeddings(size=128)), BaseEmbeddingsFactory)


def test_embeddings_factory_make_embeddings_returns_embeddings() -> None:
    embeddings = FakeEmbeddings(size=128)
    assert EmbeddingsFactory(embeddings).make_embeddings() is embeddings


def test_embeddings_factory_make_embeddings_returns_same_instance() -> None:
    embeddings = FakeEmbeddings(size=128)
    factory = EmbeddingsFactory(embeddings)
    assert factory.make_embeddings() is factory.make_embeddings()


def test_embeddings_factory_make_embeddings_returns_none() -> None:
    assert EmbeddingsFactory(FakeEmbeddings(size=128)).make_embeddings() is not None


def test_embeddings_factory_stores_embeddings() -> None:
    embeddings = FakeEmbeddings(size=128)
    assert EmbeddingsFactory(embeddings)._embeddings is embeddings


def test_embeddings_factory_different_instances_independent() -> None:
    embeddings_a = FakeEmbeddings(size=128)
    embeddings_b = FakeEmbeddings(size=128)
    factory_a = EmbeddingsFactory(embeddings_a)
    factory_b = EmbeddingsFactory(embeddings_b)
    assert factory_a.make_embeddings() is not factory_b.make_embeddings()


# --- __repr__ ---


def test_embeddings_factory_repr_contains_class_name() -> None:
    factory = EmbeddingsFactory(FakeEmbeddings(size=128))
    assert repr(factory).startswith("EmbeddingsFactory(")


def test_embeddings_factory_repr_contains_embeddings() -> None:
    factory = EmbeddingsFactory(FakeEmbeddings(size=128))
    assert "FakeEmbeddings" in repr(factory)


# --- __str__ ---


def test_embeddings_factory_str_contains_class_name() -> None:
    factory = EmbeddingsFactory(FakeEmbeddings(size=128))
    assert str(factory).startswith("EmbeddingsFactory(")


def test_embeddings_factory_str_differs_from_repr() -> None:
    factory = EmbeddingsFactory(FakeEmbeddings(size=128))
    assert str(factory) != repr(factory)
