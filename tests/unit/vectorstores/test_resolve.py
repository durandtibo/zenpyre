from __future__ import annotations

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from zenpyre.vectorstores import resolve_vector_store

MODULE = "zenpyre.vectorstores.resolve"
IN_MEMORY_VECTOR_STORE_TARGET = "langchain_core.vectorstores.InMemoryVectorStore"


def _make_vector_store() -> InMemoryVectorStore:
    """Return an InMemoryVectorStore instance for testing."""
    return InMemoryVectorStore(FakeEmbeddings(size=128))


##########################################
#     Tests for resolve_vector_store     #
##########################################


# --- Pass-through ---


def test_resolve_vector_store_returns_vector_store_instance() -> None:
    assert isinstance(resolve_vector_store(_make_vector_store()), VectorStore)


def test_resolve_vector_store_passthrough_returns_same_instance() -> None:
    vector_store = _make_vector_store()
    assert resolve_vector_store(vector_store) is vector_store


# --- From dict ---


def test_resolve_vector_store_from_dict_returns_vector_store() -> None:
    result = resolve_vector_store(
        {
            "_target_": IN_MEMORY_VECTOR_STORE_TARGET,
            "embedding": FakeEmbeddings(size=128),
        }
    )
    assert isinstance(result, VectorStore)


def test_resolve_vector_store_from_dict_returns_correct_type() -> None:
    result = resolve_vector_store(
        {
            "_target_": IN_MEMORY_VECTOR_STORE_TARGET,
            "embedding": FakeEmbeddings(size=128),
        }
    )
    assert isinstance(result, InMemoryVectorStore)


# --- Invalid input ---


def test_resolve_vector_store_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a VectorStore instance"):
        resolve_vector_store("not-a-vector-store")
