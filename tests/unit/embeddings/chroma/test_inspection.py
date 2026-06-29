"""Unit tests for inspect_embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock

from zenpyre.embeddings.chroma import inspect_embeddings
from zenpyre.testing.fixtures import langchain_chroma_available
from zenpyre.utils.imports import is_langchain_chroma_available

if is_langchain_chroma_available():
    from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_store(
    n: int = 2,
    dims: int = 6,
    embeddings: list | None = None,
) -> MagicMock:
    """Return a mock Chroma vector store with n entries."""
    mock = MagicMock(spec=Chroma)
    mock.get.return_value = {
        "embeddings": (
            embeddings
            if embeddings is not None
            else [[float(i + d) for d in range(dims)] for i in range(n)]
        ),
        "documents": [f"Document {i}" for i in range(n)],
        "metadatas": [{"source": f"file_{i}.txt"} for i in range(n)],
        "ids": [f"id_{i}" for i in range(n)],
    }
    return mock


def _make_empty_vector_store() -> MagicMock:
    """Return a mock Chroma vector store with no embeddings."""
    mock = MagicMock(spec=Chroma)
    mock.get.return_value = {
        "embeddings": None,
        "documents": None,
        "metadatas": None,
        "ids": None,
    }
    return mock


###########################################
#     Tests for inspect_embeddings        #
###########################################


@langchain_chroma_available
def test_inspect_embeddings_returns_none() -> None:
    assert inspect_embeddings(_make_vector_store()) is None


@langchain_chroma_available
def test_inspect_embeddings_calls_get() -> None:
    vector_store = _make_vector_store()
    inspect_embeddings(vector_store)
    vector_store.get.assert_called_once()


@langchain_chroma_available
def test_inspect_embeddings_calls_get_with_correct_include() -> None:
    vector_store = _make_vector_store()
    inspect_embeddings(vector_store)
    _, kwargs = vector_store.get.call_args
    assert kwargs["include"] == ["embeddings", "documents", "metadatas"]


@langchain_chroma_available
def test_inspect_embeddings_empty_vector_store_returns_none() -> None:
    assert inspect_embeddings(_make_empty_vector_store()) is None


@langchain_chroma_available
def test_inspect_embeddings_empty_vector_store_does_not_raise() -> None:
    inspect_embeddings(_make_empty_vector_store())


@langchain_chroma_available
def test_inspect_embeddings_default_n() -> None:
    inspect_embeddings(_make_vector_store(n=10))


@langchain_chroma_available
def test_inspect_embeddings_custom_n() -> None:
    inspect_embeddings(_make_vector_store(n=10), n=5)


@langchain_chroma_available
def test_inspect_embeddings_n_larger_than_embeddings() -> None:
    inspect_embeddings(_make_vector_store(n=2), n=10)


@langchain_chroma_available
def test_inspect_embeddings_single_embedding() -> None:
    inspect_embeddings(_make_vector_store(n=1))


@langchain_chroma_available
def test_inspect_embeddings_short_vector() -> None:
    inspect_embeddings(_make_vector_store(n=1, dims=3))
