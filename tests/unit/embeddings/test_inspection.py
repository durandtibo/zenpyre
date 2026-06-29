"""Unit tests for inspect_embeddings."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from zenpyre.embeddings import inspect_embeddings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Define a custom FakeEmbeddings because FakeEmbeddings fails if numpy is not installed


class MyFakeEmbeddings(FakeEmbeddings):
    def _get_embedding(self) -> list[float]:
        return [1.0] * self.size


@pytest.fixture(scope="module")
def vector_store() -> InMemoryVectorStore:
    """Return a real InMemoryVectorStore populated with cat
    documents."""
    vs = InMemoryVectorStore(MyFakeEmbeddings(size=6))
    vs.add_documents(
        [
            Document(
                page_content="Cats sleep up to 16 hours a day.", metadata={"source": "cats.txt"}
            ),
            Document(page_content="Cats are obligate carnivores.", metadata={"source": "cats.txt"}),
            Document(
                page_content="Cats are known for their agility.", metadata={"source": "cats.txt"}
            ),
        ]
    )
    return vs


###########################################
#     Tests for inspect_embeddings        #
###########################################


def test_inspect_embeddings_returns_none(vector_store: InMemoryVectorStore) -> None:
    assert inspect_embeddings(vector_store) is None


def test_inspect_embeddings_does_not_raise(vector_store: InMemoryVectorStore) -> None:
    inspect_embeddings(vector_store)


def test_inspect_embeddings_custom_n(vector_store: InMemoryVectorStore) -> None:
    inspect_embeddings(vector_store, n=1)


def test_inspect_embeddings_n_larger_than_documents(vector_store: InMemoryVectorStore) -> None:
    inspect_embeddings(vector_store, n=100)


def test_inspect_embeddings_empty_vector_store_returns_none() -> None:
    vs = InMemoryVectorStore(FakeEmbeddings(size=6))
    assert inspect_embeddings(vs) is None


def test_inspect_embeddings_empty_vector_store_does_not_raise() -> None:
    vs = InMemoryVectorStore(FakeEmbeddings(size=6))
    inspect_embeddings(vs)


def test_inspect_embeddings_calls_similarity_search(vector_store: InMemoryVectorStore) -> None:
    with patch.object(
        vector_store, "similarity_search", wraps=vector_store.similarity_search
    ) as mock:
        inspect_embeddings(vector_store, n=2)
    mock.assert_called_once_with("", k=2)
