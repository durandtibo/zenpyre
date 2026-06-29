from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from zenpyre.embeddings.chroma import inspect_embeddings
from zenpyre.testing.fixtures import langchain_chroma_available
from zenpyre.utils.imports import is_langchain_chroma_available

if is_langchain_chroma_available():
    from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vector_store() -> Chroma:
    """Return a real in-memory Chroma vector store populated with cat
    documents."""
    return Chroma.from_documents(
        documents=[
            Document(
                page_content="Cats sleep up to 16 hours a day.", metadata={"source": "cats.txt"}
            ),
            Document(page_content="Cats are obligate carnivores.", metadata={"source": "cats.txt"}),
            Document(
                page_content="Cats are known for their agility.", metadata={"source": "cats.txt"}
            ),
        ],
        embedding=FakeEmbeddings(size=6),
    )


@pytest.fixture(scope="module")
def empty_vector_store() -> Chroma:
    """Return a real in-memory Chroma vector store populated with cat
    documents."""
    return Chroma.from_documents(
        documents=[],
        embedding=FakeEmbeddings(size=6),
    )


###########################################
#     Tests for inspect_embeddings        #
###########################################


@langchain_chroma_available
def test_inspect_embeddings_chroma_returns_none(vector_store: Chroma) -> None:
    assert inspect_embeddings(vector_store) is None


@langchain_chroma_available
def test_inspect_embeddings_chroma_custom_n(vector_store: Chroma) -> None:
    inspect_embeddings(vector_store, n=1)


@langchain_chroma_available
def test_inspect_embeddings_chroma_n_larger_than_embeddings(vector_store: Chroma) -> None:
    inspect_embeddings(vector_store, n=100)


@langchain_chroma_available
def test_inspect_embeddings_empty_vector_store_does_not_raise(empty_vector_store: Chroma) -> None:
    inspect_embeddings(empty_vector_store)
