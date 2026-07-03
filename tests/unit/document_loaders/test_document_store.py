"""Unit tests for DocumentStoreLoader."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.documents import Document

from zenpyre.document_loaders import DocumentStoreLoader
from zenpyre.document_stores import InMemoryDocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(id="1", page_content="Hello", metadata={"author": "Alice"}),
        Document(id="2", page_content="World", metadata={"author": "Bob"}),
        Document(id="3", page_content="Foo", metadata={"author": "Alice"}),
    ]


##################################################
#     Tests for DocumentStoreLoader              #
##################################################

# --- repr/str ---


def test_document_store_loader_repr(store: InMemoryDocumentStore) -> None:
    assert repr(DocumentStoreLoader(store)).startswith("DocumentStoreLoader(")


def test_document_store_loader_str(store: InMemoryDocumentStore) -> None:
    assert str(DocumentStoreLoader(store)).startswith("DocumentStoreLoader(")


# --- lazy_load ---


def test_document_store_loader_lazy_load_empty_store_yields_nothing(
    store: InMemoryDocumentStore,
) -> None:
    loader = DocumentStoreLoader(store)
    assert list(loader.lazy_load()) == []


def test_document_store_loader_lazy_load_returns_iterator(store: InMemoryDocumentStore) -> None:
    loader = DocumentStoreLoader(store)
    result = loader.lazy_load()
    assert isinstance(result, Iterator)


def test_document_store_loader_lazy_load_yields_all_documents(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    result = list(loader.lazy_load())
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


def test_document_store_loader_lazy_load_yields_document_instances(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    result = list(loader.lazy_load())
    assert all(isinstance(doc, Document) for doc in result)


def test_document_store_loader_lazy_load_preserves_page_content_and_metadata(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    result = {doc.id: doc for doc in loader.lazy_load()}
    for doc in docs:
        assert result[doc.id].page_content == doc.page_content
        assert result[doc.id].metadata == doc.metadata


def test_document_store_loader_lazy_load_matches_store_all(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    loaded = sorted(loader.lazy_load(), key=lambda d: d.id)
    all_docs = sorted(store.all(), key=lambda d: d.id)
    assert loaded == all_docs


def test_document_store_loader_lazy_load_reflects_store_state_at_call_time(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    """Since lazy_load is a generator, documents added before iteration
    begins should still be picked up if the store is populated between
    creating the loader and calling load()."""
    loader = DocumentStoreLoader(store)
    store.add_documents(docs)
    result = loader.load()
    assert len(result) == len(docs)


# --- load ---


def test_document_store_loader_load_returns_list(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    result = loader.load()
    assert isinstance(result, list)


def test_document_store_loader_load_empty_store_returns_empty_list(
    store: InMemoryDocumentStore,
) -> None:
    loader = DocumentStoreLoader(store)
    assert loader.load() == []


def test_document_store_loader_load_does_not_mutate_store(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    loader.load()
    assert store.count() == len(docs)


def test_document_store_loader_load_single_document(store: InMemoryDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Solo")])
    loader = DocumentStoreLoader(store)
    result = loader.load()
    assert len(result) == 1
    assert result[0].id == "1"


def test_document_store_loader_load_mutating_result_does_not_affect_store(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    """InMemoryDocumentStore deep-copies on read, so mutating a loaded
    Document must not leak back into the store."""
    store.add_documents(docs)
    loader = DocumentStoreLoader(store)
    result = loader.load()
    result[0].page_content = "Mutated"
    assert store.get(result[0].id).page_content != "Mutated"
