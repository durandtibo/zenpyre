"""Unit tests for DocumentListLoader."""

from __future__ import annotations

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from zenpyre.document_loaders import DocumentListLoader


def _make_docs() -> list[Document]:
    return [Document(page_content="Hello"), Document(page_content="World")]


##########################################
#     Tests for DocumentListLoader       #
##########################################


# --- Inheritance ---


def test_document_list_loader_is_base_loader() -> None:
    assert isinstance(DocumentListLoader(_make_docs()), BaseLoader)


# --- lazy_load ---


def test_document_list_loader_lazy_load_returns_iterator() -> None:
    loader = DocumentListLoader(_make_docs())
    assert hasattr(loader.lazy_load(), "__iter__")


def test_document_list_loader_lazy_load_returns_all_documents() -> None:
    docs = _make_docs()
    assert list(DocumentListLoader(docs).lazy_load()) == docs


def test_document_list_loader_lazy_load_returns_same_instances() -> None:
    docs = _make_docs()
    result = list(DocumentListLoader(docs).lazy_load())
    for original, loaded in zip(docs, result):
        assert loaded is original


def test_document_list_loader_lazy_load_empty_sequence() -> None:
    assert list(DocumentListLoader([]).lazy_load()) == []


def test_document_list_loader_lazy_load_single_document() -> None:
    doc = Document(page_content="Hello")
    assert list(DocumentListLoader([doc]).lazy_load()) == [doc]


def test_document_list_loader_lazy_load_preserves_order() -> None:
    docs = [Document(page_content=str(i)) for i in range(5)]
    result = list(DocumentListLoader(docs).lazy_load())
    assert result == docs


def test_document_list_loader_lazy_load_preserves_metadata() -> None:
    docs = [Document(page_content="Hello", metadata={"source": "test.txt"})]
    result = list(DocumentListLoader(docs).lazy_load())
    assert result[0].metadata == {"source": "test.txt"}


# --- load ---


def test_document_list_loader_load_returns_list() -> None:
    assert isinstance(DocumentListLoader(_make_docs()).load(), list)


def test_document_list_loader_load_returns_all_documents() -> None:
    docs = _make_docs()
    assert DocumentListLoader(docs).load() == docs


def test_document_list_loader_load_empty_sequence() -> None:
    assert DocumentListLoader([]).load() == []


def test_document_list_loader_load_single_document() -> None:
    doc = Document(page_content="Hello")
    assert DocumentListLoader([doc]).load() == [doc]


def test_document_list_loader_load_equals_lazy_load() -> None:
    docs = _make_docs()
    loader = DocumentListLoader(docs)
    assert loader.load() == list(loader.lazy_load())
