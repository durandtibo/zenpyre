"""Unit tests for assign_ids."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import assign_ids


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="Hello"),
        Document(page_content="World"),
        Document(page_content="Cats are great."),
    ]


##############################
#     Tests for assign_ids   #
##############################


# --- Return value ---


def test_assign_ids_returns_list(docs: list[Document]) -> None:
    assert isinstance(assign_ids(docs), list)


def test_assign_ids_returns_same_list(docs: list[Document]) -> None:
    assert assign_ids(docs) is docs


# --- Default behaviour (force=False) ---


def test_assign_ids_sets_id_on_docs_without_id(docs: list[Document]) -> None:
    assign_ids(docs)
    assert all(doc.id is not None for doc in docs)


def test_assign_ids_preserves_existing_id() -> None:
    doc = Document(page_content="Hello", id="existing-id")
    assign_ids([doc])
    assert doc.id == "existing-id"


def test_assign_ids_only_sets_missing_ids() -> None:
    docs = [
        Document(page_content="Hello", id="existing-id"),
        Document(page_content="World"),
    ]
    assign_ids(docs)
    assert docs[0].id == "existing-id"
    assert docs[1].id is not None


def test_assign_ids_assigned_id_is_str(docs: list[Document]) -> None:
    assign_ids(docs)
    assert all(isinstance(doc.id, str) for doc in docs)


def test_assign_ids_same_doc_same_id() -> None:
    doc = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assign_ids([doc])
    first_id = doc.id
    doc.id = None
    assign_ids([doc])
    assert doc.id == first_id


def test_assign_ids_different_docs_different_ids(docs: list[Document]) -> None:
    assign_ids(docs)
    ids = [doc.id for doc in docs]
    assert len(set(ids)) == len(ids)


# --- force=True ---


def test_assign_ids_force_overwrites_existing_id() -> None:
    doc = Document(page_content="Hello", id="existing-id")
    assign_ids([doc], force=True)
    assert doc.id != "existing-id"


def test_assign_ids_force_sets_id_on_all_docs(docs: list[Document]) -> None:
    assign_ids(docs, force=True)
    assert all(doc.id is not None for doc in docs)


def test_assign_ids_force_produces_deterministic_ids() -> None:
    doc = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assign_ids([doc], force=True)
    first_id = doc.id
    assign_ids([doc], force=True)
    assert doc.id == first_id


# --- Edge cases ---


def test_assign_ids_empty_list() -> None:
    assert assign_ids([]) == []


def test_assign_ids_single_doc_no_id() -> None:
    doc = Document(page_content="Hello")
    assign_ids([doc])
    assert doc.id is not None


def test_assign_ids_single_doc_with_id() -> None:
    doc = Document(page_content="Hello", id="my-id")
    assign_ids([doc])
    assert doc.id == "my-id"
