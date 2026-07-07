from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import assign_ids, copy_ids_to_metadata


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="Hello"),
        Document(page_content="World"),
        Document(page_content="Cats are great."),
    ]


@pytest.fixture
def docs_with_id() -> list[Document]:
    return [
        Document(page_content="Hello", id="id-1"),
        Document(page_content="World", id="id-2"),
        Document(page_content="Cats are great.", id="id-3"),
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


##########################################
#   Tests for copy_ids_to_metadata       #
##########################################


# --- Return value ---


def test_copy_ids_to_metadata_returns_list(docs_with_id: list[Document]) -> None:
    assert isinstance(copy_ids_to_metadata(docs_with_id), list)


def test_copy_ids_to_metadata_returns_same_list(docs_with_id: list[Document]) -> None:
    assert copy_ids_to_metadata(docs_with_id) is docs_with_id


# --- Default behaviour (metadata_key="source_id") ---


def test_copy_ids_to_metadata_sets_default_key(docs_with_id: list[Document]) -> None:
    copy_ids_to_metadata(docs_with_id)
    assert all(doc.metadata["source_id"] == doc.id for doc in docs_with_id)


def test_copy_ids_to_metadata_preserves_other_metadata() -> None:
    doc = Document(page_content="Hello", id="id-1", metadata={"source": "cats.txt"})
    copy_ids_to_metadata([doc])
    assert doc.metadata["source"] == "cats.txt"
    assert doc.metadata["source_id"] == "id-1"


def test_copy_ids_to_metadata_does_not_change_id(docs_with_id: list[Document]) -> None:
    original_ids = [doc.id for doc in docs_with_id]
    copy_ids_to_metadata(docs_with_id)
    assert [doc.id for doc in docs_with_id] == original_ids


# --- Custom metadata_key ---


def test_copy_ids_to_metadata_sets_custom_key() -> None:
    doc = Document(page_content="Hello", id="id-1")
    copy_ids_to_metadata([doc], metadata_key="parent_id")
    assert doc.metadata["parent_id"] == "id-1"


def test_copy_ids_to_metadata_custom_key_does_not_set_default_key() -> None:
    doc = Document(page_content="Hello", id="id-1")
    copy_ids_to_metadata([doc], metadata_key="parent_id")
    assert "source_id" not in doc.metadata


# --- Documents without an id ---


def test_copy_ids_to_metadata_skips_docs_without_id() -> None:
    doc = Document(page_content="Hello")
    copy_ids_to_metadata([doc])
    assert "source_id" not in doc.metadata


def test_copy_ids_to_metadata_only_sets_key_on_docs_with_id() -> None:
    docs = [
        Document(page_content="Hello", id="id-1"),
        Document(page_content="World"),
    ]
    copy_ids_to_metadata(docs)
    assert docs[0].metadata["source_id"] == "id-1"
    assert "source_id" not in docs[1].metadata


# --- Overwriting existing metadata ---


def test_copy_ids_to_metadata_overwrites_existing_key() -> None:
    doc = Document(page_content="Hello", id="id-1", metadata={"source_id": "stale"})
    copy_ids_to_metadata([doc])
    assert doc.metadata["source_id"] == "id-1"


# --- Edge cases ---


def test_copy_ids_to_metadata_empty_list() -> None:
    assert copy_ids_to_metadata([]) == []


def test_copy_ids_to_metadata_single_doc_no_id() -> None:
    doc = Document(page_content="Hello")
    copy_ids_to_metadata([doc])
    assert doc.metadata == {}


def test_copy_ids_to_metadata_single_doc_with_id() -> None:
    doc = Document(page_content="Hello", id="my-id")
    copy_ids_to_metadata([doc])
    assert doc.metadata["source_id"] == "my-id"
