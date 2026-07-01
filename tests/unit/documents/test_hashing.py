"""Unit tests for hash_document."""

from __future__ import annotations

import re

import pytest
from langchain_core.documents import Document

from zenpyre.documents import hash_document, hash_document_uuid, hash_documents

UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(page_content="Hello", metadata={"source": "a.txt"}),
        Document(page_content="World", metadata={"source": "b.txt"}),
    ]


#################################
#     Tests for hash_document   #
#################################


def test_hash_document_returns_str() -> None:
    assert isinstance(hash_document(Document(page_content="Hello")), str)


def test_hash_document_default_length() -> None:
    assert len(hash_document(Document(page_content="Hello"))) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min-valid"),
        pytest.param(32, id="middle"),
        pytest.param(64, id="default"),
        pytest.param(128, id="max-valid"),
    ],
)
def test_hash_document_length(length: int) -> None:
    assert len(hash_document(Document(page_content="Hello"), length=length)) == length


def test_hash_document_same_document_same_hash() -> None:
    doc = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assert hash_document(doc) == hash_document(doc)


def test_hash_document_equal_documents_same_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assert hash_document(doc_a) == hash_document(doc_b)


def test_hash_document_different_content_different_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="World", metadata={"source": "cats.txt"})
    assert hash_document(doc_a) != hash_document(doc_b)


def test_hash_document_different_metadata_different_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="Hello", metadata={"source": "dogs.txt"})
    assert hash_document(doc_a) != hash_document(doc_b)


def test_hash_document_metadata_order_independent() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt", "category": "Science"})
    doc_b = Document(page_content="Hello", metadata={"category": "Science", "source": "cats.txt"})
    assert hash_document(doc_a) == hash_document(doc_b)


def test_hash_document_empty_metadata_same_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={})
    doc_b = Document(page_content="Hello", metadata={})
    assert hash_document(doc_a) == hash_document(doc_b)


def test_hash_document_empty_content_same_hash() -> None:
    doc_a = Document(page_content="")
    doc_b = Document(page_content="")
    assert hash_document(doc_a) == hash_document(doc_b)


def test_hash_document_nested_metadata_same_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"info": {"year": 2024, "topic": "cats"}})
    doc_b = Document(page_content="Hello", metadata={"info": {"topic": "cats", "year": 2024}})
    assert hash_document(doc_a) == hash_document(doc_b)


#####################################
#     Tests for hash_document_uuid  #
#####################################


def test_hash_document_uuid_returns_str() -> None:
    assert isinstance(hash_document_uuid(Document(page_content="Hello")), str)


def test_hash_document_uuid_is_valid_uuid_v5() -> None:
    result = hash_document_uuid(Document(page_content="Hello"))
    assert UUID_PATTERN.match(result)


def test_hash_document_uuid_same_document_same_hash() -> None:
    doc = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assert hash_document_uuid(doc) == hash_document_uuid(doc)


def test_hash_document_uuid_equal_documents_same_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="Hello", metadata={"source": "cats.txt"})
    assert hash_document_uuid(doc_a) == hash_document_uuid(doc_b)


def test_hash_document_uuid_different_content_different_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="World", metadata={"source": "cats.txt"})
    assert hash_document_uuid(doc_a) != hash_document_uuid(doc_b)


def test_hash_document_uuid_different_metadata_different_hash() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt"})
    doc_b = Document(page_content="Hello", metadata={"source": "dogs.txt"})
    assert hash_document_uuid(doc_a) != hash_document_uuid(doc_b)


def test_hash_document_uuid_metadata_order_independent() -> None:
    doc_a = Document(page_content="Hello", metadata={"source": "cats.txt", "category": "Science"})
    doc_b = Document(page_content="Hello", metadata={"category": "Science", "source": "cats.txt"})
    assert hash_document_uuid(doc_a) == hash_document_uuid(doc_b)


def test_hash_document_uuid_empty_metadata() -> None:
    doc_a = Document(page_content="Hello", metadata={})
    doc_b = Document(page_content="Hello", metadata={})
    assert hash_document_uuid(doc_a) == hash_document_uuid(doc_b)


def test_hash_document_uuid_empty_content() -> None:
    doc_a = Document(page_content="")
    doc_b = Document(page_content="")
    assert hash_document_uuid(doc_a) == hash_document_uuid(doc_b)


def test_hash_document_uuid_nested_metadata_order_independent() -> None:
    doc_a = Document(page_content="Hello", metadata={"info": {"year": 2024, "topic": "cats"}})
    doc_b = Document(page_content="Hello", metadata={"info": {"topic": "cats", "year": 2024}})
    assert hash_document_uuid(doc_a) == hash_document_uuid(doc_b)


##################################
#     Tests for hash_documents   #
##################################


def test_hash_documents_returns_str(docs: list[Document]) -> None:
    assert isinstance(hash_documents(docs), str)


def test_hash_documents_default_length(docs: list[Document]) -> None:
    assert len(hash_documents(docs)) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min-valid"),
        pytest.param(32, id="middle"),
        pytest.param(64, id="default"),
        pytest.param(128, id="max-valid"),
    ],
)
def test_hash_documents_custom_length(docs: list[Document], length: int) -> None:
    assert len(hash_documents(docs, length=length)) == length


def test_hash_documents_same_list_same_hash(docs: list[Document]) -> None:
    assert hash_documents(docs) == hash_documents(docs)


def test_hash_documents_equal_lists_same_hash(docs: list[Document]) -> None:
    docs_copy = [
        Document(page_content="Hello", metadata={"source": "a.txt"}),
        Document(page_content="World", metadata={"source": "b.txt"}),
    ]
    assert hash_documents(docs) == hash_documents(docs_copy)


def test_hash_documents_different_content_different_hash(docs: list[Document]) -> None:
    other = [
        Document(page_content="Different", metadata={"source": "a.txt"}),
        Document(page_content="World", metadata={"source": "b.txt"}),
    ]
    assert hash_documents(docs) != hash_documents(other)


def test_hash_documents_different_metadata_different_hash(docs: list[Document]) -> None:
    other = [
        Document(page_content="Hello", metadata={"source": "other.txt"}),
        Document(page_content="World", metadata={"source": "b.txt"}),
    ]
    assert hash_documents(docs) != hash_documents(other)


def test_hash_documents_order_matters(docs: list[Document]) -> None:
    reversed_docs = list(reversed(docs))
    assert hash_documents(docs) != hash_documents(reversed_docs)


def test_hash_documents_different_length_different_hash(docs: list[Document]) -> None:
    other = docs[:1]
    assert hash_documents(docs) != hash_documents(other)


def test_hash_documents_empty_list_returns_str() -> None:
    assert isinstance(hash_documents([]), str)


def test_hash_documents_empty_list_returns_correct_length() -> None:
    assert len(hash_documents([])) == 64


def test_hash_documents_empty_list_is_stable() -> None:
    assert hash_documents([]) == hash_documents([])


def test_hash_documents_single_doc_differs_from_two_docs(docs: list[Document]) -> None:
    assert hash_documents(docs[:1]) != hash_documents(docs)


def test_hash_documents_metadata_order_independent() -> None:
    docs_reordered = [
        Document(page_content="Hello", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="World", metadata={"source": "b.txt", "page": 2}),
    ]
    docs_same_reversed_meta = [
        Document(page_content="Hello", metadata={"page": 1, "source": "a.txt"}),
        Document(page_content="World", metadata={"page": 2, "source": "b.txt"}),
    ]
    assert hash_documents(docs_reordered) == hash_documents(docs_same_reversed_meta)
