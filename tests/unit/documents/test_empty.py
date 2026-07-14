from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents import is_document_empty, is_document_whitespace_only

#######################################
#     Tests for is_document_empty     #
#######################################


def test_is_document_empty_true() -> None:
    assert is_document_empty(Document(page_content=""))


def test_is_document_empty_false() -> None:
    assert not is_document_empty(Document(page_content="hello"))


def test_is_document_empty_non_string_content() -> None:
    doc = Document(page_content="x")
    doc.page_content = None
    assert is_document_empty(doc)


def test_is_document_empty_whitespace_only_ignored_by_default() -> None:
    assert not is_document_empty(Document(page_content="   "))


def test_is_document_empty_whitespace_only_treated_as_empty() -> None:
    assert is_document_empty(Document(page_content="   "), treat_whitespace_as_empty=True)


#################################################
#     Tests for is_document_whitespace_only     #
#################################################


def test_is_document_whitespace_only_true() -> None:
    assert is_document_whitespace_only(Document(page_content="  \n\t"))


def test_is_document_whitespace_only_false_non_whitespace() -> None:
    assert not is_document_whitespace_only(Document(page_content="hello"))


def test_is_document_whitespace_only_false_empty_string() -> None:
    assert not is_document_whitespace_only(Document(page_content=""))


def test_is_document_whitespace_only_non_string_content() -> None:
    doc = Document(page_content="x")
    doc.page_content = None
    assert not is_document_whitespace_only(doc)
