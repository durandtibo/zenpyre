from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents import is_document_empty


def test_is_document_empty_true() -> None:
    assert is_document_empty(Document(page_content="")) is True


def test_is_document_empty_false() -> None:
    assert is_document_empty(Document(page_content="hello")) is False


def test_is_document_empty_non_string_content() -> None:
    doc = Document(page_content="x")
    doc.page_content = None  # type: ignore[assignment]
    assert is_document_empty(doc) is True


def test_is_document_empty_whitespace_only_ignored_by_default() -> None:
    assert is_document_empty(Document(page_content="   ")) is False


def test_is_document_empty_whitespace_only_treated_as_empty() -> None:
    assert is_document_empty(Document(page_content="   "), treat_whitespace_as_empty=True) is True
