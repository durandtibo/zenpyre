from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import DocumentConsistencyError, check_document_consistency

###############################################
#     Tests for check_document_consistency     #
###############################################


def test_check_document_consistency_empty_list_returns_true() -> None:
    assert check_document_consistency([]) is True


def test_check_document_consistency_no_shared_ids_returns_true() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="2", page_content="B", metadata={"source": "b.txt"}),
    ]
    assert check_document_consistency(docs) is True


def test_check_document_consistency_identical_documents_same_id_returns_true() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
    ]
    assert check_document_consistency(docs) is True


def test_check_document_consistency_none_ids_are_ignored() -> None:
    docs = [
        Document(page_content="A", metadata={"source": "a.txt"}),
        Document(page_content="B", metadata={"source": "b.txt"}),
    ]
    assert check_document_consistency(docs) is True


def test_check_document_consistency_metadata_key_order_does_not_affect_equality() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"a": 1, "b": 2}),
        Document(id="1", page_content="A", metadata={"b": 2, "a": 1}),
    ]
    assert check_document_consistency(docs) is True


def test_check_document_consistency_different_page_content_returns_false() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="B", metadata={"source": "a.txt"}),
    ]
    assert check_document_consistency(docs) is False


def test_check_document_consistency_different_metadata_returns_false() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "b.txt"}),
    ]
    assert check_document_consistency(docs) is False


def test_check_document_consistency_warns_with_document_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="B", metadata={"source": "a.txt"}),
    ]
    with caplog.at_level("WARNING"):
        check_document_consistency(docs)
    assert "1" in caplog.text


def test_check_document_consistency_continues_after_warning() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={}),
        Document(id="1", page_content="B", metadata={}),
        Document(id="2", page_content="C", metadata={}),
        Document(id="2", page_content="D", metadata={}),
    ]
    assert check_document_consistency(docs) is False


def test_check_document_consistency_raise_error_true_raises_on_first_inconsistency() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="B", metadata={"source": "a.txt"}),
    ]
    with pytest.raises(DocumentConsistencyError, match="1"):
        check_document_consistency(docs, raise_error=True)


def test_check_document_consistency_raise_error_true_stops_at_first_inconsistency() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={}),
        Document(id="1", page_content="B", metadata={}),
        Document(id="2", page_content="C", metadata={}),
        Document(id="2", page_content="D", metadata={}),
    ]
    with pytest.raises(DocumentConsistencyError, match="1"):
        check_document_consistency(docs, raise_error=True)


def test_check_document_consistency_raise_error_true_consistent_returns_true() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
    ]
    assert check_document_consistency(docs, raise_error=True) is True
