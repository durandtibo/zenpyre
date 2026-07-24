from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from zenpyre.documents import deduplicate_documents

if TYPE_CHECKING:
    import pytest

if TYPE_CHECKING:
    import pytest

###########################################
#     Tests for deduplicate_documents     #
###########################################


def test_deduplicate_documents_empty_list_returns_empty_list() -> None:
    assert deduplicate_documents([]) == []


def test_deduplicate_documents_no_duplicates_keeps_all_in_order() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="2", page_content="B", metadata={"source": "b.txt"}),
        Document(id="3", page_content="C", metadata={"source": "c.txt"}),
    ]
    result = deduplicate_documents(docs)
    assert result == docs


def test_deduplicate_documents_exact_duplicate_keeps_first_occurrence() -> None:
    first = Document(id="1", page_content="A", metadata={"source": "a.txt"})
    duplicate = Document(id="1", page_content="A", metadata={"source": "a.txt"})
    result = deduplicate_documents([first, duplicate])
    assert result == [first]


def test_deduplicate_documents_different_id_not_duplicate() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="2", page_content="A", metadata={"source": "a.txt"}),
    ]
    assert deduplicate_documents(docs) == docs


def test_deduplicate_documents_different_page_content_not_duplicate() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="B", metadata={"source": "a.txt"}),
    ]
    assert deduplicate_documents(docs) == docs


def test_deduplicate_documents_different_metadata_not_duplicate() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "b.txt"}),
    ]
    assert deduplicate_documents(docs) == docs


def test_deduplicate_documents_none_ids_are_duplicates_of_each_other() -> None:
    first = Document(page_content="A", metadata={"source": "a.txt"})
    duplicate = Document(page_content="A", metadata={"source": "a.txt"})
    result = deduplicate_documents([first, duplicate])
    assert result == [first]


def test_deduplicate_documents_metadata_key_order_does_not_affect_equality() -> None:
    first = Document(id="1", page_content="A", metadata={"a": 1, "b": 2})
    duplicate = Document(id="1", page_content="A", metadata={"b": 2, "a": 1})
    result = deduplicate_documents([first, duplicate])
    assert result == [first]


def test_deduplicate_documents_preserves_order_with_scattered_duplicates() -> None:
    a = Document(id="1", page_content="A", metadata={})
    b = Document(id="2", page_content="B", metadata={})
    c = Document(id="3", page_content="C", metadata={})
    a_dup = Document(id="1", page_content="A", metadata={})
    b_dup = Document(id="2", page_content="B", metadata={})
    result = deduplicate_documents([a, b, a_dup, c, b_dup])
    assert result == [a, b, c]


def test_deduplicate_documents_does_not_mutate_input_list() -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
    ]
    original = list(docs)
    deduplicate_documents(docs)
    assert docs == original


def test_deduplicate_documents_log_true_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="2", page_content="B", metadata={"source": "b.txt"}),
    ]
    with caplog.at_level(logging.INFO):
        result = deduplicate_documents(docs, log=True)
    assert result == [docs[0], docs[2]]
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "3" in message
    assert "2" in message
    assert "1" in message


def test_deduplicate_documents_log_false_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
    ]
    with caplog.at_level(logging.INFO):
        deduplicate_documents(docs)
    assert len(caplog.records) == 0


def test_deduplicate_documents_log_true_no_duplicates_logs_zero_removed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    docs = [
        Document(id="1", page_content="A", metadata={"source": "a.txt"}),
        Document(id="2", page_content="B", metadata={"source": "b.txt"}),
    ]
    with caplog.at_level(logging.INFO):
        result = deduplicate_documents(docs, log=True)
    assert result == docs
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "2" in message
    assert "0" in message


def test_deduplicate_documents_log_true_empty_list_logs_zeros(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = deduplicate_documents([], log=True)
    assert result == []
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "0" in message
