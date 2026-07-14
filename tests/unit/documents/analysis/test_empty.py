from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents.analysis import find_empty_document_ids, find_empty_documents


def test_find_empty_documents_no_empty() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="b", page_content="world", metadata={"source": "x"}),
    ]
    assert find_empty_documents(docs) == []


def test_find_empty_documents_empty_string() -> None:
    empty = Document(id="empty", page_content="", metadata={"source": "x"})
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        empty,
    ]
    assert find_empty_documents(docs) == [empty]


def test_find_empty_documents_preserves_order() -> None:
    empty1 = Document(id="e1", page_content="", metadata={})
    empty2 = Document(id="e2", page_content="", metadata={})
    docs = [
        empty1,
        Document(id="a", page_content="hello", metadata={}),
        empty2,
    ]
    assert find_empty_documents(docs) == [empty1, empty2]


def test_find_empty_documents_non_string_content() -> None:
    empty = Document(id="none", page_content="hello", metadata={})
    empty.page_content = None  # type: ignore[assignment]
    docs = [Document(id="a", page_content="hello", metadata={}), empty]
    assert find_empty_documents(docs) == [empty]


def test_find_empty_documents_whitespace_only_ignored_by_default() -> None:
    docs = [Document(id="ws", page_content="   \t\n", metadata={})]
    assert find_empty_documents(docs) == []


def test_find_empty_documents_whitespace_only_treated_as_empty() -> None:
    ws = Document(id="ws", page_content="   \t\n", metadata={})
    docs = [Document(id="a", page_content="hello", metadata={}), ws]
    assert find_empty_documents(docs, treat_whitespace_as_empty=True) == [ws]


def test_find_empty_documents_empty_input() -> None:
    assert find_empty_documents([]) == []


def test_find_empty_documents_generator_input() -> None:
    empty = Document(id="empty", page_content="", metadata={})

    def gen() -> object:
        yield Document(id="a", page_content="hello", metadata={})
        yield empty

    assert find_empty_documents(gen()) == [empty]


def test_find_empty_document_ids_no_empty() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="b", page_content="world", metadata={"source": "x"}),
    ]
    assert find_empty_document_ids(docs) == []


def test_find_empty_document_ids_empty_string() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="empty", page_content="", metadata={"source": "x"}),
    ]
    assert find_empty_document_ids(docs) == ["empty"]


def test_find_empty_document_ids_preserves_order() -> None:
    docs = [
        Document(id="e1", page_content="", metadata={}),
        Document(id="a", page_content="hello", metadata={}),
        Document(id="e2", page_content="", metadata={}),
    ]
    assert find_empty_document_ids(docs) == ["e1", "e2"]


def test_find_empty_document_ids_non_string_content() -> None:
    none_doc = Document(id="none", page_content="hello", metadata={})
    none_doc.page_content = None  # type: ignore[assignment]
    docs = [Document(id="a", page_content="hello", metadata={}), none_doc]
    assert find_empty_document_ids(docs) == ["none"]


def test_find_empty_document_ids_whitespace_only_ignored_by_default() -> None:
    docs = [Document(id="ws", page_content="   \t\n", metadata={})]
    assert find_empty_document_ids(docs) == []


def test_find_empty_document_ids_whitespace_only_treated_as_empty() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={}),
        Document(id="ws", page_content="   \t\n", metadata={}),
    ]
    assert find_empty_document_ids(docs, treat_whitespace_as_empty=True) == ["ws"]


def test_find_empty_document_ids_empty_input() -> None:
    assert find_empty_document_ids([]) == []


def test_find_empty_document_ids_generator_input() -> None:
    def gen() -> object:
        yield Document(id="a", page_content="hello", metadata={})
        yield Document(id="empty", page_content="", metadata={})

    assert find_empty_document_ids(gen()) == ["empty"]
