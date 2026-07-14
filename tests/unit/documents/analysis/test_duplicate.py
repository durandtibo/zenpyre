from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents.analysis import find_duplicate_content_document_ids

#########################################################
#     Tests for find_duplicate_content_document_ids     #
#########################################################


def test_find_duplicate_content_document_ids_no_duplicates() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="b", page_content="world", metadata={"source": "x"}),
    ]
    assert find_duplicate_content_document_ids(docs) == []


def test_find_duplicate_content_document_ids_one_group() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="b", page_content="hello", metadata={"source": "x"}),
    ]
    assert find_duplicate_content_document_ids(docs) == [["a", "b"]]


def test_find_duplicate_content_document_ids_group_includes_all_occurrences() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={}),
        Document(id="b", page_content="hello", metadata={}),
        Document(id="c", page_content="hello", metadata={}),
    ]
    assert find_duplicate_content_document_ids(docs) == [["a", "b", "c"]]


def test_find_duplicate_content_document_ids_multiple_groups_in_order() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={}),
        Document(id="b", page_content="world", metadata={}),
        Document(id="c", page_content="hello", metadata={}),
        Document(id="d", page_content="world", metadata={}),
    ]
    assert find_duplicate_content_document_ids(docs) == [["a", "c"], ["b", "d"]]


def test_find_duplicate_content_document_ids_empty_input() -> None:
    assert find_duplicate_content_document_ids([]) == []


def test_find_duplicate_content_document_ids_single_document() -> None:
    docs = [Document(id="a", page_content="hello", metadata={})]
    assert find_duplicate_content_document_ids(docs) == []


def test_find_duplicate_content_document_ids_metadata_ignored() -> None:
    docs = [
        Document(id="a", page_content="hello", metadata={"source": "x"}),
        Document(id="b", page_content="hello", metadata={"source": "y"}),
    ]
    assert find_duplicate_content_document_ids(docs) == [["a", "b"]]


def test_find_duplicate_content_document_ids_none_id() -> None:
    docs = [
        Document(id=None, page_content="hello", metadata={}),
        Document(id=None, page_content="hello", metadata={}),
    ]
    assert find_duplicate_content_document_ids(docs) == [[None, None]]


def test_find_duplicate_content_document_ids_generator_input() -> None:
    def gen() -> object:
        yield Document(id="a", page_content="hello", metadata={})
        yield Document(id="b", page_content="hello", metadata={})

    assert find_duplicate_content_document_ids(gen()) == [["a", "b"]]
