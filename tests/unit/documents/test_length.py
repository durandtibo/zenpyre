from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents import (
    compute_document_lengths,
    get_document_length,
    get_shortest_document,
)

##########################################
#     Tests for get_document_length     #
##########################################


def test_get_document_length_basic() -> None:
    assert get_document_length(Document(page_content="hello")) == 5


def test_get_document_length_empty() -> None:
    assert get_document_length(Document(page_content="")) == 0


def test_get_document_length_none_content() -> None:
    doc = Document(page_content="x")
    doc.page_content = None
    assert get_document_length(doc) == 0


##############################################
#     Tests for compute_document_lengths     #
##############################################


def test_compute_document_lengths_empty() -> None:
    assert compute_document_lengths([]) == []


def test_compute_document_lengths_basic() -> None:
    docs = [
        Document(id="a", page_content="hello"),
        Document(id="b", page_content="hello world"),
    ]
    assert compute_document_lengths(docs) == [("a", 5), ("b", 11)]


def test_compute_document_lengths_none_content() -> None:
    doc = Document(id="a", page_content="x")
    doc.page_content = None
    assert compute_document_lengths([doc]) == [("a", 0)]


def test_compute_document_lengths_generator() -> None:
    def gen() -> object:
        yield Document(id="a", page_content="ab")
        yield Document(id="b", page_content="")

    assert compute_document_lengths(gen()) == [("a", 2), ("b", 0)]


def test_compute_document_lengths_sort_true() -> None:
    docs = [
        Document(id="a", page_content="hello world"),
        Document(id="b", page_content=""),
        Document(id="c", page_content="hi"),
    ]
    assert compute_document_lengths(docs, sort=True) == [("b", 0), ("c", 2), ("a", 11)]


def test_compute_document_lengths_sort_false_default() -> None:
    docs = [
        Document(id="a", page_content="hello world"),
        Document(id="b", page_content="hi"),
    ]
    assert compute_document_lengths(docs) == [("a", 11), ("b", 2)]


def test_compute_document_lengths_sort_stable_ties() -> None:
    docs = [
        Document(id="a", page_content="ab"),
        Document(id="b", page_content="cd"),
    ]
    assert compute_document_lengths(docs, sort=True) == [("a", 2), ("b", 2)]


############################################
#     Tests for get_shortest_document     #
############################################


def test_get_shortest_document_empty_iterable() -> None:
    assert get_shortest_document([]) is None


def test_get_shortest_document_basic() -> None:
    docs = [
        Document(id="a", page_content="hello world"),
        Document(id="b", page_content="hi"),
        Document(id="c", page_content="hello"),
    ]
    assert get_shortest_document(docs).id == "b"


def test_get_shortest_document_ties_first_occurrence() -> None:
    docs = [
        Document(id="a", page_content="ab"),
        Document(id="b", page_content="cd"),
    ]
    assert get_shortest_document(docs).id == "a"


def test_get_shortest_document_generator() -> None:
    def gen() -> object:
        yield Document(id="a", page_content="hello")
        yield Document(id="b", page_content="hi")

    assert get_shortest_document(gen()).id == "b"


def test_get_shortest_document_ignore_empty_false_default() -> None:
    docs = [
        Document(id="a", page_content="hello"),
        Document(id="b", page_content=""),
        Document(id="c", page_content="  "),
    ]
    assert get_shortest_document(docs).id == "b"


def test_get_shortest_document_ignore_empty_true() -> None:
    docs = [
        Document(id="a", page_content="hello world"),
        Document(id="b", page_content=""),
        Document(id="c", page_content="  "),
        Document(id="d", page_content="hi"),
    ]
    assert get_shortest_document(docs, ignore_empty=True).id == "c"


def test_get_shortest_document_ignore_empty_all_ignored() -> None:
    docs = [
        Document(id="a", page_content=""),
    ]
    assert get_shortest_document(docs, ignore_empty=True) is None


def test_get_shortest_document_ignore_empty_treat_whitespace_as_empty_true() -> None:
    docs = [
        Document(id="a", page_content="hello world"),
        Document(id="b", page_content=""),
        Document(id="c", page_content="  "),
        Document(id="d", page_content="hi"),
    ]
    assert get_shortest_document(docs, ignore_empty=True, treat_whitespace_as_empty=True).id == "d"


def test_get_shortest_document_ignore_empty_treat_whitespace_as_empty_all_ignored() -> None:
    docs = [
        Document(id="a", page_content=""),
        Document(id="b", page_content="   "),
    ]
    assert get_shortest_document(docs, ignore_empty=True, treat_whitespace_as_empty=True) is None


def test_get_shortest_document_treat_whitespace_as_empty_no_effect_without_ignore_empty() -> None:
    docs = [
        Document(id="a", page_content=" "),
        Document(id="b", page_content="hi"),
    ]
    assert get_shortest_document(docs, treat_whitespace_as_empty=True).id == "a"
