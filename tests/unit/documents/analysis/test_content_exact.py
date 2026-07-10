from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents.analysis import (
    ExactDocContentStats,
    compute_doc_content_stats_exact,
)


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(id="short", page_content="a" * 10, metadata={"source": "x"}),
        Document(id="mid", page_content="b" * 20, metadata={"source": "x"}),
        Document(id="long", page_content="c" * 30, metadata={"source": "x"}),
    ]


@pytest.fixture
def duplicate_docs() -> list[Document]:
    return [
        Document(id="a", page_content="same content", metadata={"source": "x"}),
        Document(id="b", page_content="same content", metadata={"source": "x"}),
        Document(id="c", page_content="different", metadata={"source": "x"}),
    ]


@pytest.fixture
def messy_docs() -> list[Document]:
    return [
        Document(id="ok", page_content="real content", metadata={"source": "x"}),
        Document(id="empty", page_content="", metadata={"source": "x"}),
        Document(id="whitespace", page_content="   \t\n", metadata={"source": "x"}),
        Document(id=None, page_content="no id", metadata={"source": "x"}),
        Document(id="no_meta", page_content="content", metadata={}),
    ]


#####################################################
#     Tests for compute_doc_content_stats_exact     #
#####################################################


# --- Return type and non-mutation ---


def test_compute_doc_content_stats_exact_returns_dict(docs: list[Document]) -> None:
    assert isinstance(compute_doc_content_stats_exact(docs), dict)


def test_compute_doc_content_stats_exact_does_not_mutate_input(docs: list[Document]) -> None:
    original_len = len(docs)
    compute_doc_content_stats_exact(docs)
    assert len(docs) == original_len


# --- Empty input ---


def test_compute_doc_content_stats_exact_empty_list() -> None:
    assert compute_doc_content_stats_exact([]) == {
        "count": 0,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": None,
        "max_chars": None,
        "min_doc_id": None,
        "max_doc_id": None,
        "p50_chars": None,
        "p90_chars": None,
        "p99_chars": None,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_empty_generator() -> None:
    def gen() -> object:
        return
        yield  # pragma: no cover

    assert compute_doc_content_stats_exact(gen()) == {
        "count": 0,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": None,
        "max_chars": None,
        "min_doc_id": None,
        "max_doc_id": None,
        "p50_chars": None,
        "p90_chars": None,
        "p99_chars": None,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Core analysis ---


def test_compute_doc_content_stats_exact_core_stats(docs: list[Document]) -> None:
    assert compute_doc_content_stats_exact(docs) == {
        "count": 3,
        "total_chars": 60,
        "avg_chars": 20,
        "std_dev_chars": pytest.approx(8.16496580927726),
        "min_chars": 10,
        "max_chars": 30,
        "min_doc_id": "short",
        "max_doc_id": "long",
        "p50_chars": 20,
        "p90_chars": 30,
        "p99_chars": 30,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_single_doc() -> None:
    docs = [Document(id="1", page_content="hello", metadata={"source": "x"})]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 1,
        "total_chars": 5,
        "avg_chars": 5,
        "std_dev_chars": 0,
        "min_chars": 5,
        "max_chars": 5,
        "min_doc_id": "1",
        "max_doc_id": "1",
        "p50_chars": 5,
        "p90_chars": 5,
        "p99_chars": 5,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_std_dev_known_value() -> None:
    lengths = [2, 4, 4, 4, 5, 5, 7, 9]
    docs = [
        Document(id=str(i), page_content="x" * length, metadata={"source": "x"})
        for i, length in enumerate(lengths)
    ]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 8,
        "total_chars": 40,
        "avg_chars": 5,
        "std_dev_chars": pytest.approx(2.0),
        "min_chars": 2,
        "max_chars": 9,
        "min_doc_id": "0",
        "max_doc_id": "7",
        "p50_chars": 5,
        "p90_chars": 7,
        "p99_chars": 9,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 3,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_percentiles_one_to_ten() -> None:
    docs = [
        Document(id=str(n), page_content="x" * n, metadata={"source": "x"}) for n in range(1, 11)
    ]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 10,
        "total_chars": 55,
        "avg_chars": 5.5,
        "std_dev_chars": pytest.approx(2.8722813232690143),
        "min_chars": 1,
        "max_chars": 10,
        "min_doc_id": "1",
        "max_doc_id": "10",
        "p50_chars": 5,
        "p90_chars": 9,
        "p99_chars": 10,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Tie-breaking ---


def test_compute_doc_content_stats_exact_min_max_id_first_on_tie() -> None:
    docs = [
        Document(id="first_min", page_content="aa", metadata={"source": "x"}),
        Document(id="second_min", page_content="aa", metadata={"source": "x"}),
        Document(id="first_max", page_content="zz", metadata={"source": "x"}),
        Document(id="second_max", page_content="zz", metadata={"source": "x"}),
    ]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 4,
        "total_chars": 8,
        "avg_chars": 2,
        "std_dev_chars": 0,
        "min_chars": 2,
        "max_chars": 2,
        "min_doc_id": "first_min",
        "max_doc_id": "first_min",
        "p50_chars": 2,
        "p90_chars": 2,
        "p99_chars": 2,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 2,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Data quality flags ---


def test_compute_doc_content_stats_exact_messy_docs(messy_docs: list[Document]) -> None:
    assert compute_doc_content_stats_exact(messy_docs) == {
        "count": 5,
        "total_chars": 29,
        "avg_chars": 5.8,
        "std_dev_chars": pytest.approx(3.8678159211627436),
        "min_chars": 0,
        "max_chars": 12,
        "min_doc_id": "empty",
        "max_doc_id": "ok",
        "p50_chars": 5,
        "p90_chars": 12,
        "p99_chars": 12,
        "empty_count": 1,
        "whitespace_only_count": 1,
        "none_or_non_str_content_count": 0,
        "none_id_count": 1,
        "missing_metadata_count": 1,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_none_content_treated_as_empty() -> None:
    doc = Document(id="1", page_content="placeholder", metadata={"source": "x"})
    doc.page_content = None
    assert compute_doc_content_stats_exact([doc]) == {
        "count": 1,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": 0,
        "max_chars": 0,
        "min_doc_id": "1",
        "max_doc_id": "1",
        "p50_chars": 0,
        "p90_chars": 0,
        "p99_chars": 0,
        "empty_count": 1,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 1,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_non_string_content() -> None:
    doc = Document(id="1", page_content="placeholder", metadata={"source": "x"})
    doc.page_content = 12345
    assert compute_doc_content_stats_exact([doc]) == {
        "count": 1,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": 0,
        "max_chars": 0,
        "min_doc_id": "1",
        "max_doc_id": "1",
        "p50_chars": 0,
        "p90_chars": 0,
        "p99_chars": 0,
        "empty_count": 1,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 1,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Duplicate detection ---


def test_compute_doc_content_stats_exact_duplicates(duplicate_docs: list[Document]) -> None:
    assert compute_doc_content_stats_exact(duplicate_docs) == {
        "count": 3,
        "total_chars": 33,
        "avg_chars": 11,
        "std_dev_chars": pytest.approx(1.4142135623730951),
        "min_chars": 9,
        "max_chars": 12,
        "min_doc_id": "c",
        "max_doc_id": "a",
        "p50_chars": 12,
        "p90_chars": 12,
        "p99_chars": 12,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 1,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_empty_strings_are_duplicates_of_each_other() -> None:
    docs = [Document(id=str(i), page_content="", metadata={"source": "x"}) for i in range(3)]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 3,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": 0,
        "max_chars": 0,
        "min_doc_id": "0",
        "max_doc_id": "0",
        "p50_chars": 0,
        "p90_chars": 0,
        "p99_chars": 0,
        "empty_count": 3,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 2,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_many_duplicates_same_content() -> None:
    docs = [Document(id=str(i), page_content="repeat", metadata={"source": "x"}) for i in range(5)]
    assert compute_doc_content_stats_exact(docs) == {
        "count": 5,
        "total_chars": 30,
        "avg_chars": 6,
        "std_dev_chars": 0,
        "min_chars": 6,
        "max_chars": 6,
        "min_doc_id": "0",
        "max_doc_id": "0",
        "p50_chars": 6,
        "p90_chars": 6,
        "p99_chars": 6,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 4,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Iterator / generator support ---


def test_compute_doc_content_stats_exact_generator_input() -> None:
    def gen() -> object:
        yield Document(id="1", page_content="aaaa", metadata={"source": "x"})
        yield Document(id="2", page_content="bb", metadata={"source": "x"})

    assert compute_doc_content_stats_exact(gen()) == {
        "count": 2,
        "total_chars": 6,
        "avg_chars": 3,
        "std_dev_chars": pytest.approx(1.0),
        "min_chars": 2,
        "max_chars": 4,
        "min_doc_id": "2",
        "max_doc_id": "1",
        "p50_chars": 2,
        "p90_chars": 4,
        "p99_chars": 4,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_compute_doc_content_stats_exact_generator_consumed_only_once() -> None:
    def gen() -> object:
        yield Document(id="1", page_content="x", metadata={"source": "x"})

    g = gen()
    compute_doc_content_stats_exact(g)
    assert list(g) == []


def test_compute_doc_content_stats_exact_map_iterator() -> None:
    raw_texts = ["one", "two", "three"]
    docs_iter = (Document(id=t, page_content=t, metadata={"source": "x"}) for t in raw_texts)
    assert compute_doc_content_stats_exact(docs_iter) == {
        "count": 3,
        "total_chars": 11,
        "avg_chars": pytest.approx(3.6666666666666665),
        "std_dev_chars": pytest.approx(0.9428090415820634),
        "min_chars": 3,
        "max_chars": 5,
        "min_doc_id": "one",
        "max_doc_id": "three",
        "p50_chars": 3,
        "p90_chars": 5,
        "p99_chars": 5,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


# --- Larger scale sanity check ---


def test_compute_doc_content_stats_exact_thousand_docs() -> None:
    docs = [
        Document(id=str(i), page_content="x" * (i % 50 + 1), metadata={"source": "x"})
        for i in range(1000)
    ]
    result = compute_doc_content_stats_exact(docs)
    assert result["count"] == 1000
    assert result["min_chars"] == 1
    assert result["max_chars"] == 50
    assert result["total_chars"] == sum((i % 50 + 1) for i in range(1000))


##########################################
#     Tests for ExactDocContentStats     #
##########################################


def test_exact_doc_content_stats_starts_at_zero() -> None:
    assert ExactDocContentStats().to_dict() == compute_doc_content_stats_exact([])


def test_exact_doc_content_stats_manual_update_calls() -> None:
    stats = ExactDocContentStats()
    stats.update(Document(id="1", page_content="aa", metadata={"source": "x"}))
    stats.update(Document(id="2", page_content="bbbb", metadata={"source": "x"}))
    assert stats.to_dict() == {
        "count": 2,
        "total_chars": 6,
        "avg_chars": 3,
        "std_dev_chars": pytest.approx(1.0),
        "min_chars": 2,
        "max_chars": 4,
        "min_doc_id": "1",
        "max_doc_id": "2",
        "p50_chars": 2,
        "p90_chars": 4,
        "p99_chars": 4,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


def test_exact_doc_content_stats_to_dict_is_idempotent() -> None:
    stats = ExactDocContentStats()
    stats.update(Document(id="1", page_content="abc", metadata={"source": "x"}))
    assert stats.to_dict() == stats.to_dict()
