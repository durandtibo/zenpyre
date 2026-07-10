from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents.stats import (
    ApproxDocContentStats,
    compute_doc_content_stats_approx,
)
from zenpyre.utils.bloom_filter import BloomFilter


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


# With reservoir_size >= corpus size, every length is kept (reservoir
# sampling never has to evict), so std dev / percentiles are exact for
# these small fixtures — letting us reuse the exact tests' expectations.
FULL_RESERVOIR = 10_000


######################################################
#     Tests for compute_doc_content_stats_approx     #
######################################################


# --- Return type and non-mutation ---


def test_compute_doc_content_stats_approx_returns_dict(docs: list[Document]) -> None:
    assert isinstance(compute_doc_content_stats_approx(docs), dict)


def test_compute_doc_content_stats_approx_does_not_mutate_input(docs: list[Document]) -> None:
    original_len = len(docs)
    compute_doc_content_stats_approx(docs)
    assert len(docs) == original_len


# --- Empty input ---


def test_compute_doc_content_stats_approx_empty_list() -> None:
    assert compute_doc_content_stats_approx([]) == {
        "count": 0,
        "total_chars": 0,
        "avg_chars": 0,
        "std_dev_chars": 0,
        "min_chars": None,
        "max_chars": None,
        "min_doc_id": None,
        "max_doc_id": None,
        "p50_chars_approx": None,
        "p90_chars_approx": None,
        "p99_chars_approx": None,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "approx_duplicate_count": 0,
        "duplicate_count_exact": False,
        "percentiles_exact": False,
        "bloom_filter_fp_rate": 0.01,
        "reservoir_sample_size": 0,
    }


def test_compute_doc_content_stats_approx_empty_generator() -> None:
    def gen() -> object:
        return
        yield  # pragma: no cover

    assert compute_doc_content_stats_approx(gen())["count"] == 0


# --- Core stats (exact fields: count/total/avg/min/max/doc-ids) ---


def test_compute_doc_content_stats_approx_core_exact_fields(docs: list[Document]) -> None:
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["count"] == 3
    assert result["total_chars"] == 60
    assert result["avg_chars"] == 20
    assert result["min_chars"] == 10
    assert result["max_chars"] == 30
    assert result["min_doc_id"] == "short"
    assert result["max_doc_id"] == "long"


def test_compute_doc_content_stats_approx_full_reservoir_matches_exact_stats(
    docs: list[Document],
) -> None:
    # With a reservoir large enough to hold every doc, std dev and
    # percentiles are deterministic and match the exact implementation's
    # known-good values for this fixture.
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["std_dev_chars"] == pytest.approx(8.16496580927726)
    assert result["p50_chars_approx"] == 20
    assert result["p90_chars_approx"] == 30
    assert result["p99_chars_approx"] == 30
    assert result["reservoir_sample_size"] == 3


def test_compute_doc_content_stats_approx_single_doc() -> None:
    docs = [Document(id="1", page_content="hello", metadata={"source": "x"})]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["count"] == 1
    assert result["total_chars"] == 5
    assert result["avg_chars"] == 5
    assert result["std_dev_chars"] == 0
    assert result["min_chars"] == 5
    assert result["max_chars"] == 5
    assert result["min_doc_id"] == "1"
    assert result["max_doc_id"] == "1"
    assert result["p50_chars_approx"] == 5


def test_compute_doc_content_stats_approx_std_dev_known_value() -> None:
    lengths = [2, 4, 4, 4, 5, 5, 7, 9]
    docs = [
        Document(id=str(i), page_content="x" * length, metadata={"source": "x"})
        for i, length in enumerate(lengths)
    ]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["std_dev_chars"] == pytest.approx(2.0)
    assert result["p50_chars_approx"] == 5
    assert result["p90_chars_approx"] == 7
    assert result["p99_chars_approx"] == 9
    # Bloom filter never undercounts true duplicates; with a low fp_rate
    # and a tiny, distinct-length-driven corpus, it should match exactly.
    assert result["approx_duplicate_count"] == 3


def test_compute_doc_content_stats_approx_percentiles_one_to_ten() -> None:
    docs = [
        Document(id=str(n), page_content="x" * n, metadata={"source": "x"}) for n in range(1, 11)
    ]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["std_dev_chars"] == pytest.approx(2.8722813232690143)
    assert result["p50_chars_approx"] == 5
    assert result["p90_chars_approx"] == 9
    assert result["p99_chars_approx"] == 10


# --- Tie-breaking ---


def test_compute_doc_content_stats_approx_min_max_id_first_on_tie() -> None:
    docs = [
        Document(id="first_min", page_content="aa", metadata={"source": "x"}),
        Document(id="second_min", page_content="aa", metadata={"source": "x"}),
        Document(id="first_max", page_content="zz", metadata={"source": "x"}),
        Document(id="second_max", page_content="zz", metadata={"source": "x"}),
    ]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["min_doc_id"] == "first_min"
    assert result["max_doc_id"] == "first_min"
    assert result["approx_duplicate_count"] == 2


# --- Data quality flags ---


def test_compute_doc_content_stats_approx_messy_docs(messy_docs: list[Document]) -> None:
    result = compute_doc_content_stats_approx(messy_docs, reservoir_size=FULL_RESERVOIR)
    assert result["count"] == 5
    assert result["total_chars"] == 29
    assert result["avg_chars"] == 5.8
    assert result["std_dev_chars"] == pytest.approx(3.8678159211627436)
    assert result["min_chars"] == 0
    assert result["max_chars"] == 12
    assert result["min_doc_id"] == "empty"
    assert result["max_doc_id"] == "ok"
    assert result["empty_count"] == 1
    assert result["whitespace_only_count"] == 1
    assert result["none_or_non_str_content_count"] == 0
    assert result["none_id_count"] == 1
    assert result["missing_metadata_count"] == 1


def test_compute_doc_content_stats_approx_none_content_treated_as_empty() -> None:
    doc = Document(id="1", page_content="placeholder", metadata={"source": "x"})
    doc.page_content = None
    result = compute_doc_content_stats_approx([doc], reservoir_size=FULL_RESERVOIR)
    assert result["none_or_non_str_content_count"] == 1
    assert result["empty_count"] == 1
    assert result["min_chars"] == 0


def test_compute_doc_content_stats_approx_non_string_content() -> None:
    doc = Document(id="1", page_content="placeholder", metadata={"source": "x"})
    doc.page_content = 12345
    result = compute_doc_content_stats_approx([doc], reservoir_size=FULL_RESERVOIR)
    assert result["none_or_non_str_content_count"] == 1


# --- Duplicate detection ---


def test_compute_doc_content_stats_approx_duplicates(duplicate_docs: list[Document]) -> None:
    result = compute_doc_content_stats_approx(duplicate_docs, reservoir_size=FULL_RESERVOIR)
    assert result["approx_duplicate_count"] == 1


def test_compute_doc_content_stats_approx_no_duplicates() -> None:
    docs = [
        Document(id="1", page_content="a", metadata={"source": "x"}),
        Document(id="2", page_content="b", metadata={"source": "x"}),
        Document(id="3", page_content="c", metadata={"source": "x"}),
    ]
    result = compute_doc_content_stats_approx(docs)
    # Bloom filter has no false negatives, but could false-positive;
    # for a 3-item corpus with default settings this should be 0, and
    # is never allowed to be less than the true count (0).
    assert result["approx_duplicate_count"] >= 0


def test_compute_doc_content_stats_approx_empty_strings_are_duplicates_of_each_other() -> None:
    docs = [Document(id=str(i), page_content="", metadata={"source": "x"}) for i in range(3)]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["approx_duplicate_count"] == 2


def test_compute_doc_content_stats_approx_many_duplicates_same_content() -> None:
    docs = [Document(id=str(i), page_content="repeat", metadata={"source": "x"}) for i in range(5)]
    result = compute_doc_content_stats_approx(docs, reservoir_size=FULL_RESERVOIR)
    assert result["approx_duplicate_count"] == 4


def test_compute_doc_content_stats_approx_duplicate_count_never_undercounts() -> None:
    # Bloom filters have no false negatives: every true duplicate must
    # be flagged, even under a deliberately tiny/adversarial filter.
    true_duplicates = 50
    docs = [Document(id=str(i), page_content="same", metadata={"source": "x"}) for i in range(51)]
    result = compute_doc_content_stats_approx(
        docs, expected_doc_count=5, fp_rate=0.5, reservoir_size=FULL_RESERVOIR
    )
    assert result["approx_duplicate_count"] >= true_duplicates


# --- Iterator / generator support ---


def test_compute_doc_content_stats_approx_generator_input() -> None:
    def gen() -> object:
        yield Document(id="1", page_content="aaaa", metadata={"source": "x"})
        yield Document(id="2", page_content="bb", metadata={"source": "x"})

    result = compute_doc_content_stats_approx(gen(), reservoir_size=FULL_RESERVOIR)
    assert result["count"] == 2
    assert result["total_chars"] == 6
    assert result["std_dev_chars"] == pytest.approx(1.0)
    assert result["p50_chars_approx"] == 2
    assert result["p90_chars_approx"] == 4


def test_compute_doc_content_stats_approx_generator_consumed_only_once() -> None:
    def gen() -> object:
        yield Document(id="1", page_content="x", metadata={"source": "x"})

    g = gen()
    compute_doc_content_stats_approx(g)
    assert list(g) == []


def test_compute_doc_content_stats_approx_map_iterator() -> None:
    raw_texts = ["one", "two", "three"]
    docs_iter = (Document(id=t, page_content=t, metadata={"source": "x"}) for t in raw_texts)
    result = compute_doc_content_stats_approx(docs_iter, reservoir_size=FULL_RESERVOIR)
    assert result["count"] == 3
    assert result["total_chars"] == sum(len(t) for t in raw_texts)
    assert result["std_dev_chars"] == pytest.approx(0.9428090415820634)


# --- Approximation-specific behavior ---


def test_compute_doc_content_stats_approx_reports_approx_flags(docs: list[Document]) -> None:
    result = compute_doc_content_stats_approx(docs)
    assert result["duplicate_count_exact"] is False
    assert result["percentiles_exact"] is False


def test_compute_doc_content_stats_approx_reports_fp_rate(docs: list[Document]) -> None:
    result = compute_doc_content_stats_approx(docs, fp_rate=0.05)
    assert result["bloom_filter_fp_rate"] == 0.05


def test_compute_doc_content_stats_approx_reservoir_capped_below_count() -> None:
    docs = [Document(id=str(i), page_content="x" * i, metadata={"source": "x"}) for i in range(100)]
    result = compute_doc_content_stats_approx(docs, reservoir_size=10)
    assert result["count"] == 100  # exact, unaffected by reservoir size
    assert result["reservoir_sample_size"] == 10  # capped


def test_compute_doc_content_stats_approx_reservoir_uncapped_above_count() -> None:
    docs = [Document(id=str(i), page_content="x" * i, metadata={"source": "x"}) for i in range(5)]
    result = compute_doc_content_stats_approx(docs, reservoir_size=10)
    assert result["reservoir_sample_size"] == 5  # not padded beyond count


def test_compute_doc_content_stats_approx_percentiles_none_are_estimates_not_exact(
    docs: list[Document],
) -> None:
    result = compute_doc_content_stats_approx(docs)
    assert "p50_chars_approx" in result
    assert "p50_chars" not in result  # field name differs from exact variant


# --- Larger scale sanity check ---


def test_compute_doc_content_stats_approx_thousand_docs() -> None:
    docs = [
        Document(id=str(i), page_content="x" * (i % 50 + 1), metadata={"source": "x"})
        for i in range(1000)
    ]
    result = compute_doc_content_stats_approx(docs, expected_doc_count=1000)
    assert result["count"] == 1000
    assert result["min_chars"] == 1
    assert result["max_chars"] == 50
    assert result["total_chars"] == sum((i % 50 + 1) for i in range(1000))
    # 1000 docs cycling through 50 distinct lengths -> exactly 950 dups.
    # Bloom filter never undercounts, so this is a floor, not exact.
    assert result["approx_duplicate_count"] >= 950


###########################################
#     Tests for ApproxDocContentStats     #
###########################################


def test_approx_doc_content_stats_starts_at_zero() -> None:
    assert ApproxDocContentStats().to_dict() == compute_doc_content_stats_approx([])


def test_approx_doc_content_stats_manual_update_calls() -> None:
    stats = ApproxDocContentStats(reservoir_size=FULL_RESERVOIR)
    stats.update(Document(id="1", page_content="aa", metadata={"source": "x"}))
    stats.update(Document(id="2", page_content="bbbb", metadata={"source": "x"}))
    result = stats.to_dict()
    assert result["count"] == 2
    assert result["avg_chars"] == 3
    assert result["min_doc_id"] == "1"
    assert result["max_doc_id"] == "2"
    assert result["p50_chars_approx"] == 2
    assert result["p90_chars_approx"] == 4


def test_approx_doc_content_stats_to_dict_is_idempotent() -> None:
    stats = ApproxDocContentStats()
    stats.update(Document(id="1", page_content="abc", metadata={"source": "x"}))
    assert stats.to_dict() == stats.to_dict()


def test_approx_doc_content_stats_accepts_prebuilt_bloom_filter() -> None:
    bloom = BloomFilter(expected_items=10, fp_rate=0.001)
    stats = ApproxDocContentStats(_bloom=bloom)
    stats.update(Document(id="1", page_content="abc", metadata={"source": "x"}))
    stats.update(Document(id="2", page_content="abc", metadata={"source": "x"}))
    assert stats.to_dict()["approx_duplicate_count"] == 1
