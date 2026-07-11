from __future__ import annotations

import pytest
from rich.console import Console

from zenpyre.records.analysis import print_metadata_stats_report
from zenpyre.records.analysis.metadata_print import _format_pct, _format_value


@pytest.fixture
def console() -> Console:
    # Fixed width avoids terminal-size-dependent wrapping in test output.
    return Console(width=100, record=True)


@pytest.fixture
def full_stats() -> dict:
    return {
        "count": 6,
        "missing_metadata_count": 1,
        "avg_keys": 1.67,
        "min_keys": 0,
        "max_keys": 3,
        "distinct_keys_seen": 3,
        "per_key": {
            "source": {
                "present_in_docs": 5,
                "missing_in_docs": 1,
                "value_types": ["str"],
                "none_or_empty_count": 0,
                "unique_values_sample": ["a.pdf", "b.pdf", "c.pdf"],
                "unique_values_sample_truncated": True,
            },
            "page": {
                "present_in_docs": 4,
                "missing_in_docs": 2,
                "value_types": ["int"],
                "none_or_empty_count": 0,
                "unique_values_sample": [1, 2, 3],
                "unique_values_sample_truncated": False,
            },
            "author": {
                "present_in_docs": 3,
                "missing_in_docs": 3,
                "value_types": ["NoneType", "str"],
                "none_or_empty_count": 2,
                "unique_values_sample": ["", None, "alice"],
                "unique_values_sample_truncated": False,
            },
        },
    }


#################################
#     Tests for _format_pct     #
#################################


def test_format_pct_basic() -> None:
    assert _format_pct(2, 4) == "2 (50.0%)"


def test_format_pct_zero_part() -> None:
    assert _format_pct(0, 4) == "0 (0.0%)"


def test_format_pct_part_equals_total() -> None:
    assert _format_pct(4, 4) == "4 (100.0%)"


def test_format_pct_zero_total() -> None:
    assert _format_pct(0, 0) == "0"


def test_format_pct_zero_total_nonzero_part() -> None:
    # Defensive case - shouldn't happen in practice, but must not raise.
    assert _format_pct(3, 0) == "3"


def test_format_pct_rounding() -> None:
    assert _format_pct(1, 3) == "1 (33.3%)"


###################################
#     Tests for _format_value     #
###################################


def test_format_value_empty_string() -> None:
    assert _format_value("") == "<empty string>"


def test_format_value_none() -> None:
    assert _format_value(None) == "<None>"


def test_format_value_regular_string() -> None:
    assert _format_value("a.pdf") == "a.pdf"


def test_format_value_int() -> None:
    assert _format_value(42) == "42"


def test_format_value_float() -> None:
    assert _format_value(3.14) == "3.14"


def test_format_value_bool_true() -> None:
    # bool is not str/None, falls through to str(value).
    assert _format_value(True) == "True"


def test_format_value_bool_false() -> None:
    # False == 0 and False == "" is False, so this must not be mistaken
    # for the empty-string case.
    assert _format_value(False) == "False"


def test_format_value_zero_int() -> None:
    # 0 == "" is False in Python, so this must not hit the empty-string branch.
    assert _format_value(0) == "0"


#################################################
#     Tests for print_metadata_stats_report     #
#################################################


def test_print_metadata_stats_report_does_not_raise_full_stats(
    console: Console, full_stats: dict
) -> None:
    print_metadata_stats_report(full_stats, console=console)


def test_print_metadata_stats_report_does_not_raise_empty_stats(console: Console) -> None:
    print_metadata_stats_report({"count": 0}, console=console)


def test_print_metadata_stats_report_does_not_raise_no_per_key(console: Console) -> None:
    stats = {
        "count": 2,
        "missing_metadata_count": 2,
        "avg_keys": 0,
        "min_keys": 0,
        "max_keys": 0,
        "distinct_keys_seen": 0,
        "per_key": {},
    }
    print_metadata_stats_report(stats, console=console)


def test_print_metadata_stats_report_does_not_raise_missing_per_key_field(
    console: Console,
) -> None:
    # "per_key" absent entirely - relies on .get("per_key", {}) fallback.
    stats = {
        "count": 1,
        "missing_metadata_count": 0,
        "avg_keys": 1,
        "min_keys": 1,
        "max_keys": 1,
        "distinct_keys_seen": 1,
    }
    print_metadata_stats_report(stats, console=console)


def test_print_metadata_stats_report_does_not_raise_default_console(full_stats: dict) -> None:
    # No console passed - exercises the get_console() fallback path.
    print_metadata_stats_report(full_stats)


def test_print_metadata_stats_report_does_not_raise_custom_title(
    console: Console, full_stats: dict
) -> None:
    print_metadata_stats_report(full_stats, title="Custom Title", console=console)


def test_print_metadata_stats_report_does_not_raise_no_missing_metadata(
    console: Console,
) -> None:
    stats = {
        "count": 3,
        "missing_metadata_count": 0,
        "avg_keys": 2,
        "min_keys": 2,
        "max_keys": 2,
        "distinct_keys_seen": 1,
        "per_key": {
            "source": {
                "present_in_docs": 3,
                "missing_in_docs": 0,
                "value_types": ["str"],
                "none_or_empty_count": 0,
                "unique_values_sample": ["a", "b", "c"],
                "unique_values_sample_truncated": False,
            },
        },
    }
    print_metadata_stats_report(stats, console=console)


def test_print_metadata_stats_report_does_not_raise_empty_value_sample(
    console: Console,
) -> None:
    # unique_values_sample empty (e.g. all values were unhashable).
    stats = {
        "count": 1,
        "missing_metadata_count": 0,
        "avg_keys": 1,
        "min_keys": 1,
        "max_keys": 1,
        "distinct_keys_seen": 1,
        "per_key": {
            "tags": {
                "present_in_docs": 1,
                "missing_in_docs": 0,
                "value_types": ["list"],
                "none_or_empty_count": 0,
                "unique_values_sample": [],
                "unique_values_sample_truncated": True,
            },
        },
    }
    print_metadata_stats_report(stats, console=console)


def test_print_metadata_stats_report_does_not_raise_produces_output(
    console: Console, full_stats: dict
) -> None:
    print_metadata_stats_report(full_stats, console=console)
    output = console.export_text()
    assert output.strip() != ""
