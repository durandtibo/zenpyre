from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.table import Table
from rich.text import Text

from zenpyre.documents.analysis import print_content_stats_report
from zenpyre.documents.analysis.content_print import (
    _build_approx_footnote_items,
    _build_bar_chart_items,
    _build_doc_ids_grid,
    _build_overview_line,
    _build_stats_table,
    _build_status_line,
    _format_percentage_of_total,
    _format_value,
    _render_bar_chart_rows,
)


@pytest.fixture
def console() -> Console:
    # file=io.StringIO() avoids writing to the real terminal during tests
    return Console(file=io.StringIO(), width=100)


@pytest.fixture
def exact_stats() -> dict:
    return {
        "count": 1000,
        "total_chars": 542_300,
        "avg_chars": 542.3,
        "std_dev_chars": 128.7,
        "min_chars": 12,
        "max_chars": 3200,
        "min_doc_id": "doc-0417",
        "max_doc_id": "doc-0912",
        "p50_chars": 510,
        "p90_chars": 890,
        "p99_chars": 1400,
        "empty_count": 3,
        "whitespace_only_count": 1,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 12,
        "duplicate_count": 27,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }


@pytest.fixture
def approx_stats() -> dict:
    return {
        "count": 1000,
        "total_chars": 542_300,
        "avg_chars": 542.3,
        "std_dev_chars": 128.7,
        "min_chars": 12,
        "max_chars": 3200,
        "min_doc_id": "doc-0417",
        "max_doc_id": "doc-0912",
        "p50_chars_approx": 510,
        "p90_chars_approx": 890,
        "p99_chars_approx": 1400,
        "empty_count": 3,
        "whitespace_only_count": 1,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 12,
        "approx_duplicate_count": 27,
        "duplicate_count_exact": False,
        "percentiles_exact": False,
        "bloom_filter_fp_rate": 0.01,
        "reservoir_sample_size": 1000,
    }


@pytest.fixture
def empty_stats() -> dict:
    return {
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


###################################
#     Tests for _format_value     #
###################################


def test_format_value_none() -> None:
    assert _format_value(None) == "[dim]—[/dim]"


def test_format_value_bool_true() -> None:
    assert _format_value(True) == "[green]yes[/green]"


def test_format_value_bool_false() -> None:
    assert _format_value(False) == "[yellow]no[/yellow]"


def test_format_value_float() -> None:
    assert _format_value(3.14159) == "3.14"


def test_format_value_float_thousands_separator() -> None:
    assert _format_value(542300.5) == "542,300.50"


def test_format_value_int() -> None:
    assert _format_value(1000) == "1,000"


def test_format_value_int_no_separator_needed() -> None:
    assert _format_value(5) == "5"


def test_format_value_string() -> None:
    assert _format_value("doc-1") == "doc-1"


def test_format_value_zero_int() -> None:
    assert _format_value(0) == "0"


#################################################
#     Tests for _format_percentage_of_total     #
#################################################


def test_format_percentage_of_total_normal() -> None:
    result = _format_percentage_of_total(3, 1000)
    assert "0.3%" in result


def test_format_percentage_of_total_zero_total() -> None:
    assert _format_percentage_of_total(0, 0) == ""


def test_format_percentage_of_total_zero_count() -> None:
    result = _format_percentage_of_total(0, 100)
    assert "0.0%" in result


def test_format_percentage_of_total_full() -> None:
    result = _format_percentage_of_total(100, 100)
    assert "100.0%" in result


def test_format_percentage_of_total_has_leading_space() -> None:
    result = _format_percentage_of_total(1, 10)
    assert result.startswith(" ")


############################################
#     Tests for _render_bar_chart_rows     #
############################################


def test_render_bar_chart_rows_normal_points() -> None:
    rows = _render_bar_chart_rows([12, 510, 890, 1400, 3200])
    assert len(rows) == 5
    assert all(len(r) == 40 for r in rows)


def test_render_bar_chart_rows_empty_points() -> None:
    assert _render_bar_chart_rows([]) == []


def test_render_bar_chart_rows_single_point() -> None:
    assert _render_bar_chart_rows([42]) == []


def test_render_bar_chart_rows_all_equal_points() -> None:
    rows = _render_bar_chart_rows([5, 5, 5], width=10, height=3)
    assert len(rows) == 3
    assert all(len(r) == 10 for r in rows)


def test_render_bar_chart_rows_custom_width_height() -> None:
    rows = _render_bar_chart_rows([1, 2, 3], width=10, height=3)
    assert len(rows) == 3
    assert all(len(r) == 10 for r in rows)


def test_render_bar_chart_rows_bottom_row_reaches_max() -> None:
    # the highest point should fully fill the bottom row somewhere
    rows = _render_bar_chart_rows([0, 100], width=5, height=1)
    assert "█" in rows[0]


def test_render_bar_chart_rows_two_points_minimum() -> None:
    rows = _render_bar_chart_rows([1, 2])
    assert len(rows) == 5  # default height


########################################
#     Tests for _build_stats_table     #
########################################


def test_build_stats_table_returns_table_and_bool(exact_stats: dict) -> None:
    table, any_issues = _build_stats_table(
        exact_stats, is_exact=True, p_suffix="", dup_key="duplicate_count", count=1000
    )
    assert isinstance(table, Table)
    assert isinstance(any_issues, bool)


def test_build_stats_table_flags_issues_when_present(exact_stats: dict) -> None:
    _, any_issues = _build_stats_table(
        exact_stats, is_exact=True, p_suffix="", dup_key="duplicate_count", count=1000
    )
    assert any_issues is True  # exact_stats has empty_count=3, etc.


def test_build_stats_table_no_issues_when_all_zero() -> None:
    clean_stats = {
        "avg_chars": 5,
        "std_dev_chars": 1,
        "min_chars": 4,
        "max_chars": 6,
        "p50_chars": 5,
        "p90_chars": 6,
        "p99_chars": 6,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
    }
    _, any_issues = _build_stats_table(
        clean_stats, is_exact=True, p_suffix="", dup_key="duplicate_count", count=10
    )
    assert any_issues is False


def test_build_stats_table_approx_uses_approx_dup_key(approx_stats: dict) -> None:
    table, any_issues = _build_stats_table(
        approx_stats,
        is_exact=False,
        p_suffix="_approx",
        dup_key="approx_duplicate_count",
        count=1000,
    )
    assert isinstance(table, Table)
    assert any_issues is True


def test_build_stats_table_missing_keys_does_not_raise() -> None:
    # sparse dict should not raise, since the function uses .get(...)
    table, any_issues = _build_stats_table(
        {}, is_exact=True, p_suffix="", dup_key="duplicate_count", count=0
    )
    assert isinstance(table, Table)
    assert any_issues is False


##########################################
#     Tests for _build_overview_line     #
##########################################


def test_build_overview_line_returns_text(exact_stats: dict) -> None:
    result = _build_overview_line(exact_stats, 1000)
    assert isinstance(result, Text)


def test_build_overview_line_contains_doc_count(exact_stats: dict) -> None:
    result = _build_overview_line(exact_stats, 1000)
    assert "1,000 docs" in result.plain


def test_build_overview_line_contains_char_count(exact_stats: dict) -> None:
    result = _build_overview_line(exact_stats, 1000)
    assert "542,300 chars" in result.plain


def test_build_overview_line_missing_total_chars_does_not_raise() -> None:
    result = _build_overview_line({}, 5)
    assert isinstance(result, Text)


#########################################
#     Tests for _build_doc_ids_grid     #
#########################################


def test_build_doc_ids_grid_returns_table(exact_stats: dict) -> None:
    result = _build_doc_ids_grid(exact_stats)
    assert isinstance(result, Table)


def test_build_doc_ids_grid_missing_keys_does_not_raise() -> None:
    result = _build_doc_ids_grid({})
    assert isinstance(result, Table)


########################################
#     Tests for _build_status_line     #
########################################


def test_build_status_line_no_issues() -> None:
    result = _build_status_line(False)
    assert "no issues detected" in result.plain


def test_build_status_line_with_issues() -> None:
    result = _build_status_line(True)
    assert "potential issues detected" in result.plain


def test_build_status_line_returns_text() -> None:
    assert isinstance(_build_status_line(False), Text)
    assert isinstance(_build_status_line(True), Text)


############################################
#     Tests for _build_bar_chart_items     #
############################################


def test_build_bar_chart_items_normal_stats(exact_stats: dict) -> None:
    items = _build_bar_chart_items(exact_stats, p_suffix="")
    assert len(items) > 0
    assert all(isinstance(item, Text) for item in items)


def test_build_bar_chart_items_approx_suffix(approx_stats: dict) -> None:
    items = _build_bar_chart_items(approx_stats, p_suffix="_approx")
    assert len(items) > 0


def test_build_bar_chart_items_single_distinct_point_returns_empty() -> None:
    stats = {
        "min_chars": 100,
        "max_chars": 100,
        "p50_chars": 100,
        "p90_chars": 100,
        "p99_chars": 100,
    }
    assert _build_bar_chart_items(stats, p_suffix="") == []


def test_build_bar_chart_items_all_none_returns_empty() -> None:
    stats = {"min_chars": None, "max_chars": None}
    assert _build_bar_chart_items(stats, p_suffix="") == []


def test_build_bar_chart_items_two_distinct_points_enough() -> None:
    stats = {"min_chars": 1, "max_chars": 10}
    items = _build_bar_chart_items(stats, p_suffix="")
    assert len(items) > 0


def test_build_bar_chart_items_includes_label_header(exact_stats: dict) -> None:
    items = _build_bar_chart_items(exact_stats, p_suffix="")
    assert items[0].plain == "length shape"


##################################################
#     Tests for _build_approx_footnote_items     #
##################################################


def test_build_approx_footnote_items_exact_returns_empty(exact_stats: dict) -> None:
    assert _build_approx_footnote_items(exact_stats, is_exact=True, count=1000) == []


def test_build_approx_footnote_items_approx_returns_one_item(approx_stats: dict) -> None:
    items = _build_approx_footnote_items(approx_stats, is_exact=False, count=1000)
    assert len(items) == 1
    assert isinstance(items[0], Text)


def test_build_approx_footnote_items_contains_fp_rate(approx_stats: dict) -> None:
    items = _build_approx_footnote_items(approx_stats, is_exact=False, count=1000)
    assert "0.01" in items[0].plain


def test_build_approx_footnote_items_contains_sample_size(approx_stats: dict) -> None:
    items = _build_approx_footnote_items(approx_stats, is_exact=False, count=1000)
    assert "1000" in items[0].plain


def test_build_approx_footnote_items_missing_keys_does_not_raise() -> None:
    items = _build_approx_footnote_items({}, is_exact=False, count=0)
    assert len(items) == 1


################################################
#     Tests for print_content_stats_report     #
################################################


def test_print_content_stats_report_exact_runs_without_error(
    exact_stats: dict, console: Console
) -> None:
    print_content_stats_report(exact_stats, console=console)


def test_print_content_stats_report_approx_runs_without_error(
    approx_stats: dict, console: Console
) -> None:
    print_content_stats_report(approx_stats, console=console)


def test_print_content_stats_report_empty_runs_without_error(
    empty_stats: dict, console: Console
) -> None:
    print_content_stats_report(empty_stats, console=console)


def test_print_content_stats_report_creates_own_console_if_none_given(
    exact_stats: dict,
) -> None:
    # no console passed -> should fall back to a real Console() internally
    print_content_stats_report(exact_stats)


def test_print_content_stats_report_custom_title(exact_stats: dict, console: Console) -> None:
    print_content_stats_report(exact_stats, title="Custom Title", console=console)


def test_print_content_stats_report_no_issues(exact_stats: dict, console: Console) -> None:
    healthy_stats = {
        **exact_stats,
        "empty_count": 0,
        "whitespace_only_count": 0,
        "none_or_non_str_content_count": 0,
        "none_id_count": 0,
        "missing_metadata_count": 0,
        "duplicate_count": 0,
    }
    print_content_stats_report(healthy_stats, console=console)


def test_print_content_stats_report_missing_optional_keys(console: Console) -> None:
    # a minimal dict missing several optional keys should not raise,
    # since the function uses .get(...) with defaults throughout
    minimal_stats = {
        "count": 5,
        "total_chars": 50,
        "avg_chars": 10,
        "std_dev_chars": 2,
        "min_chars": 5,
        "max_chars": 15,
        "duplicate_count_exact": True,
        "percentiles_exact": True,
    }
    print_content_stats_report(minimal_stats, console=console)


def test_print_content_stats_report_single_point_no_chart(
    exact_stats: dict, console: Console
) -> None:
    # min == max means only one distinct point -> chart section should
    # be skipped gracefully rather than erroring
    single_point_stats = {**exact_stats, "min_chars": 100, "max_chars": 100}
    print_content_stats_report(single_point_stats, console=console)


def test_print_content_stats_report_does_not_mutate_input(
    exact_stats: dict, console: Console
) -> None:
    original = dict(exact_stats)
    print_content_stats_report(exact_stats, console=console)
    assert exact_stats == original
