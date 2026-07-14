from __future__ import annotations

from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available

from zenpyre.utils.dataframe import drop_null_cols

if is_polars_available():
    import polars as pl
    from polars.testing import assert_frame_equal

######################################
#     Tests for drop_null_cols       #
######################################


@polars_available
def test_drop_null_cols_drops_all_null_column() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [None, None, None]}, schema={"a": pl.Int64, "b": pl.Int64}
    )
    result = drop_null_cols(df)
    assert_frame_equal(result, pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int64}))


@polars_available
def test_drop_null_cols_keeps_column_with_at_least_one_non_null_value() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [None, 5, None]})
    result = drop_null_cols(df)
    assert_frame_equal(result, df)


@polars_available
def test_drop_null_cols_drops_multiple_all_null_columns() -> None:
    df = pl.DataFrame(
        {"a": [1, 2], "b": [None, None], "c": [None, None], "d": [3, 4]},
        schema={"a": pl.Int64, "b": pl.Int64, "c": pl.Utf8, "d": pl.Int64},
    )
    result = drop_null_cols(df)
    assert_frame_equal(
        result, pl.DataFrame({"a": [1, 2], "d": [3, 4]}, schema={"a": pl.Int64, "d": pl.Int64})
    )


@polars_available
def test_drop_null_cols_no_columns_are_all_null_returns_all_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = drop_null_cols(df)
    assert_frame_equal(result, df)


@polars_available
def test_drop_null_cols_preserves_column_order() -> None:
    df = pl.DataFrame(
        {"a": [None, None], "b": [1, 2], "c": [None, None], "d": [3, 4]},
        schema={"a": pl.Int64, "b": pl.Int64, "c": pl.Utf8, "d": pl.Int64},
    )
    result = drop_null_cols(df)
    assert_frame_equal(
        result, pl.DataFrame({"b": [1, 2], "d": [3, 4]}, schema={"b": pl.Int64, "d": pl.Int64})
    )


@polars_available
def test_drop_null_cols_single_row_all_null_column_is_dropped() -> None:
    df = pl.DataFrame({"a": [1], "b": [None]}, schema={"a": pl.Int64, "b": pl.Int64})
    result = drop_null_cols(df)
    assert_frame_equal(result, pl.DataFrame({"a": [1]}, schema={"a": pl.Int64}))


# --- Edge cases ---


@polars_available
def test_drop_null_cols_empty_dataframe_zero_rows_keeps_all_columns() -> None:
    # Regression test: with 0 rows, every column's null_count() is 0,
    # which trivially equals df.height (also 0). Without an explicit
    # guard, this would make every column look "all null" and drop it.
    df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.Utf8})
    result = drop_null_cols(df)
    assert_frame_equal(result, df)


@polars_available
def test_drop_null_cols_no_columns_dataframe() -> None:
    df = pl.DataFrame()
    result = drop_null_cols(df)
    assert_frame_equal(result, df)


@polars_available
def test_drop_null_cols_all_columns_all_null_returns_no_columns() -> None:
    df = pl.DataFrame({"a": [None, None], "b": [None, None]}, schema={"a": pl.Int64, "b": pl.Int64})
    result = drop_null_cols(df)
    assert_frame_equal(result, pl.DataFrame())


@polars_available
def test_drop_null_cols_does_not_mutate_input() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [None, None]}, schema={"a": pl.Int64, "b": pl.Int64})
    expected = df.clone()
    drop_null_cols(df)
    assert_frame_equal(df, expected)
