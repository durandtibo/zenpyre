from __future__ import annotations

from dataclasses import dataclass

import pytest
from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available

from zenpyre.utils.dataclass import dataclasses_to_dataframe

if is_polars_available():
    import polars as pl
    from polars.testing import assert_frame_equal


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int | None


##############################################
#     Tests for dataclasses_to_dataframe     #
##############################################


@polars_available
def test_dataclasses_to_dataframe_returns_dataframe() -> None:
    assert isinstance(dataclasses_to_dataframe([Point(1, 2)]), pl.DataFrame)


@polars_available
def test_dataclasses_to_dataframe_single_item() -> None:
    df = dataclasses_to_dataframe([Point(1, 2)])
    assert_frame_equal(df, pl.DataFrame({"x": [1], "y": [2]}))


@polars_available
def test_dataclasses_to_dataframe_multiple_items() -> None:
    points = [Point(1, 2), Point(3, 4), Point(-1, 0)]
    df = dataclasses_to_dataframe(points)
    assert_frame_equal(df, pl.DataFrame({"x": [1, 3, -1], "y": [2, 4, 0]}))


@polars_available
def test_dataclasses_to_dataframe_with_none_field() -> None:
    companies = [Company(ticker="AAPL", cik=320193), Company(ticker="XYZ", cik=None)]
    df = dataclasses_to_dataframe(companies)
    assert_frame_equal(df, pl.DataFrame({"ticker": ["AAPL", "XYZ"], "cik": [320193, None]}))


@polars_available
def test_dataclasses_to_dataframe_column_names() -> None:
    df = dataclasses_to_dataframe([Point(1, 2)])
    assert df.columns == ["x", "y"]


@polars_available
def test_dataclasses_to_dataframe_row_count() -> None:
    df = dataclasses_to_dataframe([Point(1, 2), Point(3, 4)])
    assert df.height == 2


@polars_available
def test_dataclasses_to_dataframe_empty_list() -> None:
    df = dataclasses_to_dataframe([])
    assert df.is_empty()


@polars_available
def test_dataclasses_to_dataframe_non_dataclass_raises() -> None:
    with pytest.raises(TypeError, match=r"must be dataclass instances"):
        dataclasses_to_dataframe(["not a dataclass"])


@polars_available
def test_dataclasses_to_dataframe_mixed_list_raises() -> None:
    with pytest.raises(TypeError, match=r"must be dataclass instances"):
        dataclasses_to_dataframe([Point(1, 2), "not a dataclass"])
