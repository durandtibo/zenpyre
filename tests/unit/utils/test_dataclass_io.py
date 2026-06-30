"""Unit tests for load_dataclasses and save_dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from zenpyre.utils.dataclass_io import load_dataclasses, save_dataclasses


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int | None


############################################
#     Tests for load_dataclasses          #
############################################


def test_load_dataclasses_returns_list(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    save_dataclasses([Point(1, 2)], path)
    assert isinstance(load_dataclasses(path, Point), list)


def test_load_dataclasses_round_trip_single_item(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    points = [Point(1, 2)]
    save_dataclasses(points, path)
    assert load_dataclasses(path, Point) == points


def test_load_dataclasses_round_trip_multiple_items(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    points = [Point(1, 2), Point(3, 4), Point(-1, 0)]
    save_dataclasses(points, path)
    assert load_dataclasses(path, Point) == points


def test_load_dataclasses_round_trip_with_none_field(tmp_path: Path) -> None:
    path = tmp_path / "companies.json"
    companies = [Company(ticker="AAPL", cik=320193), Company(ticker="XYZ", cik=None)]
    save_dataclasses(companies, path)
    assert load_dataclasses(path, Company) == companies


def test_load_dataclasses_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    points = [Point(1, 2)]
    save_dataclasses(points, path)
    assert load_dataclasses(str(path), Point) == points


def test_load_dataclasses_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataclasses(tmp_path / "nonexistent.json", Point)


def test_load_dataclasses_non_dataclass_cls_raises(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    save_dataclasses([Point(1, 2)], path)
    with pytest.raises(TypeError, match="cls must be a dataclass type"):
        load_dataclasses(path, str)  # type: ignore[type-var]


def test_load_dataclasses_missing_field_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text('[{"x": 1}]')  # missing required field "y"
    with pytest.raises(ValueError, match="Could not parse"):
        load_dataclasses(path, Point)


def test_load_dataclasses_not_a_list_raises_type_error(tmp_path: Path) -> None:
    path = tmp_path / "not_a_list.json"
    path.write_text('{"x": 1}')
    with pytest.raises(TypeError, match="Expected a JSON array"):
        load_dataclasses(path, Point)


############################################
#     Tests for save_dataclasses          #
############################################


def test_save_dataclasses_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    save_dataclasses([Point(1, 2)], path)
    assert path.exists()


def test_save_dataclasses_returns_none(tmp_path: Path) -> None:
    assert save_dataclasses([Point(1, 2)], tmp_path / "points.json") is None


def test_save_dataclasses_accepts_str_path(tmp_path: Path) -> None:
    path = str(tmp_path / "points.json")
    save_dataclasses([Point(1, 2)], path)
    assert Path(path).exists()


def test_save_dataclasses_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    save_dataclasses([], path)
    assert load_dataclasses(path, Point) == []


def test_save_dataclasses_non_dataclass_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be dataclass instances"):
        save_dataclasses(["not a dataclass"], tmp_path / "bad.json")  # type: ignore[list-item]


def test_save_dataclasses_mixed_list_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be dataclass instances"):
        save_dataclasses([Point(1, 2), "not a dataclass"], tmp_path / "bad.json")  # type: ignore[list-item]


def test_save_dataclasses_does_not_create_file_on_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    with pytest.raises(TypeError):
        save_dataclasses(["not a dataclass"], path)  # type: ignore[list-item]
    assert not path.exists()
