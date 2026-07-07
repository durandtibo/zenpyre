from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from zenpyre.ingestors import DataclassIngestor
from zenpyre.utils.dataclass_io import save_dataclasses

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "zenpyre.ingestors.dataclass"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@pytest.fixture
def points() -> list[Point]:
    return [Point(1, 2), Point(3, 4)]


@pytest.fixture
def points_path(tmp_path: Path, points: list[Point]) -> Path:
    path = tmp_path / "points.json"
    save_dataclasses(points, path)
    return path


########################################
#     Tests for DataclassIngestor      #
########################################


# --- Constructor ---


def test_dataclass_ingestor_stores_path(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    ingestor = DataclassIngestor(path=path, cls=Point)
    assert ingestor._path == path


def test_dataclass_ingestor_stores_cls(tmp_path: Path) -> None:
    ingestor = DataclassIngestor(path=tmp_path / "points.json", cls=Point)
    assert ingestor._cls is Point


def test_dataclass_ingestor_repr_contains_class_name(tmp_path: Path) -> None:
    ingestor = DataclassIngestor(path=tmp_path / "points.json", cls=Point)
    assert "DataclassIngestor" in repr(ingestor)


def test_dataclass_ingestor_str_contains_class_name(tmp_path: Path) -> None:
    ingestor = DataclassIngestor(path=tmp_path / "points.json", cls=Point)
    assert "DataclassIngestor" in str(ingestor)


def test_dataclass_ingestor_repr_contains_path(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    assert str(path) in repr(DataclassIngestor(path=path, cls=Point))


# --- ingest: successful load ---


def test_dataclass_ingestor_ingest_returns_list(points_path: Path) -> None:
    assert isinstance(DataclassIngestor(path=points_path, cls=Point).ingest(), list)


def test_dataclass_ingestor_ingest_returns_correct_data(
    points_path: Path, points: list[Point]
) -> None:
    assert DataclassIngestor(path=points_path, cls=Point).ingest() == points


def test_dataclass_ingestor_ingest_returns_correct_type(points_path: Path) -> None:
    result = DataclassIngestor(path=points_path, cls=Point).ingest()
    assert all(isinstance(p, Point) for p in result)


def test_dataclass_ingestor_ingest_calls_load_dataclasses(points_path: Path) -> None:
    with patch(f"{MODULE}.load_dataclasses", return_value=[]) as mock_load:
        DataclassIngestor(path=points_path, cls=Point).ingest()
    mock_load.assert_called_once_with(points_path, Point)


def test_dataclass_ingestor_ingest_can_be_called_multiple_times(points_path: Path) -> None:
    ingestor = DataclassIngestor(path=points_path, cls=Point)
    assert ingestor.ingest() == ingestor.ingest()


def test_dataclass_ingestor_ingest_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    save_dataclasses([], path)
    assert DataclassIngestor(path=path, cls=Point).ingest() == []


# --- ingest: file not found ---


def test_dataclass_ingestor_ingest_raises_file_not_found_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        DataclassIngestor(path=path, cls=Point).ingest()
