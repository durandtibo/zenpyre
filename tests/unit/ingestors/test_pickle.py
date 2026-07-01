from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iden.io import save_pickle

from zenpyre.ingestors import PickleIngestor

if TYPE_CHECKING:
    from pathlib import Path

DATA = {
    "cats": ["Whiskers", "Luna"],
    "dogs": ["Rex", "Buddy"],
}


################################
#   Tests for PickleIngestor   #
################################

# --- Constructor ---


def test_pickle_ingestor_stores_path(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    ingestor = PickleIngestor(path=path)
    assert ingestor._path == path


def test_pickle_ingestor_repr_contains_class_name(tmp_path: Path) -> None:
    ingestor = PickleIngestor(path=tmp_path / "data.pkl")
    assert "PickleIngestor" in repr(ingestor)


def test_pickle_ingestor_str_contains_class_name(tmp_path: Path) -> None:
    ingestor = PickleIngestor(path=tmp_path / "data.pkl")
    assert "PickleIngestor" in str(ingestor)


def test_pickle_ingestor_repr_contains_path(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    assert str(path) in repr(PickleIngestor(path=path))


# --- ingest: successful load ---


def test_pickle_ingestor_ingest_returns_data(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    save_pickle(DATA, path)
    assert PickleIngestor(path=path).ingest() == DATA


def test_pickle_ingestor_ingest_returns_correct_keys(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    save_pickle(DATA, path)
    assert set(PickleIngestor(path=path).ingest().keys()) == {"cats", "dogs"}


def test_pickle_ingestor_ingest_can_be_called_multiple_times(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    save_pickle(DATA, path)
    ingestor = PickleIngestor(path=path)
    assert ingestor.ingest() == ingestor.ingest()


# --- ingest: file not found ---


def test_pickle_ingestor_ingest_raises_file_not_found_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError, match=r"missing.pkl"):
        PickleIngestor(path=path).ingest()


def test_pickle_ingestor_ingest_error_message_contains_full_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError, match=str(path)):
        PickleIngestor(path=path).ingest()


def test_pickle_ingestor_ingest_error_message_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError, match=r"ingest"):
        PickleIngestor(path=path).ingest()
