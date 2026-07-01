from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zenpyre.ingestors import TextIngestor

if TYPE_CHECKING:
    from pathlib import Path

CONTENT = "blabla\nBla."


################################
#   Tests for TextIngestor     #
################################

# --- Constructor ---


def test_text_ingestor_stores_path(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    assert TextIngestor(path=path)._path == path


def test_text_ingestor_default_encoding_is_utf8(tmp_path: Path) -> None:
    assert TextIngestor(path=tmp_path / "filing.md")._encoding == "utf-8"


def test_text_ingestor_accepts_custom_encoding(tmp_path: Path) -> None:
    assert TextIngestor(path=tmp_path / "filing.md", encoding="latin-1")._encoding == "latin-1"


def test_text_ingestor_repr_contains_class_name(tmp_path: Path) -> None:
    assert "TextIngestor" in repr(TextIngestor(path=tmp_path / "filing.md"))


def test_text_ingestor_str_contains_class_name(tmp_path: Path) -> None:
    assert "TextIngestor" in str(TextIngestor(path=tmp_path / "filing.md"))


def test_text_ingestor_repr_contains_path(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    assert str(path) in repr(TextIngestor(path=path))


def test_text_ingestor_repr_contains_encoding(tmp_path: Path) -> None:
    ingestor = TextIngestor(path=tmp_path / "filing.md", encoding="latin-1")
    assert "latin-1" in repr(ingestor)


# --- ingest: successful load ---


def test_text_ingestor_ingest_returns_str(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    path.write_text(CONTENT, encoding="utf-8")
    result = TextIngestor(path=path).ingest()
    assert isinstance(result, str)


def test_text_ingestor_ingest_returns_file_content(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    path.write_text(CONTENT, encoding="utf-8")
    assert TextIngestor(path=path).ingest() == CONTENT


def test_text_ingestor_ingest_returns_empty_string_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    path.write_text("", encoding="utf-8")
    assert TextIngestor(path=path).ingest() == ""


def test_text_ingestor_ingest_respects_encoding(tmp_path: Path) -> None:
    content = "Résultats financiers: €1.2B"
    path = tmp_path / "filing.md"
    path.write_text(content, encoding="utf-8")
    assert TextIngestor(path=path, encoding="utf-8").ingest() == content


def test_text_ingestor_ingest_is_consistent_across_multiple_calls(tmp_path: Path) -> None:
    path = tmp_path / "filing.md"
    path.write_text(CONTENT, encoding="utf-8")
    ingestor = TextIngestor(path=path)
    assert ingestor.ingest() == ingestor.ingest()


# --- ingest: file not found ---


def test_text_ingestor_ingest_raises_file_not_found_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError, match=r"missing.md"):
        TextIngestor(path=path).ingest()


def test_text_ingestor_ingest_error_message_contains_full_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError, match=str(path)):
        TextIngestor(path=path).ingest()


def test_text_ingestor_ingest_error_message_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError, match=r"ingest"):
        TextIngestor(path=path).ingest()
