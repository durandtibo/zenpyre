from __future__ import annotations

from typing import Any

import pytest

from zenpyre.ingestors import resolve_ingestor
from zenpyre.ingestors.base import BaseIngestor
from zenpyre.utils.config import Config

MINIMAL_INGESTOR_TARGET = "tests.unit.ingestors.test_resolve.MinimalIngestor"


class MinimalIngestor(BaseIngestor[Any]):
    """Minimal concrete BaseIngestor for testing."""

    def ingest(self) -> Any:
        return {"hello": "world"}


##############################################
#       Tests for resolve_ingestor           #
##############################################


# --- Pass-through ---


def test_resolve_ingestor_returns_base_ingestor_instance() -> None:
    assert isinstance(resolve_ingestor(MinimalIngestor()), BaseIngestor)


def test_resolve_ingestor_passthrough_returns_same_instance() -> None:
    ingestor = MinimalIngestor()
    assert resolve_ingestor(ingestor) is ingestor


# --- From dict ---


def test_resolve_ingestor_from_dict_returns_base_ingestor() -> None:
    result = resolve_ingestor({"_target_": MINIMAL_INGESTOR_TARGET})
    assert isinstance(result, BaseIngestor)


def test_resolve_ingestor_from_dict_returns_correct_type() -> None:
    result = resolve_ingestor({"_target_": MINIMAL_INGESTOR_TARGET})
    assert isinstance(result, MinimalIngestor)


# --- From BaseConfig ---


def test_resolve_ingestor_from_base_config_returns_base_ingestor() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_INGESTOR_TARGET)
    result = resolve_ingestor(config)
    assert isinstance(result, BaseIngestor)


def test_resolve_ingestor_from_base_config_returns_correct_type() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_INGESTOR_TARGET)
    result = resolve_ingestor(config)
    assert isinstance(result, MinimalIngestor)


# --- Invalid input ---


def test_resolve_ingestor_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseIngestor instance"):
        resolve_ingestor("not-an-ingestor")
