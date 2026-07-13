from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal

from zenpyre.agents.factory import AgentFactory, BaseAgentFactory, CachingAgentFactory
from zenpyre.utils.config import Config

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "zenpyre.agents.factory.cache"

MINIMAL_AGENT_FACTORY_TARGET = "tests.unit.agents.factory.test_cache.MinimalAgentFactory"


class MinimalAgentFactory(BaseAgentFactory):
    """Minimal concrete BaseAgentFactory for testing."""

    def make_agent(self) -> Any:
        return AgentFactory(MagicMock())


def _make_agent_factory() -> MagicMock:
    """Return a MagicMock standing in for a BaseAgentFactory."""
    return MagicMock(spec=BaseAgentFactory)


def _make_factory(**overrides: Any) -> CachingAgentFactory:
    """Return a CachingAgentFactory instance for testing."""
    kwargs = {
        "agent_factory": _make_agent_factory(),
        "cache_dir": None,
        "key_fn": None,
        "ignore_none": False,
    }
    kwargs.update(overrides)
    return CachingAgentFactory(**kwargs)


#########################################
#     Tests for CachingAgentFactory     #
#########################################


# --- Inheritance ---


def test_caching_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(_make_factory(), BaseAgentFactory)


# --- __init__ stores args ---


def test_caching_agent_factory_stores_agent_factory() -> None:
    agent_factory = _make_agent_factory()
    factory = _make_factory(agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


def test_caching_agent_factory_stores_cache_dir(tmp_path: Path) -> None:
    factory = _make_factory(cache_dir=tmp_path)
    assert factory._cache_dir == tmp_path


def test_caching_agent_factory_default_cache_dir_is_none() -> None:
    factory = _make_factory()
    assert factory._cache_dir is None


def test_caching_agent_factory_stores_key_fn() -> None:
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(key_fn=key_fn)
    assert factory._key_fn is key_fn


def test_caching_agent_factory_default_ignore_none_is_false() -> None:
    factory = _make_factory()
    assert factory._ignore_none is False


def test_caching_agent_factory_stores_ignore_none_true() -> None:
    factory = _make_factory(ignore_none=True)
    assert factory._ignore_none is True


# --- __init__ resolves agent_factory ---


def test_caching_agent_factory_resolves_agent_factory_from_dict() -> None:
    factory = _make_factory(agent_factory={"_target_": MINIMAL_AGENT_FACTORY_TARGET})
    assert isinstance(factory._agent_factory, MinimalAgentFactory)


def test_caching_agent_factory_resolves_agent_factory_from_base_config() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_AGENT_FACTORY_TARGET)
    factory = _make_factory(agent_factory=config)
    assert isinstance(factory._agent_factory, MinimalAgentFactory)


def test_caching_agent_factory_invalid_agent_factory_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseAgentFactory instance"):
        _make_factory(agent_factory="not-an-agent-factory")


# --- make_agent wiring ---


def test_caching_agent_factory_make_agent_builds_agent_from_factory() -> None:
    agent_factory = _make_agent_factory()
    factory = _make_factory(agent_factory=agent_factory)
    with patch(f"{MODULE}.CachingRunnable"):
        factory.make_agent()
        agent_factory.make_agent.assert_called_once_with()


def test_caching_agent_factory_make_agent_wraps_in_caching_runnable(tmp_path: Path) -> None:
    agent_factory = _make_agent_factory()
    cache_dir = tmp_path
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(
        agent_factory=agent_factory, cache_dir=cache_dir, key_fn=key_fn, ignore_none=True
    )
    with patch(f"{MODULE}.CachingRunnable") as mock_caching_runnable_cls:
        factory.make_agent()
        mock_caching_runnable_cls.assert_called_once_with(
            runnable=agent_factory.make_agent.return_value,
            cache_dir=cache_dir,
            key_fn=key_fn,
            ignore_none=True,
        )


def test_caching_agent_factory_make_agent_returns_caching_runnable() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.CachingRunnable") as mock_caching_runnable_cls:
        result = factory.make_agent()
        assert result is mock_caching_runnable_cls.return_value


# --- _get_repr_kwargs ---


def test_caching_agent_factory_get_repr_kwargs(tmp_path: Path) -> None:
    agent_factory = _make_agent_factory()
    cache_dir = tmp_path
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(
        agent_factory=agent_factory, cache_dir=cache_dir, key_fn=key_fn, ignore_none=True
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "agent_factory": agent_factory,
            "cache_dir": cache_dir,
            "key_fn": key_fn,
            "ignore_none": True,
        },
    )


# --- __repr__ and __str__ ---


def test_caching_agent_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("CachingAgentFactory(")


def test_caching_agent_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("CachingAgentFactory(")
