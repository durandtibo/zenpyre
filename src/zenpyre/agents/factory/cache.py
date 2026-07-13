r"""Provide a concrete agent factory that wraps another agent factory
with caching."""

from __future__ import annotations

__all__ = ["CachingAgentFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin

from zenpyre.agents.factory.base import BaseAgentFactory
from zenpyre.runnables import CachingRunnable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.runnables import Runnable


class CachingAgentFactory(BaseAgentFactory, MultilineDisplayMixin):
    """A concrete agent factory that wraps another agent factory and
    caches the resulting agent's outputs via
    :class:`~zenpyre.runnables.CachingRunnable`.

    Each call to :meth:`make_agent` builds a fresh agent from
    ``agent_factory`` and wraps it in a
    :class:`~zenpyre.runnables.CachingRunnable`, so repeated calls to
    the wrapped agent with the same (or equivalent, per ``key_fn``)
    input are served from the cache under ``cache_dir`` instead of
    re-invoking the underlying agent.

    Args:
        agent_factory: The factory used to build the underlying agent
            to cache.
        cache_dir: The directory used to persist cached results. If
            ``None``, caching behavior falls back to
            :class:`~zenpyre.runnables.CachingRunnable`'s default
            (e.g. in-memory only, depending on its implementation).
        key_fn: An optional function used to compute a cache key from
            the agent's input. If ``None``,
            :class:`~zenpyre.runnables.CachingRunnable`'s default
            key-computation strategy is used.
        ignore_none: If ``True``, ``None`` values are excluded when
            computing the cache key. Defaults to ``False``.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from zenpyre.agents import AgentChatModel
        >>> from zenpyre.agents.factory import AgentFactory, CachingAgentFactory
        >>> inner_agent = AgentChatModel(model=FakeListChatModel(responses=["hello"]))
        >>> factory = CachingAgentFactory(
        ...     agent_factory=AgentFactory(inner_agent),
        ...     cache_dir=Path("/tmp/my_app/cache"),  # doctest: +SKIP
        ... )
        >>> agent = factory.make_agent()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        agent_factory: BaseAgentFactory,
        cache_dir: Path | str | None = None,
        key_fn: Callable[[dict[str, Any]], str] | None = None,
        ignore_none: bool = False,
    ) -> None:
        self._agent_factory = agent_factory
        self._cache_dir = cache_dir
        self._key_fn = key_fn
        self._ignore_none = ignore_none

    def make_agent(self) -> Runnable[dict[str, Any], dict[str, Any]]:
        agent = self._agent_factory.make_agent()
        return CachingRunnable(
            runnable=agent,
            cache_dir=self._cache_dir,
            key_fn=self._key_fn,
            ignore_none=self._ignore_none,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "agent_factory": self._agent_factory,
            "cache_dir": self._cache_dir,
            "key_fn": self._key_fn,
            "ignore_none": self._ignore_none,
        }
