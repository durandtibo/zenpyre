from __future__ import annotations

__all__ = ["TrackingRunnable"]

from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


class TrackingRunnable(Runnable[str, str | None]):
    """A runnable that records how it was called (invoke vs.

    batch vs. ainvoke vs. abatch), so tests can verify CachingRunnable
    calls the wrapped runnable's batch/abatch for cache misses instead
    of looping invoke/ainvoke per miss.
    """

    def __init__(self, none_on: str | None = None) -> None:
        self.none_on = none_on
        self.invoke_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.ainvoke_calls: list[str] = []
        self.abatch_calls: list[list[str]] = []

    def _apply(self, x: str) -> str | None:
        return None if x == self.none_on else x.upper()

    def invoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> str | None:
        self.invoke_calls.append(input)
        return self._apply(input)

    def batch(
        self,
        inputs: list[str],
        config: RunnableConfig | list[RunnableConfig] | None = None,  # noqa: ARG002
        *,
        return_exceptions: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str | None]:
        self.batch_calls.append(list(inputs))
        return [self._apply(x) for x in inputs]

    async def ainvoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> str | None:
        self.ainvoke_calls.append(input)
        return self._apply(input)

    async def abatch(
        self,
        inputs: list[str],
        config: RunnableConfig | list[RunnableConfig] | None = None,  # noqa: ARG002
        *,
        return_exceptions: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str | None]:
        self.abatch_calls.append(list(inputs))
        return [self._apply(x) for x in inputs]
