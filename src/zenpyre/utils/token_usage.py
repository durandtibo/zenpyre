r"""Contain utilities to compute token usage."""

from __future__ import annotations

__all__ = ["get_batch_token_usage", "get_invoke_token_usage"]

from typing import Any

from langchain_core.messages import AIMessage, UsageMetadata


def get_invoke_token_usage(result: dict[str, Any]) -> UsageMetadata:
    """Sum token usage across all AI messages produced during an agent
    run.

    Model-agnostic: relies only on the standard ``usage_metadata`` field
    that LangChain populates on ``AIMessage`` objects, regardless of the
    underlying provider (OpenAI, Anthropic, Ollama, etc.). If a given
    message has no usage metadata (e.g. the provider didn't report it),
    it is silently skipped.

    ``total_tokens`` is summed independently from ``input_tokens`` and
    ``output_tokens`` rather than derived from them, since some
    providers may report additional token categories (e.g. cached or
    reasoning tokens) that make ``total_tokens`` differ from their
    simple sum.

    Args:
        result: The dict returned by ``agent.invoke(...)``, expected to
            contain a ``"messages"`` key with the full message list.

    Returns:
        A ``UsageMetadata`` dict with ``input_tokens``, ``output_tokens``,
            and ``total_tokens`` summed across all AI messages found.
    """
    total_input = 0
    total_output = 0
    total_tokens = 0

    for msg in result.get("messages", []):
        if not isinstance(msg, AIMessage):
            continue
        usage = msg.usage_metadata
        if not usage:
            continue
        total_input += usage.get("input_tokens", 0)
        total_output += usage.get("output_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    return UsageMetadata(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
    )


def get_batch_token_usage(results: list[dict[str, Any]]) -> UsageMetadata:
    """Sum token usage across all results from an ``agent.batch(...)``
    call.

    Delegates to :func:`get_invoke_token_usage` for each individual
    result, then sums the per-result totals across the batch.

    Args:
        results: The list of dicts returned by ``agent.batch(...)``, one
            per input in the batch.

    Returns:
        A ``UsageMetadata`` dict with token counts summed across every
            run in the batch.
    """
    total_input = 0
    total_output = 0
    total_tokens = 0

    for result in results:
        usage = get_invoke_token_usage(result)
        total_input += usage["input_tokens"]
        total_output += usage["output_tokens"]
        total_tokens += usage["total_tokens"]

    return UsageMetadata(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
    )
