r"""Contain utilities to compute token usage."""

from __future__ import annotations

__all__ = [
    "format_token_usage",
    "get_batch_token_usage",
    "get_invoke_token_usage",
    "get_token_usage",
    "log_token_usage",
]

import logging
from typing import Any

from coola.iterator import dfs_iterate, filter_by_type
from langchain_core.messages import AIMessage, BaseMessage, UsageMetadata

logger: logging.Logger = logging.getLogger(__name__)


def format_token_usage(usage: UsageMetadata) -> str:
    """Format token usage as a human-readable string for terminal
    display.

    Args:
        usage: The token usage to format, as returned by
            :func:`~zenpyre.utils.tokens.get_invoke_token_usage` or
            :func:`~zenpyre.utils.tokens.get_batch_token_usage`.

    Returns:
        A multi-line, aligned string summarizing input, output, and
        total token counts, suitable for printing to a terminal.

    Example:
        ```pycon
        >>> from langchain_core.messages import UsageMetadata
        >>> from zenpyre.utils.token_usage import format_token_usage
        >>> usage = UsageMetadata(input_tokens=1234, output_tokens=567, total_tokens=1801)
        >>> print(format_token_usage(usage))
        Token usage
          Input tokens:  1,234
          Output tokens:   567
          Total tokens:  1,801

        ```
    """
    labels = ["Input tokens", "Output tokens", "Total tokens"]
    values = [
        f"{usage.get('input_tokens', 0):,}",
        f"{usage.get('output_tokens', 0):,}",
        f"{usage.get('total_tokens', 0):,}",
    ]

    # Compute both column widths programmatically so alignment holds
    # even if a label changes length.
    label_width = max(len(label) for label in labels)
    value_width = max(len(value) for value in values)

    lines = ["Token usage"]
    lines.extend(
        f"  {label + ':':<{label_width + 1}} {value:>{value_width}}"
        for label, value in zip(labels, values, strict=True)
    )
    return "\n".join(lines)


def _sum_usage(messages: list[AIMessage]) -> UsageMetadata:
    """Sum token usage across a collection of AI messages.

    Messages without ``usage_metadata`` (e.g. the provider didn't
    report it) are silently skipped.

    ``total_tokens`` is summed independently from ``input_tokens`` and
    ``output_tokens`` rather than derived from them, since some
    providers may report additional token categories (e.g. cached or
    reasoning tokens) that make ``total_tokens`` differ from their
    simple sum.

    Args:
        messages: The AI messages to sum usage over.

    Returns:
        A ``UsageMetadata`` dict with ``input_tokens``, ``output_tokens``,
            and ``total_tokens`` summed across all given messages.
    """
    total_input = 0
    total_output = 0
    total_tokens = 0

    for msg in messages:
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


def get_invoke_token_usage(result: dict[str, Any] | BaseMessage) -> UsageMetadata:
    """Sum token usage across all AI messages produced during an agent
    run.

    Model-agnostic: relies only on the standard ``usage_metadata`` field
    that LangChain populates on ``AIMessage`` objects, regardless of the
    underlying provider (OpenAI, Anthropic, Ollama, etc.). If a given
    message has no usage metadata (e.g. the provider didn't report it),
    it is silently skipped.

    Args:
        result: Either the dict returned by ``agent.invoke(...)``,
            expected to contain a ``"messages"`` key with the full
            message list, or a single ``BaseMessage`` (e.g. the direct
            return value of ``model.invoke(...)``).

    Returns:
        A ``UsageMetadata`` dict with ``input_tokens``, ``output_tokens``,
            and ``total_tokens`` summed across all AI messages found.
    """
    messages = [result] if isinstance(result, BaseMessage) else result.get("messages", [])
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    return _sum_usage(ai_messages)


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


def get_token_usage(result: Any) -> UsageMetadata:
    """Sum token usage across every AI message found anywhere within
    ``result``.

    Unlike :func:`get_invoke_token_usage` and :func:`get_batch_token_usage`,
    this function does not branch on the shape of ``result``. Instead it
    walks ``result`` depth-first (via ``coola``'s ``dfs_iterate``) and
    collects every ``AIMessage`` instance found, regardless of how deeply
    it is nested. This makes it a convenient, shape-agnostic entry point
    that works for a single ``BaseMessage``, the dict returned by
    ``agent.invoke(...)``, the list returned by ``agent.batch(...)``, or
    any other structure that contains ``AIMessage`` objects.

    If ``result`` contains no ``AIMessage`` instances (including cases
    where ``result`` is of an unrecognized type, e.g. an ``int`` or a
    plain ``str``), this function returns all-zero usage rather than
    raising an error.

    Args:
        result: Any object potentially containing ``AIMessage``
            instances, such as a ``BaseMessage``, the dict returned by
            ``agent.invoke(...)``, or the list returned by
            ``agent.batch(...)``.

    Returns:
        A ``UsageMetadata`` dict with token counts summed across every
            ``AIMessage`` found within ``result``.
    """
    ai_messages = list(filter_by_type(dfs_iterate(result), AIMessage))
    return _sum_usage(ai_messages)


def log_token_usage(result: Any) -> None:
    """Log the token usage for a single invocation or a batch of them.

    Args:
        result: The dict returned by ``agent.invoke(...)``, or the list
            of dicts returned by ``agent.batch(...)``.
    """
    usage = get_token_usage(result)
    logger.info(format_token_usage(usage))
