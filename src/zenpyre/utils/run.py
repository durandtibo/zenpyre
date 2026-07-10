r"""Contain utility functions to generate run identifiers."""

from __future__ import annotations

__all__ = ["generate_run_id"]

import uuid
from typing import TYPE_CHECKING, Any

from zenpyre.utils.hashing import hash_dict_uuid

if TYPE_CHECKING:
    from collections.abc import Mapping


def generate_run_id(config: Mapping[str, Any] | None = None) -> str:
    r"""Generate a unique identifier for a run.

    If a configuration is given, the identifier is derived
    deterministically from its content, so the same configuration
    always produces the same run ID. Otherwise, a random identifier
    is generated.

    Args:
        config: The configuration used to derive a deterministic
            run ID. If ``None``, a random run ID is generated.

    Returns:
        The generated run ID.

    Example:
        ```pycon
        >>> from zenpyre.utils.run import generate_run_id
        >>> generate_run_id()  # doctest: +SKIP
        '3f2504e0-4f89-11d3-9a0c-0305e82c3301'
        >>> generate_run_id({"lr": 0.01, "batch_size": 32})  # doctest: +SKIP
        'a94a8fe5-ccb1-4bb2-8bb1-3e8b79d7de35'

        ```
    """
    if config is None:
        return str(uuid.uuid4())
    return hash_dict_uuid(config)
