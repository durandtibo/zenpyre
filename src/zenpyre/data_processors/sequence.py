r"""Define a processor that applies another processor to each item in a
sequence."""

from __future__ import annotations

__all__ = ["SequenceProcessor"]

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from coola.display import MultilineDisplayMixin

from zenpyre.data_processors.base import BaseProcessor
from zenpyre.utils.rich import make_progressbar

U = TypeVar("U")
T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


class SequenceProcessor(BaseProcessor[Sequence[U], list[T]], MultilineDisplayMixin):
    """Processor that applies another processor to each item in a
    sequence and returns the list of results.

    Useful for applying any :class:`~zenpyre.data_processors.base.BaseProcessor`
    to every element of a sequence without writing a dedicated class.
    Unlike :class:`~zenpyre.data_processors.LambdaProcessor`, which
    applies ``processor`` to the whole input at once, this class maps
    ``processor`` over each item individually.

    Args:
        processor: A :class:`~zenpyre.data_processors.base.BaseProcessor`
            that accepts a single item of type ``U`` and returns a
            value of type ``T``.
        progress_description: The description shown on the progress bar
            while processing.  Defaults to ``"Processing items..."``.

    Example:
        ```pycon
        >>> from zenpyre.data_processors import LambdaProcessor, SequenceProcessor
        >>> processor = SequenceProcessor(processor=LambdaProcessor(fn=str.upper))
        >>> processor.process(["hello", "world"])
        ['HELLO', 'WORLD']
        >>> processor = SequenceProcessor(processor=LambdaProcessor(fn=len))
        >>> processor.process(["a", "bb", "ccc"])
        [1, 2, 3]

        ```
    """

    def __init__(
        self, processor: BaseProcessor[U, T], progress_description: str = "Processing items..."
    ) -> None:
        self._processor = processor
        self._progress_description = progress_description

    def process(self, data: Sequence[U]) -> list[T]:
        """Apply ``processor`` to each item in ``data`` and return the
        results.

        Args:
            data: The sequence of items to process.

        Returns:
            A list of results in the same order as ``data``, where each
            element is the output of ``processor.process`` applied to
            the corresponding input item.
        """
        results = []

        with make_progressbar() as progress:
            task = progress.add_task(self._progress_description, total=len(data))
            for item in data:
                results.append(self._processor.process(item))
                progress.advance(task)

        logger.info("Processed %s item(s).", f"{len(results):,}")
        return results

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"processor": self._processor, "progress_description": self._progress_description}
