from __future__ import annotations

from flexrag.processors.chunkers import Chunk

from .base import TypedHandle


class ChunkerHandle(TypedHandle):
    """Typed proxy for chunker resources."""

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        """Split text into chunks.

        :param text: Text to split.
        :param return_str: Whether to return plain chunk strings instead of
            ``Chunk`` objects.
        :returns: Chunks produced by the raw chunker.
        """
        return self._target.call("chunk", text, return_str=return_str)
