from __future__ import annotations

from flexrag.processors.chunkers import Chunk

from .base import TypedHandle


class ChunkerHandle(TypedHandle):
    """Typed proxy for chunker resources."""

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        :param text: Text to split.
        :returns: Chunks produced by the raw chunker.
        """
        return self._target.call("chunk", text)
