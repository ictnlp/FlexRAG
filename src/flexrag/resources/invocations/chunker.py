from typing import Any

from flexrag.processors.chunkers import Chunk


class ChunkerInvocation:
    """Invocation semantics for managed chunkers."""

    def __init__(self, runtime: Any) -> None:
        """Create a chunker invocation.

        :param runtime: Runtime adapter used to execute chunker calls.
        """
        self.runtime = runtime
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        """Split text into chunks.

        :param text: Text to split.
        :param return_str: Whether to return chunk strings instead of chunk
            objects.
        :return: Chunk objects or chunk strings.
        """
        return self.runtime.call("chunk", text, return_str=return_str)
