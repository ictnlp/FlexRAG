from __future__ import annotations

from flexrag.common.dataclasses import RetrievedContext

from .base import TypedHandle


class RefinerHandle(TypedHandle):
    """Typed proxy for refiner resources."""

    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Refine retrieved contexts.

        :param contexts: Retrieved contexts to refine.
        :returns: Refined contexts returned by the raw refiner.
        """
        return self._target.call("refine", contexts)

    async def async_refine(
        self, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """Asynchronously refine retrieved contexts.

        :param contexts: Retrieved contexts to refine.
        :returns: Refined contexts returned by the raw refiner.
        """
        return await self._target.async_call("async_refine", contexts)
