from typing import Protocol

from flexrag.common import Register
from flexrag.common.dataclasses import RetrievedContext


class RefinerProtocol(Protocol):
    """Protocol for directly usable context refiners."""

    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Refine retrieved contexts.

        :param contexts: Retrieved contexts to refine.
        :return: Refined contexts.
        """
        ...

    async def async_refine(
        self, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """Refine retrieved contexts asynchronously.

        :param contexts: Retrieved contexts to refine.
        :return: Refined contexts.
        """
        ...


REFINERS = Register[RefinerProtocol]("refiner")
