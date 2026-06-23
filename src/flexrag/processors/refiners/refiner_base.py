from abc import ABC, abstractmethod
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


class RefinerBase(ABC):
    """The base class for context refiners.
    The subclasses should implement the ``refine`` method.
    """

    @abstractmethod
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Refine the contexts.

        :param contexts: The retrieved contexts to refine.
        :type contexts: list[RetrievedContext]
        :return: The refined contexts.
        :rtype: list[RetrievedContext]
        """
        return


REFINERS = Register[RefinerProtocol]("refiner")
