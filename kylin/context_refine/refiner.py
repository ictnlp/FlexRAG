from abc import ABC, abstractmethod

from kylin.retriever import RetrievedContext
from kylin.utils import Register


class RefinerBase(ABC):
    @abstractmethod
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        return


REFINERS = Register("refiner")
