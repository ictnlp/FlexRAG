from abc import ABC, abstractmethod

from librarian.retriever import RetrievedContext
from librarian.utils import Register


class RefinerBase(ABC):
    @abstractmethod
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        return


REFINERS = Register("refiner")
