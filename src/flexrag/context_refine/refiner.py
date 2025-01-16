from abc import ABC, abstractmethod

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import Register


class RefinerBase(ABC):
    @abstractmethod
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        return


REFINERS = Register("refiner")
