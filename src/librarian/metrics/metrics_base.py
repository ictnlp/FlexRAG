from abc import ABC, abstractmethod

from librarian.utils import Register
from librarian.retriever import RetrievedContext


class MetricsBase(ABC):
    def __call__(
        self,
        questions: list[str] = None,
        responses: list[str] = None,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | RetrievedContext]] = None,
        golden_contexts: list[list[str]] = None,
    ) -> dict[str, float]:
        return self.compute(
            questions=questions,
            responses=responses,
            golden_responses=golden_responses,
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
        )

    @abstractmethod
    def compute(
        self,
        questions: list[str] = None,
        responses: list[str] = None,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | RetrievedContext]] = None,
        golden_contexts: list[list[str]] = None,
    ) -> tuple[float, object]:
        """
        Compute the metric value.

        Args:
            questions (list[str], optional): A list of questions. Defaults to None.
            responses (list[str], optional): A list of responses. Defaults to None.
            golden_responses (list[list[str]], optional): A list of golden responses. Defaults to None.
            retrieved_contexts (list[list[str | RetrievedContext]], optional): A list of retrieved contexts. Defaults to None.
            golden_contexts (list[list[str]], optional): A list of golden contexts. Defaults to None.

        Returns:
            score (float): The metric value.
            metadata (object): The metadata of the metric.
        """
        return


METRICS = Register[MetricsBase]("metrics")
