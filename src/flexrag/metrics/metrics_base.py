from abc import ABC, abstractmethod

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import Register


class MetricsBase(ABC):
    def __call__(
        self,
        questions: list[str] = None,
        responses: list[str] = None,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | RetrievedContext]] = None,
        golden_contexts: list[list[str]] = None,
    ) -> tuple[dict[str, float], dict]:
        """
        Compute the metric value.

        :param questions: A list of questions. Defaults to None.
        :param responses: A list of responses. Defaults to None.
        :param golden_responses: A list of golden responses. Defaults to None.
        :param retrieved_contexts: A list of retrieved contexts. Defaults to None.
        :param golden_contexts: A list of golden contexts. Defaults to None.
        :type questions: list[str], optional
        :type responses: list[str], optional
        :type golden_responses: list[list[str]], optional
        :type retrieved_contexts: list[list[str | RetrievedContext]], optional
        :type golden_contexts: list[list[str]], optional
        :return: The metric scores and the metadata of the metric.
        :rtype: tuple[dict[str, float], dict]
        """
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
    ) -> tuple[dict[str, float], dict]:
        """
        Compute the metric value.

        :param questions: A list of questions. Defaults to None.
        :param responses: A list of responses. Defaults to None.
        :param golden_responses: A list of golden responses. Defaults to None.
        :param retrieved_contexts: A list of retrieved contexts. Defaults to None.
        :param golden_contexts: A list of golden contexts. Defaults to None.
        :type questions: list[str], optional
        :type responses: list[str], optional
        :type golden_responses: list[list[str]], optional
        :type retrieved_contexts: list[list[str | RetrievedContext]], optional
        :type golden_contexts: list[list[str]], optional
        :return: The metric scores and the metadata of the metric.
        :rtype: tuple[dict[str, float], dict]
        """
        return


METRICS = Register[MetricsBase]("metrics")
