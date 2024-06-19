from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from .metrics_base import MetricsBase


class RetrievalMetric(MetricsBase):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--relevance_check",
            type=str,
            default="contain",
            choices=["contain"],
            help="The method to check the relevance of the retrieved documents.",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.relevance_check = args.relevance_check
        return

    def __call__(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> dict[str, float]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            relevant (list[list[str]]): The relevant documents.
            retrieved (list[list[str]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        assert len(evidences) == len(
            retrieved
        ), "The length of y_true and y_pred should be the same"
        evidences = [[self.preprocess_text(y_) for y_ in y] for y in evidences]
        retrieved = [[self.preprocess_text(y_) for y_ in y] for y in retrieved]
        return self.compute(evidences, retrieved)

    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, object]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[str]]): The evidence documents.
            retrieved (list[list[str]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        relevance_map: list[list[list[bool]]] = []
        for evs, rets in zip(evidences, retrieved):
            relevance_map.append([])
            for ret in rets:
                relevance_map[-1].append([self.is_relevance(ev, ret) for ev in evs])
        return self._get_score(relevance_map), relevance_map

    @abstractmethod
    def _get_score(self, relevance_map: list[list[list[bool]]]) -> float:
        return

    def is_relevance(self, evidence: str, retrieved: str):
        if self.relevance_check == "contain":
            if evidence in retrieved:
                return True
            return False
        else:
            raise NotImplementedError


class SuccessRate(RetrievalMetric):
    def _get_score(self, relevance_map: list[list[list[bool]]]) -> float:
        scores = 0.0
        for line in relevance_map:
            for ret_line in line:
                if any(ret_line):
                    scores += 1.0
                    break
        return scores / len(relevance_map)


# class RetrievalPrecision(RetrievalMetric):
#     def _get_score(self, relevance_map: list[list[list[bool]]]) -> float:
#         scores = []
#         for line in relevance_map:
#             scores.append(sum(line) / len(line))
#         return sum(scores) / len(scores)
