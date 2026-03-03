from collections.abc import MutableMapping
from typing import Any, Protocol, TypeAlias, runtime_checkable

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.common.dataclasses import RetrievedContext

from .metrics_base import METRICS, MetricsBase

logger = LOGGER_MANAGER.get_logger("flexrag.metrics")
MetricConfig = METRICS.make_config(allow_multiple=True)


@configure
class EvaluatorConfig(MetricConfig):
    round: int = 2


MetricReturn: TypeAlias = tuple[dict[str, float], dict[str, Any]]
SimpleMetricReturn: TypeAlias = float | int | MetricReturn


@runtime_checkable
class MetricCallable(Protocol):
    def __call__(
        self,
        *,
        questions: list[str] | None = None,
        responses: list[str] | None = None,
        golden_responses: list[list[str]] | None = None,
        retrieved_contexts: list[list[Any]] | None = None,
        golden_contexts: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> SimpleMetricReturn: ...


class Evaluator(MutableMapping[str, MetricCallable]):
    """Evaluator is a container and orchestrator for multiple evaluation metrics.

    It manages a collection of metrics invoked by the configuration or added dynamically,
    and provides a unified interface to compute these metrics on the given data.

    Besides using the built-in metrics initialized via `EvaluatorConfig`, it also
    accepts custom functions as long as they conform to the `MetricCallable` protocol.

    Example:
        ```python
        evaluator = Evaluator({})

        # Custom metric functions MUST accept `**kwargs` to ignore unrelated arguments
        def exact_match(responses: list[str], golden_responses: list[list[str]], **kwargs):
            score = sum(r == g[0] for r, g in zip(responses, golden_responses)) / len(responses)
            return {"exact_match": score}, {}

        evaluator["em"] = exact_match
        results, details = evaluator.evaluate(responses=[...], golden_responses=[...])
        ```
    """

    def __init__(self, cfg: EvaluatorConfig | dict[str, MetricCallable]) -> None:
        self.metrics: dict[str, MetricCallable] = {}
        if isinstance(cfg, EvaluatorConfig):
            for name, metric in zip(cfg.metrics_type, METRICS.load(cfg)):
                self.metrics[name] = metric
            self.round = cfg.round
        else:
            for name, metric in cfg.items():
                assert isinstance(
                    metric, MetricCallable
                ), f"Metric {name} must implement the MetricCallable protocol."
            self.metrics = {name: metric for name, metric in cfg.items()}
            self.round = 2
        return

    def evaluate(
        self,
        *,
        questions: list[str] = None,
        responses: list[str] = None,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | RetrievedContext]] = None,
        golden_contexts: list[list[str]] = None,
        log: bool = True,
    ):
        """Evaluate the generated responses against the ground truth responses.

        :param questions: A list of questions. Defaults to None.
        :param responses: A list of responses. Defaults to None.
        :param golden_responses: A list of golden responses. Defaults to None.
        :param retrieved_contexts: A list of retrieved contexts. Defaults to None.
        :param golden_contexts: A list of golden contexts. Defaults to None.
        :param log: Whether to log the evaluation results. Defaults to True.
        :type questions: list[str], optional
        :type responses: list[str], optional
        :type golden_responses: list[list[str]], optional
        :type retrieved_contexts: list[list[str | RetrievedContext]], optional
        :type golden_contexts: list[list[str]], optional
        :type log: bool, optional
        :return: The evaluation results and the evaluation details.
        :rtype: tuple[dict[str, float], dict[str, Any]]
        """
        # check the input arguments
        not_none_args = [
            arg
            for arg in [
                questions,
                responses,
                golden_responses,
                retrieved_contexts,
                golden_contexts,
            ]
            if arg is not None
        ]
        assert len(not_none_args) > 1, "At least one argument must be provided."
        assert all(
            len(i) == len(not_none_args[0]) for i in not_none_args
        ), "All arguments must have the same length."

        # evaluate
        evaluation_results = {}
        evaluation_details = {}
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](
                questions=questions,
                responses=responses,
                golden_responses=golden_responses,
                retrieved_contexts=retrieved_contexts,
                golden_contexts=golden_contexts,
            )
            if log:
                for name, score in r.items():
                    logger.info(f"{name}: {score*100:.{self.round}f}%")
            evaluation_results.update(r)
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details

    async def async_evaluate(self):
        raise NotImplementedError("Async evaluation is not implemented yet.")

    def __getitem__(self, key: str) -> MetricCallable:
        return self.metrics[key]

    def __setitem__(self, key: str, value: MetricCallable):
        assert isinstance(
            value, MetricCallable
        ), f"Metric {key} must implement the MetricCallable protocol."
        self.metrics[key] = value
        return

    def __delitem__(self, key: str):
        del self.metrics[key]
        return

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __repr__(self) -> str:
        return f"Evaluator(metrics={list(self.metrics.keys())})"
