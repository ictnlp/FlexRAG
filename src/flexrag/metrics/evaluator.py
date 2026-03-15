import inspect
from collections.abc import MutableMapping
from typing import Any

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.common.dataclasses import RetrievedContext

from .metrics_base import METRICS, MetricCallable

logger = LOGGER_MANAGER.get_logger("flexrag.metrics")
MetricConfig = METRICS.make_config(allow_multiple=True)


@configure
class EvaluatorConfig(MetricConfig):
    round: int = 2


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
        log: bool = True,
        **kwargs: Any,
    ):
        """Evaluate the provided data using registered metrics.

        :param log: Whether to log the evaluation results. Defaults to True.
        :type log: bool, optional
        :param kwargs: Keyword arguments to be passed to the metric functions.
                       All list arguments must have the same length.
        :return: The evaluation results and the evaluation details.
        :rtype: tuple[dict[str, float], dict[str, Any]]
        """
        # check the input arguments
        list_args = [v for v in kwargs.values() if isinstance(v, list)]
        if not list_args:
            raise ValueError("At least one list argument must be provided.")
        lengths = {len(v) for v in list_args}
        assert len(lengths) == 1, "All list arguments must have the same length."

        # evaluate
        evaluation_results = {}
        evaluation_details = {}
        for name, metric in self.metrics.items():
            name = str(name)  # make json serializable

            # Use inspect.signature to find required params
            # metric might be an object with __call__ or a function
            callable_target = (
                metric
                if inspect.isfunction(metric)
                else getattr(metric, "__call__", metric)
            )
            sig = inspect.signature(callable_target)

            metric_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name in kwargs:
                    metric_kwargs[param_name] = kwargs[param_name]
                elif param.default == inspect.Parameter.empty and param.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    if param_name == "self":
                        continue
                    raise ValueError(
                        f"Metric '{name}' requires '{param_name}', but it was not provided."
                    )

            res = metric(**metric_kwargs)
            if isinstance(res, (float, int)):
                r = {name: float(res)}
                r_detail = {}
            else:
                r, r_detail = res

            if log:
                for metric_name, score in r.items():
                    logger.info(f"{metric_name}: {score:.{self.round}f}")
            evaluation_results.update(r)
            evaluation_details[name] = r_detail
        return evaluation_results, evaluation_details

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
