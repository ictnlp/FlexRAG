from dataclasses import dataclass, field

from flexrag.common_dataclass import RetrievedContext
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER

from .metrics_base import METRICS, MetricsBase

logger = LOGGER_MANAGER.get_logger("flexrag.metrics")
MetricConfig = METRICS.make_config(allow_multiple=True)


@dataclass
class EvaluatorConfig(MetricConfig):
    round: int = 2
    response_preprocess: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig) -> None:
        self.metrics: dict[str, MetricsBase] = {
            name: metric for name, metric in zip(cfg.metrics_type, METRICS.load(cfg))
        }
        self.response_pipeline = TextProcessPipeline(cfg.response_preprocess)
        self.round = cfg.round
        return

    def evaluate(
        self,
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
        if responses is not None:
            responses = [self.response_pipeline(res) for res in responses]
        if golden_responses is not None:
            golden_responses = [
                [self.response_pipeline(g) for g in golds] for golds in golden_responses
            ]
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
