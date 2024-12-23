from dataclasses import dataclass, field

from librarian.data import TextProcessPipeline, TextProcessPipelineConfig
from librarian.retriever import RetrievedContext
from librarian.utils import LOGGER_MANAGER

from .metrics_base import METRICS

logger = LOGGER_MANAGER.get_logger("librarian.metrics")
MetricConfig = METRICS.make_config(allow_multiple=True)


@dataclass
class RAGEvaluatorConfig(MetricConfig):
    round: int = 2
    response_preprocess: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore


class RAGEvaluator:
    def __init__(self, cfg: RAGEvaluatorConfig) -> None:
        self.metrics = {
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

        Args:
            questions (list[str]): A list contains the questions.
            responses (list[str]): A list contains the generated responses.
            golden_responses (list[list[str]]): A list contains the golden responses for each question.
            retrieved_contexts (list[list[str | RetrievedContext]]): A list contains the retrieved contexts.
            golden_contexts (list[list[str | | RetrievedContext]]): A list contains the golden contexts.
            log (bool, optional): Whether to log the evaluation results. Defaults to True.

        Returns:
            eval_result (dict[str, float]): A dict contains the evaluation results.
            eval_details (dict[str, Any]): A dict contains the evaluation details.
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
        responses = [self.response_pipeline(res) for res in responses]
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
                logger.info(f"{metric}: {r*100:.{self.round}f}%")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details
