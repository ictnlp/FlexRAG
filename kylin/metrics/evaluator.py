from dataclasses import dataclass, field
from typing import Any

from kylin.retriever import RetrievedContext
from kylin.text_process import PipelineConfig, Pipeline
from kylin.utils import Choices, Optional, LOGGER_MANAGER

from .matching_metrics import (
    F1,
    F1Config,
    Accuracy,
    AccuracyConfig,
    ExactMatch,
    ExactMatchConfig,
    Precision,
    PrecisionConfig,
    Recall,
    RecallConfig,
)
from .generation_metrics import (
    BLEU,
    BLEUConfig,
    chrF,
    chrFConfig,
    Rouge1,
    Rouge2,
    RougeL,
    RougeConfig,
)
from .retrieval_metrics import (
    SuccessRate,
    SuccessRateConfig,
    RetrievalPrecision,
    RetrievalPrecisionConfig,
)


logger = LOGGER_MANAGER.get_logger("kylin.metrics")


@dataclass
class ResponseEvaluatorConfig:
    response_metrics: list[
        Choices(  # type: ignore
            [
                "f1",
                "recall",
                "em",
                "precision",
                "accuracy",
                "bleu",
                "chrf",
                "rouge-1",
                "rouge-2",
                "rouge-l",
            ]
        )
    ] = field(default_factory=list)
    f1_config: F1Config = field(default_factory=F1Config)
    recall_config: RecallConfig = field(default_factory=RecallConfig)
    em_config: ExactMatchConfig = field(default_factory=ExactMatchConfig)
    precision_config: PrecisionConfig = field(default_factory=PrecisionConfig)
    accuracy_config: AccuracyConfig = field(default_factory=AccuracyConfig)
    bleu_config: BLEUConfig = field(default_factory=BLEUConfig)
    chrf_config: chrFConfig = field(default_factory=chrFConfig)
    rouge_config: RougeConfig = field(default_factory=RougeConfig)
    round: int = 2
    answer_preprocess_pipeline: PipelineConfig = field(default_factory=PipelineConfig)  # type: ignore


class ResponseEvaluator:
    def __init__(self, cfg: ResponseEvaluatorConfig) -> None:
        self.metrics = {}
        for metric in cfg.response_metrics:
            match metric:
                # shortform metrics
                case "f1":
                    self.metrics[metric] = F1(cfg.f1_config)
                case "recall":
                    self.metrics[metric] = Recall(cfg.recall_config)
                case "em":
                    self.metrics[metric] = ExactMatch(cfg.em_config)
                case "precision":
                    self.metrics[metric] = Precision(cfg.precision_config)
                case "accuracy":
                    self.metrics[metric] = Accuracy(cfg.accuracy_config)
                # longform metrics
                case "bleu":
                    self.metrics[metric] = BLEU(cfg.bleu_config)
                case "chrf":
                    self.metrics[metric] = chrF(cfg.chrf_config)
                case "rouge-1":
                    self.metrics[metric] = Rouge1(cfg.rouge_config)
                case "rouge-2":
                    self.metrics[metric] = Rouge2(cfg.rouge_config)
                case "rouge-l":
                    self.metrics[metric] = RougeL(cfg.rouge_config)
                case _:
                    raise ValueError(f"Invalid metric type: {metric}")
        self.round = cfg.round
        self.preprocess_pipeline = Pipeline(cfg.answer_preprocess_pipeline)
        return

    def evaluate(
        self,
        trues: list[list[str]],
        preds: list[str],
        log: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Evaluate the generated responses against the ground truth responses.

        Args:
            trues (list[list[str]]): A list contains ground truth responses for each sample.
            preds (list[str]): A list contains generated responses.
            log (bool, optional): Whether to log the evaluation results. Defaults to True.

        Returns:
            tuple[dict[str, float], dict[str, object]]: A tuple contains the evaluation results and details.
        """
        evaluation_results = {}
        evaluation_details = {}
        preds = [self.preprocess_pipeline(p) for p in preds]
        trues = [[self.preprocess_pipeline(t_) for t_ in t] for t in trues]
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](trues, preds)
            if log:
                logger.info(f"{metric}: {r*100:.{self.round}f}%")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details


@dataclass
class RetrievalEvaluatorConfig:
    retrieval_metrics: list[
        Choices(  # type: ignore
            [
                "success_rate",
                "precision",
            ]
        )
    ] = field(default_factory=list)
    success_config: SuccessRateConfig = field(default_factory=SuccessRateConfig)
    precision_config: RetrievalPrecisionConfig = field(default_factory=RetrievalPrecisionConfig)  # fmt: skip
    round: int = 2
    text_preprocess_pipeline: PipelineConfig = field(default_factory=PipelineConfig)  # type: ignore
    evaluate_field: Optional[str] = None


class RetrievalEvaluator:
    def __init__(self, cfg: RetrievalEvaluatorConfig) -> None:
        self.metrics = {}
        for metric in cfg.retrieval_metrics:
            match metric:
                case "success_rate":
                    self.metrics[metric] = SuccessRate(cfg.success_config)
                case "precision":
                    self.metrics[metric] = RetrievalPrecision(cfg.precision_config)
                case _:
                    raise ValueError(f"Invalid metric type: {metric}")
        self.round = cfg.round
        self.preprocess_pipeline = Pipeline(cfg.text_preprocess_pipeline)
        self.eval_field = cfg.evaluate_field
        return

    def evaluate(
        self,
        evidences: list[list[Any | RetrievedContext]],
        retrieved: list[list[Any | RetrievedContext]],
        log: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        evaluation_results = {}
        evaluation_details = {}

        if (self.eval_field is not None) and isinstance(
            evidences[0][0], RetrievedContext
        ):
            evidences_ = [[ctx.data[self.eval_field] for ctx in e] for e in evidences]
        else:
            evidences_ = evidences
        if (self.eval_field is not None) and isinstance(
            retrieved[0][0], RetrievedContext
        ):
            retrieved_ = [[ctx.data[self.eval_field] for ctx in r] for r in retrieved]
        else:
            retrieved_ = retrieved

        evidences_ = [[self.preprocess_pipeline(y_) for y_ in y] for y in evidences_]
        retrieved_ = [[self.preprocess_pipeline(y_) for y_ in y] for y in retrieved_]
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](evidences_, retrieved_)
            if log:
                logger.info(f"{metric}: {r*100:.{self.round}f}%")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details
