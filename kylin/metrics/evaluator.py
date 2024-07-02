import logging
from dataclasses import dataclass, field

from kylin.utils import Choices

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
from .retrieval_metrics import SuccessRate, SuccessRateConfig


logger = logging.getLogger(__name__)


@dataclass
class ShortFormEvaluatorConfig:
    short_form_metrics: list[Choices(["f1", "recall", "em", "precision", "accuracy"])] = field(default_factory=list)  # type: ignore
    f1_config: F1Config = field(default_factory=F1Config)
    recall_config: RecallConfig = field(default_factory=RecallConfig)
    em_config: ExactMatchConfig = field(default_factory=ExactMatchConfig)
    precision_config: PrecisionConfig = field(default_factory=PrecisionConfig)
    accuracy_config: AccuracyConfig = field(default_factory=AccuracyConfig)


class ShortFormEvaluator:
    def __init__(self, cfg: ShortFormEvaluatorConfig) -> None:
        self.metrics = {}
        for metric in cfg.short_form_metrics:
            match metric:
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
                case _:
                    raise ValueError(f"Invalid metric type: {metric}")
        return

    def evaluate(
        self,
        trues: list[list[str]],
        preds: list[str],
        log: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        evaluation_results = {}
        evaluation_details = {}
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](trues, preds)
            if log:
                logger.info(f"{metric}: {r}")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details


@dataclass
class LongFormEvaluatorConfig:
    long_form_metrics: list[Choices(["bleu", "chrf", "rouge-1", "rouge-2", "rouge-l"])] = field(default_factory=list)  # type: ignore
    bleu_config: BLEUConfig = field(default_factory=BLEUConfig)
    chrf_config: chrFConfig = field(default_factory=chrFConfig)
    rouge_config: RougeConfig = field(default_factory=RougeConfig)


class LongFormEvaluator:
    def __init__(self, cfg: LongFormEvaluatorConfig) -> None:
        self.metrics = {}
        for metric in cfg.long_form_metrics:
            match metric:
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
        return

    def evaluate(
        self,
        trues: list[list[str]],
        preds: list[str],
        log: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        evaluation_results = {}
        evaluation_details = {}
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](trues, preds)
            if log:
                logger.info(f"{metric}: {r}")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details


@dataclass
class RetrievalEvaluatorConfig:
    retrieval_metrics: list[Choices(["success_rate"])] = field(default_factory=list)  # type: ignore
    success_config: SuccessRateConfig = field(default_factory=SuccessRateConfig)


class RetrievalEvaluator:
    def __init__(self, cfg: RetrievalEvaluatorConfig) -> None:
        self.metrics = {}
        for metric in cfg.retrieval_metrics:
            match metric:
                case "success_rate":
                    self.metrics[metric] = SuccessRate(cfg.success_config)
                case _:
                    raise ValueError(f"Invalid metric type: {metric}")
        return

    def evaluate(
        self,
        evidences: list[list[str]],
        retrieved: list[list[str]],
        log: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        evaluation_results = {}
        evaluation_details = {}
        for metric in self.metrics:
            metric = str(metric)  # make json serializable
            r, r_detail = self.metrics[metric](evidences, retrieved)
            if log:
                logger.info(f"{metric}: {r}")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details
