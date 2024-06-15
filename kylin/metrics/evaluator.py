import logging
from argparse import ArgumentParser, Namespace

from .matching_metrics import F1, Accuracy, ExactMatch, Precision, Recall
from .generation_metrics import BLEU, Rouge1, Rouge2, RougeL, chrF
from .metrics_base import MetricsBase


logger = logging.getLogger(__name__)


class ShortFormEvaluator:
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--short_form_metrics",
            type=str,
            nargs="+",
            default=["f1"],
            choices=["f1", "recall", "em", "precision", "accuracy"],
            help="The metrics to use",
        )
        parser = MetricsBase.add_args(parser)
        return parser

    def __init__(self, args: Namespace) -> None:
        self.metrics = {}
        for metric in args.short_form_metrics:
            match metric:
                case "f1":
                    self.metrics[metric] = F1(args)
                case "recall":
                    self.metrics[metric] = Recall(args)
                case "em":
                    self.metrics[metric] = ExactMatch(args)
                case "precision":
                    self.metrics[metric] = Precision(args)
                case "accuracy":
                    self.metrics[metric] = Accuracy(args)
                case _:
                    raise ValueError(f"Invalid metric type: {args.metrics}")
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
            r, r_detail = self.metrics[metric](trues, preds)
            if log:
                logger.info(f"{metric}: {r}")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details


class LongFormEvaluator:
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--long_form_metrics",
            type=str,
            nargs="+",
            default=["bleu"],
            choices=["bleu", "rouge-1", "rouge-2", "rouge-l", "chrf"],
            help="The metrics to use",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        self.metrics = {}
        for metric in args.long_form_metrics:
            match metric:
                case "bleu":
                    self.metrics[metric] = BLEU(args)
                case "rouge-1":
                    self.metrics[metric] = Rouge1(args)
                case "rouge-2":
                    self.metrics[metric] = Rouge2(args)
                case "rouge-l":
                    self.metrics[metric] = RougeL(args)
                case "chrf":
                    self.metrics[metric] = chrF(args)
                case _:
                    raise ValueError(f"Invalid metric type: {args.metrics}")
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
            r, r_detail = self.metrics[metric](trues, preds)
            if log:
                logger.info(f"{metric}: {r}")
            evaluation_results[metric] = r
            evaluation_details[metric] = r_detail
        return evaluation_results, evaluation_details


# TODO
class RetrievalEvaluator: ...

