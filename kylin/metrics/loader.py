from argparse import ArgumentParser, Namespace

from .matching_metrics import F1, ContainsMatch, ExactMatch, Precision, Recall
from .metrics_base import MetricsBase


def add_args_for_metrics(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["f1"],
        choices=["f1", "recall", "em", "precision", "contains"],
        help="The metrics to use",
    )
    parser = MetricsBase.add_args(parser)
    return parser


def load_metrics(args: Namespace) -> dict[str, MetricsBase]:
    metrics = {}
    for metric in args.metrics:
        match metric:
            case "f1":
                metrics[metric] = F1(args)
            case "recall":
                metrics[metric] = Recall(args)
            case "em":
                metrics[metric] = ExactMatch(args)
            case "precision":
                metrics[metric] = Precision(args)
            case "contains":
                metrics[metric] = ContainsMatch(args)
            case _:
                raise ValueError(f"Invalid metric type: {args.metrics}")
    return metrics
