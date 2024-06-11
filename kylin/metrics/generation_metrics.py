from argparse import ArgumentParser, Namespace

import sacrebleu
import rouge

from .metrics_base import MetricsBase


class BLEU(MetricsBase):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--bleu_tokenizer",
            type=str,
            default=sacrebleu.BLEU.TOKENIZER_DEFAULT,
            choices=sacrebleu.BLEU.TOKENIZERS,
            help="The tokenizer to use for BLEU",
        )
        return parser

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.tokenizer = args.bleu_tokenizer
        return

    def compute(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[float, dict[str, float]]:
        bleu = sacrebleu.corpus_bleu(
            hypotheses=y_preds,
            references=y_trues,
            tokenize=self.tokenizer,
        )
        return bleu.score, vars(bleu)


class chrF(MetricsBase):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--chrf_beta",
            type=float,
            default=1.0,
            help="The beta value for chrF",
        )
        parser.add_argument(
            "--chrf_char_order",
            type=int,
            default=sacrebleu.CHRF.CHAR_ORDER,
            help="Character n-gram order.",
        )
        parser.add_argument(
            "--chrf_word_order",
            type=int,
            default=sacrebleu.CHRF.WORD_ORDER,
            help="Word n-gram order. If equals to 2, the metric is referred to as chrF++.",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.beta = args.chrf_beta
        self.char_order = args.chrf_char_order
        self.word_order = args.chrf_word_order
        return

    def compute(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[float, dict[str, float]]:
        chrf = sacrebleu.corpus_chrf(
            hypotheses=y_preds,
            references=y_trues,
            beta=self.beta,
        )
        return chrf.score, vars(chrf)


class Rouge(MetricsBase):
    scorer: rouge.Rouge

    def compute(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[float, dict[str, float]]:
        score_dict = {"r": [], "p": [], "f": []}
        for y_t, y_p in zip(y_trues, y_preds):
            rouge_score = self.compute_item(y_t, y_p)
            for key in score_dict.keys():
                score_dict[key].append(rouge_score[key])
        for key in score_dict.keys():
            score_dict[key] = sum(score_dict[key]) / len(score_dict[key])
        return score_dict["f"], score_dict

    def compute_item(
        self, y_trues: list[str], y_pred: str
    ) -> tuple[float, dict[str, float]]:
        score_dict = {"r": 0.0, "p": 0.0, "f": 0.0}
        for y_true in y_trues:
            rouge_score = self.scorer.get_scores(y_pred, y_true)
            for key in score_dict.keys():
                score_dict[key] = max(score_dict[key], rouge_score[0][key])
        return score_dict["f"], score_dict


class Rouge1(MetricsBase):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.scorer = rouge.Rouge(metrics=["rouge-1"])
        return


class Rouge2(MetricsBase):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.scorer = rouge.Rouge(metrics=["rouge-2"])
        return


class RougeL(MetricsBase):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.scorer = rouge.Rouge(metrics=["rouge-l"])
        return
