from dataclasses import dataclass

import sacrebleu
import rouge

from kylin.utils import Choices
from .metrics_base import MetricsBase, MetricsConfig


@dataclass
class BLEUConfig(MetricsConfig):
    bleu_tokenizer: Choices(sacrebleu.BLEU.TOKENIZERS) = sacrebleu.BLEU.TOKENIZER_DEFAULT  # type: ignore


class BLEU(MetricsBase):
    def __init__(self, cfg: BLEUConfig):
        super().__init__(cfg)
        self.tokenizer = cfg.bleu_tokenizer
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


@dataclass
class chrFConfig:
    chrf_beta: float = 1.0
    chrf_char_order: int = sacrebleu.CHRF.CHAR_ORDER
    chrf_word_order: int = sacrebleu.CHRF.WORD_ORDER


class chrF(MetricsBase):
    def __init__(self, cfg: chrFConfig) -> None:
        super().__init__(cfg)
        self.beta = cfg.chrf_beta
        self.char_order = cfg.chrf_char_order
        self.word_order = cfg.chrf_word_order
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


RougeConfig = MetricsConfig


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
    def __init__(self, cfg: RougeConfig) -> None:
        super().__init__(cfg)
        self.scorer = rouge.Rouge(metrics=["rouge-1"])
        return


class Rouge2(MetricsBase):
    def __init__(self, cfg: RougeConfig) -> None:
        super().__init__(cfg)
        self.scorer = rouge.Rouge(metrics=["rouge-2"])
        return


class RougeL(MetricsBase):
    def __init__(self, cfg: RougeConfig) -> None:
        super().__init__(cfg)
        self.scorer = rouge.Rouge(metrics=["rouge-l"])
        return
