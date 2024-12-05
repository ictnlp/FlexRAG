from dataclasses import dataclass

import rouge
import sacrebleu

from librarian.utils import Choices, TIME_METER

from .metrics_base import MetricsBase, METRICS


@dataclass
class BLEUConfig:
    tokenizer: Choices(sacrebleu.BLEU.TOKENIZERS) = sacrebleu.BLEU.TOKENIZER_DEFAULT  # type: ignore


@METRICS("generation_bleu", config_class=BLEUConfig)
class BLEU(MetricsBase):
    def __init__(self, cfg: BLEUConfig):
        super().__init__(cfg)
        self.tokenizer = cfg.tokenizer
        return

    @TIME_METER("metrics.generation_bleu")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[float, dict[str, float]]:
        bleu = sacrebleu.corpus_bleu(
            hypotheses=responses,
            references=golden_responses,
            tokenize=self.tokenizer,
        )
        return bleu.score, vars(bleu)


@dataclass
class chrFConfig:
    chrf_beta: float = 1.0
    chrf_char_order: int = sacrebleu.CHRF.CHAR_ORDER
    chrf_word_order: int = sacrebleu.CHRF.WORD_ORDER


@METRICS("generation_chrf", config_class=chrFConfig)
class chrF(MetricsBase):
    def __init__(self, cfg: chrFConfig) -> None:
        super().__init__(cfg)
        self.beta = cfg.chrf_beta
        self.char_order = cfg.chrf_char_order
        self.word_order = cfg.chrf_word_order
        return

    @TIME_METER("metrics.generation_chrf")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[float, dict[str, float]]:
        chrf = sacrebleu.corpus_chrf(
            hypotheses=responses,
            references=golden_responses,
            beta=self.beta,
        )
        return chrf.score, vars(chrf)


class Rouge(MetricsBase):
    scorer: rouge.Rouge

    @TIME_METER("metrics.generation_rouge")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[float, dict[str, float]]:
        score_dict = {"r": [], "p": [], "f": []}
        for golds, response in zip(golden_responses, responses):
            rouge_score = self.compute_item(golds, response)
            for key in score_dict.keys():
                score_dict[key].append(rouge_score[key])
        for key in score_dict.keys():
            score_dict[key] = sum(score_dict[key]) / len(score_dict[key])
        return score_dict["f"], score_dict

    def compute_item(
        self, golds: list[str], response: str
    ) -> tuple[float, dict[str, float]]:
        score_dict = {"r": 0.0, "p": 0.0, "f": 0.0}
        for gold in golds:
            rouge_score = self.scorer.get_scores(response, gold)
            for key in score_dict.keys():
                score_dict[key] = max(score_dict[key], rouge_score[0][key])
        return score_dict["f"], score_dict


@METRICS("generation_rouge-1")
class Rouge1(Rouge):
    def __init__(self) -> None:
        self.scorer = rouge.Rouge(metrics=["rouge-1"])
        return


@METRICS("generation_rouge-2")
class Rouge2(Rouge):
    def __init__(self) -> None:
        self.scorer = rouge.Rouge(metrics=["rouge-2"])
        return


@METRICS("generation_rouge-l")
class RougeL(Rouge):
    def __init__(self) -> None:
        self.scorer = rouge.Rouge(metrics=["rouge-l"])
        return
