from dataclasses import dataclass

import rouge
import sacrebleu

from flexrag.utils import Choices, TIME_METER

from .metrics_base import MetricsBase, METRICS


@dataclass
class BLEUConfig:
    """Configuration for ``BLEU`` metric.
    The computation of BLEU score is based on `sacrebleu <https://github.com/mjpost/sacrebleu>`_.

    :param tokenizer: The tokenizer to use. Defaults to sacrebleu.BLEU.TOKENIZER_DEFAULT.
        Available choices: Please refer to sacrebleu.BLEU.TOKENIZERS.
    :type tokenizer: str
    """

    tokenizer: Choices(sacrebleu.BLEU.TOKENIZERS) = sacrebleu.BLEU.TOKENIZER_DEFAULT  # type: ignore


@METRICS("generation_bleu", config_class=BLEUConfig)
class BLEU(MetricsBase):
    """The BLEU metric."""

    def __init__(self, cfg: BLEUConfig):
        super().__init__(cfg)
        self.tokenizer = cfg.tokenizer
        return

    @TIME_METER("metrics.generation_bleu")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[dict[str, float], dict[str, float]]:
        bleu = sacrebleu.corpus_bleu(
            hypotheses=responses,
            references=golden_responses,
            tokenize=self.tokenizer,
        )
        return {"response_bleu": bleu.score}, vars(bleu)


@dataclass
class chrFConfig:
    """Configuration for ``chrF`` metric.
    The computation of chrF score is based on `sacrebleu <https://github.com/mjpost/sacrebleu>`_.

    :param chrf_beta: The beta value for the F-score. Defaults to 1.0.
    :type chrf_beta: float
    :param chrf_char_order: The order of characters. Defaults to sacrebleu.CHRF.CHAR_ORDER.
    :type chrf_char_order: int
    :param chrf_word_order: The order of words. Defaults to sacrebleu.CHRF.WORD_ORDER.
    :type chrf_word_order: int
    """

    chrf_beta: float = 1.0
    chrf_char_order: int = sacrebleu.CHRF.CHAR_ORDER
    chrf_word_order: int = sacrebleu.CHRF.WORD_ORDER


@METRICS("generation_chrf", config_class=chrFConfig)
class chrF(MetricsBase):
    """The chrF metric."""

    def __init__(self, cfg: chrFConfig) -> None:
        super().__init__(cfg)
        self.beta = cfg.chrf_beta
        self.char_order = cfg.chrf_char_order
        self.word_order = cfg.chrf_word_order
        return

    @TIME_METER("metrics.generation_chrf")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[dict[str, float], dict[str, float]]:
        chrf = sacrebleu.corpus_chrf(
            hypotheses=responses,
            references=golden_responses,
            beta=self.beta,
        )
        return {"response_chrf": chrf.score}, vars(chrf)


@METRICS("generation_rouge")
class Rouge(MetricsBase):
    """The Rouge metric.
    The computation of Rouge score is based on `rouge <https://github.com/pltrdy/rouge>`_.
    This metric will return the average of the Rouge-1, Rouge-2, and Rouge-L F1 scores.
    """

    def __init__(self) -> None:
        self.scorer = rouge.Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
        return

    @TIME_METER("metrics.generation_rouge")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[dict[str, float], dict[str, float]]:
        score_dict = {
            "rouge-1": {"r": [], "p": [], "f": []},
            "rouge-2": {"r": [], "p": [], "f": []},
            "rouge-l": {"r": [], "p": [], "f": []},
        }
        # collect all the scores
        for golds, response in zip(golden_responses, responses):
            details = self.compute_item(golds, response)
            for metric in score_dict.keys():
                for key in ["r", "p", "f"]:
                    score_dict[metric][key].append(details[metric][key])
        # average the scores
        for metric in score_dict.keys():
            for key in ["r", "p", "f"]:
                score_dict[metric][key] = sum(score_dict[metric][key]) / len(
                    score_dict[metric][key]
                )
        return {
            "rouge-1": score_dict["rouge-1"]["f"],
            "rouge-2": score_dict["rouge-2"]["f"],
            "rouge-l": score_dict["rouge-l"]["f"],
        }, score_dict

    def compute_item(
        self, golds: list[str], response: str
    ) -> tuple[dict[str, float], dict[str, float]]:
        # as rouge score does not support multiple references, we take the max score.
        score_dict = {
            "rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0},
        }
        for gold in golds:
            rouge_score = self.scorer.get_scores(response, gold)[0]
            for metric in score_dict.keys():
                for key in ["r", "p", "f"]:
                    score_dict[metric][key] = max(
                        score_dict[metric][key], rouge_score[metric][key]
                    )
        return score_dict
