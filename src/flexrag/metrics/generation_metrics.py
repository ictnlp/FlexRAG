from itertools import zip_longest
from typing import Annotated

import sacrebleu
from rouge_score import rouge_scorer

from flexrag.common import Choices, configure, trace
from flexrag.models.tokenizer import TokenizerProtocol

from .metrics_base import METRICS


@configure
class BLEUConfig:
    """Configuration for ``BLEU`` metric.
    The computation of BLEU score is based on `sacrebleu <https://github.com/mjpost/sacrebleu>`_.

    :param tokenizer: The tokenizer to use. Defaults to sacrebleu.BLEU.TOKENIZER_DEFAULT.
        Available choices: Please refer to sacrebleu.BLEU.TOKENIZERS.
    :type tokenizer: str
    """

    tokenizer: Annotated[str, Choices(*sacrebleu.BLEU.TOKENIZERS)] = (
        sacrebleu.BLEU.TOKENIZER_DEFAULT
    )


@METRICS("generation_bleu", config_class=BLEUConfig)
class BLEU:
    """The BLEU metric."""

    def __init__(self, cfg: BLEUConfig):
        self.tokenizer = cfg.tokenizer
        return

    @trace("metrics.generation_bleu")
    def __call__(
        self, responses: list[str], golden_responses: list[list[str]]
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not golden_responses:
            refs = []
        else:
            refs = [list(item) for item in zip_longest(*golden_responses, fillvalue="")]
        bleu = sacrebleu.corpus_bleu(
            hypotheses=responses,
            references=refs,
            tokenize=self.tokenizer,
        )
        return {"response_bleu": bleu.score}, vars(bleu)


@configure
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
class chrF:
    """The chrF metric."""

    def __init__(self, cfg: chrFConfig) -> None:
        self.beta = cfg.chrf_beta
        self.char_order = cfg.chrf_char_order
        self.word_order = cfg.chrf_word_order
        return

    @trace("metrics.generation_chrf")
    def __call__(
        self, responses: list[str], golden_responses: list[list[str]]
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not golden_responses:
            refs = []
        else:
            refs = [list(item) for item in zip_longest(*golden_responses, fillvalue="")]
        chrf = sacrebleu.corpus_chrf(
            hypotheses=responses,
            references=refs,
            beta=self.beta,
            char_order=self.char_order,
            word_order=self.word_order,
        )
        return {"response_chrf": chrf.score}, vars(chrf)


@configure
class RougeConfig:
    """Configuration for ``Rouge`` metric.
    The computation of Rouge score is based on `rouge-score
    <https://github.com/google-research/google-research/tree/master/rouge>`_.
    """


@METRICS("generation_rouge", config_class=RougeConfig)
class Rouge:
    """The Rouge metric.
    The computation of Rouge score is based on `rouge-score
    <https://github.com/google-research/google-research/tree/master/rouge>`_.
    This metric returns the average F1 scores for Rouge-1, Rouge-2, and Rouge-L.
    """

    def __init__(self, cfg: RougeConfig, tokenizer: TokenizerProtocol) -> None:
        self.tokenizer = tokenizer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            tokenizer=self.tokenizer,
            use_stemmer=False,
        )
        return

    @trace("metrics.generation_rouge")
    def __call__(
        self, responses: list[str], golden_responses: list[list[str]]
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
    ) -> dict[str, dict[str, float]]:
        score_dict = {
            "rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0},
        }
        if not self.tokenizer.tokenize(response):
            return score_dict

        valid_golds = [gold for gold in golds if self.tokenizer.tokenize(gold)]
        if not valid_golds:
            return score_dict

        rouge_score = self.scorer.score_multi(valid_golds, response)
        metric_mapping = {
            "rouge-1": "rouge1",
            "rouge-2": "rouge2",
            "rouge-l": "rougeL",
        }
        for metric, rouge_name in metric_mapping.items():
            score = rouge_score[rouge_name]
            score_dict[metric]["r"] = score.recall
            score_dict[metric]["p"] = score.precision
            score_dict[metric]["f"] = score.fmeasure
        return score_dict
