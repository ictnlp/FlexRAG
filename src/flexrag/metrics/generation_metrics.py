from dataclasses import field
from itertools import zip_longest
from typing import Annotated

import rouge
import sacrebleu

from flexrag.common import TIME_METER, Choices, configure
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig

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

    @TIME_METER("metrics.generation_bleu")
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

    @TIME_METER("metrics.generation_chrf")
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
    The computation of Rouge score is based on `rouge <https://github.com/pltrdy/rouge>`_.

    :param tokenizer_config: The tokenizer used for splitting the answer into tokens.
        Defaults to a space tokenizer.
    :type tokenizer_config: TokenizerConfig
    """

    tokenizer_config: TokenizerConfig = field(
        default_factory=lambda: TokenizerConfig(tokenizer_type="space")
    )


@METRICS("generation_rouge", config_class=RougeConfig)
class Rouge:
    """The Rouge metric.
    The computation of Rouge score is based on `rouge <https://github.com/pltrdy/rouge>`_.
    This metric will return the average of the Rouge-1, Rouge-2, and Rouge-L F1 scores.
    """

    def __init__(self, cfg: RougeConfig) -> None:
        self.scorer = rouge.Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
        self.tokenizer = TOKENIZERS.load(cfg.tokenizer_config)
        return

    @TIME_METER("metrics.generation_rouge")
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
    ) -> tuple[dict[str, float], dict[str, float]]:
        # as rouge score does not support multiple references, we take the max score.
        score_dict = {
            "rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0},
        }
        response = " ".join(self.tokenizer.tokenize(response))
        if not response.strip():
            return score_dict

        for gold in golds:
            gold = " ".join(self.tokenizer.tokenize(gold))
            if not gold.strip():
                continue
            rouge_score = self.scorer.get_scores(response, gold)[0]
            for metric in score_dict.keys():
                for key in ["r", "p", "f"]:
                    score_dict[metric][key] = max(
                        score_dict[metric][key], rouge_score[metric][key]
                    )
        return score_dict
