from typing import Annotated

import torch
import numpy as np
from transformers import GenerationConfig as HFGenerationConfig

from flexrag.common import Choices, configure, TIME_METER

from ..hf_utils import HFModelConfig, load_hf_model
from .scorer_base import SCORERS, PairScorerBase


@configure
class HFLogitsScorerConfig(HFModelConfig):
    """The configuration for the Scorer that uses logits for scoring.

    :param model_type: the type of the model.
        Choices are "seq2seq" and "causal". Default is "seq2seq".
    :type model_type: str
    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    :param input_template: the input template for the seq2seq model.
        Default is "Query: {query} Document: {candidate} Relevant:".
    :type input_template: str
    :param positive_token: the positive token for the seq2seq model. Default is "▁true".
    :type positive_token: str
    :param negative_token: the negative token for the seq2seq model. Default is "▁false".
    :type negative_token: str
    """

    model_type: Annotated[str, Choices("seq2seq", "causal")] = "seq2seq"
    max_encode_length: int = 512
    input_template: str = "Query: {query} Document: {candidate} Relevant:"
    positive_token: str = "▁true"
    negative_token: str = "▁false"


@SCORERS("hf_logits", config_class=HFLogitsScorerConfig)
class HFLogitsScorer(PairScorerBase):
    def __init__(self, cfg: HFLogitsScorerConfig):
        # load model
        super().__init__(cfg)
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type=cfg.model_type,
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        self.input_template = cfg.input_template
        self.positive_token = self.tokenizer.convert_tokens_to_ids(cfg.positive_token)
        self.negative_token = self.tokenizer.convert_tokens_to_ids(cfg.negative_token)
        self.generation_config = HFGenerationConfig(
            max_new_tokens=1, output_logits=True
        )
        return

    @TIME_METER("scorer.hf_logits")
    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # prepare prompts
        input_texts = [
            self.input_template.format(query=pair[0], candidate=pair[1])
            for pair in pairs
        ]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
        )
        logits = outputs.logits[0]
        positive_scores = logits[:, self.positive_token : self.positive_token + 1]
        negative_scores = logits[:, self.negative_token : self.negative_token + 1]
        scores = torch.softmax(
            torch.cat([positive_scores, negative_scores], dim=1), dim=1
        )[:, 0].cpu().numpy()  # fmt: skip
        return scores
