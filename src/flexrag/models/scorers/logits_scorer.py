from typing import Annotated

import numpy as np
import torch

from flexrag.common import Choices, configure, trace

from ..hf_utils import HFModelConfig, load_hf_model
from .scorer_base import LocalPairScorerBase


@configure
class HFLogitsScorerConfig(HFModelConfig):
    """The configuration for the Scorer that uses logits for scoring.

    :param model_type: the type of the model.
        Choices are "seq2seq" and "causal_lm". Default is "seq2seq".
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
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the scorer is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    """

    model_type: Annotated[str, Choices("seq2seq", "causal_lm")] = "seq2seq"
    max_encode_length: int = 512
    input_template: str = "Query: {query} Document: {candidate} Relevant:"
    positive_token: str = "▁true"
    negative_token: str = "▁false"
    batch_size: int = 32


class HFLogitsScorer(LocalPairScorerBase):
    def __init__(self, cfg: HFLogitsScorerConfig):
        super().__init__(batch_size=cfg.batch_size)

        # load model
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type=cfg.model_type,
            device_map=cfg.device_map,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        self.input_template = cfg.input_template
        self.positive_token: int = self.tokenizer.convert_tokens_to_ids(
            cfg.positive_token
        )
        self.negative_token: int = self.tokenizer.convert_tokens_to_ids(
            cfg.negative_token
        )
        return

    @trace("scorer.hf_logits")
    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
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
            padding_side="left",
        )
        inputs = inputs.to(self.model.device)
        if getattr(self.model.config, "is_encoder_decoder", False):
            decoder_start_token_id = self.model.config.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = self.tokenizer.pad_token_id
            decoder_input_ids = torch.full(
                (len(input_texts), 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=self.model.device,
            )
            outputs = self.model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
            ).logits[:, 0, :]
        else:
            outputs = self.model(**inputs).logits[:, -1, :]
        positive_scores = outputs[:, self.positive_token : self.positive_token + 1]
        negative_scores = outputs[:, self.negative_token : self.negative_token + 1]
        scores = torch.softmax(
            torch.cat([positive_scores, negative_scores], dim=1), dim=1
        )[:, 0]
        return np.atleast_1d(scores.cpu().float().numpy())
