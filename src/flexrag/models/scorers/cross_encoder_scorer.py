import numpy as np
import torch

from flexrag.common import TIME_METER, configure

from ..hf_utils import HFModelConfig, load_hf_model
from .scorer_base import SCORERS, PairScorerBase


@configure
class HFCrossEncoderScorerConfig(HFModelConfig):
    """The configuration for the Cross Encoder scorer.

    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    """

    max_encode_length: int = 512


@SCORERS("hf_cross_encoder", config_class=HFCrossEncoderScorerConfig)
class HFCrossEncoderScorer(PairScorerBase):
    """HFCrossEncoderScorer: The scorer based on the HuggingFace Cross Encoder model."""

    def __init__(self, cfg: HFCrossEncoderScorerConfig):
        # load model
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="sequence_classification",
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        return

    @TIME_METER("scorer.cross_encoder")
    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # score the candidates
        inputs = self.tokenizer(
            pairs,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.model.device)
        scores = self.model(**inputs).logits.squeeze().cpu().numpy()
        return scores
