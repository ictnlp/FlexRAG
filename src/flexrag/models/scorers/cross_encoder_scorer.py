import numpy as np
import torch

from flexrag.common import configure, trace

from ..hf_utils import HFModelConfig, load_hf_model
from .scorer_base import SCORERS, LocalPairScorerBase


@configure
class HFCrossEncoderScorerConfig(HFModelConfig):
    """The configuration for the Cross Encoder scorer.

    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the scorer is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    """

    max_encode_length: int = 512
    batch_size: int = 32


@SCORERS("hf_cross_encoder", config_class=HFCrossEncoderScorerConfig)
class HFCrossEncoderScorer(LocalPairScorerBase):
    """HFCrossEncoderScorer: The scorer based on the HuggingFace Cross Encoder model."""

    def __init__(self, cfg: HFCrossEncoderScorerConfig):
        super().__init__(batch_size=cfg.batch_size)

        # load model
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="sequence_classification",
            device_map=cfg.device_map,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        return

    @trace("scorer.cross_encoder")
    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # score the candidates
        inputs = self.tokenizer(
            pairs,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.model.device)
        scores = self.model(**inputs).logits.view(-1).cpu().numpy()
        return np.atleast_1d(scores)
