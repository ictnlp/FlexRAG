import numpy as np
import torch

from flexrag.common import configure, trace

from ..hf_utils import HFModelConfig, load_hf_model
from .local_process_scorer_base import LocalProcessScorerBase
from .scorer_base import SCORERS


@configure
class HFCrossEncoderScorerConfig(HFModelConfig):
    """The configuration for the Cross Encoder scorer.

    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    """

    max_encode_length: int = 512


class HFCrossEncoderScorerImpl:
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

    @trace("scorer.cross_encoder")
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
        scores = self.model(**inputs).logits.view(-1).cpu().numpy()
        return np.atleast_1d(scores)


@SCORERS("hf_cross_encoder", config_class=HFCrossEncoderScorerConfig)
class HFCrossEncoderScorer(LocalProcessScorerBase):
    impl_cls = HFCrossEncoderScorerImpl
