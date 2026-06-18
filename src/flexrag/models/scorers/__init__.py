from .colbert_scorer import HFColBertScorer, HFColBertScorerConfig
from .cross_encoder_scorer import HFCrossEncoderScorer, HFCrossEncoderScorerConfig
from .logits_scorer import HFLogitsScorer, HFLogitsScorerConfig
from .scorer_base import SCORERS, LocalPairScorerBase, PairScorerProtocol

ScorerConfig = SCORERS.make_config(config_name="ScorerConfig")


__all__ = [
    "HFColBertScorer",
    "HFColBertScorerConfig",
    "HFCrossEncoderScorer",
    "HFCrossEncoderScorerConfig",
    "HFLogitsScorer",
    "HFLogitsScorerConfig",
    "LocalPairScorerBase",
    "SCORERS",
    "PairScorerProtocol",
    "ScorerConfig",
]
