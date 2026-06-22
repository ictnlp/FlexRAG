"""Built-in resource metadata registrations."""

from flexrag.models.encoders import (
    HFClipEncoder,
    HFClipEncoderConfig,
    HFEncoder,
    HFEncoderConfig,
    LiteLLMEncoder,
    LiteLLMEncoderConfig,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
)
from flexrag.models.generators import (
    HFGenerator,
    HFGeneratorConfig,
    LiteLLMGenerator,
    LiteLLMGeneratorConfig,
)
from flexrag.models.scorers import (
    HFColBertScorer,
    HFColBertScorerConfig,
    HFCrossEncoderScorer,
    HFCrossEncoderScorerConfig,
    HFLogitsScorer,
    HFLogitsScorerConfig,
)

from .registry import Resources
from .runtime_adapters import (
    ProcessEncoderAdapter,
    ProcessGeneratorAdapter,
    ProcessScorerAdapter,
    RemoteEncoderRuntimeAdapter,
    RemoteGeneratorRuntimeAdapter,
)

Resources.register(
    "hf_encoder",
    config_class=HFEncoderConfig,
    runtime_adapter_cls=ProcessEncoderAdapter,
)(HFEncoder)

Resources.register(
    "hf_clip_encoder",
    config_class=HFClipEncoderConfig,
    runtime_adapter_cls=ProcessEncoderAdapter,
)(HFClipEncoder)

Resources.register(
    "sentence_transformer_encoder",
    config_class=SentenceTransformerEncoderConfig,
    runtime_adapter_cls=ProcessEncoderAdapter,
)(SentenceTransformerEncoder)

Resources.register(
    "litellm_encoder",
    config_class=LiteLLMEncoderConfig,
    runtime_adapter_cls=RemoteEncoderRuntimeAdapter,
)(LiteLLMEncoder)

Resources.register(
    "hf_generator",
    config_class=HFGeneratorConfig,
    runtime_adapter_cls=ProcessGeneratorAdapter,
)(HFGenerator)

Resources.register(
    "litellm_generator",
    config_class=LiteLLMGeneratorConfig,
    runtime_adapter_cls=RemoteGeneratorRuntimeAdapter,
)(LiteLLMGenerator)

Resources.register(
    "hf_cross_encoder_scorer",
    config_class=HFCrossEncoderScorerConfig,
    runtime_adapter_cls=ProcessScorerAdapter,
)(HFCrossEncoderScorer)

Resources.register(
    "hf_colbert_scorer",
    config_class=HFColBertScorerConfig,
    runtime_adapter_cls=ProcessScorerAdapter,
)(HFColBertScorer)

Resources.register(
    "hf_logits_scorer",
    config_class=HFLogitsScorerConfig,
    runtime_adapter_cls=ProcessScorerAdapter,
)(HFLogitsScorer)
