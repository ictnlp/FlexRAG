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
from flexrag.processors.rankers import (
    HFRanker,
    HFRankerConfig,
    LiteLLMRanker,
    LiteLLMRankerConfig,
    RankGPTRanker,
    RankGPTRankerConfig,
)
from flexrag.processors.refiners import (
    AbstractiveSummarizer,
    AbstractiveSummarizerConfig,
    ContextArranger,
    ContextArrangerConfig,
    RecompExtractiveSummarizer,
    RecompExtractiveSummarizerConfig,
)

from .registry import Resources
from .runtime_adapters import (
    DirectRuntimeAdapter,
    ProcessRuntimeAdapter,
    RemoteRuntimeAdapter,
)

Resources.register(
    "hf_encoder",
    interface="encoder",
    config_class=HFEncoderConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFEncoder)

Resources.register(
    "hf_clip_encoder",
    interface="encoder",
    config_class=HFClipEncoderConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFClipEncoder)

Resources.register(
    "sentence_transformer_encoder",
    interface="encoder",
    config_class=SentenceTransformerEncoderConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(SentenceTransformerEncoder)

Resources.register(
    "litellm_encoder",
    interface="encoder",
    config_class=LiteLLMEncoderConfig,
    runtime_adapter_cls=RemoteRuntimeAdapter,
)(LiteLLMEncoder)

Resources.register(
    "hf_generator",
    interface="generator",
    config_class=HFGeneratorConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFGenerator)

Resources.register(
    "litellm_generator",
    interface="generator",
    config_class=LiteLLMGeneratorConfig,
    runtime_adapter_cls=RemoteRuntimeAdapter,
)(LiteLLMGenerator)

Resources.register(
    "hf_cross_encoder_scorer",
    interface="scorer",
    config_class=HFCrossEncoderScorerConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFCrossEncoderScorer)

Resources.register(
    "hf_colbert_scorer",
    interface="scorer",
    config_class=HFColBertScorerConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFColBertScorer)

Resources.register(
    "hf_logits_scorer",
    interface="scorer",
    config_class=HFLogitsScorerConfig,
    runtime_adapter_cls=ProcessRuntimeAdapter,
)(HFLogitsScorer)

Resources.register(
    "hf_ranker",
    interface="ranker",
    config_class=HFRankerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(HFRanker)

Resources.register(
    "rank_gpt_ranker",
    interface="ranker",
    config_class=RankGPTRankerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(RankGPTRanker)

Resources.register(
    "litellm_ranker",
    interface="ranker",
    config_class=LiteLLMRankerConfig,
    runtime_adapter_cls=RemoteRuntimeAdapter,
)(LiteLLMRanker)

Resources.register(
    "context_arranger",
    interface="refiner",
    config_class=ContextArrangerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(ContextArranger)

Resources.register(
    "abstractive_summarizer",
    interface="refiner",
    config_class=AbstractiveSummarizerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(AbstractiveSummarizer)

Resources.register(
    "extractive_summarizer",
    interface="refiner",
    config_class=RecompExtractiveSummarizerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(RecompExtractiveSummarizer)
