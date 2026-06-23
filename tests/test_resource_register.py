from dataclasses import dataclass

import pytest

from flexrag.resources.registry import _ResourceRegister


@dataclass
class FakeConfig:
    value: int = 1


@dataclass
class OtherConfig:
    value: int = 2


class FakeImpl:
    pass


class OtherImpl:
    pass


class FakeRuntimeAdapter:
    pass


class OtherRuntimeAdapter:
    pass


def test_resource_register_decorator_registers_metadata():
    register = _ResourceRegister()

    returned_cls = register.register(
        "fake",
        "fake_alias",
        config_class=FakeConfig,
        runtime_adapter_cls=FakeRuntimeAdapter,
    )(FakeImpl)

    assert returned_cls is FakeImpl
    assert register.names == ("fake", "fake_alias")
    assert len(register.entries) == 1

    entry = register.resolve(FakeConfig())
    assert register.resolve_config_class(FakeConfig) is entry
    assert register.resolve_name("fake") is entry
    assert register.resolve_name("fake_alias") is entry
    assert entry.short_names == ("fake", "fake_alias")
    assert entry.config_class is FakeConfig
    assert entry.impl_cls is FakeImpl
    assert entry.runtime_adapter_cls is FakeRuntimeAdapter


def test_resource_register_rejects_invalid_names():
    register = _ResourceRegister()

    with pytest.raises(ValueError, match="At least one short name"):
        register.register(
            config_class=FakeConfig,
            runtime_adapter_cls=FakeRuntimeAdapter,
        )

    with pytest.raises(ValueError, match="must not be empty"):
        register.register(
            "",
            config_class=FakeConfig,
            runtime_adapter_cls=FakeRuntimeAdapter,
        )

    with pytest.raises(ValueError, match="Duplicate short name"):
        register.register(
            "fake",
            "fake",
            config_class=FakeConfig,
            runtime_adapter_cls=FakeRuntimeAdapter,
        )


def test_resource_register_rejects_duplicate_short_name_and_config_class():
    register = _ResourceRegister()
    register.register(
        "fake",
        config_class=FakeConfig,
        runtime_adapter_cls=FakeRuntimeAdapter,
    )(FakeImpl)

    with pytest.raises(ValueError, match="already registered"):
        register.register(
            "other",
            config_class=FakeConfig,
            runtime_adapter_cls=OtherRuntimeAdapter,
        )

    with pytest.raises(ValueError, match="already registered"):
        register.register(
            "fake",
            config_class=OtherConfig,
            runtime_adapter_cls=OtherRuntimeAdapter,
        )


def test_resource_register_rejects_unknown_config_and_name():
    register = _ResourceRegister()
    register.register(
        "fake",
        config_class=FakeConfig,
        runtime_adapter_cls=FakeRuntimeAdapter,
    )(FakeImpl)

    with pytest.raises(KeyError, match="not registered"):
        register.resolve(OtherConfig())

    with pytest.raises(KeyError, match="not registered"):
        register.resolve_config_class(OtherConfig)

    with pytest.raises(KeyError, match="not registered"):
        register.resolve_name("missing")


def test_builtin_resource_registrations_are_available():
    from flexrag.models.encoders import (
        HFEncoder,
        HFEncoderConfig,
        LiteLLMEncoder,
        LiteLLMEncoderConfig,
    )
    from flexrag.models.generators import (
        HFGenerator,
        HFGeneratorConfig,
        LiteLLMGenerator,
        LiteLLMGeneratorConfig,
    )
    from flexrag.models.scorers import HFCrossEncoderScorer, HFCrossEncoderScorerConfig
    from flexrag.processors.rankers import (
        HFRanker,
        HFRankerConfig,
        LiteLLMRanker,
        LiteLLMRankerConfig,
        RankGPTRanker,
        RankGPTRankerConfig,
    )
    from flexrag.resources import Resources
    from flexrag.resources.runtime_adapters import (
        ProcessEncoderAdapter,
        ProcessGeneratorAdapter,
        ProcessScorerAdapter,
        RankerRuntimeAdapter,
        RemoteEncoderRuntimeAdapter,
        RemoteGeneratorRuntimeAdapter,
        RemoteRankerRuntimeAdapter,
    )

    hf_generator = Resources.resolve_name("hf_generator")
    assert hf_generator.impl_cls is HFGenerator
    assert hf_generator.config_class is HFGeneratorConfig
    assert hf_generator.runtime_adapter_cls is ProcessGeneratorAdapter

    litellm_generator = Resources.resolve_name("litellm_generator")
    assert litellm_generator.impl_cls is LiteLLMGenerator
    assert litellm_generator.config_class is LiteLLMGeneratorConfig
    assert litellm_generator.runtime_adapter_cls is RemoteGeneratorRuntimeAdapter

    hf_encoder = Resources.resolve_name("hf_encoder")
    assert hf_encoder.impl_cls is HFEncoder
    assert hf_encoder.config_class is HFEncoderConfig
    assert hf_encoder.runtime_adapter_cls is ProcessEncoderAdapter

    litellm_encoder = Resources.resolve_name("litellm_encoder")
    assert litellm_encoder.impl_cls is LiteLLMEncoder
    assert litellm_encoder.config_class is LiteLLMEncoderConfig
    assert litellm_encoder.runtime_adapter_cls is RemoteEncoderRuntimeAdapter

    scorer = Resources.resolve_name("hf_cross_encoder_scorer")
    assert scorer.impl_cls is HFCrossEncoderScorer
    assert scorer.config_class is HFCrossEncoderScorerConfig
    assert scorer.runtime_adapter_cls is ProcessScorerAdapter

    hf_ranker = Resources.resolve_name("hf_ranker")
    assert hf_ranker.impl_cls is HFRanker
    assert hf_ranker.config_class is HFRankerConfig
    assert hf_ranker.runtime_adapter_cls is RankerRuntimeAdapter

    rank_gpt_ranker = Resources.resolve_name("rank_gpt_ranker")
    assert rank_gpt_ranker.impl_cls is RankGPTRanker
    assert rank_gpt_ranker.config_class is RankGPTRankerConfig
    assert rank_gpt_ranker.runtime_adapter_cls is RankerRuntimeAdapter

    litellm_ranker = Resources.resolve_name("litellm_ranker")
    assert litellm_ranker.impl_cls is LiteLLMRanker
    assert litellm_ranker.config_class is LiteLLMRankerConfig
    assert litellm_ranker.runtime_adapter_cls is RemoteRankerRuntimeAdapter


def test_builtin_resource_registrations_resolve_by_config_instance():
    from flexrag.models.generators import HFGeneratorConfig
    from flexrag.resources import Resources

    entry = Resources.resolve(HFGeneratorConfig())

    assert entry is Resources.resolve_config_class(HFGeneratorConfig)
    assert entry is Resources.resolve_name("hf_generator")
