import numpy as np
import pytest

from flexrag.common.dataclasses import RetrievedContext
from flexrag.processors.refiners import (
    AbstractiveSummarizer,
    AbstractiveSummarizerConfig,
    ContextArranger,
    ContextArrangerConfig,
    RecompExtractiveSummarizer,
    RecompExtractiveSummarizerConfig,
)


class FakeGenerator:
    def __init__(self) -> None:
        self.prefixes: list[list[str]] = []
        self.async_prefixes: list[list[str]] = []
        return

    def generate(self, prefixes, generation_config=None):
        self.prefixes.append(prefixes)
        return [[f"summary:{prefix}"] for prefix in prefixes]

    async def async_generate(self, prefixes, generation_config=None):
        self.async_prefixes.append(prefixes)
        return [[f"summary:{prefix}"] for prefix in prefixes]

    def chat(self, messages, generation_config=None):
        raise NotImplementedError

    async def async_chat(self, messages, generation_config=None):
        raise NotImplementedError


class FakeEncoder:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        self.async_inputs: list[list[str]] = []
        return

    def encode(self, inputs):
        self.inputs.append(inputs)
        return self._encode(inputs)

    def _encode(self, inputs):
        rows = []
        for item in inputs:
            if "query" in item or "useful" in item:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])
        return np.array(rows)

    async def async_encode(self, inputs):
        self.async_inputs.append(inputs)
        return self._encode(inputs)

    @property
    def embedding_size(self):
        return 2


def _contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            context_id="low",
            query="query",
            data={"text": "A less relevant context."},
            score=0.1,
        ),
        RetrievedContext(
            context_id="high",
            query="query",
            data={"text": "A more relevant context."},
            score=0.9,
        ),
    ]


@pytest.mark.asyncio
async def test_context_arranger_orders_contexts_by_score():
    refiner = ContextArranger(ContextArrangerConfig(order="descending"))

    refined = refiner.refine(_contexts())
    async_refined = await refiner.async_refine(_contexts())

    assert [context.context_id for context in refined] == ["high", "low"]
    assert [context.context_id for context in async_refined] == ["high", "low"]


@pytest.mark.asyncio
async def test_abstractive_summarizer_uses_injected_generator_without_mutating_inputs():
    generator = FakeGenerator()
    refiner = AbstractiveSummarizer(
        AbstractiveSummarizerConfig(
            template="summarize ${content} for ${query}",
            refined_field="text",
        ),
        generator=generator,
    )
    contexts = _contexts()

    refined = refiner.refine(contexts)
    async_refined = await refiner.async_refine(contexts)

    assert (
        refined[0].data["text"]
        == "summary:summarize A less relevant context. for query"
    )
    assert (
        refined[1].data["text"]
        == "summary:summarize A more relevant context. for query"
    )
    assert async_refined == refined
    assert contexts[0].data["text"] == "A less relevant context."
    assert generator.prefixes == [
        [
            "summarize A less relevant context. for query",
            "summarize A more relevant context. for query",
        ]
    ]
    assert generator.async_prefixes == generator.prefixes


@pytest.mark.asyncio
async def test_extractive_summarizer_uses_injected_encoder_without_mutating_inputs():
    encoder = FakeEncoder()
    refiner = RecompExtractiveSummarizer(
        RecompExtractiveSummarizerConfig(
            preserved_sents=1,
            refined_field="text",
            substitute=True,
        ),
        encoder=encoder,
    )
    contexts = [
        RetrievedContext(
            context_id="ctx",
            query="query",
            data={"text": "Useful sentence is useful. Distractor sentence is extra."},
            score=1.0,
        )
    ]

    refined = refiner.refine(contexts)
    async_refined = await refiner.async_refine(contexts)

    assert refined[0].data["text"] == "Useful sentence is useful."
    assert async_refined == refined
    assert contexts[0].data["text"] == (
        "Useful sentence is useful. Distractor sentence is extra."
    )
    assert encoder.async_inputs == encoder.inputs
