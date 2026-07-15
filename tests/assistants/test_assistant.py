import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import flexrag.assistants as assistants
from flexrag.common import ChatMessages, ChatTurn, Context, RetrievedContext
from flexrag.processors.rankers import RankingResult


def _messages(text: str = "question") -> ChatMessages:
    return ChatMessages.from_list([{"role": "user", "content": text}])


def _hit(context: Context) -> RetrievedContext:
    return RetrievedContext(context_id=context.context_id, data=context.data)


class _Generator:
    def __init__(self) -> None:
        self.active = self.max_active = 0

    async def async_chat(self, messages, **kwargs):
        del messages, kwargs
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return [[ChatTurn(role="assistant", content="answer")]]


class _Retriever:
    def __init__(self, contexts=()) -> None:
        self.contexts = list(contexts)
        self.clears = 0

    async def async_search(self, queries, **kwargs):
        del kwargs
        return [[_hit(context) for context in self.contexts] for _ in queries]

    async def async_add_contexts(self, contexts):
        self.contexts.extend(contexts)

    async def async_clear(self):
        self.clears += 1
        self.contexts.clear()


@pytest.mark.asyncio
async def test_modular_pipeline_memory_and_concurrency():
    generator, memory = _Generator(), _Retriever()
    knowledge = _Retriever([RetrievedContext(context_id="k", data={"text": "k"})])

    async def rank(query, candidates, **kwargs):
        return RankingResult(query=query, candidates=list(reversed(candidates)))

    async def refine(contexts):
        contexts[0].metadata["refined"] = True
        return contexts

    assistant = assistants.ModularAssistant(
        assistants.ModularAssistantConfig(used_fields=["text"]),
        generator=generator,
        retriever=knowledge,
        memory_retriever=memory,
        ranker=type("Ranker", (), {"async_rank": staticmethod(rank)})(),
        refiners=(type("Refiner", (), {"async_refine": staticmethod(refine)})(),),
    )
    history = ChatMessages.from_list(
        [{"role": "user", "content": "past"}], metadata={"session_id": "s1"}
    )

    async with assistant:
        await assistant.add_histories([history])
        await assistant.add_contexts([Context(data={"text": "memory"})])
        results = await asyncio.gather(
            assistant.answer(_messages("one")), assistant.answer(_messages("two"))
        )
        assert generator.max_active == 2
        ids = [item.context_id for item in results[0].contexts]
        assert ids == ["k", "context-2", "history-1"]
        assert memory.contexts[0].metadata["session_id"] == "s1"
        assert results[0].contexts[0].metadata["refined"]

        generator.max_active = 0
        await asyncio.gather(
            assistant.run(_messages("three")), assistant.run(_messages("four"))
        )
        assert generator.max_active == 1
        assert [c.context_id for c in memory.contexts[-2:]] == ["turn-3", "turn-4"]

    assert memory.clears == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["jina", "perplexity"])
async def test_online_assistant_qa_boundary(monkeypatch, provider):
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "answer"}}],
            "citations": ["https://example.test/source"],
        },
        request=httpx.Request("POST", "https://example.test"),
    )
    client = MagicMock(post=AsyncMock(return_value=response), aclose=AsyncMock())
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    assistant = (
        assistants.JinaDeepSearch(assistants.JinaDeepSearchConfig(api_key="key"))
        if provider == "jina"
        else assistants.PerplexityAssistant(
            assistants.PerplexityAssistantConfig(api_key="key")
        )
    )
    async with assistant:
        results = await asyncio.gather(
            assistant.answer(_messages("one")), assistant.answer(_messages("two"))
        )
        assert [result.response.text_content for result in results] == ["answer"] * 2
        assert bool(results[0].contexts) is (provider == "perplexity")
        with pytest.raises(RuntimeError, match="stateful run"):
            await assistant.run(_messages())
    assert client.post.await_count == 2
    client.aclose.assert_awaited_once()
