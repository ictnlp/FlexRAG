import json
import types

import numpy as np
import pytest


@pytest.fixture
def mock_litellm_client(mocker):
    calls: dict[str, list[dict]] = {
        "acompletion": [],
        "atext_completion": [],
        "aembedding": [],
        "arerank": [],
    }

    async def mock_acompletion(*, model, messages, **kwargs):
        calls["acompletion"].append(
            {"model": model, "messages": messages, "kwargs": kwargs}
        )
        sample_num = kwargs.get("n", 1)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        role="assistant",
                        content=f"Mocked LiteLLM chat response {i}",
                        tool_calls=None,
                        reasoning_content=None,
                        thinking_blocks=None,
                    ),
                    finish_reason="stop",
                )
                for i in range(sample_num)
            ],
            usage=None,
        )

    async def mock_atext_completion(*, model, prompt, **kwargs):
        calls["atext_completion"].append(
            {"model": model, "prompt": prompt, "kwargs": kwargs}
        )
        sample_num = kwargs.get("n", 1)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    text=f"Mocked LiteLLM text completion {i}",
                    message=types.SimpleNamespace(
                        content=f"Mocked LiteLLM text completion {i}"
                    ),
                )
                for i in range(sample_num)
            ]
        )

    async def mock_aembedding(
        *, model, input, dimensions=None, input_type=None, **kwargs
    ):
        calls["aembedding"].append(
            {
                "model": model,
                "input": input,
                "dimensions": dimensions,
                "input_type": input_type,
                "kwargs": kwargs,
            }
        )
        response = mocker.MagicMock()
        response.data = []
        embedding_dim = dimensions or 12
        for item in input:
            data_item = mocker.MagicMock()
            key = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
            np.random.seed(hash(key) % 2**32)
            data_item.embedding = np.random.randn(embedding_dim).tolist()
            response.data.append(data_item)
        return response

    async def mock_arerank(
        *, model, query, documents, top_n, return_documents=False, **kwargs
    ):
        calls["arerank"].append(
            {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents,
                "kwargs": kwargs,
            }
        )
        scores = []
        for doc in documents:
            rng = np.random.default_rng(hash(doc) % 2**32)
            scores.append(rng.random())
        scores = np.array(scores)
        indices = np.argsort(scores)[-top_n:][::-1]
        return types.SimpleNamespace(
            results=[
                types.SimpleNamespace(index=idx, relevance_score=scores[idx])
                for idx in indices
            ]
        )

    import litellm

    mocked_module = types.SimpleNamespace(
        acompletion=mocker.patch.object(
            litellm, "acompletion", side_effect=mock_acompletion
        ),
        atext_completion=mocker.patch.object(
            litellm, "atext_completion", side_effect=mock_atext_completion
        ),
        aembedding=mocker.patch.object(
            litellm, "aembedding", side_effect=mock_aembedding
        ),
        arerank=mocker.patch.object(litellm, "arerank", side_effect=mock_arerank),
        completion=mocker.patch.object(litellm, "completion"),
        text_completion=mocker.patch.object(litellm, "text_completion"),
        embedding=mocker.patch.object(litellm, "embedding"),
        rerank=mocker.patch.object(litellm, "rerank"),
    )
    return {"module": mocked_module, "calls": calls}
