import re
from typing import Optional

import numpy as np
import pytest


@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client for testing without actual API calls."""

    def create_mock_chat_completion(model, messages, **kwargs):
        """Mock chat completion response."""
        n = kwargs.get("n", 1)

        mock_response = mocker.MagicMock()
        mock_response.choices = []
        if (messages[0]["role"] == "system") and ("RankGPT" in messages[0]["content"]):
            # for rankgpt, return a mocked response with passage numbers
            passage_num = int(re.match(".*(\d+)", messages[1]["content"]).group(1))
            mocked_content = " > ".join([f"[{i + 1}]" for i in range(passage_num)])
            for i in range(n):
                choice = mocker.MagicMock()
                choice.message.content = mocked_content
                mock_response.choices.append(choice)
        else:
            # for normal chat completions, return a mocked response
            for i in range(n):
                choice = mocker.MagicMock()
                choice.message.content = f"Mocked response {i+1} for model {model}"
                mock_response.choices.append(choice)

        return mock_response

    def create_mock_text_completion(model, prompt, **kwargs):
        """Mock text completion response."""
        n = kwargs.get("n", 1)
        mock_response = mocker.MagicMock()
        mock_response.choices = []

        for i in range(n):
            choice = mocker.MagicMock()
            choice.text = f"Mocked text completion {i+1} for model {model}"
            mock_response.choices.append(choice)

        return mock_response

    def create_mock_embedding(model, input, dimensions=None, **kwargs):
        """Mock embedding response with configurable dimensions."""
        if not isinstance(input, list):
            input = [input]

        # Default embedding size based on common OpenAI models
        default_dim = 1536 if "ada" in model.lower() else 1024
        embedding_dim = dimensions if dimensions is not None else default_dim

        mock_response = mocker.MagicMock()
        mock_response.data = []

        for i, text in enumerate(input):
            data_item = mocker.MagicMock()
            # Create deterministic embedding based on text hash for consistency
            np.random.seed(hash(text) % 2**32)
            data_item.embedding = np.random.randn(embedding_dim).tolist()
            mock_response.data.append(data_item)

        return mock_response

    def create_mock_models_list():
        """Mock models list response."""
        mock_response = mocker.MagicMock()
        mock_response.data = [
            mocker.MagicMock(id="gpt-3.5-turbo"),
            mocker.MagicMock(id="gpt-4"),
            mocker.MagicMock(id="gpt-4.1"),
            mocker.MagicMock(id="text-davinci-003"),
            mocker.MagicMock(id="text-embedding-ada-002"),
            mocker.MagicMock(id="text-embedding-3-small"),
        ]
        return mock_response

    # Create the mock client
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create = create_mock_chat_completion
    mock_client.completions.create = create_mock_text_completion
    mock_client.embeddings.create = create_mock_embedding
    mock_client.models.list = create_mock_models_list
    mocker.patch("flexrag.models.openai_model.OpenAI", return_value=mock_client)
    mocker.patch("flexrag.models.openai_model.AzureOpenAI", return_value=mock_client)

    return mock_client


@pytest.fixture
def mock_ollama_client(mocker):
    """Mock Ollama client for testing without actual server connection."""

    def create_mock_chat_response(model, messages, options=None):
        """Mock chat response."""
        mock_response = mocker.MagicMock()
        mock_response.message.content = f"Mocked chat response for model {model}"
        return mock_response

    def create_mock_generate_response(model, prompt, raw=True, options=None):
        """Mock generate response."""
        mock_response = mocker.MagicMock()
        mock_response.response = f"Mocked generate response for model {model}"
        return mock_response

    def create_mock_embeddings_response(model, prompt):
        """Mock embeddings response with deterministic output."""
        # Create deterministic embedding based on text hash for consistency
        rng = np.random.default_rng(hash(prompt) % 2**32)
        embedding_dim = 768  # Default embedding size for Ollama
        embedding = rng.random(embedding_dim).tolist()
        return {"embedding": embedding}

    def create_mock_list_response():
        """Mock models list response."""
        return {
            "models": [
                {"model": "contriever", "name": "contriever"},
                {"model": "llama2", "name": "llama2"},
                {"model": "mistral", "name": "mistral"},
            ]
        }

    # Create the mock client
    mock_client = mocker.MagicMock()
    mock_client.chat = create_mock_chat_response
    mock_client.generate = create_mock_generate_response
    mock_client.embeddings = create_mock_embeddings_response
    mock_client.list = create_mock_list_response

    # Mock the entire ollama module to handle lazy import
    mock_ollama_module = mocker.MagicMock()
    mock_ollama_module.Client = mocker.MagicMock(return_value=mock_client)

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"ollama": mock_ollama_module})
    return mock_client


@pytest.fixture
def mock_jina_client(mocker):
    """Mock JinaEncoder HTTP clients to avoid actual API calls."""

    # Store base_url for both sync and async clients
    client_configs = {"sync_base_url": None, "async_base_url": None}

    def create_mock_response(url, json: dict):
        """Create mock response data for Jina API."""
        if "rerank" in url:
            assert "query" in json, "Query must be provided in json"
            scores = []
            for doc in json["documents"]:
                # Create deterministic scores based on document content
                rng = np.random.default_rng(hash(doc) % 2**32)
                scores.append(rng.random())
            scores = np.array(scores)
            indices = np.argsort(scores)[-json["top_n"] :][::-1]
            return {
                "model": json["model"],
                "usage": np.random.randint(100, 1000),
                "results": [
                    {"index": i, "relevance_score": scores[i]} for i in indices
                ],
            }
        elif "embed" in url:
            # Mock embeddings with fixed dimensions (1024 as per default config)
            embeddings = []
            for i, text in enumerate(json["input"]):
                # Create deterministic embeddings based on text content
                rng = np.random.default_rng(hash(text) % 2**32)
                embedding = rng.random(1024).tolist()
                embeddings.append({"embedding": embedding})
            return {
                "data": embeddings,
                "model": "jina-embeddings-v3",
                "usage": {
                    "total_tokens": sum(len(text.split()) for text in json["input"])
                },
            }
        else:
            raise ValueError(f"Unexpected URL: {url}")

    # Mock synchronous client
    mock_sync_response = mocker.MagicMock()
    mock_sync_response.raise_for_status.return_value = None

    def mock_sync_post(url, json=None):
        base_url = client_configs["sync_base_url"]
        full_url = (
            url if url.startswith("http") else f"{base_url}{url}" if base_url else url
        )
        mock_sync_response.json.return_value = create_mock_response(full_url, json)
        return mock_sync_response

    # Mock asynchronous client
    mock_async_response = mocker.AsyncMock()
    mock_async_response.raise_for_status.return_value = None

    async def mock_async_post(url, json=None):
        base_url = client_configs["async_base_url"]
        full_url = (
            url if url.startswith("http") else f"{base_url}{url}" if base_url else url
        )
        mock_async_response.json.return_value = create_mock_response(full_url, json)
        return mock_async_response

    # Create mock client instances that capture base_url
    def create_mock_sync_client(base_url=None, **kwargs):
        client_configs["sync_base_url"] = base_url
        mock_instance = mocker.MagicMock()
        mock_instance.base_url = base_url
        mock_instance.post = mock_sync_post
        return mock_instance

    def create_mock_async_client(base_url=None, **kwargs):
        client_configs["async_base_url"] = base_url
        mock_instance = mocker.AsyncMock()
        mock_instance.base_url = base_url
        mock_instance.post = mock_async_post
        return mock_instance

    # Patch httpx clients
    mock_sync_client = mocker.patch("flexrag.models.jina_model.httpx.Client")
    mock_async_client = mocker.patch("flexrag.models.jina_model.httpx.AsyncClient")

    # Configure mock clients to use our factory functions
    mock_sync_client.side_effect = create_mock_sync_client
    mock_async_client.side_effect = create_mock_async_client

    # Configure mock clients
    mock_sync_client.return_value.post = mock_sync_post
    mock_async_client.return_value.post = mock_async_post

    return {
        "sync_client": mock_sync_client,
        "async_client": mock_async_client,
        "sync_response": mock_sync_response,
        "async_response": mock_async_response,
    }


@pytest.fixture
def mock_cohere_client(mocker):
    """Mock Cohere ClientV2 for testing CohereEncoder"""

    # Mock embedding response
    class MockEmbeddingResponse:
        def __init__(
            self,
            texts: list[str],
            model: str = "embed-v4.0",
            dimention: Optional[int] = None,
        ):
            match model:
                case "embed-multilingual-light-v3.0":
                    dim = 384
                case "embed-multilingual-v3.0":
                    dim = 1024
                case "embed-english-light-v3.0":
                    dim = 384
                case "embed-english-v3.0":
                    dim = 1024
                case "embed-v4.0":
                    if dimention is not None:
                        dim = dimention
                    dim = 1536
            self.embeddings = mocker.MagicMock()
            self.embeddings.float = []
            for text in texts:
                rng = np.random.default_rng(hash(text) % 2**32)
                self.embeddings.float.append(rng.random(dim).tolist())

    # Mock rerank response
    class MockRerankResponse:
        def __init__(self, documents: list[str], top_n: int, **kwargs):
            scores = []
            for doc in documents:
                rng = np.random.default_rng(hash(doc) % 2**32)
                scores.append(rng.random())
            scores = np.array(scores)
            indices = np.argsort(scores)[-top_n:][::-1]
            documents = [documents[i] for i in indices]
            self.results = [{"index": i, "relevance_score": scores[i]} for i in indices]
            self.id = "07734bd2-2473-4f07-94e1-0d9f0e6843cf"
            self.meta = {
                "api_version": {"version": "2", "is_experimental": False},
                "billed_units": {"search_units": 1},
            }

    # Mock ClientV2
    mock_client = mocker.MagicMock()

    def mock_embed(
        texts,
        model=None,
        output_dimension=None,
        **kwargs,
    ):
        return MockEmbeddingResponse(
            texts,
            model=model,
            dimention=output_dimension,
        )

    def mock_rerank(
        query: str,
        documents: list[str],
        model: str = "rerank-v3.5",
        top_n: int = 10,
        **kwargs,
    ):
        return MockRerankResponse(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
            **kwargs,
        )

    mock_client.embed = mock_embed
    mock_client.rerank = mock_rerank

    # Mock the entire cohere module to handle lazy import
    mock_cohere_module = mocker.MagicMock()
    mock_cohere_module.ClientV2.return_value = mock_client

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"cohere": mock_cohere_module})
    return mock_client


@pytest.fixture
def mock_anthropic_client(mocker):
    """Mock Anthropic client for testing AnthropicGenerator"""

    # Mock the response content structure
    mock_content = mocker.MagicMock()
    mock_content.text = "This is a mocked response from Anthropic."

    mock_response = mocker.MagicMock()
    mock_response.content = [mock_content]

    # Mock the Anthropic client
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value = mock_response

    # Mock the entire anthropic module to handle lazy import
    mock_anthropic_module = mocker.MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_module})

    return mock_client


@pytest.fixture
def mock_es_client(mocker):
    client_state = {
        "indexes": {},
    }

    def mocked_msearch(**kwargs):
        searched_meta = [i for i in kwargs.get("body") if "index" in i]
        searched_data = [i for i in kwargs.get("body") if "index" not in i]
        responses = []
        for meta, data in zip(searched_meta, searched_data):
            top_k = data.get("size", 10)
            responses.append(
                {
                    "took": 123,
                    "timed_out": False,
                    "_shards": {"total": 1, "successful": 1, "failed": 0, "skipped": 0},
                    "hits": {
                        "total": {"value": len(client_state["indexes"][meta["index"]])},
                        "max_score": 1.0,
                        "hits": [
                            {
                                "_index": meta["index"],
                                "_id": str(i),
                                "_score": 1.0,
                                "_source": client_state["indexes"][meta["index"]][i],
                            }
                            for i in range(
                                min(top_k, len(client_state["indexes"][meta["index"]]))
                            )
                        ],
                    },
                    "status": 200,
                }
            )
        return {"responses": responses}

    def mocked_bulk(**kwargs):
        inserted_meta = [i for i in kwargs.get("operations", []) if "index" in i]
        inserted_data = [i for i in kwargs.get("operations", []) if "index" not in i]
        items = []
        for meta, data in zip(inserted_meta, inserted_data):
            client_state["indexes"][meta["index"]["_index"]].append(data)
            items.append(
                {
                    "index": {
                        "_index": meta["index"]["_index"],
                        "_id": data["_id"] if "_id" in data else None,
                        "_version": 1,
                        "status": 201,
                        "result": "created",
                        "_shards": {"total": 2, "successful": 1, "failed": 0},
                        "_seq_no": 0,
                        "_primary_term": 1,
                    }
                }
            )
        returned_obj = mocker.MagicMock()
        returned_obj.body = {"errors": False, "took": 123456, "items": items}
        return returned_obj

    def mocked_create(**kwargs):
        if kwargs.get("index") in client_state["indexes"]:
            raise ValueError("Index already exists")
        client_state["indexes"][kwargs.get("index")] = []
        create_obj = mocker.MagicMock()
        create_obj.body = {
            "acknowledged": True,
            "shards_acknowledged": True,
            "index": "mocked_index",
        }
        return create_obj

    def mocked_delete(**kwargs):
        if kwargs.get("index") not in client_state["indexes"]:
            raise ValueError("Index does not exist")
        client_state["indexes"].pop(kwargs.get("index"))
        delete_obj = mocker.MagicMock()
        delete_obj.body = {"acknowledged": True}
        return delete_obj

    def mocked_count(**kwargs):
        index_name = kwargs.get("index")
        if index_name not in client_state["indexes"]:
            raise ValueError("Index does not exist")
        # count_obj = mocker.MagicMock()
        # count_obj.body = {"count": len(client_state["indexes"][index_name])}
        # count_obj.__getitem__ = lambda self, key: self.body[key]
        return {"count": len(client_state["indexes"][index_name])}

    def mocked_exists(**kwargs):
        index_name = kwargs.get("index")
        exists_obj = mocker.MagicMock()
        exists_obj.body = index_name in client_state["indexes"]
        exists_obj.__bool__ = lambda self: self.body
        return exists_obj

    def mocked_cat_indices(**kwargs):
        # Return a mocked list of indices
        body = [{"index": index} for index in client_state["indexes"].keys()]
        return body

    # create mock client
    mock_client = mocker.MagicMock()

    # mock the Elasticsearch.indices
    mock_client.indices = mocker.MagicMock()

    # mock the Elasticsearch.indices.exists
    mock_client.indices.exists = mocked_exists

    # mock the Elasticsearch.indices.create
    mock_client.indices.create = mocked_create

    # mock the Elasticsearch.indices.delete
    mock_client.indices.delete = mocked_delete

    # mock the Elasticsearch.bulk
    mock_client.bulk = mocked_bulk

    # mock the Elasticsearch.count
    mock_client.count = mocked_count

    # mock the Elasticsearch.msearch
    mock_client.msearch = mocked_msearch

    # mock the Elasticsearch.cat
    mock_client.cat = mocker.MagicMock()
    mock_client.cat.indices = mocked_cat_indices

    # substitute the original Elasticsearch client with the mock
    mocker.patch(
        "flexrag.retriever.elastic_retriever.Elasticsearch", return_value=mock_client
    )
    return mock_client


@pytest.fixture
def mock_ts_client(mocker):
    """Mock Typesense client for testing without actual server connection."""

    client_state = {
        "collections": {},
    }

    def mocked_collections_retrieve():
        """Mock collections retrieve method."""
        return [
            {"name": name, "num_documents": len(docs)}
            for name, docs in client_state["collections"].items()
        ]

    def mocked_collections_create(schema):
        """Mock collections create method."""
        collection_name = schema["name"]
        if collection_name in client_state["collections"]:
            raise ValueError(f"Collection {collection_name} already exists")
        client_state["collections"][collection_name] = []

        # Mock collection object
        mock_collection = mocker.MagicMock()
        mock_collection.name = collection_name
        return mock_collection

    def mocked_collections_getitem(instance, collection_name):
        """Mock collections __getitem__ method."""
        if collection_name not in client_state["collections"]:
            raise KeyError(f"Collection {collection_name} not found")

        # Create mock collection with documents attribute
        mock_collection = mocker.MagicMock()
        mock_collection.name = collection_name

        def mocked_documents_import(documents):
            """Mock documents import method."""
            client_state["collections"][collection_name].extend(documents)
            return [{"success": True} for _ in documents]

        def mocked_collection_retrieve():
            """Mock collection retrieve method."""
            return {
                "name": collection_name,
                "num_documents": len(client_state["collections"][collection_name]),
                "fields": [
                    {"name": "id", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "text", "type": "string"},
                    {"name": "section", "type": "string"},
                ],
            }

        def mocked_collection_delete():
            """Mock collection delete method."""
            if collection_name in client_state["collections"]:
                del client_state["collections"][collection_name]
            return {"acknowledged": True}

        # Configure mock collection
        mock_collection.documents = mocker.MagicMock()
        mock_collection.documents.import_ = mocked_documents_import
        mock_collection.retrieve = mocked_collection_retrieve
        mock_collection.delete = mocked_collection_delete

        return mock_collection

    def mocked_multisearch_perform(search_queries, common_params=None):
        """Mock multi-search perform method."""
        searches = search_queries.get("searches", [])
        results = []

        for search in searches:
            collection_name = search["collection"]
            query = search["q"]
            per_page = search.get("per_page", 10)

            if collection_name not in client_state["collections"]:
                results.append({"hits": [], "found": 0})
                continue

            # Simple mock search - return first per_page documents with mock scores
            documents = client_state["collections"][collection_name]
            hits = []

            for i, doc in enumerate(documents[:per_page]):
                # Create deterministic score based on query and document
                import hashlib

                score_seed = hashlib.md5(f"{query}{doc}".encode()).hexdigest()
                score = int(score_seed[:8], 16) / (16**8)  # Normalize to 0-1

                hits.append(
                    {"document": doc.copy(), "text_match": score, "highlights": []}
                )

            results.append(
                {
                    "hits": hits,
                    "found": len(hits),
                    "out_of": len(documents),
                    "page": 1,
                    "request_params": search,
                    "search_time_ms": 1,
                }
            )

        return {"results": results}

    # Create mock client
    mock_client = mocker.MagicMock()

    # Mock the collections interface
    mock_client.collections = mocker.MagicMock()
    mock_client.collections.retrieve = mocked_collections_retrieve
    mock_client.collections.create = mocked_collections_create
    mock_client.collections.__getitem__ = mocked_collections_getitem

    # Mock the multi_search interface
    mock_client.multi_search = mocker.MagicMock()
    mock_client.multi_search.perform = mocked_multisearch_perform

    # Mock exceptions
    mock_typesense_client_error = type("TypesenseClientError", (Exception,), {})

    # Mock the entire typesense module to handle lazy import
    mock_typesense_module = mocker.MagicMock()
    mock_typesense_module.Client = mocker.MagicMock(return_value=mock_client)
    mock_typesense_module.exceptions = mocker.MagicMock()
    mock_typesense_module.exceptions.TypesenseClientError = mock_typesense_client_error

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"typesense": mock_typesense_module})

    return mock_client


@pytest.fixture
def mock_mixedbread_client(mocker):
    """Mock Mixedbread client for testing MixedbreadRanker"""

    # Mock the MixedbreadAI client
    mock_client = mocker.MagicMock()

    def mock_reranking(input, model, top_k, **kwargs):
        scores = []
        for doc in input:
            rng = np.random.default_rng(hash(doc) % 2**32)
            scores.append(rng.random())
        scores = np.array(scores)
        indices = np.argsort(scores)[-top_k:][::-1]
        return mocker.MagicMock(
            data=[
                mocker.MagicMock(score=scores[idx], index=idx, input=None)
                for idx in indices
            ],
            model=model,
            usage=mocker.MagicMock(
                total_tokens=np.random.randint(100, 1000),
                prompt_tokens=np.random.randint(50, 500),
                completion_tokens=np.random.randint(50, 500),
            ),
        )

    mock_client.rerank = mock_reranking

    # Mock the entire mixedbread module to handle lazy import
    mock_mixedbread_module = mocker.MagicMock()
    mock_mixedbread_module.Mixedbread.return_value = mock_client

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"mixedbread": mock_mixedbread_module})
    return mock_client


@pytest.fixture
def mock_voyage_client(mocker):
    """Mock Mixedbread client for testing MixedbreadRanker"""

    # Mock the Voyage client
    mock_client = mocker.MagicMock()

    def mock_reranking(documents, top_k, **kwargs):
        scores = []
        for doc in documents:
            rng = np.random.default_rng(hash(doc) % 2**32)
            scores.append(rng.random())
        scores = np.array(scores)
        indices = np.argsort(scores)[-top_k:][::-1]
        return mocker.MagicMock(
            results=[
                mocker.MagicMock(
                    relevance_score=scores[idx],
                    index=idx,
                    document=documents[idx],
                )
                for idx in indices
            ],
            total_tokens=np.random.randint(100, 1000),
        )

    mock_client.rerank = mock_reranking

    # Mock the entire voyage module to handle lazy import
    mock_voyage_module = mocker.MagicMock()
    mock_voyage_module.Client.return_value = mock_client

    # Patch the module import itself
    mocker.patch.dict("sys.modules", {"voyageai": mock_voyage_module})
    return mock_client
