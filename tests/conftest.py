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
        "flexrag.retrievers.elastic_retriever.Elasticsearch", return_value=mock_client
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
