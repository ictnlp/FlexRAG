import pytest


@pytest.fixture
def mock_es_client(mocker):
    client_state = {
        "indexes": {},
    }

    async def mocked_msearch(**kwargs):
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

    async def mocked_bulk(**kwargs):
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

    async def mocked_create(**kwargs):
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

    async def mocked_delete(**kwargs):
        if kwargs.get("index") not in client_state["indexes"]:
            raise ValueError("Index does not exist")
        client_state["indexes"].pop(kwargs.get("index"))
        delete_obj = mocker.MagicMock()
        delete_obj.body = {"acknowledged": True}
        return delete_obj

    async def mocked_count(**kwargs):
        index_name = kwargs.get("index")
        if index_name not in client_state["indexes"]:
            raise ValueError("Index does not exist")
        # count_obj = mocker.MagicMock()
        # count_obj.body = {"count": len(client_state["indexes"][index_name])}
        # count_obj.__getitem__ = lambda self, key: self.body[key]
        return {"count": len(client_state["indexes"][index_name])}

    async def mocked_exists(**kwargs):
        index_name = kwargs.get("index")
        exists_obj = mocker.MagicMock()
        exists_obj.body = index_name in client_state["indexes"]
        exists_obj.__bool__ = lambda self: self.body
        return exists_obj

    async def mocked_get_mapping(**kwargs):
        index_name = kwargs.get("index")
        return {
            index_name: {
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "section": {"type": "text"},
                    }
                }
            }
        }

    async def mocked_get(**kwargs):
        index_name = kwargs.get("index")
        context_id = kwargs.get("id")
        try:
            data = client_state["indexes"][index_name][int(context_id)]
        except (KeyError, IndexError, ValueError):
            raise KeyError(context_id)
        return {
            "_index": index_name,
            "_id": context_id,
            "_source": data,
        }

    async def mocked_cat_indices(**kwargs):
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

    # mock the Elasticsearch.indices.get_mapping
    mock_client.indices.get_mapping = mocked_get_mapping

    # mock the Elasticsearch.bulk
    mock_client.bulk = mocked_bulk

    # mock the Elasticsearch.count
    mock_client.count = mocked_count

    # mock the Elasticsearch.get
    mock_client.get = mocked_get

    # mock the Elasticsearch.msearch
    mock_client.msearch = mocked_msearch

    # mock the Elasticsearch.cat
    mock_client.cat = mocker.MagicMock()
    mock_client.cat.indices = mocked_cat_indices
    mock_client.close = mocker.AsyncMock()

    # substitute the original Elasticsearch client with the mock
    mocker.patch(
        "flexrag.retrievers.elastic_retriever.AsyncElasticsearch",
        return_value=mock_client,
    )
    return mock_client
