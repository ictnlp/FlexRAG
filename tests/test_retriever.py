import tempfile
from pathlib import Path

from flexrag.common import Context
from flexrag.datasets.reader import LineDelimitedReader
from flexrag.models import ENCODERS, EncoderConfig, LiteLLMEncoderConfig
from flexrag.retrievers import (
    EditableRetriever,
    ElasticRetriever,
    ElasticRetrieverConfig,
    FlexRetriever,
    FlexRetrieverConfig,
    IndexFieldsConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from flexrag.retrievers.index import (
    RETRIEVER_INDEX,
    BM25IndexConfig,
    FaissIndexConfig,
    RetrieverIndexConfig,
)


def load_test_corpus_slice(path: Path, start: int, stop: int) -> list[Context]:
    contexts = []
    reader = LineDelimitedReader(path)
    for idx, data in enumerate(reader):
        if idx < start:
            continue
        if idx >= stop:
            break
        payload = dict(data)
        context_id = payload.pop("id")
        contexts.append(Context(context_id=context_id, data=payload))
    return contexts


def litellm_encoder_config() -> EncoderConfig:
    return EncoderConfig(
        encoder_type="litellm",
        litellm_config=LiteLLMEncoderConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            embedding_size=8,
        ),
    )


def contriever_fields_config() -> IndexFieldsConfig:
    return IndexFieldsConfig(
        indexed_fields=["text"],
        merge_method="max",
    )


def bm25_fields_config() -> IndexFieldsConfig:
    return IndexFieldsConfig(
        indexed_fields=["title", "section", "text"],
        merge_method="max",
    )


def build_contriever_index(encoder):
    index = RETRIEVER_INDEX.load(
        RetrieverIndexConfig(
            index_type="faiss",
            faiss_config=FaissIndexConfig(
                indexed_fields=["text"],
                merge_method="max",
            ),
        ),
        query_encoder=encoder,
    )
    assert index.query_encoder is encoder
    assert index.passage_encoder is encoder
    return index


def build_bm25_index():
    return RETRIEVER_INDEX.load(
        RetrieverIndexConfig(
            index_type="bm25",
            bm25_config=BM25IndexConfig(
                indexed_fields=["title", "section", "text"],
                merge_method="max",
            ),
        )
    )


class TestRetrievers:
    query = [
        "Who is Bruce Wayne?",
        "What is the capital of China?",
    ]

    def run_retriever(self, retriever: EditableRetriever):
        retriever.clear()
        assert len(retriever) == 0

        # load corpus
        data_path = Path(__file__).parent / "testcorp" / "testcorp.jsonl"
        dataset1 = load_test_corpus_slice(data_path, 0, 10000)
        dataset2 = load_test_corpus_slice(data_path, 10000, 20000)

        # testing add_passages
        retriever.add_passages(dataset1)
        assert len(retriever) == 10000

        # testing search without top_k option
        ctxs = retriever.search(self.query, disable_cache=True)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 10

        # testing search with top_k option
        ctxs = retriever.search(self.query, disable_cache=True, top_k=5)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 5

        # testing add_passages
        retriever.add_passages(dataset2)
        assert len(retriever) == 20000

        # testing search without top_k option
        ctxs = retriever.search(self.query, disable_cache=True)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 10

        # testing search with top_k option
        ctxs = retriever.search(self.query, disable_cache=True, top_k=5)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 5

        # testing clear method
        retriever.clear()
        assert len(retriever) == 0
        return

    def test_flex_retriever(self, mock_litellm_client):
        # load datasets
        data_path = Path(__file__).parent / "testcorp" / "testcorp.jsonl"
        dataset1 = load_test_corpus_slice(data_path, 0, 1000)
        dataset2 = load_test_corpus_slice(data_path, 1000, 2000)
        encoder = ENCODERS.load(litellm_encoder_config())
        with tempfile.TemporaryDirectory() as tempdir:
            # in mem retriever
            cfg = FlexRetrieverConfig(
                batch_size=512,
            )
            retriever = FlexRetriever(cfg)
            retriever.add_passages(dataset1, log_interval=1000)
            retriever.add_index(
                "contriever",
                build_contriever_index(encoder),
            )
            assert len(retriever) == 1000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # add new passages
            retriever.add_passages(dataset2, log_interval=1000)
            assert len(retriever) == 2000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # add new index
            retriever.add_index(
                "bm25",
                build_bm25_index(),
            )
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
                used_indexes=["contriever", "bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
                used_indexes=["contriever"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # save index to local
            retriever.save_to_local(tempdir)
            retriever.database.close()
            del retriever
            assert Path(tempdir).exists()
            assert Path(tempdir, "indexes").exists()
            assert Path(tempdir, "indexes", "contriever").exists()
            assert Path(tempdir, "indexes", "bm25").exists()
            assert Path(tempdir, "database.lmdb").exists()
            assert Path(
                tempdir, "indexes", "contriever", "context_mapping.pkl"
            ).exists()
            assert Path(tempdir, "indexes", "contriever", "raw").exists()
            assert Path(
                tempdir, "indexes", "contriever", "raw", "config.yaml"
            ).exists()
            assert not Path(
                tempdir, "indexes", "contriever", "index_fields_config.yaml"
            ).exists()
            assert not Path(
                tempdir, "indexes", "contriever", "multi_field_index_config.yaml"
            ).exists()
            dense_config = Path(
                tempdir, "indexes", "contriever", "config.yaml"
            ).read_text()
            assert "query_encoder_config" not in dense_config
            assert "passage_encoder_config" not in dense_config

        with tempfile.TemporaryDirectory() as tempdir:
            cfg = FlexRetrieverConfig(
                batch_size=512,
            )
            retriever = FlexRetriever(cfg)
            retriever.add_passages(dataset1 + dataset2, log_interval=1000)
            retriever.add_index(
                "bm25",
                build_bm25_index(),
            )
            retriever.save_to_local(tempdir)
            retriever.database.close()
            del retriever
            assert Path(tempdir, "indexes", "bm25", "context_mapping.pkl").exists()
            assert Path(tempdir, "indexes", "bm25", "raw").exists()
            assert not Path(
                tempdir, "indexes", "bm25", "index_fields_config.yaml"
            ).exists()
            assert not Path(
                tempdir, "indexes", "bm25", "multi_field_index_config.yaml"
            ).exists()
            retriever = FlexRetriever.load_from_local(tempdir)
            assert len(retriever) == 2000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                top_k=5,
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5
        return

    def test_flex_retriever_rebuilds_non_addable_index_with_full_database(self):
        retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2))
        retriever.add_index(
            "bm25",
            RETRIEVER_INDEX.load(
                RetrieverIndexConfig(
                    index_type="bm25",
                    bm25_config=BM25IndexConfig(
                        indexed_fields=["text"],
                        show_progress=False,
                    ),
                )
            ),
        )
        retriever.add_passages(
            [Context(context_id="bruce", data={"text": "Bruce Wayne guards Gotham."})]
        )
        retriever.add_passages(
            [
                Context(
                    context_id="capital",
                    data={"text": "Beijing is the capital of China."},
                )
            ]
        )

        results = retriever.search(
            ["Bruce Wayne", "capital of China"],
            disable_cache=True,
            top_k=1,
            used_indexes=["bm25"],
        )

        assert results[0][0].context_id == "bruce"
        assert results[1][0].context_id == "capital"

    def test_elastic_retriever(self, mock_es_client):
        # load retriever
        retriever = ElasticRetriever(
            ElasticRetrieverConfig(
                host="http://127.0.0.1:9200",
                index_name="testing",
            )
        )
        self.run_retriever(retriever)
        return

    def test_typesense_retriever(self, mock_ts_client):
        # load retriever
        retriever = TypesenseRetriever(
            TypesenseRetrieverConfig(
                api_key="test_api_key",
                host="127.0.0.1",
                port=8108,
                index_name="testing",
            )
        )
        self.run_retriever(retriever)
        return
