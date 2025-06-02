import tempfile
from pathlib import Path

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.models import EncoderConfig, OpenAIEncoderConfig
from flexrag.retriever import (
    EditableRetriever,
    ElasticRetriever,
    ElasticRetrieverConfig,
    FlexRetriever,
    FlexRetrieverConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from flexrag.retriever.index import (
    BM25IndexConfig,
    FaissIndexConfig,
    MultiFieldIndexConfig,
    RetrieverIndexConfig,
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
        cfg1 = RAGCorpusDatasetConfig(
            file_paths=[Path(__file__).parent / "testcorp" / "testcorp.jsonl"],
            data_ranges=[[0, 10000]],
            id_field="id",
        )
        cfg2 = RAGCorpusDatasetConfig(
            file_paths=[Path(__file__).parent / "testcorp" / "testcorp.jsonl"],
            data_ranges=[[10000, 20000]],
            id_field="id",
        )
        dataset1 = RAGCorpusDataset(cfg1)
        dataset2 = RAGCorpusDataset(cfg2)

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

    def test_flex_retriever(self, mock_openai_client):
        # load datasets
        cfg1 = RAGCorpusDatasetConfig(
            file_paths=[Path(__file__).parent / "testcorp" / "testcorp.jsonl"],
            data_ranges=[[0, 1000]],
            id_field="id",
        )
        cfg2 = RAGCorpusDatasetConfig(
            file_paths=[Path(__file__).parent / "testcorp" / "testcorp.jsonl"],
            data_ranges=[[1000, 2000]],
            id_field="id",
        )
        dataset1 = RAGCorpusDataset(cfg1)
        dataset2 = RAGCorpusDataset(cfg2)
        with tempfile.TemporaryDirectory() as tempdir:
            # in mem retriever
            cfg = FlexRetrieverConfig(
                batch_size=512,
                used_indexes=["contriever"],
                top_k=5,
                log_interval=1000,
            )
            retriever = FlexRetriever(cfg)
            retriever.add_passages(dataset1)
            retriever.add_index(
                "contriever",
                index_config=RetrieverIndexConfig(
                    index_type="faiss",
                    faiss_config=FaissIndexConfig(
                        index_type="auto",
                        batch_size=512,
                        query_encoder_config=EncoderConfig(
                            encoder_type="openai",
                            openai_config=OpenAIEncoderConfig(
                                model_name="text-embedding-3-small",
                            ),
                        ),
                        passage_encoder_config=EncoderConfig(
                            encoder_type="openai",
                            openai_config=OpenAIEncoderConfig(
                                model_name="text-embedding-3-small",
                            ),
                        ),
                    ),
                ),
                indexed_fields_config=MultiFieldIndexConfig(
                    indexed_fields=["text"],
                    merge_method="max",
                ),
            )
            assert len(retriever) == 1000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"]
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # add new passages
            retriever.add_passages(dataset2)
            assert len(retriever) == 2000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"]
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # add new index
            retriever.add_index(
                "bm25",
                index_config=RetrieverIndexConfig(
                    index_type="bm25",
                    bm25_config=BM25IndexConfig(batch_size=512),
                ),
                indexed_fields_config=MultiFieldIndexConfig(
                    indexed_fields=["title", "section", "text"],
                    merge_method="max",
                ),
            )
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                used_indexes=["contriever", "bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                used_indexes=["contriever"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # save index to loacl
            retriever.save_to_local(tempdir)
            del retriever
            assert Path(tempdir).exists()
            assert Path(tempdir, "indexes").exists()
            assert Path(tempdir, "indexes", "contriever").exists()
            assert Path(tempdir, "indexes", "bm25").exists()
            assert Path(tempdir, "database.lmdb").exists()

            # load index from local
            retriever: FlexRetriever = FlexRetriever.load_from_local(tempdir)
            assert len(retriever) == 2000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"]
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # remove one index
            retriever.remove_index("contriever")
            assert not Path(tempdir, "indexes", "contriever").exists()
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # detach retriever from the disk
            retriever.detach()
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"],
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # remove another index
            retriever.remove_index("bm25")
            assert Path(tempdir, "indexes", "bm25").exists()
        return

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
