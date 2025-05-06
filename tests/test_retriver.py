import os
import tempfile
import uuid
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.models import EncoderConfig, HFEncoderConfig
from flexrag.retriever import (
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
from flexrag.utils import ConfigureBase


@dataclass
class RetrieverTestConfig(ConfigureBase):
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)  # fmt: skip
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)  # fmt: skip


@pytest.fixture
def setup_elastic():
    TestRetrievers.cfg.elastic_config.index_name = str(uuid.uuid4())
    retriever = ElasticRetriever(TestRetrievers.cfg.elastic_config)
    yield retriever
    retriever.clear()
    return


@pytest.fixture
def setup_typesense():
    TestRetrievers.cfg.typesense_config.index_name = str(uuid.uuid4())
    retriever = TypesenseRetriever(TestRetrievers.cfg.typesense_config)
    yield retriever
    retriever.clear()
    return


class TestRetrievers:
    cfg: RetrieverTestConfig = OmegaConf.merge(
        OmegaConf.structured(RetrieverTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "retriever.yaml")
        ),
    )
    query = [
        "Who is Bruce Wayne?",
        "What is the capital of China?",
    ]

    def test_flex_retriever(self):
        # load datasets
        cfg1 = RAGCorpusDatasetConfig(
            file_paths=[
                os.path.join(os.path.dirname(__file__), "testcorp", "testcorp.jsonl")
            ],
            data_ranges=[[0, 10000]],
            id_field="id",
        )
        cfg2 = RAGCorpusDatasetConfig(
            file_paths=[
                os.path.join(os.path.dirname(__file__), "testcorp", "testcorp.jsonl")
            ],
            data_ranges=[[10000, 20000]],
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
                            encoder_type="hf",
                            hf_config=HFEncoderConfig(
                                model_path="facebook/contriever-msmarco",
                                device_id=[2, 3],
                            ),
                        ),
                        passage_encoder_config=EncoderConfig(
                            encoder_type="hf",
                            hf_config=HFEncoderConfig(
                                model_path="facebook/contriever-msmarco",
                                device_id=[2, 3],
                            ),
                        ),
                    ),
                ),
                indexed_fields_config=MultiFieldIndexConfig(
                    indexed_fields=["text"],
                    merge_method="max",
                ),
            )
            assert len(retriever) == 10000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"]
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # add new passages
            retriever.add_passages(dataset2)
            assert len(retriever) == 20000
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
            assert os.path.exists(tempdir)
            assert os.path.exists(os.path.join(tempdir, "indexes"))
            assert os.path.exists(os.path.join(tempdir, "indexes", "contriever"))
            assert os.path.exists(os.path.join(tempdir, "indexes", "bm25"))
            assert os.path.exists(os.path.join(tempdir, "database.lance"))

            # load index from local
            retriever: FlexRetriever = FlexRetriever.load_from_local(tempdir)
            assert len(retriever) == 20000
            ctxs = retriever.search(
                ["Who is Bruce Wayne?", "What is the capital of France?"]
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            # remove one index
            retriever.remove_index("contriever")
            assert not os.path.exists(os.path.join(tempdir, "indexes", "contriever"))
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
            assert os.path.exists(os.path.join(tempdir, "indexes", "bm25"))
        return

    def test_elastic_retriever(self, setup_elastic):
        # load retriever
        retriever: ElasticRetriever = setup_elastic

        # build index
        data_cfg = RAGCorpusDatasetConfig(
            file_paths=[
                os.path.join(os.path.dirname(__file__), "testcorp", "testcorp.jsonl")
            ],
            id_field="id",
        )
        corpus = RAGCorpusDataset(data_cfg)
        retriever.add_passages(corpus)

        # search
        r = retriever.search(self.query, disable_cache=True)
        assert len(r) == 2
        assert len(r[0]) == 10
        assert len(r[1]) == 10
        return

    def test_typesense_retriever(self, setup_typesense):
        # load retriever
        retriever: TypesenseRetriever = setup_typesense

        # build index
        data_cfg = RAGCorpusDatasetConfig(
            file_paths=[
                os.path.join(os.path.dirname(__file__), "testcorp", "testcorp.jsonl")
            ],
            id_field="id",
        )
        corpus = RAGCorpusDataset(data_cfg)
        retriever.add_passages(corpus)

        # search
        r = retriever.search(self.query, disable_cache=True)
        assert len(r) == 2
        return
