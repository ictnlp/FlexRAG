import os
import tempfile
import uuid
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.models import EncoderConfig, HFEncoderConfig
from flexrag.retriever import (
    BM25SRetriever,
    BM25SRetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
    ElasticRetriever,
    ElasticRetrieverConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)


@dataclass
class RetrieverTestConfig:
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)  # fmt: skip
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)  # fmt: skip


@pytest.fixture
def setup_elastic():
    TestRetrievers.cfg.elastic_config.index_name = str(uuid.uuid4())
    retriever = ElasticRetriever(TestRetrievers.cfg.elastic_config)
    yield retriever
    retriever.clean()
    return


@pytest.fixture
def setup_typesense():
    TestRetrievers.cfg.typesense_config.index_name = str(uuid.uuid4())
    retriever = TypesenseRetriever(TestRetrievers.cfg.typesense_config)
    yield retriever
    retriever.clean()
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

    def test_dense_retriever(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # load retriever
            retriever_cfg = DenseRetrieverConfig(
                top_k=10,
                database_path=tempdir,
                query_encoder_config=EncoderConfig(
                    encoder_type="hf",
                    hf_config=HFEncoderConfig(
                        model_path="sentence-transformers/all-MiniLM-L6-v2",
                        device_id=[0],
                    ),
                ),
                passage_encoder_config=EncoderConfig(
                    encoder_type="hf",
                    hf_config=HFEncoderConfig(
                        model_path="sentence-transformers/all-MiniLM-L6-v2",
                        device_id=[0],
                    ),
                ),
                encode_fields=["text"],
                index_type="faiss",
                batch_size=512,
            )
            retriever = DenseRetriever(retriever_cfg)

            # build index
            data_cfg = RAGCorpusDatasetConfig(
                file_paths=[
                    os.path.join(
                        os.path.dirname(__file__), "testcorp", "testcorp.jsonl"
                    )
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

    def test_bm25s_retriever(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # load retriever
            retriever_cfg = BM25SRetrieverConfig(
                top_k=10,
                database_path=tempdir,
                method="lucene",
                idf_method="lucene",
                indexed_fields=["text"],
            )
            retriever = BM25SRetriever(retriever_cfg)

            # build index
            data_cfg = RAGCorpusDatasetConfig(
                file_paths=[
                    os.path.join(
                        os.path.dirname(__file__), "testcorp", "testcorp.jsonl"
                    )
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
