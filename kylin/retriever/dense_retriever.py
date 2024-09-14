import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Iterable, Optional

import numpy as np
import tables
from omegaconf import MISSING

from kylin.utils import SimpleProgressLogger, Choices
from kylin.models import EncoderBase, HFEncoder, HFEncoderConfig
from .index import (
    FaissIndex,
    FaissIndexConfig,
    ScaNNIndex,
    ScaNNIndexConfig,
    DenseIndex,
)
from .retriever_base import LocalRetriever, LocalRetrieverConfig, RetrievedContext
from .fingerprint import Fingerprint


logger = logging.getLogger("DenseRetriever")


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig):
    read_only: bool = True
    database_path: str = MISSING
    index_type: Choices(["faiss", "scann"]) = "faiss"  # type: ignore
    faiss_index_config: FaissIndexConfig = field(default_factory=FaissIndexConfig)
    scann_index_config: ScaNNIndexConfig = field(default_factory=ScaNNIndexConfig)
    query_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_query_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)
    passage_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_passage_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)
    source: str = MISSING


class DenseRetriever(LocalRetriever):
    name = "Dense Retrieval"
    index: DenseIndex

    def __init__(self, cfg: DenseRetrieverConfig) -> None:
        super().__init__(cfg)
        # set args
        self.database_path = cfg.database_path
        self.source = cfg.source

        # load encoder
        self.query_encoder = self.load_encoder(
            cfg.query_encoder_type, cfg.hf_query_encoder_config
        )
        self.passage_encoder = self.load_encoder(
            cfg.passage_encoder_type, cfg.hf_passage_encoder_config
        )

        # load database
        self.read_only = cfg.read_only
        db_path = os.path.join(self.database_path, "database.h5")
        self.db_file, self.db_table = self.load_database(db_path)

        # load fingerprint
        self._fingerprint = Fingerprint(
            features={
                "database_path": cfg.database_path,
                "source": cfg.source,
                "index_type": cfg.index_type,
            }
        )

        # load / build index
        self.index = self.load_index(
            index_type=cfg.index_type,
            faiss_index_config=cfg.faiss_index_config,
            scann_index_config=cfg.scann_index_config,
        )

        # consistency check
        assert len(self.db_table) == len(self.index), "Inconsistent database and index"
        return

    def load_encoder(
        self,
        encoder_type: str,
        hf_encoder_config: HFEncoderConfig,
    ) -> EncoderBase:
        match encoder_type:
            case "hf":
                return HFEncoder(hf_encoder_config)
            case None:
                return None
            case _:
                raise ValueError(f"Encoder type {encoder_type} is not supported")

    def load_database(self, database_path: str) -> tuple[tables.File, tables.Table]:
        # open database file
        if os.path.exists(database_path):
            mode = "r" if self.read_only else "r+"
            h5file = tables.open_file(database_path, mode=mode)
        else:
            assert not self.read_only, "Database does not exist"
            h5file = tables.open_file(
                database_path, mode="w", title="Retriever Database"
            )

        # load table from database
        node_path = f"/{self.source}"
        if node_path in h5file:
            table = h5file.get_node(f"/{self.source}")
        else:

            class RetrieveTableWithEmb(tables.IsDescription):
                title = tables.StringCol(itemsize=self.max_query_length * 4, pos=1)
                section = tables.StringCol(itemsize=self.max_query_length * 4, pos=2)
                text = tables.StringCol(itemsize=self.max_passage_length * 4, pos=3)
                embedding = tables.Float16Col(self.embedding_size, pos=4)

            table = h5file.create_table("/", self.source, RetrieveTableWithEmb)
        return h5file, table

    def load_index(
        self,
        index_type: str,
        faiss_index_config: FaissIndexConfig,
        scann_index_config: ScaNNIndexConfig,
    ) -> DenseIndex:
        match index_type:
            case "faiss":
                index_path = os.path.join(self.database_path, f"faiss_{self.source}")
                if (faiss_index_config.embedding_size is None) and (
                    self.passage_encoder is not None
                ):
                    faiss_index_config.embedding_size = self.embedding_size
                return FaissIndex(index_path, faiss_index_config)
            case "scann":
                index_path = os.path.join(self.database_path, f"scann_{self.source}")
                if (scann_index_config.embedding_size is None) and (
                    self.passage_encoder is not None
                ):
                    scann_index_config.embedding_size = self.embedding_size
                return ScaNNIndex(index_path, scann_index_config)
            case _:
                raise ValueError(f"Index type {index_type} is not supported")

    def _add_passages(self, passages: Iterable[dict[str, str]]):
        """
        Add passages to the retriever database
        """

        def get_batch() -> Generator[list[dict[str, str]]]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                batch.append(passage)
            if batch:
                yield batch
            return

        # generate embeddings
        assert self.passage_encoder is not None, "Passage encoder is not provided"
        total_length = len(passages) if isinstance(passages, list) else None
        p_logger = SimpleProgressLogger(
            logger, total=total_length, interval=self.log_interval
        )
        start_idx = len(self.db_table)
        for batch in get_batch():
            texts = [i["text"] for i in batch]
            embeddings = self.passage_encoder.encode(texts)
            p_logger.update(step=self.batch_size, desc="Encoding passages")

            # add embeddings to database
            for i, p in enumerate(batch):
                row = self.db_table.row
                row["title"] = p["title"].encode("utf-8")
                row["section"] = p["section"].encode("utf-8")
                row["text"] = p["text"].encode("utf-8")
                row["embedding"] = embeddings[i]
                row.append()
            self.db_table.flush()
            self._fingerprint.update(texts)

        if not self.index.is_trained:  # train index
            logger.info("Training index")
            logger.warning("Training index will consume a lot of memory")
            full_embeddings = np.array(self.db_table.cols.embedding[:])
            self.index.build_index(full_embeddings)
        else:  # add embeddings to index
            full_embeddings = np.array(self.db_table.cols.embedding[start_idx:])
            self.index.add_embeddings(full_embeddings)
            self.index.serialize()
            logger.info("Finished adding passages")
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        embeddings = self.query_encoder.encode(query)
        indices, scores = self.index.search(embeddings, top_k, **search_kwargs)
        results = [
            [
                RetrievedContext(
                    retriever=self.name,
                    query=q,
                    source=self.source,
                    chunk_id=int(chunk_id),
                    score=float(s),
                    title=self.db_table[chunk_id]["title"].decode(),
                    section=self.db_table[chunk_id]["section"].decode(),
                    text=self.db_table[chunk_id]["text"].decode(),
                    full_text=self.db_table[chunk_id]["text"].decode(),
                )
                for chunk_id, s in zip(idx, score)
            ]
            for q, idx, score in zip(query, indices, scores)
        ]
        return results

    def clean(self) -> None:
        self.db_table.remove_rows(0, len(self.db_table))
        self.db_table.flush()
        self.index.clean()
        self._fingerprint.clean()
        return

    def close(self):
        self.db_table.flush()
        self.db_file.close()
        return

    def __len__(self):
        return len(self.db_table)

    @property
    def embedding_size(self) -> int:
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if hasattr(self, "index"):
            return self.index.embedding_size
        if hasattr(self, "db_table"):
            return self.db_table.description.embedding.shape[0]
        raise ValueError(
            "No encoder or database is provided, embedding size can not be determined."
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()
