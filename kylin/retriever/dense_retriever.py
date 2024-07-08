import logging
import os
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Iterable, Optional

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
from .retriever_base import LocalRetriever, LocalRetrieverConfig


logger = logging.getLogger("DenseRetriever")


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig):
    read_only: bool = False
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
        fp_path = os.path.join(self.database_path, "fingerprint")
        self._fingerprint = self.load_fingerprint(fp_path)

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
                title = tables.StringCol(itemsize=self.max_query_length, pos=1)
                section = tables.StringCol(itemsize=self.max_query_length, pos=2)
                text = tables.StringCol(itemsize=self.max_passage_length, pos=3)
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

    def load_fingerprint(self, fingerprint_path: str) -> str:
        if os.path.exists(fingerprint_path):
            with open(fingerprint_path, "r") as f:
                return f.read()
        fp = uuid4().hex
        if len(self) > 0:
            logger.warning("Fingerprint is missing, regenerate the fingerprint")
        with open(fingerprint_path, "w") as f:
            f.write(fp)
        return fp

    def add_passages(
        self, passages: Iterable[dict[str, str]] | Iterable[str], reinit: bool = False
    ):
        """
        Add passages to the retriever database
        """
        # reinitialize database
        if reinit:
            logger.info("Reinitializing database")
            self.db_table.remove_rows(0, len(self.db_table))
            self.index.clear()
            self.db_table.flush()

        # generate embeddings
        assert self.passage_encoder is not None, "Passage encoder is not provided"
        total_length = len(passages) if isinstance(passages, list) else None
        p_logger = SimpleProgressLogger(
            logger, total=total_length, interval=self.log_interval
        )
        start_idx = len(self.db_table)
        for batch, sources in self._get_batch(passages):
            embeddings = self.passage_encoder.encode(batch)
            p_logger.update(step=self.batch_size, desc="Encoding passages")

            # add embeddings to database
            for i, p in enumerate(sources):
                row = self.db_table.row
                row["title"] = self._safe_encode(p.get("title", ""))
                row["section"] = self._safe_encode(p.get("section", ""))
                row["text"] = self._safe_encode(p.get("text", ""))
                row["embedding"] = embeddings[i]
                row.append()
            self.db_table.flush()
            self._fingerprint = uuid4()  # update fingerprint

        indices = np.arange(start_idx, len(self.db_table))
        if not self.index.is_trained:  # train index
            logger.info("Training index")
            logger.warning("Training index will consume a lot of memory")
            full_embeddings = np.array(self.db_table.cols.embedding[start_idx:])
            self.index.build_index(full_embeddings, indices)
        else:  # add embeddings to index
            self.index.add_embeddings(full_embeddings, indices, self.batch_size)
            self.index.serialize()
            logger.info("Finished adding passages")
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[dict[str, str]]]:
        texts = [self._prepare_text(q) for q in query]
        embeddings = self.query_encoder.encode(texts)
        indices, scores = self.index.search(embeddings, top_k, **search_kwargs)
        results = [
            [
                {
                    "retriever": self.name,
                    "query": q,
                    "source": self.source,
                    "chunk_id": chunk_id,
                    "score": float(s),
                    "title": self.db_table[chunk_id]["title"].decode(),
                    "section": self.db_table[chunk_id]["section"].decode(),
                    "text": self.db_table[chunk_id]["text"].decode(),
                    "full_text": self.db_table[chunk_id]["text"].decode(),
                }
                for chunk_id, s in zip(idx, score)
            ]
            for q, idx, score in zip(query, indices, scores)
        ]
        return results

    def close(self):
        self.db_table.flush()
        self.db_file.close()
        return

    def _safe_encode(self, text: str) -> bytes:
        text = text.encode("utf-8")
        if len(text) <= self.max_passage_length:
            return text

        # do not truncate in the middle of a character
        trunc_point = self.max_passage_length
        while trunc_point > 0 and (text[trunc_point] & 0xC0) == 0x80:
            trunc_point -= 1

        # ensure utf-8 encoding
        while trunc_point > 0:
            try:
                _ = text[:trunc_point].decode("utf-8")
                break
            except:
                trunc_point -= 1
        return text[:trunc_point]

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

    def _get_batch(
        self, passages: Iterable[dict[str, str] | str]
    ) -> Iterable[tuple[list[str], list[dict[str, str]]]]:
        batch = []
        sources = []
        for passage in passages:
            passage = {"text": passage} if isinstance(passage, str) else passage
            if len(batch) == self.batch_size:
                yield batch, sources
                batch = []
                sources = []
            batch.append(self._prepare_text(passage))
            sources.append(passage)
        if batch:
            yield batch, sources
        return

    @property
    def fingerprint(self) -> str:
        return self._fingerprint
