import atexit
import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Iterable, Optional

import numpy as np
import tables
from omegaconf import MISSING

from kylin.models import EncoderConfig, load_encoder
from kylin.utils import Choices, SimpleProgressLogger

from .fingerprint import Fingerprint
from .index import (
    DenseIndex,
    FaissIndex,
    FaissIndexConfig,
    ScaNNIndex,
    ScaNNIndexConfig,
)
from .retriever_base import (
    SEMANTIC_RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = logging.getLogger("DenseRetriever")


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig):
    read_only: bool = True
    database_path: str = MISSING
    index_type: Choices(["faiss", "scann"]) = "faiss"  # type: ignore
    faiss_index_config: FaissIndexConfig = field(default_factory=FaissIndexConfig)
    scann_index_config: ScaNNIndexConfig = field(default_factory=ScaNNIndexConfig)
    query_encoder_config: Optional[EncoderConfig] = None  # type: ignore
    passage_encoder_config: Optional[EncoderConfig] = None  # type: ignore
    source: str = MISSING


@SEMANTIC_RETRIEVERS("dense", config_class=DenseRetrieverConfig)
class DenseRetriever(LocalRetriever):
    name = "Dense Retrieval"
    index: DenseIndex

    def __init__(self, cfg: DenseRetrieverConfig) -> None:
        super().__init__(cfg)
        # set args
        self.database_path = cfg.database_path
        self.source = cfg.source

        # load encoder
        if cfg.query_encoder_config is not None:
            self.query_encoder = load_encoder(cfg.query_encoder_config)
        if cfg.passage_encoder_config is not None:
            self.passage_encoder = load_encoder(cfg.passage_encoder_config)

        # load database
        self.read_only = cfg.read_only
        self.db_file, self.titles, self.sections, self.texts, self.embeddings = (
            self.load_database()
        )
        atexit.register(self._close)

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
        assert len(self.titles) == len(self.index), "Inconsistent database and index"
        assert len(self.sections) == len(self.index), "Inconsistent database and index"
        assert len(self.texts) == len(self.index), "Inconsistent database and index"
        assert len(self.embeddings) == len(self.index), "Inconsistent database and index"  # fmt: skip
        return

    def load_database(self) -> tuple[
        tables.File,
        tables.VLArray,
        tables.VLArray,
        tables.VLArray,
        tables.EArray,
    ]:
        # open database file
        h5path = os.path.join(self.database_path, "database.h5")
        if os.path.exists(h5path):
            mode = "r" if self.read_only else "r+"
            h5file = tables.open_file(h5path, mode=mode)
        else:
            assert not self.read_only, "Database does not exist"
            if not os.path.exists(self.database_path):
                os.makedirs(self.database_path)
            h5file = tables.open_file(h5path, mode="w", title="Retriever Database")

        # load table from database
        node_path = f"/{self.source}"
        if node_path in h5file:
            group = h5file.get_node(f"/{self.source}")
            titles = group.titles
            sections = group.sections
            texts = group.texts
            embeddings = group.embeddings
        else:
            group = h5file.create_group("/", self.source)
            titles = h5file.create_vlarray(group, "titles", tables.VLUnicodeAtom())
            sections = h5file.create_vlarray(group, "sections", tables.VLUnicodeAtom())
            texts = h5file.create_vlarray(group, "texts", tables.VLUnicodeAtom())
            embeddings = h5file.create_earray(
                group,
                "embeddings",
                tables.Float16Atom(),
                shape=(0, self.embedding_size),
            )
        return h5file, titles, sections, texts, embeddings

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

        def get_batch() -> Generator[list[dict[str, str]], None, None]:
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
        for batch in get_batch():
            texts = [i["text"] for i in batch]
            embeddings = self.passage_encoder.encode(texts)
            p_logger.update(step=self.batch_size, desc="Encoding passages")

            # add data to database
            for p in batch:
                self.titles.append(p["title"])
                self.sections.append(p["section"])
                self.texts.append(p["text"])
            self.embeddings.append(embeddings)
            self._fingerprint.update(texts)

            # add embeddings to index
            if self.index.is_trained:
                self.index.add_embeddings(embeddings, serialize=False)

        if not self.index.is_trained:  # train index from scratch
            logger.info("Training index")
            logger.warning("Training index will consume a lot of memory")
            full_embeddings = np.array(self.embeddings[:])
            self.index.build_index(full_embeddings)
        else:
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
                    title=self.titles[chunk_id],
                    section=self.sections[chunk_id],
                    text=self.texts[chunk_id],
                    full_text=self.texts[chunk_id],
                )
                for chunk_id, s in zip(idx, score)
            ]
            for q, idx, score in zip(query, indices, scores)
        ]
        return results

    def clean(self) -> None:
        embed_dim = self.embedding_size
        self.db_file.remove_node(f"/{self.source}", recursive=True)
        group = self.db_file.create_group("/", self.source)
        self.titles = self.db_file.create_vlarray(
            group, "titles", tables.VLUnicodeAtom()
        )
        self.sections = self.db_file.create_vlarray(
            group, "sections", tables.VLUnicodeAtom()
        )
        self.texts = self.db_file.create_vlarray(group, "texts", tables.VLUnicodeAtom())
        self.embeddings = self.db_file.create_earray(
            group,
            "embeddings",
            tables.Float16Atom(),
            shape=(0, embed_dim),
        )
        self._fingerprint.clean()
        return

    def close(self):
        atexit.unregister(self._close)
        self._close()
        return

    def _close(self):
        logger.info("Closing DenseRetriever")
        self.db_file.close()
        return

    def __len__(self):
        return self.embeddings.nrows

    @property
    def embedding_size(self) -> int:
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if hasattr(self, "index"):
            return self.index.embedding_size
        if hasattr(self, "embeddings"):
            return self.embeddings.ndim
        raise ValueError(
            "No encoder or database is provided, embedding size can not be determined."
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()
