import atexit
import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Iterable

import h5py
import numpy as np
from omegaconf import MISSING
from scipy.spatial.distance import cdist

from kylin.models import EncoderConfig, load_encoder
from kylin.utils import SimpleProgressLogger, TimeMeter

from .fingerprint import Fingerprint
from .index import DenseIndex, DenseIndexConfig, load_index
from .retriever_base import (
    SEMANTIC_RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = logging.getLogger("DenseRetriever")


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig, DenseIndexConfig):
    inference_only: bool = True
    database_path: str = MISSING
    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    passage_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    source: str = MISSING
    refine_factor: int = 1


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
        if cfg.inference_only:
            self.query_encoder = load_encoder(cfg.query_encoder_config)
            self.passage_encoder = None
        else:
            self.query_encoder = None
            self.passage_encoder = load_encoder(cfg.passage_encoder_config)

        # load database
        self.inference_only = cfg.inference_only
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
        index_path = os.path.join(self.database_path, f"{cfg.index_type}_{cfg.source}")
        self.index = load_index(index_path, self.embedding_size, cfg)
        self.refine_factor = cfg.refine_factor
        self.distance_function = self.index.distance_function

        # consistency check
        assert len(self.titles) == len(self.index), "Inconsistent database and index"
        assert len(self.sections) == len(self.index), "Inconsistent database and index"
        assert len(self.texts) == len(self.index), "Inconsistent database and index"
        assert len(self.embeddings) == len(self.index), "Inconsistent database and index"  # fmt: skip
        return

    def load_database(self) -> tuple[
        h5py.File,
        h5py.Dataset,
        h5py.Dataset,
        h5py.Dataset,
        h5py.Dataset,
    ]:
        # open database file
        h5path = os.path.join(self.database_path, "database.h5")
        if os.path.exists(h5path):
            mode = "r" if self.inference_only else "r+"
            h5file = h5py.File(h5path, mode=mode)
        else:
            assert not self.inference_only, "Database does not exist"
            if not os.path.exists(self.database_path):
                os.makedirs(self.database_path)
            h5file = h5py.File(h5path, mode="w", title="Retriever Database")

        # load table from database
        if self.source in h5file:
            group = h5file[self.source]
            titles = group["titles"]
            sections = group["sections"]
            texts = group["texts"]
            embeddings = group["embeddings"]
        else:
            group = h5file.create_group(self.source)
            titles = group.create_dataset(
                "titles",
                shape=(0,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                maxshape=(None,),
            )
            sections = group.create_dataset(
                "sections",
                shape=(0,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                maxshape=(None,),
            )
            texts = group.create_dataset(
                "texts",
                shape=(0,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                maxshape=(None,),
            )
            embeddings = group.create_dataset(
                "embeddings",
                shape=(0, self.embedding_size),
                dtype=np.float16,
                maxshape=(None, self.embedding_size),
            )
        return h5file, titles, sections, texts, embeddings

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
            cur_len = len(self.texts)
            self.titles.resize((cur_len + len(batch),))
            self.titles[cur_len:] = [p["title"] for p in batch]
            self.sections.resize((cur_len + len(batch),))
            self.sections[cur_len:] = [p["section"] for p in batch]
            self.texts.resize((cur_len + len(batch),))
            self.texts[cur_len:] = [p["text"] for p in batch]
            self.embeddings.resize((cur_len + len(batch), self.embedding_size))
            self.embeddings[cur_len:] = embeddings
            self._fingerprint.update(texts)

            # add embeddings to index
            if self.index.is_trained:
                self.index.add_embeddings(embeddings, serialize=False)

        if not self.index.is_trained:  # train index from scratch
            logger.info("Training index")
            logger.warning("Training index may consume a lot of memory")
            self.index.build_index(self.embeddings)
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
        emb_q = self.query_encoder.encode(query)
        indices, scores = self.index.search(
            emb_q, top_k * self.refine_factor, **search_kwargs
        )
        if self.refine_factor > 1:
            refined_indices, refined_scores = self.refine_index(emb_q, indices)
            indices = refined_indices[:, :top_k]
            scores = refined_scores[:, :top_k]
        results = [
            [
                RetrievedContext(
                    retriever=self.name,
                    query=q,
                    source=self.source,
                    chunk_id=int(chunk_id),
                    score=float(s),
                    title=self.titles[chunk_id].decode(),
                    section=self.sections[chunk_id].decode(),
                    text=self.texts[chunk_id].decode(),
                    full_text=self.texts[chunk_id].decode(),
                )
                for chunk_id, s in zip(idx, score)
            ]
            for q, idx, score in zip(query, indices, scores)
        ]
        return results

    def clean(self) -> None:
        self.titles.resize((0,))
        self.sections.resize((0,))
        self.texts.resize((0,))
        self.embeddings.resize((0, self.embedding_size))
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
        return self.embeddings.shape[0]

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

    @TimeMeter("retrieve", "refine-index")
    def refine_index(
        self,
        query: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine the retrieved indices based on the distance between the query and the retrieved embeddings.

        Args:
            query (np.ndarray): The query embeddings with shape [bsz, emb_size]
            indices (np.ndarray): The retrieved indices with shape [bsz, top_k * refine_factor]

        Returns:
            indices (np.ndarray): The refined indices with shape [bsz, top_k]
            scores (np.ndarray): The refined scores with shape [bsz, top_k]
        """
        # as h5py does not fully support fancy indexing, we process the indices individually
        kf = indices.shape[1]
        new_indices = []
        new_scores = []
        for q, idx in zip(query, indices):
            # extract the embeddings of the retrieved indices
            idx_order = np.argsort(idx)
            ordered_idx = idx[idx_order]
            emb_d_ = self.embeddings[ordered_idx]  # [kf, emb_size]
            emb_d = np.empty_like(emb_d_)
            emb_d[idx_order] = emb_d_  # recover the original order

            # compute the distance between the query and the retrieved embeddings
            q = np.expand_dims(q, 0).repeat(kf, axis=0)  # [kf, emb_size]
            match self.distance_function:
                case "L2":
                    dis = cdist(q, emb_d, "euclidean")
                case "COSINE":
                    dis = -cdist(q, emb_d, "cosine")
                case "IP":
                    dis = -np.sum(q * emb_d, axis=1)
                case "HAMMING":
                    dis = cdist(q, emb_d, "hamming")
                case "MANHATTAN":
                    dis = cdist(q, emb_d, "cityblock")
                case _:
                    raise ValueError("Unsupported distance function")

            # sort the indices & scores based on the distance
            new_order = np.argsort(dis)
            new_indices.append(idx[new_order])
            new_scores.append(dis[new_order])
        new_indices = np.stack(new_indices, axis=0)
        new_scores = np.stack(new_scores, axis=0)
        return new_indices, new_scores
