import os
import shutil
from dataclasses import dataclass, field
from typing import Generator, Iterable, Optional

import lance
import numpy as np
import pandas as pd
from omegaconf import MISSING
from scipy.spatial.distance import cdist

from librarian.models import ENCODERS, EncoderBase
from librarian.utils import SimpleProgressLogger, TIME_METER, LOGGER_MANAGER

from .index import DenseIndexBase, DENSE_INDEX
from .retriever_base import (
    RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = LOGGER_MANAGER.get_logger("librarian.retreviers.dense")


DenseIndexConfig = DENSE_INDEX.make_config()
EncoderConfig = ENCODERS.make_config(default=None)


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig, DenseIndexConfig):
    database_path: str = MISSING
    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    passage_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    refine_factor: int = 1
    encode_fields: Optional[list[str]] = None


@RETRIEVERS("dense", config_class=DenseRetrieverConfig)
class DenseRetriever(LocalRetriever):
    name = "Dense Retrieval"
    index: DenseIndexBase
    query_encoder: EncoderBase
    passage_encoder: EncoderBase

    def __init__(self, cfg: DenseRetrieverConfig, no_check: bool = False) -> None:
        super().__init__(cfg)
        # set args
        self.database_path = cfg.database_path
        self.encode_fields = cfg.encode_fields

        # load encoder
        self.query_encoder = ENCODERS.load(cfg.query_encoder_config)
        self.passage_encoder = ENCODERS.load(cfg.passage_encoder_config)

        # load database
        self.db_path = os.path.join(self.database_path, "database.lance")
        if os.path.exists(self.db_path):
            self.database = lance.dataset(self.db_path)
        else:
            self.database = None

        # load index
        index_path = os.path.join(self.database_path, f"index.{cfg.index_type}")
        self.index = DENSE_INDEX.load(cfg, index_path=index_path)
        self.refine_factor = cfg.refine_factor
        self.distance_function = self.index.distance_function

        # consistency check
        if not no_check:
            self._check_consistency()
        return

    @TIME_METER("dense_retriever", "add-passages")
    def add_passages(self, passages: Iterable[dict[str, str]]):
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
        assert self.passage_encoder is not None, "Passage encoder is not provided."
        p_logger = SimpleProgressLogger(logger, interval=self.log_interval)
        for batch in get_batch():
            # encode passages
            if len(self.encode_fields) > 1:
                data_to_encode = [
                    " ".join([f"{key}:{i[key]}" for key in self.encode_fields])
                    for i in batch
                ]
            else:
                data_to_encode = [i[self.encode_fields[0]] for i in batch]
            embeddings = self.passage_encoder.encode(data_to_encode)
            p_logger.update(step=self.batch_size, desc="Encoding passages")

            # add data to database
            for n, emb in enumerate(embeddings):
                batch[n]["vector"] = emb
            data_to_add = pd.DataFrame(batch)
            if self.database is None:
                lance.write_dataset(data_to_add, uri=self.db_path, mode="create")
                self.database = lance.dataset(self.db_path)
            else:
                self.database = lance.write_dataset(
                    data_to_add,
                    uri=self.db_path,
                    mode="append",
                    schema=self.database.schema,
                )

            # add embeddings to index
            if self.index.is_trained:
                self.index.add_embeddings(embeddings, serialize=False)

        if not self.index.is_trained:  # train index from scratch
            self.build_index()
        else:
            self.index.serialize()
        logger.info("Finished adding passages")
        return

    @TIME_METER("dense_retriever", "search")
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        top_k = search_kwargs.get("top_k", self.top_k)
        emb_q = self.query_encoder.encode(query)
        indices, scores = self.index.search(
            emb_q, top_k * self.refine_factor, **search_kwargs
        )
        if self.refine_factor > 1:
            refined_indices, refined_scores = self.refine_index(emb_q, indices)
            indices = refined_indices[:, :top_k]
            scores = refined_scores[:, :top_k]
        retrieved = self.database.take(
            indices.flatten(), columns=self.fields
        ).to_pandas()
        results = []
        for i, (q, score) in enumerate(zip(query, scores)):
            results.append(
                [
                    RetrievedContext(
                        retriever=self.name,
                        query=q,
                        score=float(s),
                        data=retrieved.iloc[i * top_k + j].to_dict(),
                    )
                    for j, s in enumerate(score)
                ]
            )
        return results

    def clean(self) -> None:
        self.index.clean()
        shutil.rmtree(self.database_path)
        self.database = None
        return

    def __len__(self) -> int:
        if self.database is None:
            return 0
        return self.database.count_rows()

    @property
    def embedding_size(self) -> int:
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if self.database is not None:
            return self.database.head(num_rows=1).to_pandas()["vector"][0].shape[0]
        if hasattr(self, "index"):
            return self.index.embedding_size
        raise ValueError(
            "No encoder or database is provided, embedding size can not be determined."
        )

    @property
    def fields(self) -> list[str]:
        fields = self.database.head(num_rows=1).to_pandas().columns.to_list()
        fields.remove("vector")
        return fields

    @TIME_METER("dense_retriever", "refine-index")
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
            indices (np.ndarray): The refined indices with shape [bsz, top_k * refine_factor]
            scores (np.ndarray): The refined scores with shape [bsz, top_k * refine_factor]
        """
        bsz, kf = indices.shape
        flat_indices = indices.flatten()
        embs = np.stack(
            self.database.take(flat_indices, columns=["vector"]).to_pandas()["vector"]
        )  # [bsz * kf, emb_size]
        query = np.expand_dims(query, 1).repeat(kf, axis=1).reshape(bsz * kf, -1)

        # compute the distance between the query and the retrieved embeddings
        match self.distance_function:
            case "L2":
                dis = cdist(query, embs, "euclidean")
            case "COSINE":
                dis = -cdist(query, embs, "cosine")
            case "IP":
                dis = -np.sum(query * embs, axis=1)
            case "HAMMING":
                dis = cdist(query, embs, "hamming")
            case "MANHATTAN":
                dis = cdist(query, embs, "cityblock")
            case _:
                raise ValueError("Unsupported distance function")
        dis = dis.reshape(bsz, kf)
        new_order = np.argsort(dis, axis=1)
        new_indices = np.take_along_axis(indices, new_order, axis=1)
        new_scores = np.take_along_axis(dis, new_order, axis=1)
        return new_indices, new_scores

    @TIME_METER("dense_retriever", "build-index")
    def build_index(self) -> None:
        logger.info("Copying embeddings to memory map")
        embeddings = np.memmap(
            os.path.join(self.database_path, f"_embeddings.npy"),
            dtype=np.float32,
            mode="w+",
            shape=(self.database.count_rows(), self.embedding_size),
        )
        idx = 0
        for emb_batch in self.database.to_batches(columns=["vector"]):
            emb_batch = np.stack(emb_batch.to_pandas()["vector"])
            embeddings[idx : idx + emb_batch.shape[0]] = emb_batch
            idx += emb_batch.shape[0]
            del emb_batch
        logger.info("Training index.")
        logger.warning("Training index may consume a lot of memory.")
        self.index.build_index(embeddings)
        os.remove(os.path.join(self.database_path, f"_embeddings.npy"))
        return

    def _check_consistency(self) -> None:
        assert len(self.index) == len(self), "Inconsistent index and database."
        if self.index.is_trained:
            assert (
                self.index.embedding_size == self.embedding_size
            ), "Inconsistent embedding size."
        return
