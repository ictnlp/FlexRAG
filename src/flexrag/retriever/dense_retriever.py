import datetime
import os
import shutil
from dataclasses import dataclass, field
from hashlib import sha1
from functools import cached_property
from typing import Generator, Iterable, Optional

import lance
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from lance.dataset import DatasetOptimizer
from scipy.spatial.distance import cdist

from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.models import ENCODERS, EncoderBase, EncoderConfig
from flexrag.utils import LOGGER_MANAGER, TIME_METER, SimpleProgressLogger

from .index import DENSE_INDEX, DenseIndexBase, DenseIndexConfig
from .retriever_base import RETRIEVERS, LocalRetriever, LocalRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retreviers.dense")


@dataclass
class DenseRetrieverConfig(LocalRetrieverConfig, DenseIndexConfig):
    """Configuration class for DenseRetriever.

    :param query_encoder_config: Configuration for the query encoder. Default: None.
    :type query_encoder_config: EncoderConfig
    :param passage_encoder_config: Configuration for the passage encoder. Default: None.
    :type passage_encoder_config: EncoderConfig
    :param refine_factor: Refine factor for the retrieved results. Default: 1.
    :type refine_factor: int
    :param encode_fields: Fields to be encoded. None stands for all fields. Default: None.
    :type encode_fields: Optional[list[str]]
    """

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
        self.refine_factor = cfg.refine_factor

        # load encoder
        self.query_encoder = ENCODERS.load(cfg.query_encoder_config)
        self.passage_encoder = ENCODERS.load(cfg.passage_encoder_config)

        if self.cfg.database_path is not None:
            # load the database
            db_path = os.path.join(self.cfg.database_path, "database.lance")
            if os.path.exists(db_path):
                self.database = lance.dataset(db_path)
            else:
                self.database = None

            # load the index
            index_path = os.path.join(self.database_path, f"index.{cfg.index_type}")
            self.index = DENSE_INDEX.load(cfg, index_path=index_path)
            self.embeddings = None
        else:
            self.database = None
            self.embeddings = None
            self.index = DENSE_INDEX.load(cfg, index_path=None)
        self.distance_function = self.index.distance_function

        # consistency check
        if not no_check:
            self._check_consistency()
        return

    @TIME_METER("dense_retriever", "add-passages")
    def add_passages(self, passages: Iterable[Context]):

        def get_batch() -> Generator[list[dict[str, str]], None, None]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                data = passage.data.copy()
                data[self.id_field_name] = passage.context_id
                batch.append(data)
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
            if (self.database is None) and (self.cfg.database_path is not None):
                # create a new local database
                for n, emb in enumerate(embeddings):
                    batch[n]["vector"] = emb
                data_to_add = pd.DataFrame(batch)
                db_path = os.path.join(self.cfg.database_path, "database.lance")
                lance.write_dataset(data_to_add, uri=db_path, mode="create")
                self.database = lance.dataset(db_path)
            elif isinstance(self.database, lance.LanceDataset):
                # append data to the existing database
                for n, emb in enumerate(embeddings):
                    batch[n]["vector"] = emb
                data_to_add = pd.DataFrame(batch)
                db_path = os.path.join(self.cfg.database_path, "database.lance")
                self.database = lance.write_dataset(
                    data_to_add,
                    uri=db_path,
                    mode="append",
                    schema=self.database.schema,
                )
            elif self.database is None:
                # create a new in-memory DataFrame / ndarray
                self.database = pd.DataFrame(batch)
                self.embeddings = embeddings
            else:
                # concatenate the data into in-memory DataFrame / ndarray
                self.database = pd.concat(
                    [self.database, pd.DataFrame(batch)], ignore_index=True
                )
                self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)

            # add embeddings to index
            if self.index.is_trained and self.index.is_addable:
                self.index.add_embeddings_batch(embeddings)

        if (not self.index.is_trained) or (not self.index.is_addable):
            # train index from scratch
            self.build_index()
        elif self.cfg.database_path is not None:
            # serialize the index if `database_path` is provided
            self.index.serialize()
        logger.info("Finished adding passages.")
        return

    @TIME_METER("dense_retriever", "search")
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        assert self.index.is_trained, "Index is not trained."
        assert self.query_encoder is not None, "Query encoder is not provided."
        top_k = search_kwargs.get("top_k", self.top_k)
        emb_q = self.query_encoder.encode(query)

        # retrieve using vector index
        indices, scores = self.index.search(
            emb_q, top_k * self.refine_factor, **search_kwargs
        )
        if self.refine_factor > 1:
            refined_indices, refined_scores = self.refine_index(emb_q, indices)
            indices = refined_indices[:, :top_k]
            scores = refined_scores[:, :top_k]

        # collect the retrieved data
        if isinstance(self.database, lance.LanceDataset):
            retrieved = self.database.take(
                indices.flatten(), columns=self.fields
            ).to_pandas()
        else:
            retrieved = self.database.iloc[indices.flatten()]

        # format the retrieved data
        results: list[list[RetrievedContext]] = []
        for i, (q, score) in enumerate(zip(query, scores)):
            results.append([])
            for j, s in enumerate(score):
                data = retrieved.iloc[i * top_k + j].to_dict()
                context_id = data.pop(self.id_field_name)
                results[-1].append(
                    RetrievedContext(
                        context_id=context_id,
                        retriever=self.name,
                        query=q,
                        score=float(s),
                        data=data,
                    )
                )
        return results

    def clean(self) -> None:
        self.index.clean()
        if self.cfg.database_path is not None:
            shutil.rmtree(self.cfg.database_path)
        self.database = None
        self.embeddings = None
        return

    def __len__(self) -> int:
        if self.database is None:
            return 0
        if isinstance(self.database, lance.LanceDataset):
            return self.database.count_rows()
        return self.database.shape[0]

    @property
    def embedding_size(self) -> int:
        """The embedding size of the retriever."""
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        if self.database is not None:
            return self.database.head(num_rows=1).to_pandas()["vector"][0].shape[0]
        if hasattr(self, "index"):
            return self.index.embedding_size
        raise ValueError(
            "No encoder or database is provided, embedding size can not be determined."
        )

    @property
    def fields(self) -> list[str]:
        fields: list
        if self.database is None:
            return []
        if isinstance(self.database, lance.LanceDataset):
            fields = self.database.head(num_rows=1).to_pandas().columns.to_list()
            fields.remove("vector")
        else:
            fields = self.database.columns.to_list()
        return fields

    @TIME_METER("dense_retriever", "refine-index")
    def refine_index(
        self,
        query: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine the retrieved indices based on the distance between the query and the retrieved embeddings.

        :param query: The query embeddings with shape [bsz, emb_size].
        :type query: np.ndarray
        :param indices: The retrieved indices with shape [bsz, top_k * refine_factor].
        :type indices: np.ndarray
        :return: The refined indices and scores with shape [bsz, top_k * refine_factor].
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        bsz, kf = indices.shape
        flat_indices = indices.flatten()
        if isinstance(self.database, lance.LanceDataset):
            embs = np.stack(
                self.database.take(flat_indices, columns=["vector"]).to_pandas()["vector"]  # fmt: skip
            )  # [bsz * kf, emb_size]
        else:
            embs = self.embeddings[flat_indices]  # [bsz * kf, emb_size]
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
        if self.embeddings is not None:
            embeddings = self.embeddings
        else:
            logger.info("Copying embeddings to memory map")
            embeddings = np.memmap(
                os.path.join(self.cfg.database_path, f"_embeddings.npy"),
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
        if self.embeddings is None:
            os.remove(os.path.join(self.cfg.database_path, f"_embeddings.npy"))
        return

    def _check_consistency(self) -> None:
        assert len(self.index) == len(self), "Inconsistent index and database."
        if self.index.is_trained:
            assert (
                self.index.embedding_size == self.embedding_size
            ), "Inconsistent embedding size."
        return

    @cached_property
    def id_field_name(self) -> str:
        return sha1("context_id".encode()).hexdigest()

    def _save_to_local(self, database_path: str) -> None:
        if database_path == self.cfg.database_path:
            return
        # merge embeddings to the database
        self.database["vector"] = self.embeddings.tolist()
        self.embeddings = None
        db_path = os.path.join(database_path, "database.lance")
        self.database = lance.write_dataset(self.database, uri=db_path, mode="create")

        # save the index
        self.index.serialize(
            os.path.join(database_path, f"index.{self.cfg.index_type}")
        )
        return

    def save_to_hub(
        self,
        repo_id: str,
        token: str = os.environ.get("HF_TOKEN", None),
        commit_message: str = "Update FlexRAG retriever",
        retriever_card: str = None,
        private: bool = False,
        **kwargs,
    ) -> str:
        # do additional check before saving
        DenseRetriever._check_config(self.cfg, token)

        # clear up the database to reduce the size & file number
        self._clearup_database()
        return super().save_to_hub(
            repo_id, token, commit_message, retriever_card, private, **kwargs
        )

    @staticmethod
    def _check_config(cfg: DenseRetrieverConfig, token: str = None) -> None:
        client = HfApi(token=token)
        if cfg.query_encoder_config.encoder_type is None:
            logger.warning(
                "Query encoder is not provided. "
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        elif cfg.query_encoder_config.encoder_type == "hf":
            if not client.repo_exists(cfg.query_encoder_config.hf_config.model_path):
                logger.warning(
                    "Query encoder model is not available in the HuggingFace model hub."
                    "Please make sure loading the appropriate encoder when loading the retriever."
                )
        else:
            logger.warning(
                "Query encoder is not a model hosted on the HuggingFace model hub."
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        # check the passage encoder
        if cfg.passage_encoder_config.encoder_type is None:
            logger.warning(
                "Passage encoder is not provided. "
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        elif cfg.passage_encoder_config.encoder_type == "hf":
            if not client.repo_exists(cfg.passage_encoder_config.hf_config.model_path):
                logger.warning(
                    "Passage encoder model is not available in the HuggingFace model hub."
                    "Please make sure loading the appropriate encoder when loading the retriever."
                )
        else:
            logger.warning(
                "Passage encoder is not a model hosted on the HuggingFace model hub."
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        return

    def _clearup_database(self) -> None:
        if isinstance(self.database, lance.LanceDataset):
            # compact the database
            logger.info("Compacting the database to reduce the size & file number.")
            optim = DatasetOptimizer(self.database)
            optim.compact_files()
            # clear up the database
            self.database.cleanup_old_versions(
                datetime.datetime.now() - self.database.versions()[-1]["timestamp"]
            )
        return
