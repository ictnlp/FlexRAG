import os
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha1
from typing import Generator, Iterable, Optional
from uuid import uuid4

import lance
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

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
    :param encode_fields: Fields to be encoded. None stands for all fields. Default: None.
    :type encode_fields: Optional[list[str]]
    """

    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    passage_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    encode_fields: Optional[list[str]] = None


@RETRIEVERS("dense", config_class=DenseRetrieverConfig)
class DenseRetriever(LocalRetriever):
    """DenseRetriever is a retriever that retrieves the most relevant passages based on semantic embeddings."""

    name = "DenseRetriever"
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
        else:
            self.database = None
            self.index = DENSE_INDEX.load(cfg, index_path=None)

        # consistency check
        if not no_check:
            self._check_consistency()
        return

    @TIME_METER("dense_retriever", "add-passages")
    def add_passages(self, passages: Iterable[Context]):

        def get_batch() -> Generator[list[pd.DataFrame], None, None]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield pd.DataFrame(batch)
                    batch = []
                data = passage.data.copy()
                data[self.id_field_name] = passage.context_id
                batch.append(data)
            if batch:
                yield pd.DataFrame(batch)
            return

        # generate embeddings
        assert self.passage_encoder is not None, "Passage encoder is not provided."
        p_logger = SimpleProgressLogger(logger, interval=self.log_interval)
        for batch in get_batch():
            p_logger.update(step=len(batch), desc="Adding passages.")

            # add data to database
            if self.cfg.database_path is not None:
                # save the data into lance dataset
                db_path = os.path.join(self.cfg.database_path, "database.lance")
                self.database = lance.write_dataset(
                    batch,
                    uri=db_path,
                    mode="append" if self.database is not None else "create",
                )
            else:
                # save the data into memory
                self.database = (
                    pd.concat([self.database, batch], ignore_index=True)
                    if self.database is not None
                    else batch
                )

        self.build_index()
        logger.info("Finished adding passages.")
        return

    @TIME_METER("dense_retriever", "search")
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        assert self.query_encoder is not None, "Query encoder is not provided."
        top_k = search_kwargs.pop("top_k", self.top_k)
        emb_q = self.query_encoder.encode(query)

        # retrieve using vector index
        indices, scores = self.index.search(emb_q, top_k, **search_kwargs)

        # collect the retrieved data
        if isinstance(self.database, lance.LanceDataset):
            retrieved = self.database.take(indices.flatten()).to_pandas()
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
        return

    def __len__(self) -> int:
        if self.database is None:
            return 0
        if isinstance(self.database, lance.LanceDataset):
            return self.database.count_rows()
        return self.database.shape[0]

    @property
    def fields(self) -> list[str]:
        fields: list
        if self.database is None:
            return []
        if isinstance(self.database, lance.LanceDataset):
            fields = self.database.head(num_rows=1).to_pandas().columns.to_list()
        else:
            fields = self.database.columns.to_list()
        fields = [i for i in fields if i != self.id_field_name]
        return fields

    @TIME_METER("dense_retriever", "build-index")
    def build_index(self, rebuild: bool = False) -> None:
        """Build the index for the retriever."""

        def prepare_text(data: pd.DataFrame) -> list[str]:
            if len(self.encode_fields) > 1:
                data_to_encode = [
                    " ".join([f"{key}:{i[key]}" for key in self.encode_fields])
                    for i in data.iloc
                ]
            else:
                data_to_encode = [i[self.encode_fields[0]] for i in data.iloc]
            return data_to_encode

        def get_batch(offset: int) -> Generator[list[str], None, None]:
            if isinstance(self.database, lance.LanceDataset):
                for batch in self.database.to_batches(
                    batch_size=self.batch_size, offset=offset
                ):
                    batch = batch.to_pandas()
                    yield prepare_text(batch)
            else:
                for i in range(offset, len(self.database), self.batch_size):
                    batch = self.database.iloc[i : i + self.batch_size]
                    yield prepare_text(batch)
            return

        # encode the database
        assert self.passage_encoder is not None, "Passage encoder is not provided."
        if rebuild:
            offset = 0
            total = len(self.database)
            self.index.clean()
        elif self.index.is_addable:
            offset = len(self.index)
            total = len(self.database) - offset
        else:
            offset = 0
            total = len(self.database)
        p_logger = SimpleProgressLogger(logger, total=total, interval=self.log_interval)
        embeddings = []
        for texts in get_batch(offset):
            emb = self.passage_encoder.encode(texts)
            if self.index.is_addable:
                self.index.add_embeddings_batch(emb)
            elif self.cfg.database_path is not None:
                file_name = os.path.join(self.cfg.database_path, f"{uuid4()}.npy")
                np.save(file_name, emb)
                embeddings.append(file_name)
            else:
                embeddings.append(emb)
            p_logger.update(step=len(texts), desc="Encoding passages")

        # exit if the embeddings is already added to the index
        if len(embeddings) == 0:
            self.index.serialize()
            return

        # concatenate the embeddings
        if isinstance(embeddings[0], str):
            logger.info("Copying embeddings to memory map")
            emb_path = embeddings[0]
            emb = np.load(emb_path)
            emb_map = np.memmap(
                os.path.join(self.cfg.database_path, f"_embeddings.npy"),
                dtype=np.float32,
                mode="w+",
                shape=(len(self), emb.shape[1]),
            )
            idx = 0
            for emb_path in embeddings:
                emb = np.load(emb_path)
                emb_map[idx : idx + emb.shape[0]] = emb
                idx += emb.shape[0]
                del emb
                os.remove(emb_path)
            embeddings = emb_map
        else:
            embeddings = np.concatenate(embeddings, axis=0)

        # build the index
        assert embeddings.shape[0] == len(
            self.database
        ), "Inconsistent data and embeddings."
        logger.info("Training index.")
        logger.info("Training index may consume a lot of memory.")
        self.index.build_index(embeddings)

        # clean up the embeddings
        if isinstance(embeddings, np.memmap):
            os.remove(os.path.join(self.cfg.database_path, f"_embeddings.npy"))
        else:
            del embeddings
        return

    def _check_consistency(self) -> None:
        assert len(self.index) == len(
            self
        ), "Inconsistent index and database. Please rebuild the index."
        return

    @cached_property
    def id_field_name(self) -> str:
        return sha1("context_id".encode()).hexdigest()

    def _save_to_local(self, database_path: str) -> None:
        if database_path == self.cfg.database_path:
            return
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
        if not isinstance(self.database, lance.LanceDataset):
            return
        if len(self.database.versions()) <= 1:
            return
        # compact the database
        # we find that the `compact_files` operation will change the order of the rows
        # so we compact the files directly by reading and writing the data
        logger.info("Compacting the database. This may take a while.")
        new_data_path = os.path.join(self.cfg.database_path, "tmp.lance")
        ori_data_path = os.path.join(self.cfg.database_path, "database.lance")
        lance.write_dataset(self.database, uri=new_data_path, mode="create")
        shutil.rmtree(ori_data_path)
        shutil.move(new_data_path, ori_data_path)
        self.database = lance.dataset(ori_data_path)
        return
