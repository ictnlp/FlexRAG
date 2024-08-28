import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable, Optional

from omegaconf import MISSING

from kylin.utils import Choices, SimpleProgressLogger
from kylin.models import HFEncoderConfig, EncoderBase, HFEncoder

from .retriever_base import LocalRetriever, LocalRetrieverConfig, RetrievedContext
from .fingerprint import Fingerprint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VALID_INDEX_TYPE = [
    "GPU_IVF_FLAT",
    "GPU_IVF_PQ",
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "HNSW",
    "DISKANN",
]


@dataclass
class MilvusRetrieverConfig(LocalRetrieverConfig):
    read_only: bool = True
    uri: str = MISSING
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: str = MISSING
    source: str = MISSING
    index_type: Choices(VALID_INDEX_TYPE) = "FLAT"  # type: ignore
    distance_function: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    query_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_query_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)
    passage_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_passage_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)


class MilvusRetriever(LocalRetriever):
    def __init__(self, cfg: MilvusRetrieverConfig) -> None:
        super().__init__(cfg)
        # check pymilvus
        try:
            import pymilvus

            self.pymilvus = pymilvus
        except:
            raise ImportError(
                "Please install pymilvus by running `pip install pymilvus`"
            )

        # load encoder
        self.query_encoder = self.load_encoder(
            cfg.query_encoder_type, cfg.hf_query_encoder_config
        )
        self.passage_encoder = self.load_encoder(
            cfg.passage_encoder_type, cfg.hf_passage_encoder_config
        )

        # load milvus database
        self.source = cfg.source
        client_args = {"uri": cfg.uri}
        if cfg.user is not None:
            client_args["user"] = cfg.user
        if cfg.password is not None:
            client_args["password"] = cfg.password
        self.client = self.pymilvus.MilvusClient(**client_args)
        self.schema, self.index_param = self._prep_param(
            index_type=cfg.index_type,
            distance_function=cfg.distance_function,
            embedding_dim=self.passage_encoder.embedding_size,
        )
        if self.client.has_collection(self.source):
            self.client.load_collection(self.source)
        else:
            self.client.create_collection(
                self.source,
                schema=self.schema,
                index_params=self.index_param,
            )

        # prepare fingerprint
        self._fingerprint = Fingerprint(features=cfg)
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

    def _prep_param(
        self,
        index_type: str,
        distance_function: str,
        embedding_dim: int,
    ):
        # setup schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=self.pymilvus.DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            field_name="embedding",
            datatype=self.pymilvus.DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        )
        schema.add_field(
            field_name="data",
            datatype=self.pymilvus.DataType.JSON,
        )
        # setup index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=index_type,
            metric_type=distance_function,
        )
        return schema, index_params

    def add_passages(self, passages: Iterable[dict[str, str]] | Iterable[str]):
        """
        Add passages to the retriever database
        """
        assert self.passage_encoder is not None, "Passage encoder is not provided"
        total_length = len(passages) if isinstance(passages, list) else None
        p_logger = SimpleProgressLogger(
            logger, total=total_length, interval=self.log_interval
        )
        for batch, sources in self._get_batch(passages):
            # generate embeddings
            embeddings = self.passage_encoder.encode(batch)
            p_logger.update(step=self.batch_size, desc="Encoding passages")
            data = [
                {
                    "embedding": emb,
                    "data": src,
                }
                for emb, src in zip(embeddings, sources)
            ]
            self.client.insert(collection_name=self.source, data=data)
            # update fingerprint
            self._fingerprint.update(batch)
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        embeddings = self.query_encoder.encode(query)
        results_ = self.client.search(
            collection_name=self.source,
            data=embeddings,
            limit=top_k,
            search_params=search_kwargs,
        )
        get = partial(self.client.get, collection_name=self.source)
        results = [
            [
                RetrievedContext(
                    retriever="milvus",
                    query=q,
                    text=get(id=i["id"])[0]["text"],
                    full_text=get(id=i["id"])[0]["text"],
                    title=get(id=i["id"])[0]["title"],
                    section=get(id=i["id"])[0]["section"],
                    source=self.source,
                    score=i["distance"],
                )
                for i in result_
            ]
            for q, result_ in zip(query, results_)
        ]
        return results

    def __len__(self):
        return self.client.get_collection_stats(self.source)["row_count"]

    def clean(self) -> None:
        self.client.drop_collection(self.source)
        self.client.create_collection(
            self.source,
            schema=self.schema,
            index_params=self.index_param,
        )
        self._fingerprint.clean()
        return

    def close(self):
        self.client.close()
        return

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()

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
