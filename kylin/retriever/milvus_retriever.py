import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Generator, Iterable, Optional

from omegaconf import MISSING, OmegaConf

from kylin.models import EncoderBase, HFEncoder, HFEncoderConfig
from kylin.utils import Choices, SimpleProgressLogger

from .fingerprint import Fingerprint
from .retriever_base import (
    SEMANTIC_RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VALID_INDEX_TYPE = [
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "GPU_BRUTE_FORCE",
    "GPU_IVF_FLAT",
    "GPU_IVF_PQ",
    "HNSW",
    "SCANN",
]


@dataclass
class FLATIndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore


@dataclass
class IVFIndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    nlist: int = 128
    nprobe: int = 8


@dataclass
class IVFPQIndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    nlist: int = 128
    nbits: int = 8
    m: int = 8


@dataclass
class IVFSQ8IndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    nlist: int = 128


@dataclass
class SCANNIndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    nlist: int = 8
    with_raw_data: bool = True


@dataclass
class HNSWIndexConfig:
    metric_type: Choices(["IP", "L2", "COSINE"]) = "IP"  # type: ignore
    M: int = 16
    efConstruction: int = 8


@dataclass
class MilvusRetrieverConfig(LocalRetrieverConfig):
    # database configs
    read_only: bool = True
    uri: str = MISSING
    user: Optional[str] = None
    password: Optional[str] = None
    source: str = MISSING
    index_type: Choices(VALID_INDEX_TYPE) = "FLAT"  # type: ignore
    # index configs
    flat_config: FLATIndexConfig = field(default_factory=FLATIndexConfig)
    ivf_config: IVFIndexConfig = field(default_factory=IVFIndexConfig)
    ivf_sq_config: IVFSQ8IndexConfig = field(default_factory=IVFSQ8IndexConfig)
    ivf_pq_config: IVFPQIndexConfig = field(default_factory=IVFPQIndexConfig)
    scann_config: SCANNIndexConfig = field(default_factory=SCANNIndexConfig)
    hnsw_config: HNSWIndexConfig = field(default_factory=HNSWIndexConfig)
    # encoder configs
    query_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_query_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)
    passage_encoder_type: Optional[Choices(["hf"])] = None  # type: ignore
    hf_passage_encoder_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)


@SEMANTIC_RETRIEVERS("milvus", config_class=MilvusRetrieverConfig)
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
            embedding_dim=self.passage_encoder.embedding_size,
            flat_config=cfg.flat_config,
            ivf_config=cfg.ivf_config,
            ivf_pq_config=cfg.ivf_pq_config,
            ivf_sq_config=cfg.ivf_sq_config,
            scann_config=cfg.scann_config,
            hnsw_config=cfg.hnsw_config,
        )
        if self.client.has_collection(self.source):
            self.client.load_collection(self.source)
        else:
            try:
                self.client.create_collection(
                    self.source,
                    schema=self.schema,
                    index_params=self.index_param,
                )
            except Exception as e:
                # clean up if failed
                if self.client.has_collection(self.source):
                    self.client.drop_collection(self.source)
                raise e

        # prepare fingerprint
        self._fingerprint = Fingerprint(
            features={
                "uri": cfg.uri,
                "user": cfg.user,
                "password": cfg.password,
                "source": cfg.source,
            }
        )
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
        embedding_dim: int,
        flat_config: FLATIndexConfig,
        ivf_config: IVFIndexConfig,
        ivf_pq_config: IVFPQIndexConfig,
        ivf_sq_config: IVFSQ8IndexConfig,
        scann_config: SCANNIndexConfig,
        hnsw_config: HNSWIndexConfig,
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
        match index_type:
            case "FLAT":
                index_param_ = OmegaConf.to_container(flat_config, resolve=True)
            case "IVF_FLAT":
                index_param_ = OmegaConf.to_container(ivf_config, resolve=True)
            case "IVF_SQ8":
                index_param_ = OmegaConf.to_container(ivf_sq_config, resolve=True)
            case "IVF_PQ":
                index_param_ = OmegaConf.to_container(ivf_pq_config, resolve=True)
            case "GPU_BRUTE_FORCE":
                index_param_ = OmegaConf.to_container(flat_config, resolve=True)
            case "GPU_IVF_FLAT":
                index_param_ = OmegaConf.to_container(ivf_config, resolve=True)
            case "GPU_IVF_PQ":
                index_param_ = OmegaConf.to_container(ivf_pq_config, resolve=True)
            case "SCANN":
                index_param_ = OmegaConf.to_container(scann_config, resolve=True)
            case "HNSW":
                index_param_ = OmegaConf.to_container(hnsw_config, resolve=True)
            case _:
                raise ValueError(f"Index type {index_type} is not supported")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=index_type,
            **index_param_,
        )
        return schema, index_params

    def _add_passages(self, passages: Iterable[dict[str, str]] | Iterable[str]):
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

        assert self.passage_encoder is not None, "Passage encoder is not provided"
        total_length = len(passages) if isinstance(passages, list) else None
        p_logger = SimpleProgressLogger(
            logger, total=total_length, interval=self.log_interval
        )
        for batch in get_batch():
            # generate embeddings
            texts = [i["text"] for i in batch]
            embeddings = self.passage_encoder.encode(texts)
            p_logger.update(step=self.batch_size, desc="Encoding passages")
            data = [
                {
                    "embedding": emb,
                    "data": src,
                }
                for emb, src in zip(embeddings, batch)
            ]
            self.client.insert(collection_name=self.source, data=data)
            # update fingerprint
            self._fingerprint.update(texts)
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
        try:
            self.client.create_collection(
                self.source,
                schema=self.schema,
                index_params=self.index_param,
            )
        except Exception as e:
            # clean up if failed
            if self.client.has_collection(self.source):
                self.client.drop_collection(self.source)
            raise e
        self._fingerprint.clean()
        return

    def close(self):
        self.client.close()
        return

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()
