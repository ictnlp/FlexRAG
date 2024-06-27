import logging
import os
from argparse import ArgumentParser, Namespace
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .retriever_base import LocalRetriever


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
    "BIN_FLAT",
    "BIN_IVF_FLAT",
    "DISKANN",
    "AUTOINDEX",
    "GPU_CAGRA",
    "GPU_BRUTE_FORCE",
]


class MilvusRetriever(LocalRetriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        # model arguments
        parser.add_argument(
            "--query_encoder",
            type=str,
            default="facebook/contriever-msmarco",
            help="The model to use for the query encoder",
        )
        parser.add_argument(
            "--passage_encoder",
            type=str,
            default=None,
            help="The model to use for the passage encoder",
        )
        parser.add_argument(
            "--retriever_tokenizer",
            type=str,
            default=None,
            help="The tokenizer to use for the retriever",
        )
        parser.add_argument(
            "--retriever_device_id",
            type=int,
            default=-1,
            help="The device id to use for the retriever(query and passage encoder)",
        )
        # database arguments
        parser.add_argument(
            "--read_only",
            action="store_true",
            default=False,
            help="Whether to open the database in read only mode",
        )
        # encoding arguments
        parser.add_argument(
            "--batch_size",
            type=int,
            default=512,
            help="The batch size to use for encoding",
        )
        parser.add_argument(
            "--max_query_length",
            type=int,
            default=256,
            help="The maximum length of the queries",
        )
        parser.add_argument(
            "--max_passage_length",
            type=int,
            default=512,
            help="The maximum length of the passages",
        )
        parser.add_argument(
            "--no_title",
            action="store_true",
            default=False,
            help="Whether to include the title in the passage",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            default=False,
            help="Whether to lowercase the text",
        )
        parser.add_argument(
            "--normalize_text",
            action="store_true",
            default=False,
            help="Whether to normalize the text",
        )
        parser.add_argument(
            "--metric_type",
            type=str,
            default="IP",
            choices=["IP", "L2", "cosine"],
            help="The metric type to use for the Milvus index",
        )
        parser.add_argument(
            "--index_type",
            type=str,
            default="AUTOINDEX",
            choices=VALID_INDEX_TYPE,
            help="The index type to use for the Milvus index",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        # check pymilvus
        try:
            import pymilvus

            self.pymilvus = pymilvus
        except:
            raise ImportError(
                "Please install pymilvus by running `pip install pymilvus`"
            )

        # set args
        if args.retriever_tokenizer is None:
            args.retriever_tokenizer = args.query_encoder
        self.batch_size = args.batch_size
        self.max_query_length = args.max_query_length
        self.max_passage_length = args.max_passage_length
        self.no_title = args.no_title
        self.lowercase = args.lowercase
        self.normalize_text = args.normalize_text
        self.log_interval = args.log_interval
        self.index_type = args.index_type
        self.metric_type = args.metric_type

        # prepare models
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_tokenizer)
        self.query_encoder = AutoModel.from_pretrained(args.query_encoder)
        if args.passage_encoder is not None:
            self.passage_encoder = AutoModel.from_pretrained(args.passage_encoder)
        else:
            self.passage_encoder = None
        if args.retriever_device_id >= 0:
            self.query_encoder.to(args.retriever_device_id)
            if self.passage_encoder is not None:
                self.passage_encoder.to(args.retriever_device_id)
        self.embedding_size = self.query_encoder.config.hidden_size

        # load database
        self.database_path = os.path.join(args.database_path, "database.db")
        self.indices_path = os.path.join(args.database_path, "indices.npy")
        self.client = self.pymilvus.MilvusClient(self.database_path)
        if self.client.has_collection("database"):
            self.db = self.client.load_collection("database")
            self.indices = np.load(self.indices_path)
        else:
            self.db = self._init_database()
            self.indices = np.zeros(0, dtype=np.int64)
        return

    def _init_database(self):
        # prepare schema
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
            dim=768,
        )
        schema.add_field(
            field_name="data",
            datatype=self.pymilvus.DataType.JSON,
        )

        # prepare index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
        )
        milvus_db = self.client.create_collection(
            "database",
            schema=schema,
            index_params=index_params,
        )
        return milvus_db

    def add_passages(
        self, passages: list[dict[str, str]] | list[str], source: str = None
    ):
        """
        Add passages to the retriever database
        """
        # generate embeddings
        for n, idx in enumerate(range(0, len(passages), self.batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Generating embeddings for batch {n}")

            # prepare batch
            batch = passages[idx : idx + self.batch_size]
            embeddings = self._encode_batch(batch)

            # add embeddings to database
            rows = []
            for emb, p in zip(embeddings, batch):
                row = {
                    "embedding": emb.astype(np.float32),
                    "data": {
                        "title": p["title"],
                        "text": p["text"],
                        "section": p["section"],
                    },
                }
                rows.append(row)
            r = self.client.insert(collection_name="database", data=rows)
            self.indices = np.concatenate([self.indices, r["ids"]])
        np.save(self.indices, self.indices_path)
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        embeddings = self._encode_batch(query, is_query=True)
        results_ = self.client.search(
            collection_name="database",
            data=embeddings,
            limit=top_k,
            search_params=search_kwargs,
        )
        results = [
            {
                "scores": [i["distance"] for i in result_],
                "titles": [i["entity"]["data"]["title"] for i in result_],
                "sections": [i["entity"]["data"]["section"] for i in result_],
                "texts": [i["entity"]["data"]["text"] for i in result_],
            }
            for result_ in results_
        ]
        return results

    def close(self):
        self.client.close()
        return

    @torch.no_grad()
    def _encode_batch(
        self,
        batch: Iterable[dict[str, str]] | Iterable[str],
        is_query: bool = False,
    ) -> np.ndarray:
        # prepare batch
        batch_text = [self._prepare_text(doc) for doc in batch]

        # get encoder
        if is_query:
            encoder = self.query_encoder
        else:
            encoder = self.passage_encoder

        # tokenize text
        device = encoder.device
        input_dict = self.tokenizer.batch_encode_plus(
            batch_text,
            return_tensors="pt",
            max_length=self.max_passage_length,
            padding=True,
            truncation=True,
        )
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        mask = input_dict["attention_mask"]

        # encode text
        token_embeddings = encoder(**input_dict).last_hidden_state
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def __len__(self):
        return len(self.indices)
