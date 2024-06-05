import logging
import os
import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Iterable

import numpy as np
import tables
import torch
from transformers import AutoModel, AutoTokenizer

from .index import add_index_args, load_index
from .retriever import Retriever
from .utils import normalize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
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
            "--database_path",
            type=str,
            required=True,
            help="The path to the Retriever database",
        )
        parser.add_argument(
            "--read_only",
            action="store_true",
            default=False,
            help="Whether to open the database in read only mode",
        )
        parser.add_argument(
            "--compress_database",
            action="store_true",
            default=False,
            help="Whether to compress the database",
        )
        parser.add_argument(
            "--save_embedding",
            action="store_true",
            default=False,
            help="Whether to save the embeddings into the database",
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
            default=2048,
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
        # index arguments
        parser = add_index_args(parser)
        return parser

    def __init__(self, args: Namespace) -> None:
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
        self._embedding_size = self.query_encoder.config.hidden_size

        # load database
        self.save_embedding = args.save_embedding
        self.read_only = args.read_only
        self.compress_database = args.compress_database
        database_path = os.path.join(args.database_path, "database.h5")
        self.db_file, self.db_table = self.load_database(database_path)

        # load / build index
        self.index = load_index(
            index_args=args,
            index_path=args.database_path,
            log_interval=args.log_interval,
            embedding_size=self._embedding_size,
            device_id=args.retriever_device_id,
        )
        return

    def load_database(self, database_path: str) -> tuple[tables.File, tables.Table]:
        if os.path.exists(database_path):
            logger.info(f"Loading database from {database_path}")
            mode = "r" if self.read_only else "r+"
            h5file = tables.open_file(database_path, mode=mode)
            table = h5file.root.passages.data
            return h5file, table
        logger.info(f"Initiate database from {database_path}")
        return self.init_tables(database_path)

    def init_tables(self, database_path: str) -> tuple[tables.File, tables.Table]:
        class RetrieveTableWithEmb(tables.IsDescription):
            title = tables.StringCol(self.max_passage_length, pos=1)
            section = tables.StringCol(self.max_passage_length, pos=2)
            text = tables.StringCol(self.max_passage_length, pos=3)
            embedding = tables.Float16Col(768, pos=4)

        class RetrieveTable(tables.IsDescription):
            title = tables.StringCol(self.max_passage_length, pos=1)
            section = tables.StringCol(self.max_passage_length, pos=2)
            text = tables.StringCol(self.max_passage_length, pos=3)
            embedding = tables.Float16Col(768, pos=4)

        if not os.path.exists(os.path.dirname(database_path)):
            os.makedirs(os.path.dirname(database_path))

        assert not self.read_only, "Database does not exist"
        h5file = tables.open_file(database_path, mode="w", title="Retriever Database")

        # setup compressor
        if self.compress_database:
            filters = tables.Filters(complevel=5, complib="zlib")
        else:
            filters = None

        group = h5file.create_group("/", "passages", "Passages", filters=filters)
        if self.save_embedding:
            table = h5file.create_table(
                group, "data", RetrieveTableWithEmb, "data", filters=filters
            )
        else:
            table = h5file.create_table(group, "data", RetrieveTable, "data")
        return h5file, table

    def add_passages(
        self, passages: list[dict[str, str]] | list[str], source: str = None
    ):
        """
        Add passages to the retriever database
        """
        # generate embeddings
        full_embeddings = []
        for n, idx in enumerate(range(0, len(passages), self.batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Generating embeddings for batch {n}")

            # prepare batch
            batch = passages[idx : idx + self.batch_size]
            embeddings = self._encode_batch(batch)
            full_embeddings.append(embeddings)

            # add embeddings to database
            for i, p in enumerate(batch):
                p = {"text": p} if isinstance(p, str) else p
                row = self.db_table.row
                row["title"] = self._safe_encode(p.get("title", ""))
                row["section"] = self._safe_encode(p.get("section", ""))
                row["text"] = self._safe_encode(p.get("text", ""))
                if self.save_embedding:
                    row["embedding"] = embeddings[i]
                row.append()
            self.db_table.flush()
        full_embeddings = np.concatenate(full_embeddings, axis=0)

        # update index
        if not self.index.is_trained:
            self.index.train_index(full_embeddings)
        for n, idx in enumerate(range(0, len(passages), self.batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Adding embeddings for batch {n}")
            embeds_to_add = full_embeddings[idx : idx + self.batch_size]
            self.index.add_embeddings_batch(embeds_to_add)
        self.index.serialize()
        logger.info("Finished adding passages")
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        embeddings = self._encode_batch(query, is_query=True)
        scores, indices = self.index.search_batch(embeddings, top_k, **search_kwargs)
        results = [
            {
                "query": q,
                "indices": [i for i in r],
                "scores": [i for i in s],
                "titles": [i.decode() for i in self.db_table[r]["title"]],
                "sections": [i.decode() for i in self.db_table[r]["section"]],
                "texts": [i.decode() for i in self.db_table[r]["text"]],
            }
            for q, r, s in zip(query, indices, scores)
        ]
        return results

    def close(self):
        self.db_table.flush()
        self.db_file.close()
        return

    @torch.no_grad()
    def _encode_batch(
        self,
        batch: Iterable[dict[str, str]] | Iterable[str],
        is_query: bool = False,
    ) -> np.ndarray:
        # prepare batch
        batch_text = self._prepare_texts(batch)

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

    def _prepare_texts(
        self, batch: Iterable[dict[str, str]] | Iterable[str]
    ) -> list[str]:
        # add title
        if isinstance(batch[0], dict):
            if ("title" in batch[0]) and (not self.no_title):
                batch_text = [f"{p['title']} {p['text']}" for p in batch]
            else:
                batch_text = [p["text"] for p in batch]
        else:
            assert isinstance(batch[0], str)
            batch_text = deepcopy(batch)
        # lowercase
        if self.lowercase:
            batch_text = [p.lower() for p in batch_text]
        # normalize
        if self.normalize_text:
            batch_text = [normalize(p) for p in batch_text]
        return batch_text

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="The interval to log the progress",
    )
    parser = DenseRetriever.add_args(parser)
    args = parser.parse_args()
    retriever = DenseRetriever(args)
    # while True:
    #     query = input("Query: ")
    #     if query == "quit":
    #         break
    #     r = retriever.search([query], leaves_to_search=500)[0]
    #     pass
    # retriever.test_acc(sample_num=1024, top_k=5, leaves_to_search=500)
    retriever.test_speed(sample_num=8192, top_k=5, leaves_to_search=500)
    retriever.close()
    pass
