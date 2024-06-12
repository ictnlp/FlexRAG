import logging
import os
from argparse import ArgumentParser, Namespace
from typing import Iterable

import numpy as np
import tables
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from .index import add_index_args, load_index
from .retriever_base import LocalRetriever


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetriever(LocalRetriever):
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
            nargs="+",
            default=[-1],
            help="The device id to use for the retriever(query and passage encoder)",
        )
        # database arguments
        parser.add_argument(
            "--read_only",
            action="store_true",
            default=False,
            help="Whether to open the database in read only mode",
        )
        # index arguments
        parser = add_index_args(parser)
        return parser

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
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
        self._embedding_size = self.query_encoder.config.hidden_size

        # prepare gpu
        if args.retriever_device_id != [-1]:
            assert torch.cuda.is_available(), "CUDA is not available"
            assert torch.cuda.device_count() >= len(args.retriever_device_id), (
                f"Number of devices ({len(args.retriever_device_id)}) "
                f"exceeds the number of available GPUs ({torch.cuda.device_count()})"
            )
            self.device_id = args.retriever_device_id
            self.query_encoder.to(args.retriever_device_id[0])
            if self.passage_encoder is not None:
                self.passage_encoder.to(args.retriever_device_id[0])

        # load database
        self.read_only = args.read_only
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

        # consistency check
        assert len(self.db_table) == len(self.index), "Inconsistent database and index"
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
            title = tables.VLStringAtom()
            section = tables.VLStringAtom()
            text = tables.VLStringAtom()
            embedding = tables.Float16Col(768, pos=4)
            tables.StringAtom

        if not os.path.exists(os.path.dirname(database_path)):
            os.makedirs(os.path.dirname(database_path))

        assert not self.read_only, "Database does not exist"
        h5file = tables.open_file(database_path, mode="w", title="Retriever Database")

        group = h5file.create_group("/", "passages", "Passages")
        table = h5file.create_table(group, "data", RetrieveTableWithEmb, "data")
        return h5file, table

    def add_passages(
        self, passages: list[dict[str, str]] | list[str], source: str = None
    ):
        """
        Add passages to the retriever database
        """
        # prepare DDP
        if (len(self.device_id) > 1) and (
            len(passages) > (self.batch_size * len(self.device_id))
        ):
            dist.init_process_group()
            passage_encoder = DDP(self.passage_encoder)
            batch_size = self.batch_size * torch.cuda.device_count()
        else:
            passage_encoder = self.passage_encoder
            batch_size = self.batch_size

        # generate embeddings
        start_idx = len(self.db_table)
        for n, idx in enumerate(range(0, len(passages), batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Generating embeddings for batch {n}")

            # prepare batch
            batch = passages[idx : idx + batch_size]
            embeddings = self._encode_batch(passage_encoder, batch)

            # add embeddings to database
            for i, p in enumerate(batch):
                p = {"text": p} if isinstance(p, str) else p
                row = self.db_table.row
                row["title"] = self._safe_encode(p.get("title", ""))
                row["section"] = self._safe_encode(p.get("section", ""))
                row["text"] = self._safe_encode(p.get("text", ""))
                row["embedding"] = embeddings[i]
                row.append()
            self.db_table.flush()

        # update index
        if not self.index.is_trained:
            logger.info("Training index")
            logger.warning("Training index will consume a lot of memory")
            full_embeddings = np.array(self.db_table.cols.embedding[start_idx:])
            self.index.train_index(full_embeddings)

        # add embeddings to index
        for n, idx in enumerate(range(start_idx, len(self.db_table), batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Adding embeddings for batch {n}")
            embeds_to_add = np.array(
                self.db_table.cols.embedding[idx : idx + batch_size]
            )
            self.index.add_embeddings_batch(embeds_to_add)
        self.index.serialize()
        logger.info("Finished adding passages")

        # cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    def _search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        embeddings = self._encode_batch(self.query_encoder, query)
        scores, indices = self.index.search_batch(embeddings, top_k, **search_kwargs)
        results = [
            {
                "query": q,
                "indices": [int(i) for i in r],
                "scores": [float(i) for i in s],
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
        encoder: torch.nn.Module,
        batch: Iterable[dict[str, str]] | Iterable[str],
    ) -> np.ndarray:
        # prepare batch
        batch_text = [self._prepare_text(doc) for doc in batch]

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
        return self._embedding_size


def setup_distributed():
    dist.init_process_group()


def cleanup_distributed():
    dist.destroy_process_group()
