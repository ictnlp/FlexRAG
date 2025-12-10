import json
import shutil
import zipfile
from hashlib import blake2b
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.common.misc import download_and_extract

from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase


@configure
class CRUDQADatasetConfig:
    """Configuration for CRUDQADataset.

    `CRUD-QA <https://arxiv.org/abs/2401.17043>`_ is subsets of CRUD RAG benchmark
    consisting of three knowledge-intensive datasets, 1-doc QA, 2-doc QA, and
    3-doc QA, designed to evaluate RAG systems.

    :param data_path: The path to the CRUD dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of CRUD-QA to use. Default is `questanswer_1doc`.
        Available choices are:

        - `questanswer_1doc`: QA pairs that require 1 document.
        - `questanswer_2docs`: QA pairs that require 2 documents.
        - `questanswer_3docs`: QA pairs that require 3 documents.
    :type subset: str
    :param load_corpus: Whether to load the documents from 80000_docs. Default is False.
        Note that loading the corpus is optional because CRUD does not guarantee
        all relevant documents are within 80000_docs.
    :type load_corpus: bool
    """

    data_path: Optional[str] = None
    subset: Annotated[
        str,
        Choices(
            "questanswer_1doc",
            "questanswer_2docs",
            "questanswer_3docs",
        ),
    ] = "questanswer_1doc"
    load_corpus: bool = False


RESOURCES = "https://github.com/IAAR-Shanghai/CRUD_RAG/archive/refs/heads/main.zip"


@QA_DATASETS("crud_qa", config_class=CRUDQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("crud_qa", config_class=CRUDQADatasetConfig)
class CRUDQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: CRUDQADatasetConfig):
        self._subset = config.subset
        # download the crud dataset if not exists
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "crud"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            download_and_extract(RESOURCES, data_dir)
            for file in (data_dir / "CRUD_RAG-main").iterdir():
                shutil.move(file, data_dir)
            (data_dir / "CRUD_RAG-main").rmdir()

        # load the corpus
        self._context_data = {}
        if config.load_corpus:
            corpus_dir = data_dir / "data" / "80000_docs"
            for doc_file in corpus_dir.iterdir():
                # skip hallucinated documents
                if "hallu" in doc_file.name:
                    continue
                with open(doc_file, "r", encoding="utf-8") as f:
                    for doc in f:
                        doc_idx = blake2b(
                            doc.encode("utf-8"), digest_size=16
                        ).hexdigest()
                        self._context_data[doc_idx] = Context(
                            context_id=doc_idx,
                            data={"text": doc},
                            source="crud",
                        )

        # load qa pairs
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        data_path = data_dir / "data" / "crud" / "merged.zip"
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            with zip_ref.open("merged.json", "r") as f:
                data = json.load(f)[config.subset]
        for item in data:
            query_id = item["ID"]
            self._queries[query_id] = item["questions"]
            self._answers[query_id] = [item["answers"]]
            # As CRUD does not guarantee all relevant documents within 80000_docs, we
            # have to add all relevant documents from the merged dataset to context_data.
            # Blake2b hash algorithm is used to deduplicate the documents.
            docs = []
            qrels = {}
            if "news1" in item:
                docs.append(item["news1"])
            if "news2" in item:
                docs.append(item["news2"])
            if "news3" in item:
                docs.append(item["news3"])
            for doc in docs:
                doc_idx = blake2b(doc.encode("utf-8"), digest_size=16).hexdigest()
                self._context_data[doc_idx] = Context(
                    context_id=doc_idx, data={"text": doc}, source="crud"
                )
                qrels[doc_idx] = 1.0
            self._qrels_data[query_id] = qrels
        return

    @property
    def _queries(self) -> dict[str, str]:
        return self._queries_data

    @property
    def _answers(self) -> dict[str, list[str]] | None:
        return self._answers_data

    @property
    def _qrels(self) -> dict[str, dict[str, float]]:
        return self._qrels_data

    @property
    def _contexts(self) -> dict[str, Context]:
        return self._context_data
