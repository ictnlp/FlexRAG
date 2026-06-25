import csv
import io
import shutil
import zipfile
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    Choices,
    Context,
    configure,
    download_and_extract,
)

from ...core import DATASETS, IRQASample, MappingDataset
from ...corpora.corpus_dataset import _ContextMappingCorpus
from ...reader import LineDelimitedReader

_REPO_URL = "https://github.com/primeqa/clapnq/archive/refs/heads/main.zip"
_ANSWERABILITY_OPTIONS = ("answerable", "unanswerable")


@configure
class ClapNQDatasetConfig:
    """Configuration for ClapNQDataset.

    `CLAPNQ <https://arxiv.org/abs/2404.02103>`_ is a long-form question
    answering benchmark built from Natural Questions. It provides grounded gold
    passages and a dedicated passage corpus for retrieval-augmented generation.

    :param data_path: The path to the local CLAPNQ GitHub repository. If not
        provided, the repository will be downloaded to ``FLEXRAG_CACHE_DIR``.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `dev`.
        Available choices are: `train`, `dev`.
    :type split: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "dev")] = "dev"


@DATASETS("clapnq", config_class=ClapNQDatasetConfig)
class ClapNQDataset(MappingDataset[IRQASample]):
    """Dataset for the CLAPNQ end-to-end RAG benchmark."""

    def __init__(self, config: ClapNQDatasetConfig):
        # determine data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "clapnq"
        else:
            data_dir = Path(config.data_path)

        # download if not exists
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            download_and_extract(_REPO_URL, data_dir)
            extracted_dir = data_dir / "clapnq-main"
            if extracted_dir.exists():
                for path in extracted_dir.iterdir():
                    shutil.move(path.as_posix(), data_dir / path.name)
                extracted_dir.rmdir()
        if data_dir.is_file():
            raise ValueError(f"CLAPNQ data_path must be a directory, got: {data_dir}")

        # load corpus contexts from the passages.tsv.zip file
        self._context_data: dict[str, Context] = {}
        corpus_path = data_dir / "retrieval" / "passages.tsv.zip"
        with zipfile.ZipFile(corpus_path) as zip_file:
            with zip_file.open("passages.tsv") as f:
                reader = csv.DictReader(
                    io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"
                )
                for item in reader:
                    context_id = str(item["id"])
                    self._context_data[context_id] = Context(
                        context_id=context_id,
                        data={
                            "text": item.get("text", ""),
                            "title": item.get("title", ""),
                        },
                        source="clapnq",
                    )

        # load queries / answers from the question files
        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[str]] = {}
        self._doc_ids_data: dict[str, list[str]] = {}
        self._meta_data: dict[str, dict] = {}
        self._qids: list[str] = []
        for answerability in _ANSWERABILITY_OPTIONS:
            file_name = f"question_{config.split}_{answerability}.tsv"
            question_path = data_dir / "retrieval" / config.split / file_name

            for item in LineDelimitedReader(question_path, file_format="tsv"):
                qid = str(item["id"])
                doc_ids = [
                    doc_id
                    for doc_id in item.get("doc-id-list", "").replace(",", " ").split()
                    if doc_id
                ]
                self._qids.append(qid)
                self._queries_data[qid] = item["question"]
                self._answers_data[qid] = [
                    answer.strip()
                    for answer in item.get("answers", "").split("::")
                    if answer.strip()
                ]
                self._doc_ids_data[qid] = doc_ids
                self._meta_data[qid] = {
                    "split": config.split,
                    "answerability": answerability,
                    "doc_ids": doc_ids,
                }
        return

    def __len__(self) -> int:
        return len(self._qids)

    def get_item(self, index: int) -> IRQASample:
        qid = self._qids[index]
        doc_ids = self._doc_ids_data[qid]
        return IRQASample(
            question_id=qid,
            question=self._queries_data[qid],
            answers=self._answers_data[qid],
            contexts=[self._context_data[ctx_id] for ctx_id in doc_ids],
            qrels={ctx_id: 1.0 for ctx_id in doc_ids},
            meta_data=self._meta_data[qid],
        )

    @property
    def corpus(self):
        return _ContextMappingCorpus(self._context_data)
