import csv
import os
from pathlib import Path
from typing import Annotated, Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure, download

from ...core import DATASETS, IRQASample, MappingDataset
from ...corpora.corpus_dataset import _ContextMappingCorpus
from ...reader import LineDelimitedReader

_ANNOTATION_REPO_ID = "PrimeQA/clapnq"
_CORPUS_REPO_ID = "PrimeQA/clapnq_passages"
_RETRIEVAL_URL = (
    "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval"
    "/{split}/question_{split}_{answerability}.tsv"
)
_ANSWERABILITY_OPTIONS = ("answerable", "unanswerable")


@configure
class ClapNQDatasetConfig:
    """Configuration for ClapNQDataset.

    `CLAPNQ <https://arxiv.org/abs/2404.02103>`_ is a long-form question
    answering benchmark built from Natural Questions. It provides grounded gold
    passages and a dedicated passage corpus for retrieval-augmented generation.

    :param data_path: The path to the local CLAPNQ annotation repository. If
        not provided, the annotations will be downloaded automatically.
    :type data_path: Optional[str]
    :param corpus_path: The path to the local CLAPNQ passage corpus repository
        or ``passages.tsv`` file. If not provided and ``load_corpus`` is enabled,
        the corpus will be downloaded automatically.
    :type corpus_path: Optional[str]
    :param retrieval_path: The path to the local official retrieval TSV files.
        If not provided, the retrieval TSVs will be downloaded automatically.
    :type retrieval_path: Optional[str]
    :param split: The dataset split to use. Default is `dev`.
        Available choices are: `train`, `dev`.
    :type split: str
    :param answerability: Which samples to load. Default is `all`.
        Available choices are: `all`, `answerable`, `unanswerable`.
    :type answerability: str
    :param load_corpus: Whether to load the CLAPNQ passage corpus and expose it
        through ``dataset.corpus``. Default is False.
    :type load_corpus: bool
    """

    data_path: Optional[str] = None
    corpus_path: Optional[str] = None
    retrieval_path: Optional[str] = None
    split: Annotated[str, Choices("train", "dev")] = "dev"
    answerability: Annotated[str, Choices("all", *_ANSWERABILITY_OPTIONS)] = "all"
    load_corpus: bool = False


@DATASETS("clapnq", config_class=ClapNQDatasetConfig)
class ClapNQDataset(MappingDataset[IRQASample]):
    """Dataset for the CLAPNQ end-to-end RAG benchmark."""

    def __init__(self, config: ClapNQDatasetConfig):
        data_dir = self._prepare_data_dir(config)
        retrieval_dir = self._resolve_retrieval_dir(config, data_dir)

        self._context_data: dict[str, Context] | None = None
        if config.load_corpus:
            self._context_data = self._load_corpus(config)
            self._corpus = _ContextMappingCorpus(self._context_data)
        else:
            self._corpus = None

        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[str]] = {}
        self._doc_ids_data: dict[str, list[str]] = {}
        self._passages_data: dict[str, list[dict]] = {}
        self._meta_data: dict[str, dict] = {}
        self._qids: list[str] = []

        answerability_options = (
            _ANSWERABILITY_OPTIONS
            if config.answerability == "all"
            else (config.answerability,)
        )
        for answerability in answerability_options:
            annotation_path = self._resolve_annotation_path(
                data_dir, config.split, answerability
            )
            retrieval_path = self._resolve_retrieval_path(
                retrieval_dir, config.split, answerability
            )
            retrieval_data = self._load_retrieval_data(retrieval_path)
            for item in LineDelimitedReader(annotation_path, file_format="jsonl"):
                qid = str(item["id"])
                retrieval_item = retrieval_data.get(qid)
                if retrieval_item is None:
                    raise KeyError(
                        f"CLAPNQ retrieval data for question '{qid}' not found in "
                        f"{retrieval_path}"
                    )
                outputs = item.get("output", []) or []
                answers = [
                    answer
                    for answer in (
                        str(output.get("answer", "") or "").strip()
                        for output in outputs
                    )
                    if answer
                ]
                doc_ids = self._parse_doc_ids(retrieval_item.get("doc-id-list", ""))
                passages = item.get("passages", []) or []

                self._qids.append(qid)
                self._queries_data[qid] = str(item["input"])
                self._answers_data[qid] = answers
                self._doc_ids_data[qid] = doc_ids
                self._passages_data[qid] = passages
                self._meta_data[qid] = {
                    "split": config.split,
                    "answerability": answerability,
                    "doc_ids": doc_ids,
                    "passages": passages,
                    "selected_sentences": [
                        output.get("selected_sentences", []) or [] for output in outputs
                    ],
                    "output_metadata": [
                        output.get("meta", {}) or {} for output in outputs
                    ],
                }
        return

    def _prepare_data_dir(self, config: ClapNQDatasetConfig) -> Path:
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "clapnq"
        else:
            data_dir = Path(config.data_path)

        if config.data_path is None and not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=_ANNOTATION_REPO_ID,
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
        if not data_dir.exists():
            raise FileNotFoundError(
                f"CLAPNQ annotation directory not found: {data_dir}"
            )
        if data_dir.is_file():
            raise ValueError(f"CLAPNQ data_path must be a directory, got: {data_dir}")
        return data_dir

    def _resolve_retrieval_dir(
        self, config: ClapNQDatasetConfig, data_dir: Path
    ) -> Path:
        if config.retrieval_path is not None:
            return Path(config.retrieval_path)
        if (data_dir / "retrieval").exists():
            return data_dir / "retrieval"
        return FLEXRAG_CACHE_DIR / "datasets" / "clapnq_retrieval"

    def _resolve_annotation_path(
        self, data_dir: Path, split: str, answerability: str
    ) -> Path:
        file_name = f"clapnq_{split}_{answerability}.jsonl"
        candidates = [
            data_dir / file_name,
            data_dir / "annotated_data" / file_name,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"CLAPNQ annotation file not found: {file_name} under {data_dir}"
        )

    def _resolve_retrieval_path(
        self, retrieval_dir: Path, split: str, answerability: str
    ) -> Path:
        file_name = f"question_{split}_{answerability}.tsv"
        if retrieval_dir.is_file():
            return retrieval_dir

        candidates = [
            retrieval_dir / split / file_name,
            retrieval_dir / file_name,
            retrieval_dir / "retrieval" / split / file_name,
        ]
        for path in candidates:
            if path.exists():
                return path

        path = retrieval_dir / split / file_name
        download(
            _RETRIEVAL_URL.format(split=split, answerability=answerability),
            path,
            show_progress=False,
        )
        return path

    def _load_retrieval_data(self, retrieval_path: Path) -> dict[str, dict[str, str]]:
        with open(retrieval_path, "r", encoding="utf-8", newline="") as f:
            return {str(row["id"]): row for row in csv.DictReader(f, delimiter="\t")}

    def _load_corpus(self, config: ClapNQDatasetConfig) -> dict[str, Context]:
        if config.corpus_path is None:
            corpus_path = FLEXRAG_CACHE_DIR / "datasets" / "clapnq_passages"
            if not corpus_path.exists():
                corpus_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=_CORPUS_REPO_ID,
                    repo_type="dataset",
                    local_dir=corpus_path.as_posix(),
                    token=os.getenv("HF_TOKEN"),
                )
        else:
            corpus_path = Path(config.corpus_path)

        if corpus_path.is_dir():
            corpus_path = corpus_path / "passages.tsv"
        if not corpus_path.exists():
            raise FileNotFoundError(f"CLAPNQ corpus file not found: {corpus_path}")

        contexts = {}
        for item in LineDelimitedReader(corpus_path, file_format="tsv"):
            context_id = str(item["id"])
            contexts[context_id] = Context(
                context_id=context_id,
                data={
                    "text": item.get("text", ""),
                    "title": item.get("title", ""),
                },
                source="clapnq",
            )
        return contexts

    def _parse_doc_ids(self, value: str) -> list[str]:
        if not value:
            return []
        return [doc_id for doc_id in value.replace(",", " ").split() if doc_id]

    def __len__(self) -> int:
        return len(self._qids)

    def _build_contexts(self, qid: str) -> list[Context]:
        contexts = []
        passages = self._passages_data[qid]
        for idx, doc_id in enumerate(self._doc_ids_data[qid]):
            if self._context_data is not None and doc_id in self._context_data:
                contexts.append(self._context_data[doc_id])
                continue

            passage = passages[idx] if idx < len(passages) else {}
            contexts.append(
                Context(
                    context_id=doc_id,
                    data={
                        "text": passage.get("text", ""),
                        "title": passage.get("title", ""),
                    },
                    source="clapnq",
                    meta_data={"sentences": passage.get("sentences", []) or []},
                )
            )
        return contexts

    def get_item(self, index: int) -> IRQASample:
        qid = self._qids[index]
        return IRQASample(
            question_id=qid,
            question=self._queries_data[qid],
            answers=self._answers_data[qid],
            contexts=self._build_contexts(qid),
            meta_data=self._meta_data[qid],
        )

    @property
    def corpus(self):
        return self._corpus
