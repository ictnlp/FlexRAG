import os
import tempfile
from pathlib import Path
from typing import Annotated, Optional
from zipfile import ZipFile

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure

from ...core import DATASETS, MappingDataset, QASample

_SUBSETS = ("feta", "nq", "paper_text", "paper_tab", "fin", "tat")
_ANSWER_FIELDS = {
    "feta": ("answer",),
    "nq": ("short_answer", "long_answer"),
    "paper_text": ("answer_1", "answer_2", "answer_3"),
    "paper_tab": ("answer_1", "answer_2", "answer_3"),
    "fin": ("answer_1", "answer_2"),
    "tat": ("answer",),
}
_DOC_ARCHIVES = {
    "feta": "wiki_feta_docs.zip",
    "nq": "wiki_nq_docs.zip",
    "paper_text": "paper_docs.zip",
    "paper_tab": "paper_docs.zip",
    "fin": "fin_docs.zip",
    "tat": "tat_docs.zip",
}
_DOC_MIME_TYPES = {
    "pdf": "application/pdf",
    "html": "text/html",
}
_WIKI_SUBSETS = {"feta", "nq"}
_REPO_ID = "qinchuanhui/UDA-QA"


@configure
class UDAQADatasetConfig:
    """Configuration for UDAQADataset.

    `UDA-QA <https://arxiv.org/abs/2406.15187>`_
    is a benchmark suite for question answering over real-world unstructured
    documents such as financial reports, papers, and Wikipedia-derived pages.

    :param data_path: The path to the local UDA-QA dataset repository. If not
        provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The UDA-QA subset to load. Default is `feta`.
        Available choices are: `feta`, `nq`, `paper_text`, `paper_tab`,
        `fin`, and `tat`.
    :type subset: str
    :param prefered_format: The preferred source document format for the
        Wikipedia-based subsets `feta` and `nq`. Available choices are `pdf`
        and `html`. This option is ignored by the other subsets.
    :type prefered_format: str
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices(*_SUBSETS)] = "feta"
    prefered_format: Annotated[str, Choices("pdf", "html")] = "pdf"


@DATASETS("uda_qa", config_class=UDAQADatasetConfig)
class UDAQADataset(MappingDataset[QASample]):
    """Dataset for the UDA-QA benchmark."""

    def __init__(self, config: UDAQADatasetConfig):
        self._subset = config.subset
        self._prefered_format = config.prefered_format
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "uda_qa"
        else:
            data_dir = Path(config.data_path)

        data_files = sorted(
            path.as_posix() for path in (data_dir / self._subset).glob("test*.parquet")
        )
        if not data_files and (config.data_path is None or not data_dir.exists()):
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=_REPO_ID,
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
            data_files = sorted(
                path.as_posix()
                for path in (data_dir / self._subset).glob("test*.parquet")
            )
        if not data_files:
            raise FileNotFoundError(
                f"UDA-QA parquet files not found for subset '{self._subset}' under {data_dir}"
            )

        cache_dir = Path(
            os.getenv(
                "HF_DATASETS_CACHE",
                Path(tempfile.gettempdir(), "flexrag_hf_datasets").as_posix(),
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._data = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            cache_dir=cache_dir.as_posix(),
        )

        self._data_dir = data_dir
        self._documents_dir = self._prepare_documents_dir(config)
        self._source_file_cache: dict[str, tuple[Path, str, str]] = {}
        return

    def __len__(self) -> int:
        return len(self._data)

    def _prepare_documents_dir(self, config: UDAQADatasetConfig) -> Path:
        archive_name = _DOC_ARCHIVES[self._subset]
        archive_path = self._data_dir / "src_doc_files" / archive_name
        extract_dir = archive_path.with_suffix("")

        if not extract_dir.exists() and not archive_path.exists():
            if config.data_path is None:
                snapshot_download(
                    repo_id=_REPO_ID,
                    repo_type="dataset",
                    local_dir=self._data_dir.as_posix(),
                    token=os.getenv("HF_TOKEN"),
                )
            else:
                raise FileNotFoundError(
                    f"UDA-QA archive for subset '{self._subset}' not found: {archive_path}"
                )

        if archive_path.exists() and not extract_dir.exists():
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path) as zf:
                zf.extractall(archive_path.parent)
        if not extract_dir.exists():
            raise FileNotFoundError(
                f"UDA-QA extracted documents not found for subset '{self._subset}': {extract_dir}"
            )
        return extract_dir

    def _resolve_source_file(self, doc_name: str) -> tuple[Path, str, str]:
        if doc_name in self._source_file_cache:
            return self._source_file_cache[doc_name]

        if self._subset in _WIKI_SUBSETS:
            file_format = self._prefered_format
            source_path = (
                self._documents_dir / f"{file_format}s" / f"{doc_name}.{file_format}"
            )
        else:
            file_format = "pdf"
            source_path = self._documents_dir / f"{doc_name}.pdf"

        if not source_path.exists():
            raise FileNotFoundError(
                f"UDA-QA source file for document '{doc_name}' not found: {source_path}"
            )

        resolved = (source_path, file_format, _DOC_MIME_TYPES[file_format])
        self._source_file_cache[doc_name] = resolved
        return resolved

    def _normalize_answers(self, item: dict) -> list[str]:
        def to_string_list(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [str(entry) for entry in value]
            if hasattr(value, "tolist"):
                converted = value.tolist()
                if isinstance(converted, list):
                    return [str(entry) for entry in converted]
                return [str(converted)]
            return [str(value)]

        answers = []
        for field in _ANSWER_FIELDS[self._subset]:
            for answer in to_string_list(item.get(field)):
                answer = answer.strip()
                if answer and answer not in answers:
                    answers.append(answer)
        return answers

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        answers = self._normalize_answers(item)
        source_path, source_format, source_mime_type = self._resolve_source_file(
            item["doc_name"]
        )
        metadata = {
            "subset": self._subset,
            "doc_name": item["doc_name"],
            "source_file_path": source_path.as_posix(),
            "source_file_name": source_path.name,
            "source_file_format": source_format,
            "source_mime_type": source_mime_type,
        }
        if "doc_url" in item:
            metadata["doc_url"] = item["doc_url"]
        if "answer_type" in item:
            metadata["answer_type"] = item["answer_type"]
        if "answer_scale" in item:
            metadata["answer_scale"] = item["answer_scale"]

        return QASample(
            question_id=str(item["q_uid"]),
            question=item["question"],
            answers=answers,
            metadata=metadata,
        )
