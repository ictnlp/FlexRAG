import os
import tempfile
from pathlib import Path
from typing import Annotated, Optional

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
_REPO_ID = "qinchuanhui/UDA-QA"


@configure
class UDAQADatasetConfig:
    """Configuration for UDAQADataset.

    `UDA-QA <https://huggingface.co/datasets/qinchuanhui/UDA-QA>`_
    is a benchmark suite for question answering over real-world unstructured
    documents such as financial reports, papers, and Wikipedia-derived pages.

    :param data_path: The path to the local UDA-QA dataset repository. If not
        provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The UDA-QA subset to load. Default is `feta`.
        Available choices are: `feta`, `nq`, `paper_text`, `paper_tab`,
        `fin`, and `tat`.
    :type subset: str
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices(*_SUBSETS)] = "feta"


def _to_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
        return [str(converted)]
    return [str(value)]


def _normalize_answers(item: dict, subset: str) -> list[str]:
    answers = []
    for field in _ANSWER_FIELDS[subset]:
        for answer in _to_string_list(item.get(field)):
            answer = answer.strip()
            if answer and answer not in answers:
                answers.append(answer)
    return answers


def _resolve_subset_files(data_dir: Path, subset: str) -> list[str]:
    return sorted(path.as_posix() for path in (data_dir / subset).glob("test*.parquet"))


@DATASETS("uda_qa", config_class=UDAQADatasetConfig)
class UDAQADataset(MappingDataset[QASample]):
    """Dataset for the UDA-QA benchmark."""

    def __init__(self, config: UDAQADatasetConfig):
        self._subset = config.subset
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "uda_qa"
        else:
            data_dir = Path(config.data_path)

        data_files = _resolve_subset_files(data_dir, self._subset)
        if not data_files and (config.data_path is None or not data_dir.exists()):
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=_REPO_ID,
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
            data_files = _resolve_subset_files(data_dir, self._subset)
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

        archive_name = _DOC_ARCHIVES[self._subset]
        archive_path = data_dir / "src_doc_files" / archive_name
        self._doc_archive_path = archive_path if archive_path.exists() else None
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        answers = _normalize_answers(item, self._subset)
        meta_data = {
            "subset": self._subset,
            "doc_name": item["doc_name"],
        }
        if "doc_url" in item:
            meta_data["doc_url"] = item["doc_url"]
        if "answer_type" in item:
            meta_data["answer_type"] = item["answer_type"]
        if "answer_scale" in item:
            meta_data["answer_scale"] = item["answer_scale"]
        if self._doc_archive_path is not None:
            meta_data["doc_archive_path"] = self._doc_archive_path.as_posix()

        return QASample(
            question_id=str(item["q_uid"]),
            question=item["question"],
            answers=answers or None,
            meta_data=meta_data,
        )
