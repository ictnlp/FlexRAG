import os
from pathlib import Path
from typing import Annotated, Any, Optional

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure

from ...core import DATASETS, MappingDataset, QASample

_REPO_ID = "din0s/asqa"
_SPLIT_PATTERNS = {
    "train": "train-*.parquet",
    "validation": "dev-*.parquet",
}


@configure
class ASQADatasetConfig:
    """Configuration for ASQADataset.

    `ASQA <https://arxiv.org/abs/2204.06092>`_
    is an open-domain long-form QA dataset built on ambiguous factoid
    questions. Each sample provides one ambiguous question, one or more
    long-form reference answers, disambiguated QA pairs, and supporting
    Wikipedia page metadata.

    :param data_path: The path to the local ASQA dataset directory. It may
        point to either the dataset repository root or its `data` subdirectory.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `validation`.
        Available choices are: `train`, `validation`.
        The `validation` split is mapped to ASQA's original `dev` files.
    :type split: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation")] = "validation"


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_nested(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_nested(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normalize_nested(item())
        except (TypeError, ValueError):
            pass
    return value


def _resolve_data_dir(data_path: Path) -> Path:
    if data_path.is_file():
        raise ValueError(
            f"ASQADataset data_path must be a directory, got file: {data_path}"
        )
    if (data_path / "data").is_dir():
        return data_path / "data"
    return data_path


@DATASETS("asqa", config_class=ASQADatasetConfig)
class ASQADataset(MappingDataset[QASample]):
    """Dataset for ASQA benchmark."""

    def __init__(self, config: ASQADatasetConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "asqa"
        else:
            data_path = Path(config.data_path)

        if config.data_path is None and not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=_REPO_ID,
                repo_type="dataset",
                local_dir=data_path.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )

        if not data_path.exists():
            raise FileNotFoundError(f"ASQA dataset directory not found: {data_path}")

        data_dir = _resolve_data_dir(data_path)
        parquet_paths = sorted(data_dir.glob(_SPLIT_PATTERNS[config.split]))
        if not parquet_paths:
            raise FileNotFoundError(
                f"ASQA parquet files not found for split '{config.split}' under {data_dir}"
            )

        self._data: list[QASample] = []
        for parquet_path in parquet_paths:
            rows = pq.read_table(parquet_path).to_pylist()
            for row in rows:
                row = _normalize_nested(row)
                annotations = row.get("annotations", []) or []
                sample_id = str(row.get("sample_id", len(self._data)))
                answers = []
                for annotation in annotations:
                    long_answer = str(annotation.get("long_answer", "")).strip()
                    if long_answer:
                        answers.append(long_answer)

                self._data.append(
                    QASample(
                        question_id=sample_id,
                        question=str(row["ambiguous_question"]),
                        answers=answers,
                        meta_data={
                            "sample_id": sample_id,
                            "split": config.split,
                            "qa_pairs": row.get("qa_pairs", []) or [],
                            "wikipages": row.get("wikipages", []) or [],
                            "annotations": annotations,
                        },
                    )
                )
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        return self._data[index]
