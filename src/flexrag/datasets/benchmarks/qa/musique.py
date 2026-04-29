import shutil
import zipfile
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure

from ...core import DATASETS, ContextualQASample, MappingDataset
from ...reader import LineDelimitedReader

MUSIQUE_FILE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
MUSIQUE_MANUAL_URL = (
    "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing"
)


@configure
class MuSiQueDatasetConfig:
    """Configuration for MuSiQueDataset.

    `MuSiQue <https://arxiv.org/abs/2108.00573>`_ is a multi-hop QA benchmark
    built by composing single-hop questions. This implementation currently
    targets the officially released `MuSiQue-Full` files.

    :param data_path: The path to the MuSiQue data directory. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `validation`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation", "test")] = "validation"


def _download_from_google_drive(file_id: str, save_path: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise ImportError(
            "MuSiQue automatic download requires the optional dependency `gdown`. "
            "Install it in the runtime environment before using MuSiQue."
        ) from error

    save_path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(
        id=file_id,
        output=save_path.as_posix(),
        quiet=False,
        fuzzy=False,
    )
    if result is None or not save_path.exists():
        raise RuntimeError(
            "Unable to download the MuSiQue archive automatically. "
            f"Please download it manually from {MUSIQUE_MANUAL_URL} "
            f"and extract it under {save_path.parent}."
        )
    return


def _ensure_data_dir(data_dir: Path) -> None:
    archive_path = data_dir / "musique_v1.0.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        _download_from_google_drive(MUSIQUE_FILE_ID, archive_path)

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(data_dir)

    nested_data_dir = data_dir / "data"
    if nested_data_dir.exists():
        for item in nested_data_dir.iterdir():
            shutil.move(item.as_posix(), data_dir.as_posix())
        nested_data_dir.rmdir()
    return


@DATASETS("musique", config_class=MuSiQueDatasetConfig)
class MuSiQueDataset(MappingDataset[ContextualQASample]):
    _file_name_map = {
        "train": "musique_full_v1.0_train.jsonl",
        "validation": "musique_full_v1.0_dev.jsonl",
        "test": "musique_full_v1.0_test.jsonl",
    }

    def __init__(self, config: MuSiQueDatasetConfig):
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "musique"
        else:
            data_dir = Path(config.data_path)
        data_path = data_dir / self._file_name_map[config.split]
        if not data_path.exists():
            _ensure_data_dir(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"MuSiQue file not found: {data_path}. "
                "Please verify the extracted archive contents."
            )

        self._data = []
        reader = LineDelimitedReader(data_path)
        for item in reader:
            qid = item["id"]
            paragraphs = item["paragraphs"]
            answers = []
            primary_answer = item.get("answer")
            if primary_answer is not None:
                answers.append(primary_answer)
            for alias in item.get("answer_aliases", []):
                if alias not in answers:
                    answers.append(alias)

            contexts = []
            supporting_paragraph_indices = []
            for paragraph in paragraphs:
                support_flag = paragraph.get("is_supporting")
                if support_flag:
                    supporting_paragraph_indices.append(paragraph["idx"])
                contexts.append(
                    Context(
                        context_id=f"{qid}:{paragraph['idx']}",
                        data={
                            "text": paragraph["paragraph_text"],
                            "title": paragraph["title"],
                        },
                        source="musique",
                        meta_data={
                            "idx": paragraph["idx"],
                            "is_supporting": support_flag,
                        },
                    )
                )

            self._data.append(
                ContextualQASample(
                    question_id=qid,
                    question=item["question"],
                    answers=answers,
                    contexts=contexts,
                    meta_data={
                        "answerable": item.get("answerable"),
                        "question_decomposition": item.get(
                            "question_decomposition", []
                        ),
                        "supporting_paragraph_indices": supporting_paragraph_indices,
                    },
                )
            )
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> ContextualQASample:
        return self._data[index]
