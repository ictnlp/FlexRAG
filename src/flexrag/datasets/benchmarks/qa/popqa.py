import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import DATASETS, MappingDataset, QASample
from ...reader import LineDelimitedReader


@configure
class PopQADatasetConfig:
    """Configuration for PopQADataset.

    `PopQA <https://huggingface.co/datasets/akariasai/PopQA>`_ is an
    open-domain QA benchmark containing entity-centric questions derived from
    Wikidata tuples. Alongside each question, it provides the original subject,
    relation, object annotations and popularity-related metadata.

    :param data_path: The local path to the PopQA dataset. Default is None.
        If not provided, the dataset will be loaded from Hugging Face.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


def _parse_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


@DATASETS("popqa", config_class=PopQADatasetConfig)
class PopQADataset(MappingDataset[QASample]):
    def __init__(self, config: PopQADatasetConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "popqa" / "test.tsv"
        else:
            data_path = Path(config.data_path)
            if data_path.is_dir():
                data_path = data_path / "test.tsv"

        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="akariasai/PopQA",
                repo_type="dataset",
                filename="test.tsv",
                local_dir=data_path.parent.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )

        self._data = list(LineDelimitedReader(data_path, file_format="tsv"))
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = dict(self._data[index])
        question_id = str(item.get("id", index))
        answers = _parse_string_list(item.get("possible_answers"))
        item["id"] = question_id
        item["possible_answers"] = answers
        item["s_aliases"] = _parse_string_list(item.get("s_aliases"))
        item["o_aliases"] = _parse_string_list(item.get("o_aliases"))
        return QASample(
            question_id=question_id,
            question=item["question"],
            answers=answers,
            metadata=item,
        )
