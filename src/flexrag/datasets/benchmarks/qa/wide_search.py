import csv
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import MappingDataset, QASample


def _markdown_escape(value: str) -> str:
    value = value.replace("\n", " ").replace("\r", " ")
    return value.replace("|", "\\|")


def _csv_to_markdown_table(path: Path) -> tuple[str, list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None, f"No header found in gold csv: {path}"
        headers = reader.fieldnames

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for row in rows:
        values = [_markdown_escape(str(row.get(header, ""))) for header in headers]
        body_lines.append("| " + " | ".join(values) + " |")
    markdown = "\n".join([header_line, separator_line, *body_lines])
    return markdown, rows


@configure
class WideSearchDatasetConfig:
    """Configuration for WideSearch dataset.

    `WideSearch <https://huggingface.co/datasets/ByteDance-Seed/WideSearch>`_
    is a benchmark for broad information-seeking tasks. Each sample contains
    a natural language query and a structured evaluation schema, while the gold
    answer is stored as a CSV table under the corresponding ``widesearch_gold``
    directory.

    :param data_path: The path to the local WideSearch dataset repository.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


class WideSearchDataset(MappingDataset[QASample]):
    """Dataset for the WideSearch benchmark."""

    def __init__(self, config: WideSearchDatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "widesearch"

        if not data_path.exists():
            snapshot_download(
                repo_id="ByteDance-Seed/WideSearch",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        self._raw_data = load_dataset(data_path.as_posix(), split="full")
        self._gold_dir = data_path / "widesearch_gold"
        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}
        for item in self._raw_data:
            qid = item["instance_id"]
            gold_path = self._gold_dir / f"{qid}.csv"
            if not gold_path.exists():
                raise FileNotFoundError(f"Missing gold csv for {qid}: {gold_path}")
            gold_markdown, gold_rows = _csv_to_markdown_table(gold_path)
            self._queries_data[qid] = item["query"]
            self._answers_data[qid] = [f"```markdown\n{gold_markdown}\n```"]
            self._metadata[qid] = {
                "evaluation": json.loads(item["evaluation"]),
                "language": item["language"],
                "gold_rows": gold_rows,
                "gold_csv_path": gold_path.as_posix(),
            }
        self._qids = sorted(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._qids)

    def get_item(self, index: int) -> QASample:
        qid = self._qids[index]
        return QASample(
            question=self._queries_data[qid],
            question_id=qid,
            answers=self._answers_data[qid],
            metadata=self._metadata[qid],
        )
