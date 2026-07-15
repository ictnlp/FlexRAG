import json
import os
import zipfile
from typing import Optional

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import Context

from ...core import ContextualMCSample, MappingDataset


@configure
class NovelQADatasetConfig:
    """Configuration for NovelQA Dataset.

    `NovelQA <https://arxiv.org/abs/2403.12766>`_ is a benchmark tailored for
    evaluating LLMs with complex, extended narratives.
    Constructed from English novels, NovelQA offers a unique blend of complexity,
    length, and narrative coherence, making it an ideal tool for assessing deep
    textual understanding in LLMs.

    Note that this dataset currently only exposes questions and options in the
    PublicDomain subset, and answers are not public.

    :param data_path: The path to the NovelQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically to FLEXRAG_CACHE_DIR.
    :type data_path: str
    """

    data_path: Optional[str] = None


class NovelQADataset(MappingDataset[ContextualMCSample]):
    def __init__(self, config: NovelQADatasetConfig):
        # download dataset if not exists
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "novel_qa"
        else:
            data_path = config.data_path
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            assert os.getenv("HF_TOKEN") is not None, (
                "HF_TOKEN environment variable must be set to download the NovelQA dataset."
            )
            hf_hub_download(
                repo_id="NovelQA/NovelQA",
                repo_type="dataset",
                filename="NovelQA.zip",
                local_dir=data_path.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
            with zipfile.ZipFile(data_path / "NovelQA.zip", "r") as zip_f:
                zip_f.extractall(data_path.as_posix())

        # load the contexts
        self._context_data = {}
        ctx_dir = data_path / "Books" / "PublicDomain"
        for doc_path in ctx_dir.iterdir():
            with open(doc_path, "r", encoding="utf-8") as f:
                self._context_data[doc_path.stem] = Context(
                    context_id=doc_path.stem,
                    data={"text": f.read()},
                    source="NovelQA PublicDomain",
                )

        # load the multiple-choice questions
        self._queries_data: dict[str, str] = {}
        self._choices_data: dict[str, list[str]] = {}
        self._qrels_data: dict[str, dict[str, float]] = {}
        qa_dir = data_path / "Data" / "PublicDomain"
        for qa_path in qa_dir.iterdir():
            doc_id = qa_path.stem
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
                for qid, item in qa_data.items():
                    self._queries_data[qid] = item["Question"]
                    self._choices_data[qid] = [
                        item["Options"][choice]
                        for choice in sorted(item["Options"].keys())
                    ]
                    self._qrels_data[qid] = {doc_id: 1.0}
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualMCSample:
        qid = self._qids[index]
        ctx_ids = list(self._qrels_data[qid].keys())
        return ContextualMCSample(
            question_id=qid,
            question=self._queries_data[qid],
            choices=self._choices_data[qid],
            contexts=[self._context_data[ctx_id] for ctx_id in ctx_ids],
        )
