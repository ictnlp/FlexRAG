from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context

from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase


@configure
class NarrativeQADatasetConfig:
    """Configuration for NarrativeQADataset.

    `NarrativeQA <https://arxiv.org/abs/1712.07040>`_ is a high-quality subset
    of NarrativeQA focused on literary works, designed to address issues with
    noisy documents and flawed QA pairs in the original benchmark.

    :param data_path: The path to the NarrativeQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `test`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation", "test")] = "test"


@QA_DATASETS("narrative_qa", config_class=NarrativeQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("narrative_qa", config_class=NarrativeQADatasetConfig)
class NarrativeQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: NarrativeQADatasetConfig):
        # Download the dataset if not exists
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "narrative_qa"
        else:
            data_path = Path(config.data_path)
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="deepmind/narrativeqa",
                local_dir=data_path.as_posix(),
            )
        data = load_dataset(data_path.as_posix(), split=config.split)

        self._context_data = {}
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        for idx, item in enumerate(data):
            self._queries_data[str(idx)] = item["question"]["text"]
            self._answers_data[str(idx)] = [ans["text"] for ans in item["answers"]]
            context = Context(
                context_id=item["document"]["id"],
                data={
                    "text": item["document"]["text"],
                    "summary": item["document"]["summary"]["text"],
                    "title": item["document"]["summary"]["title"],
                },
                source=item["document"].get("url", ""),
                meta_data={
                    "kind": item["document"].get("kind", ""),
                    "file_size": item["document"].get("file_size", 0),
                },
            )
            self._context_data[context.context_id] = context
            self._qrels_data[str(idx)] = {context.context_id: 1.0}
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
