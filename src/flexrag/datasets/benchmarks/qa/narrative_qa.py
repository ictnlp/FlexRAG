import re
from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS, ContextualQASample, MappingDataset


@configure
class NarrativeQADatasetConfig:
    """Configuration for NarrativeQADataset.

    `NarrativeQA <https://arxiv.org/abs/1712.07040>`_ is a reading-comprehension
    dataset that requires answering questions about entire books or movie scripts,
    emphasizing deep narrative understanding rather than shallow text matching.

    :param data_path: The path to the NarrativeQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `test`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    :param fix_line_endings: Whether to normalize document line endings and fold
        single line breaks in the NarrativeQA full text. Default is True.
    :type fix_line_endings: bool
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation", "test")] = "test"
    fix_line_endings: bool = True


def _fix_narrative_qa_doc(doc: str) -> str:
    # unify line endings
    doc = doc.replace("\r\n", "\n")
    # reserve paragraph breaks
    doc = re.sub(r"\n{2,}", "\n\n", doc)
    # remove single line breaks
    doc = re.sub(r"(?<!\n)\n(?!\n)", " ", doc)
    return doc


@DATASETS("narrative_qa", config_class=NarrativeQADatasetConfig)
class NarrativeQADataset(MappingDataset[ContextualQASample]):
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
            if config.fix_line_endings:
                doc_text = _fix_narrative_qa_doc(item["document"]["text"])
            else:
                doc_text = item["document"]["text"]
            context = Context(
                context_id=item["document"]["id"],
                data={
                    "text": doc_text,
                    "summary": item["document"]["summary"]["text"],
                    "title": item["document"]["summary"]["title"],
                },
                source=item["document"].get("url", ""),
                metadata={
                    "kind": item["document"].get("kind", ""),
                    "file_size": item["document"].get("file_size", 0),
                },
            )
            self._context_data[context.context_id] = context
            self._qrels_data[str(idx)] = {context.context_id: 1.0}
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualQASample:
        qid = str(index)
        contexts = [
            self._context_data[ctx_id] for ctx_id in self._qrels_data[qid].keys()
        ]
        return ContextualQASample(
            question=self._queries_data[qid],
            question_id=qid,
            contexts=contexts,
            answers=self._answers_data[qid],
        )
