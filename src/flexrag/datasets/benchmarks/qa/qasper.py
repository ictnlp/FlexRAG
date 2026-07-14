import json
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure
from flexrag.common.misc import download_and_extract

from ...core import DATASETS, ContextualQASample, MappingDataset

_RESOURCES = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}

_FILE_MAP = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


@configure
class QasperDatasetConfig:
    """Configuration for QasperDataset.

    `Qasper <https://aclanthology.org/2021.naacl-main.365/>`_ is a QA benchmark
    over scientific papers. Each paper contains multiple questions and annotated
    answers with evidence.

    :param data_path: Path to the extracted dataset directory. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `test`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    :param context_mode: How contexts are organized. Default is `paragraph`.
        Available choices are: `paragraph`, `paper`.
    :type context_mode: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation", "test")] = "test"
    context_mode: Annotated[str, Choices("paragraph", "paper")] = "paragraph"


def _normalize_answers(raw_answers: list[dict]) -> list[str]:
    answers = []
    for item in raw_answers:
        answer = item.get("answer", {})
        if answer.get("unanswerable"):
            candidates = ["Unanswerable"]
        elif answer.get("extractive_spans", []):
            candidates = [", ".join(answer["extractive_spans"])]
        elif answer.get("free_form_answer", ""):
            candidates = [answer["free_form_answer"]]
        elif answer.get("yes_no") is True:
            candidates = ["Yes"]
        elif answer.get("yes_no") is False:
            candidates = ["No"]
        else:
            candidates = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate:
                answers.append(candidate)
    return list(dict.fromkeys(answers))


@DATASETS("qasper", config_class=QasperDatasetConfig)
class QasperDataset(MappingDataset[ContextualQASample]):
    def __init__(self, config: QasperDatasetConfig):
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "qasper"
        else:
            data_dir = Path(config.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / _FILE_MAP[config.split]
        if not data_path.exists():
            resource = _RESOURCES["train_dev"]
            if config.split == "test":
                resource = _RESOURCES["test"]
            download_and_extract(resource, data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Qasper file not found: {data_path}")

        raw_data = json.loads(data_path.read_text(encoding="utf-8"))
        self._data = []
        for paper_id, paper in raw_data.items():
            contexts = self._build_contexts(paper_id, paper, config.context_mode)
            paper_meta = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
            }
            for qa in paper.get("qas", []):
                self._data.append(
                    ContextualQASample(
                        question_id=qa["question_id"],
                        question=qa["question"],
                        answers=_normalize_answers(qa.get("answers", [])),
                        contexts=contexts,
                        metadata={
                            **paper_meta,
                            "question_writer": qa.get("question_writer"),
                            "paper_read": qa.get("paper_read"),
                            "search_query": qa.get("search_query"),
                            "topic_background": qa.get("topic_background"),
                            "nlp_background": qa.get("nlp_background"),
                            "annotations": qa.get("answers", []),
                        },
                    )
                )
        return

    def _build_contexts(
        self, paper_id: str, paper: dict, context_mode: str
    ) -> list[Context]:
        texts = []
        contexts = []

        for sec_idx, section in enumerate(paper.get("full_text", [])):
            section_name = section.get("section_name", "")
            for para_idx, paragraph in enumerate(section.get("paragraphs", [])):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                texts.append(paragraph)
                if context_mode == "paragraph":
                    contexts.append(
                        Context(
                            context_id=f"{paper_id}:sec{sec_idx}:para{para_idx}",
                            data={"text": paragraph},
                            source="qasper",
                            metadata={
                                "paper_id": paper_id,
                                "kind": "paragraph",
                                "section_name": section_name,
                                "section_idx": sec_idx,
                                "paragraph_idx": para_idx,
                            },
                        )
                    )

        for fig_idx, item in enumerate(paper.get("figures_and_tables", [])):
            caption = item.get("caption", "").strip()
            if not caption:
                continue
            texts.append(caption)
            if context_mode == "paragraph":
                contexts.append(
                    Context(
                        context_id=f"{paper_id}:fig{fig_idx}",
                        data={"text": caption},
                        source="qasper",
                        metadata={
                            "paper_id": paper_id,
                            "kind": "figure_or_table",
                            "figure_idx": fig_idx,
                            "file": item.get("file", ""),
                        },
                    )
                )

        if context_mode == "paper":
            return [
                Context(
                    context_id=paper_id,
                    data={"text": "\n".join(texts)},
                    source="qasper",
                    metadata={
                        "paper_id": paper_id,
                        "kind": "paper",
                    },
                )
            ]
        return contexts

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> ContextualQASample:
        return self._data[index]
