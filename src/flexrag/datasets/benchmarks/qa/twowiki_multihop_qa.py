import json
from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure

from ...core import DATASETS, ContextualQASample, MappingDataset


@configure
class TwoWikiMultihopQADatasetConfig:
    """Configuration for TwoWikiMultihopQADataset.

    `2WikiMultihopQA <https://huggingface.co/datasets/xanhho/2WikiMultihopQA>`_
    is a multi-hop question answering benchmark designed to evaluate reasoning
    across multiple Wikipedia documents. Each sample includes a question, an
    answer, a set of source documents, and annotations for supporting facts and
    reasoning evidences.

    :param data_path: The path to the local 2WikiMultihopQA dataset repository.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `dev`.
        Available choices are: `train`, `dev`, `test`.
    :type split: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "dev", "test")] = "dev"


@DATASETS("2wiki_multihop_qa", config_class=TwoWikiMultihopQADatasetConfig)
class TwoWikiMultihopQADataset(MappingDataset[ContextualQASample]):
    """Dataset for the 2WikiMultihopQA benchmark."""

    def __init__(self, config: TwoWikiMultihopQADatasetConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "2wiki_multihop_qa"
        else:
            data_path = Path(config.data_path)

        data_path.mkdir(parents=True, exist_ok=True)
        parquet_path = data_path / f"{config.split}.parquet"
        if not parquet_path.exists():
            snapshot_download(
                repo_id="xanhho/2WikiMultihopQA",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        raw_dataset = load_dataset(
            "parquet",
            data_files=parquet_path.as_posix(),
            split="train",
        )

        self._data: list[ContextualQASample] = []
        for item in raw_dataset:
            qid = item["_id"]
            raw_contexts = json.loads(item["context"])
            contexts = []
            title_to_sentences: dict[str, list[str]] = {}
            for title, sentences in raw_contexts:
                title_to_sentences[title] = list(sentences)
                contexts.append(
                    Context(
                        context_id=title,
                        data={
                            "title": title,
                            "text": " ".join(sentences),
                        },
                        source="2wikimultihopqa",
                    )
                )

            supporting_facts = []
            supporting_sentences = []
            for title, sent_id in json.loads(item["supporting_facts"]):
                supporting_facts.append({"title": title, "sent_id": sent_id})
                sentences = title_to_sentences.get(title, [])
                if 0 <= sent_id < len(sentences):
                    supporting_sentences.append(
                        {
                            "title": title,
                            "sent_id": sent_id,
                            "text": sentences[sent_id],
                        }
                    )

            evidences = []
            for fact, relation, entity in json.loads(item["evidences"]):
                evidences.append(
                    {
                        "fact": fact,
                        "relation": relation,
                        "entity": entity,
                    }
                )

            answer = item.get("answer")
            answers = [answer] if answer not in {None, ""} else None

            self._data.append(
                ContextualQASample(
                    question_id=qid,
                    question=item["question"],
                    answers=answers,
                    contexts=contexts,
                    meta_data={
                        "type": item.get("type"),
                        "supporting_facts": supporting_facts,
                        "supporting_sentences": supporting_sentences,
                        "evidences": evidences,
                    },
                )
            )
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> ContextualQASample:
        return self._data[index]
