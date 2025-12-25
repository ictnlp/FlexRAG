from pathlib import Path
from typing import Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS, MappingDataset, ContextualQASample


@configure
class MultihopRAGDatasetConfig:
    """Configuration for MultihopRAGDataset.

    `Multihop RAG <https://arxiv.org/abs/2401.15391>`_ is a benchmark dataset for
    evaluating retrieval-augmented generation (RAG) systems on multihop question answering tasks.

    :param data_path: The path to the Multihop RAG dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@DATASETS("multihop_rag", config_class=MultihopRAGDatasetConfig)
class MultihopRAGDataset(MappingDataset[ContextualQASample]):
    def __init__(self, config: MultihopRAGDatasetConfig):
        # Download the dataset if not exists
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "multihop_rag"
        else:
            data_path = Path(config.data_path)
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="yixuantt/MultiHopRAG",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        # Load the corpus
        self._context_data = {}
        corpus = load_dataset(data_path.as_posix(), name="corpus", split="train")
        for item in corpus:
            ctx_id = item["title"]
            self._context_data[ctx_id] = Context(
                context_id=ctx_id,
                data={"text": item["body"], "title": item["title"]},
                source=item["source"],
                meta_data={
                    "author": item["author"],
                    "category": item["category"],
                    "url": item["url"],
                    "published_at": str(item["published_at"]),
                },
            )

        # Load the queries and answers
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        self._meta_data = {}
        data = load_dataset(data_path.as_posix(), name="MultiHopRAG", split="train")
        for idx, item in enumerate(data):
            qid = str(idx)
            self._queries_data[qid] = item["query"]
            self._answers_data[qid] = [item["answer"]]
            self._meta_data[qid] = {"question_type": item["question_type"], "facts": []}
            qrels = {}
            for ctx in item["evidence_list"]:
                qrels[ctx["title"]] = 1.0
                self._meta_data[qid]["facts"].append(ctx["fact"])
            self._qrels_data[qid] = qrels
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualQASample:
        qid = self._qids[index]
        contexts = [
            self._context_data[ctx_id] for ctx_id in self._qrels_data[qid].keys()
        ]
        return ContextualQASample(
            question=self._queries_data[qid],
            question_id=qid,
            contexts=contexts,
            answers=self._answers_data[qid],
            meta_data=self._meta_data[qid],
        )
