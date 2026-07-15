import json
from pathlib import Path
from typing import Literal, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure, download, download_and_extract
from flexrag.common.dataclasses import ChatMessages, ChatTurn, Context

from ...core import IRDialogueSample, MappingDataset
from ...corpora.corpus_dataset import _ContextMappingCorpus

RESOURCES = {
    "corpus": {
        "clapnq": "https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/passage_level/clapnq.jsonl.zip",
        "cloud": "https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/passage_level/cloud.jsonl.zip",
        "fiqa": "https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/passage_level/fiqa.jsonl.zip",
        "govt": "https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/passage_level/govt.jsonl.zip",
    },
    "data": "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/refs/heads/main/human/generation_tasks/reference.jsonl",
}


@configure
class MTRAGDatasetConfig:
    """Configuration for MTRAGDataset.

    `MTRAG <https://arxiv.org/abs/2501.03468>`_ is a human-created, end-to-end
    multi-turn retrieval-augmented generation benchmark designed to evaluate
    the full RAG pipeline across realistic conversational settings, diverse
    domains, and challenging phenomena such as later-turn reasoning, unanswerable
     and non-standalone questions.

    :param data_path: The path to the MTRAG dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of MTRAG to use. Default is `all`.
        Available choices are: `clapnq`, `cloud`, `fiqa`, `govt`, and `all`.
        The first four choices correspond to the four domains in MTRAG,
        while `all` corresponds to the full dataset containing all domains.
        Note that if `all` is chosen, the corpus for all domains will be merged.
    :type subset: str
    """

    data_path: Optional[str] = None
    subset: Literal["clapnq", "cloud", "fiqa", "govt", "all"] = "all"


class MTRAGDataset(MappingDataset[IRDialogueSample]):
    def __init__(self, config: MTRAGDatasetConfig):
        # Set basic arguments
        self._subset = config.subset

        # Prepare data path
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "mt_rag"
        else:
            data_path = Path(config.data_path)
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        # Download the dataset if not exists
        if not (data_path / "reference.jsonl").exists():
            download(RESOURCES["data"], data_path, keep_name=True)

        if self._subset != "all":
            corpus_path = data_path / f"{self._subset}.jsonl"
            if not corpus_path.exists():
                download_and_extract(RESOURCES["corpus"][self._subset], data_path)
        else:
            for domain, url in RESOURCES["corpus"].items():
                corpus_path = data_path / f"{domain}.jsonl"
                if not corpus_path.exists():
                    download_and_extract(url, data_path)

        # Load the corpus
        self._context_data = {}
        if self._subset != "all":
            domains = [self._subset]
        else:
            domains = ["clapnq", "cloud", "fiqa", "govt"]
        for domain in domains:
            corpus_path = data_path / f"{domain}.jsonl"
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = json.loads(line)
                    data = Context(
                        context_id=raw["id"],
                        data={
                            "text": raw["text"],
                            "title": raw["title"],
                        },
                        metadata={"url": raw["url"]},
                        source=f"mtrag-{domain}",
                    )
                    self._context_data[data.context_id] = data

        # Load the Conversations
        conv_path = data_path / "reference.jsonl"
        convs = [json.loads(line) for line in conv_path.open("r", encoding="utf-8")]
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        for conv in convs:
            qid = conv["task_id"]
            if self._subset != "all" and self._subset not in conv["Collection"]:
                continue
            # Parse conversation
            convs = []
            for turn in conv["input"]:
                if turn["speaker"] == "user":
                    convs.append(ChatTurn(role="user", content=turn["text"]))
                elif turn["speaker"] in {"model", "agent"}:
                    convs.append(ChatTurn(role="assistant", content=turn["text"]))
                else:
                    raise ValueError(f"Unknown speaker: {turn['speaker']}")
            convs = ChatMessages.from_list(convs)
            # Parse reference answer
            ref = ChatTurn(role="assistant", content=conv["targets"][0]["text"])
            # Parse golden contexts
            ctx_ids = {}
            for ctx in conv["contexts"]:
                assert ctx["document_id"] in self._context_data
                ctx_ids[ctx["document_id"]] = 1.0
            # Save data
            self._queries_data[qid] = convs
            self._answers_data[qid] = ref
            self._qrels_data[qid] = ctx_ids
        self._qids = sorted(self._queries_data.keys())
        self._corpus = _ContextMappingCorpus(self._context_data)
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> IRDialogueSample:
        qid = self._qids[index]
        query = self._queries_data[qid]
        answer = self._answers_data[qid]
        ctxs = [
            self._corpus.contexts[ctx_id] for ctx_id in self._qrels_data[qid].keys()
        ]
        return IRDialogueSample(
            dialogue_id=qid,
            messages=query,
            golden_responses=[answer],
            contexts=ctxs,
            qrels=dict(self._qrels_data[qid]),
        )

    @property
    def corpus(self):
        return self._corpus
