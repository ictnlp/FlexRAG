import json
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    ChatMessages,
    ChatTurn,
    Choices,
    Context,
    configure,
)

from ...core import DATASETS, ContextualDialogueSample, MappingDataset

_SUBSET_TO_FILE = {
    "coqa": "data/coqa/dev.json",
    "convfinqa": "data/convfinqa/dev.json",
    "doc2dial": "data/doc2dial/test.json",
    "doqa_cooking": "data/doqa/test_cooking.json",
    "doqa_movies": "data/doqa/test_movies.json",
    "doqa_travel": "data/doqa/test_travel.json",
    "hybridial": "data/hybridial/test.json",
    "inscit": "data/inscit/dev.json",
    "qrecc": "data/qrecc/test.json",
    "quac": "data/quac/test.json",
    "sqa": "data/sqa/test.json",
    "topiocqa": "data/topiocqa/dev.json",
}
_SUBSET_ORDER = list(_SUBSET_TO_FILE.keys())


@configure
class ChatRAGBenchDatasetConfig:
    """Configuration for ChatRAG-Bench dataset.

    `ChatRAG-Bench <https://arxiv.org/abs/2401.10225>`_ is a benchmark for
    contextual conversational QA over provided documents or retrieved context.

    :param data_path: The local ChatRAG-Bench directory. If not provided, the
        default path under FLEXRAG_CACHE_DIR will be used.
    :type data_path: Optional[str]
    :param subset: The subset to load. Default is `all`.
    :type subset: str
    :param num_ctx: If provided, only keep the first `num_ctx` contexts for
        each sample. Default is None.
    :type num_ctx: Optional[int]
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices("all", *_SUBSET_TO_FILE.keys())] = "all"
    num_ctx: Optional[int] = None


@DATASETS("chatrag_bench", config_class=ChatRAGBenchDatasetConfig)
class ChatRAGBenchDataset(MappingDataset[ContextualDialogueSample]):
    """Dataset for ChatRAG-Bench."""

    def __init__(self, config: ChatRAGBenchDatasetConfig):
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "chatrag_bench"
        else:
            data_dir = Path(config.data_path)

        if not data_dir.exists():
            raise FileNotFoundError(f"ChatRAG-Bench directory not found: {data_dir}")

        if config.subset == "all":
            subsets = _SUBSET_ORDER
        else:
            subsets = [config.subset]

        self._data: list[ContextualDialogueSample] = []
        for subset in subsets:
            file_path = data_dir / _SUBSET_TO_FILE[subset]
            if not file_path.exists():
                raise FileNotFoundError(
                    f"ChatRAG-Bench subset file not found for {subset}: {file_path}"
                )
            raw_samples = json.loads(file_path.read_text(encoding="utf-8"))
            for index, item in enumerate(raw_samples):
                self._data.append(
                    self._build_sample(item, subset, index, config.num_ctx)
                )
        return

    def _build_messages(self, item: dict) -> ChatMessages:
        turns = []
        for turn in item.get("messages", []):
            role = str(turn["role"])
            content = str(turn["content"])
            if len(turns) > 0 and turns[-1].role == role:
                turns[-1].content = str(turns[-1].content) + "\n\n" + content
                continue
            turns.append(ChatTurn(role=role, content=content))
        return ChatMessages.from_list(turns)

    def _build_contexts(
        self,
        item: dict,
        subset: str,
        dialogue_id: str,
        num_ctx: Optional[int],
    ) -> list[Context]:
        raw_contexts = item.get("ctxs", [])
        if num_ctx is not None:
            raw_contexts = raw_contexts[:num_ctx]
        contexts = []
        for idx, ctx in enumerate(raw_contexts):
            context_id = str(ctx.get("id", f"{dialogue_id}:ctx{idx}"))
            extra_meta = {
                k: v for k, v in ctx.items() if k not in {"id", "title", "text"}
            }
            contexts.append(
                Context(
                    context_id=context_id,
                    data={
                        "title": str(ctx.get("title", "")),
                        "text": str(ctx.get("text", "")),
                    },
                    source=f"chatrag_bench-{subset}",
                    meta_data=extra_meta,
                )
            )
        return contexts

    def _build_golden_responses(self, item: dict) -> list[ChatTurn]:
        if "answers" in item:
            answers = item["answers"]
        elif "answer" in item:
            answer = item["answer"]
            answers = answer if isinstance(answer, list) else [answer]
        else:
            raise ValueError("ChatRAG-Bench sample must contain `answer` or `answers`.")
        return [ChatTurn(role="assistant", content=str(answer)) for answer in answers]

    def _build_sample(
        self, item: dict, subset: str, index: int, num_ctx: Optional[int]
    ) -> ContextualDialogueSample:
        dialogue_id = str(item.get("id", f"{subset}_{index}"))
        if subset == "all":
            dialogue_id = f"{subset}_{dialogue_id}"
        meta_data = {
            "subset": subset,
            "document": item.get("document"),
            "ground_truth_ctx": item.get("ground_truth_ctx"),
        }
        for key, value in item.items():
            if key not in {"messages", "ctxs", "answers", "answer"}:
                meta_data.setdefault(key, value)
        return ContextualDialogueSample(
            dialogue_id=dialogue_id,
            messages=self._build_messages(item),
            golden_responses=self._build_golden_responses(item),
            contexts=self._build_contexts(item, subset, dialogue_id, num_ctx),
            meta_data=meta_data,
        )

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> ContextualDialogueSample:
        return self._data[index]
