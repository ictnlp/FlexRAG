import json
import re
from pathlib import Path
from typing import Annotated, Literal, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.common.misc import download

from ...core import DATASETS, ContextualQASample, MappingDataset

RESOURCES = {
    "en": {
        "mem": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/en/perltmem_en.json",
        "qa": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/en/perltqa_en.json",
    },
    "en_v2": {
        "mem": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/en_v2/perltmem_en_v2.json",
        "qa": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/en_v2/perltqa_en_v2.json",
    },
    "zh": {
        "mem": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/zh/perltmem.json",
        "qa": "https://github.com/Elvin-Yiming-Du/PerLTQA/raw/refs/heads/main/Dataset/zh/perltqa.json",
    },
}


@configure
class PerLTQADatasetConfig:
    """Configuration for PerLTQA.

    `PerLTQA <https://aclanthology.org/2024.sighan-1.18/>`_ is a comprehensive
    conversational QA dataset that integrates multiple types of long-term
    memory—world knowledge, user profiles, social relationships, events, and
    dialogue history—to model their interactions for consistent and personalized
    question answering.

    :param data_path: The path to the PerLTQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param lang: The language of the dataset to use. Default is `en_v2`.
        Available choices are: `en`, `en_v2`, `zh`.
    :type lang: Literal["en", "en_v2", "zh"]
    """

    data_path: Optional[str] = None
    lang: Literal["en", "en_v2", "zh"] = "en_v2"


@DATASETS("perltqa", config_class=PerLTQADatasetConfig)
class PerLTQADataset(MappingDataset[ContextualQASample]):
    def __init__(self, config: PerLTQADatasetConfig):
        self._lang = config.lang
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "perltqa"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset
        mem_path = data_dir / self._lang / "mem.json"
        qa_path = data_dir / self._lang / "qa.json"
        if not mem_path.exists():
            download(RESOURCES[self._lang]["mem"], mem_path)
        if not qa_path.exists():
            download(RESOURCES[self._lang]["qa"], qa_path)

        # load the contexts & QA items
        raw_mem = json.loads(mem_path.read_text(encoding="utf-8"))
        raw_qa = json.loads(qa_path.read_text(encoding="utf-8"))
        contexts = {}
        self._qa_data = []
        # parse contexts
        if isinstance(raw_mem, list):
            for item in raw_mem:
                ctx_id = item["profile"]["Protagonist"]
                ctx = Context(
                    context_id=ctx_id,
                    data=item,
                    source=f"perltqa-{self._lang}",
                )
                contexts[ctx_id] = ctx
        else:
            for ctx_id, item in raw_mem.items():
                ctx = Context(
                    context_id=ctx_id,
                    data=item,
                    source=f"perltqa-{self._lang}",
                )
                contexts[ctx_id] = ctx
        for qa in raw_qa:
            ctx_id = list(qa.keys())[0]
            # skip if context not exists
            if ctx_id not in contexts:
                continue
            # parse QA pairs
            for group_name, group in qa[ctx_id].items():
                if isinstance(group, dict):
                    group = [group]
                for qa_pair in group:
                    if group_name == "profile":
                        metadata = qa_pair.copy()
                        q, a = metadata.pop("Question"), metadata.pop("Answer")
                        self._qa_data.append(
                            ContextualQASample(
                                question=q,
                                answers=[a],
                                metadata=metadata,
                                contexts=[contexts[ctx_id]],
                            )
                        )
                    else:
                        for anchor, sub_qas in qa_pair.items():
                            for sub_qa in sub_qas:
                                metadata = sub_qa.copy()
                                q, a = metadata.pop("Question"), metadata.pop("Answer")
                                self._qa_data.append(
                                    ContextualQASample(
                                        question=q,
                                        answers=[a],
                                        metadata=metadata,
                                        contexts=[contexts[ctx_id]],
                                    )
                                )
        return

    def __len__(self) -> int:
        return len(self._qa_data)

    def get_item(self, index: int) -> ContextualQASample:
        return self._qa_data[index]
