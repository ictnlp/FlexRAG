import json
import re
from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import ChatMessages
from flexrag.common.misc import download

from ...core import DATASETS, MappingDataset, MultiSessionQASample

RESOURCE_URL = (
    "https://github.com/snap-research/locomo/raw/refs/heads/main/data/locomo10.json"
)


@configure
class LoCoMoDatasetConfig:
    """Configuration for LoCoMo.

    `LoCoMo <https://aclanthology.org/2024.acl-long.747/>`_ is a dataset of very
    long-term open-domain conversations generated through a machine-human pipeline
    and verified for long-range consistency and event grounding.

    :param data_path: The path to the LoCoMo dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@DATASETS("locomo", config_class=LoCoMoDatasetConfig)
class LoCoMoDataset(MappingDataset[MultiSessionQASample]):
    def __init__(self, config: LoCoMoDatasetConfig):
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "locomo"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset
        data_path = data_dir / "locomo10.json"
        if not data_path.exists():
            download(RESOURCE_URL, data_path)

        # load the data items
        raw_data = json.loads(data_path.read_text(encoding="utf-8"))
        self._qa_data = {}
        self._conv_data = {}
        self._meta_data = {}
        for group in raw_data:
            # parse conversation
            group_id = group["sample_id"]
            conv_keys = [
                key
                for key in group["conversation"].keys()
                if re.match(r"session_\d+$", key)
            ]
            conv_keys = sorted(conv_keys, key=lambda x: int(x.split("_")[1]))
            messages = []
            metadatas = {}
            for conv_key in conv_keys:
                conv_id = conv_key.split("_")[1]
                # parse raw conversation turns
                message = []
                for turn in group["conversation"][conv_key]:
                    message.append(
                        {
                            "role": turn["speaker"],
                            "content": turn["text"],
                            "turn_id": turn["dia_id"],
                            "strict_mode": False,
                        }
                    )
                message = ChatMessages.from_list(message, strict_mode=False)
                metadatas[conv_key] = {}
                # parse ovservation
                if f"{conv_key}_observation" in group["observation"]:
                    metadatas[conv_key]["observation"] = group["observation"][
                        f"{conv_key}_observation"
                    ]
                # parse session summary
                if f"{conv_key}_summary" in group["session_summary"]:
                    metadatas[conv_key]["session_summary"] = group["session_summary"][
                        f"{conv_key}_summary"
                    ]
                # parse event summary
                if f"events_{conv_key}" in group["event_summary"]:
                    metadatas[conv_key]["event_summary"] = group["event_summary"][
                        f"events_{conv_key}"
                    ]
                messages.append(message)
            self._conv_data[group_id] = messages
            self._meta_data[group_id] = metadatas
            # parse qa pairs
            for i, qa in enumerate(group["qa"]):
                qid = f"locomo_{group_id}_{i}"
                self._qa_data[qid] = qa
        return

    def __len__(self) -> int:
        return len(self._qa_data)

    def get_item(self, index: int) -> MultiSessionQASample:
        qid = list(self._qa_data.keys())[index]
        group_id = qid.split("_")[1]
        metadata = self._meta_data[group_id]
        metadata["evidence"] = self._qa_data[qid].get("evidence", [])
        metadata["category"] = self._qa_data[qid].get("category", 1)
        if metadata["category"] == 5:
            response = "Not Answerable"
            metadata["adversarial_answer"] = str(
                self._qa_data[qid]["adversarial_answer"]
            )
        else:
            response = str(self._qa_data[qid]["answer"])
        return MultiSessionQASample(
            question_id=qid,
            sessions=self._conv_data[group_id],
            question=self._qa_data[qid]["question"],
            answers=[response],
            meta_data=metadata,
        )
