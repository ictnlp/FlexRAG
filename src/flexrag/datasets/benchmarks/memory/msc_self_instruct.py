import json
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import ChatMessages

from ...core import MappingDataset, MultiSessionQASample


@configure
class MSCSelfInstructDatasetConfig:
    """Configuration for MSC-Self-Instruct.

    `MSC-Self-Instruct <https://arxiv.org/pdf/2310.08560>`_ is a self-instruct
    dataset of multi-session conversational (MSC) examples designed to evaluate
    models on personalized, multi-round dialogue tasks by generating
    conversation openers that reference topics from prior sessions.

    :param data_path: The path to the MSC dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


class MSCSelfInstructDataset(MappingDataset[MultiSessionQASample]):
    def __init__(self, config: MSCSelfInstructDatasetConfig):
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "msc_self_instruct"
        else:
            data_dir = Path(config.data_path)

        # download the dataset
        if not data_dir.exists():
            hf_hub_download(
                "MemGPT/MSC-Self-Instruct",
                filename="msc_self_instruct.jsonl",
                repo_type="dataset",
                local_dir=data_dir,
            )

        # load the dataset
        data_file = data_dir / "msc_self_instruct.jsonl"
        raw_data = [json.loads(line) for line in data_file.open()]

        # parse the dataset
        self._qa_samples = []
        for item in raw_data:
            # load previous sessions
            previous_sessions = []
            for n, prev in enumerate(item["previous_dialogs"]):
                if n % 2 == 0:
                    role = "speaker_1"
                else:
                    role = "speaker_2"
                msgs = [{"role": role, "content": m["text"]} for m in prev["dialog"]]
                previous_sessions.append(ChatMessages.from_list(msgs, False))
            # load current dialogue
            current_dialog = []
            for n, turn in enumerate(item["dialog"]):
                if n % 2 == 0:
                    role = "speaker_1"
                else:
                    role = "speaker_2"
                current_dialog.append({"role": role, "content": turn["text"]})
            current_dialog = ChatMessages.from_list(current_dialog, False)
            previous_sessions.append(current_dialog)
            # load qa pairs
            query = item["self_instruct"]["B"]
            response = item["self_instruct"]["A"]
            self._qa_samples.append(
                MultiSessionQASample(
                    question=query,
                    answers=[response],
                    sessions=previous_sessions,
                )
            )
        return

    def __len__(self) -> int:
        return len(self._qa_samples)

    def get_item(self, index: int) -> MultiSessionQASample:
        return self._qa_samples[index]
