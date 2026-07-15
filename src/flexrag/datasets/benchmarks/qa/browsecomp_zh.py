import base64
import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import MappingDataset, QASample

_REPO_ID = "PALIN2018/BrowseComp-ZH"


@configure
class BrowseCompZHDatasetConfig:
    """Configuration for BrowseComp-ZH dataset.

    `BrowseComp-ZH <https://arxiv.org/abs/2504.19314>`_
    is a Chinese benchmark for evaluating the web browsing and reasoning
    ability of large language models on hard-to-retrieve questions.

    :param data_path: The path to the BrowseComp-ZH dataset file or directory.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


class BrowseCompZHDataset(MappingDataset[QASample]):
    """Dataset for BrowseComp-ZH benchmark."""

    def __init__(self, config: BrowseCompZHDatasetConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "browsecomp_zh"
        else:
            data_path = Path(config.data_path)

        if data_path.is_file():
            parquet_path = data_path
        else:
            if not data_path.exists():
                data_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=_REPO_ID,
                    repo_type="dataset",
                    local_dir=data_path.as_posix(),
                    token=os.getenv("HF_TOKEN"),
                )
            parquet_path = data_path / "test.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"BrowseComp-ZH parquet file not found: {parquet_path}"
            )

        self._data = pd.read_parquet(parquet_path).to_dict(orient="records")
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        canary = item.get("canary", "")
        return QASample(
            question=_decrypt(item.get("Question", ""), canary),
            answers=[_decrypt(item.get("Answer", ""), canary)],
            metadata={"problem_topic": _decrypt(item.get("Topic", ""), canary)},
        )


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode("utf-8")


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]
