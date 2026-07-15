import base64
import hashlib
from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.misc import download

from ...core import MappingDataset, QASample
from ...reader import LineDelimitedReader


@configure
class BrowseCompDatasetConfig:
    """Configuration for BrowseComp dataset.

    `BrowseComp <https://arxiv.org/abs/2504.12516>`_
    is a challenging yet easy-to-use benchmark of 1,266 short-answer questions
    designed to evaluate an agent's persistence and creativity in navigating the
    web to find hard-to-locate, entangled information.

    :param data_path: The path to the BrowseComp dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


RESOURCES = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"  # fmt: skip


class BrowseCompDataset(MappingDataset[QASample]):
    """Dataset for BrowseComp benchmark."""

    def __init__(self, config: BrowseCompDatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "browsecomp"

        # download the dataset if not exists
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download(RESOURCES, data_path)
        data_path = data_path / "browse_comp_test_set.csv"
        self._data = list(LineDelimitedReader(data_path))
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        return QASample(
            question=_decrypt(item.get("problem", ""), item.get("canary", "")),
            answers=[_decrypt(item.get("answer", ""), item.get("canary", ""))],
            metadata={"problem_topic": item["problem_topic"]},
        )


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode("utf-8")


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]
