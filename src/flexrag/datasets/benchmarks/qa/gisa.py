import base64
import hashlib
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import DATASETS, MappingDataset, QASample
from ...reader import LineDelimitedReader

_ANSWER_ENCODINGS = {
    "63": "cp1252",
    "64": "cp1252",
}


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _get_answer_encoding(qid: str) -> str:
    return _ANSWER_ENCODINGS.get(qid, "utf-8")


def _load_answer_csv(path: Path, qid: str) -> str:
    rows = list(
        LineDelimitedReader(
            path,
            file_format="csv",
            encoding=_get_answer_encoding(qid),
        )
    )
    return json.dumps(rows, ensure_ascii=False)


@configure
class GISADatasetConfig:
    """Configuration for GISA dataset.

    `GISA <https://huggingface.co/datasets/RUC-NLPIR/GISA>`_
    is a benchmark for general information-seeking assistants. Each sample
    contains an encrypted question together with structured gold answers
    stored in per-example CSV files and a human search trajectory stored
    in a JSON file.

    In this implementation, the encrypted question is decrypted according
    to the official loading method provided in the dataset card, and the
    gold answer CSV is serialized into ``answers[0]`` as a JSON string
    containing a list of row dictionaries. Most files are loaded as UTF-8;
    only the known non-UTF-8 answer files in the current GISA snapshot are
    handled with an explicit dataset-specific encoding override.

    :param data_path: The path to the local GISA dataset repository.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@DATASETS("gisa", config_class=GISADatasetConfig)
class GISADataset(MappingDataset[QASample]):
    """Dataset for GISA benchmark."""

    def __init__(self, config: GISADatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "gisa"

        if not data_path.exists():
            snapshot_download(
                repo_id="RUC-NLPIR/GISA",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        question_path = data_path / "encrypted_question.jsonl"
        answer_dir = data_path / "answer"
        trace_dir = data_path / "trace"

        if not question_path.exists():
            raise FileNotFoundError(f"Missing question file: {question_path}")
        if not answer_dir.exists():
            raise FileNotFoundError(f"Missing answer directory: {answer_dir}")
        if not trace_dir.exists():
            raise FileNotFoundError(f"Missing trace directory: {trace_dir}")

        self._data: list[QASample] = []
        for item in LineDelimitedReader(question_path):
            qid = str(item["id"])
            answer_path = answer_dir / f"{qid}.csv"
            trace_path = trace_dir / f"{qid}.json"

            if not answer_path.exists():
                raise FileNotFoundError(
                    f"Missing answer csv for question {qid}: {answer_path}"
                )
            if not trace_path.exists():
                raise FileNotFoundError(
                    f"Missing trace json for question {qid}: {trace_path}"
                )

            question = _decrypt(str(item["question"]), str(item["canary"]))
            answer_json = _load_answer_csv(answer_path, qid)
            with open(trace_path, "r", encoding="utf-8") as trace_f:
                trace = json.load(trace_f)

            self._data.append(
                QASample(
                    question=question,
                    question_id=qid,
                    answers=[answer_json],
                    meta_data={
                        "answer_type": item["answer_type"],
                        "question_type": item["question_type"],
                        "topic": item["topic"],
                        "trace": trace,
                    },
                )
            )
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        return self._data[index]
