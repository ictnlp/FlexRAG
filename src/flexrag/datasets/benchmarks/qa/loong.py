import json
from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure, download
from flexrag.common.misc import download_and_extract

from ...core import DATASETS, ContextualQASample, MappingDataset
from ...reader import LineDelimitedReader

_DATA_URL = "https://github.com/MozerWang/Loong/raw/refs/heads/main/data/loong.jsonl"
_DOC_URL = "http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"


@configure
class LoongDatasetConfig:
    """Configuration for LoongDataset.

    `Loong <https://arxiv.org/abs/2406.17419>`_ is a long-context benchmark
    for extended multi-document question answering across financial reports,
    legal cases, and academic papers in English and Chinese.

    :param data_path: The local directory for the Loong benchmark data.
        If not provided, the data will be downloaded automatically.
    :type data_path: Optional[str]
    :param level: The task level to load. Available choices are `all`, `level1`,
        `level2`, `level3`, `level4`. Default is `all`.
    :type level: str
    :param set_id: The length set to load. Available choices are `all`, `1`, `2`,
        `3`, `4`. Default is `all`.
    :type set_id: str
    :param doc_type: The document domain to load. Available choices are `all`,
        `financial`, `legal`, `paper`. Default is `all`.
    :type doc_type: str
    :param language: The language to load. Available choices are `all`, `en`,
        `zh`. Default is `all`.
    :type language: str
    """

    data_path: Optional[str] = None
    level: Annotated[str, Choices("all", "level1", "level2", "level3", "level4")] = (
        "all"
    )
    set_id: Annotated[str, Choices("all", "1", "2", "3", "4")] = "all"
    doc_type: Annotated[str, Choices("all", "financial", "legal", "paper")] = "all"
    language: Annotated[str, Choices("all", "en", "zh")] = "all"


@DATASETS("loong", config_class=LoongDatasetConfig)
class LoongDataset(MappingDataset[ContextualQASample]):
    """Dataset for the Loong benchmark."""

    def __init__(self, config: LoongDatasetConfig):
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "loong"
        else:
            data_dir = Path(config.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_file = data_dir / "loong.jsonl"
        if not data_file.exists():
            download(_DATA_URL, data_file)

        doc_dir = data_dir / "doc"
        if not doc_dir.exists():
            download_and_extract(_DOC_URL, data_dir)
        if not doc_dir.exists():
            raise FileNotFoundError(f"Loong document directory not found: {doc_dir}")

        self._legal_docs = self._load_legal_docs(doc_dir)
        self._data: list[ContextualQASample] = []
        for item in LineDelimitedReader(data_file):
            if not self._match_config(item, config):
                continue
            self._data.append(self._build_sample(item, doc_dir))
        return

    def _match_config(self, item: dict, config: LoongDatasetConfig) -> bool:
        if config.level != "all":
            expected_level = int(config.level.removeprefix("level"))
            if item.get("level") != expected_level:
                return False
        if config.set_id != "all" and str(item.get("set")) != config.set_id:
            return False
        if config.doc_type != "all" and str(item.get("type")) != config.doc_type:
            return False
        if config.language != "all" and str(item.get("language")) != config.language:
            return False
        return True

    def _load_legal_docs(self, doc_root: Path) -> dict[str, dict]:
        legal_path = doc_root / "legal" / "legal.json"
        if not legal_path.exists():
            return {}
        return json.loads(legal_path.read_text(encoding="utf-8"))

    def _resolve_doc_path(
        self, doc_root: Path, doc_name: str, doc_type: str
    ) -> Path | list[Path]:
        type_dir = doc_root / doc_type if doc_type else doc_root
        candidates = [
            type_dir / doc_name,
            type_dir / f"{doc_name}.md",
            doc_root / doc_name,
            doc_root / f"{doc_name}.md",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = sorted(type_dir.glob(f"{doc_name}.*"))
        if len(matches) == 1:
            return matches[0]

        if doc_type == "financial":
            matches = sorted(type_dir.glob(f"*{doc_name}*"))
            if matches:
                return matches

        matches = sorted(doc_root.glob(f"**/{doc_name}"))
        if len(matches) == 1:
            return matches[0]

        matches = sorted(doc_root.glob(f"**/{doc_name}.*"))
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"Loong document not found for {doc_name} in {doc_root}"
        )

    def _build_contexts(self, item: dict, doc_dir: Path) -> list[Context]:
        contexts = []
        doc_type = str(item.get("type", ""))
        for index, doc_name in enumerate(item.get("doc", [])):
            doc_name = str(doc_name)
            if doc_type == "legal":
                doc_path = doc_dir / "legal" / "legal.json"
                legal_doc = self._legal_docs.get(doc_name)
                if legal_doc is None:
                    raise FileNotFoundError(
                        f"Loong legal document not found for {doc_name} in {doc_path}"
                    )
                text = str(legal_doc.get("content", ""))
                raw_path = doc_path.as_posix()
            else:
                doc_path = self._resolve_doc_path(doc_dir, doc_name, doc_type)
                if isinstance(doc_path, list):
                    text = "\n\n".join(
                        path.read_text(encoding="utf-8") for path in doc_path
                    )
                    raw_path = [path.as_posix() for path in doc_path]
                else:
                    text = doc_path.read_text(encoding="utf-8")
                    raw_path = doc_path.as_posix()
            contexts.append(
                Context(
                    context_id=f"{item['id']}:doc{index}",
                    data={
                        "title": Path(doc_name).stem or doc_name,
                        "text": text,
                    },
                    source="loong",
                    metadata={
                        "doc_name": doc_name,
                        "doc_index": index,
                        "raw_path": raw_path,
                    },
                )
            )
        return contexts

    def _build_answers(self, answer) -> list[str]:
        if isinstance(answer, str):
            return [answer]
        return [json.dumps(answer, ensure_ascii=False)]

    def _build_question(self, item: dict) -> str:
        instruction = str(item.get("instruction", "")).strip()
        question = str(item.get("question", "")).strip()
        if instruction and question:
            return f"{instruction}\n\nQuestion: {question}"
        if instruction:
            return instruction
        return question

    def _build_sample(self, item: dict, doc_dir: Path) -> ContextualQASample:
        return ContextualQASample(
            question_id=str(item["id"]),
            question=self._build_question(item),
            answers=self._build_answers(item.get("answer")),
            contexts=self._build_contexts(item, doc_dir),
            metadata={
                "level": item.get("level"),
                "set": item.get("set"),
                "type": item.get("type"),
                "language": item.get("language"),
                "length": item.get("length"),
                "shuffle_doc": item.get("shuffle_doc", False),
                "instruction": item.get("instruction", ""),
                "raw_question": item.get("question", ""),
                "prompt_template": item.get("prompt_template", ""),
                "doc_names": [str(doc_name) for doc_name in item.get("doc", [])],
            },
        )

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> ContextualQASample:
        return self._data[index]
