import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone

METADATA_FILE = "metadata.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IndexState:
    """Serializable index state for an on-disk retriever collection."""

    indexes: list[str] = field(default_factory=list)
    dirty_indexes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def create(cls) -> "IndexState":
        """Create an empty index state.

        :return: Empty index state.
        """
        return cls()

    @classmethod
    def load(cls, root: str) -> "IndexState":
        """Load index state from a collection root.

        :param root: Collection root directory.
        :return: Loaded index state.
        """
        path = os.path.join(root, METADATA_FILE)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        field_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in field_names})

    def save(self, root: str) -> None:
        """Atomically save index state to a collection root.

        :param root: Collection root directory.
        :return: None.
        """
        os.makedirs(root, exist_ok=True)
        self.updated_at = _now()
        path = os.path.join(root, METADATA_FILE)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
        return

    def mark_dirty(self, index_name: str) -> None:
        """Mark one index as dirty.

        :param index_name: Index name.
        :return: None.
        """
        if index_name not in self.dirty_indexes:
            self.dirty_indexes.append(index_name)
            self.dirty_indexes.sort()
        return

    def mark_clean(self, index_name: str) -> None:
        """Mark one index as clean.

        :param index_name: Index name.
        :return: None.
        """
        if index_name in self.dirty_indexes:
            self.dirty_indexes.remove(index_name)
        return

    def add_index(self, index_name: str) -> None:
        """Record an active clean index.

        :param index_name: Index name.
        :return: None.
        """
        if index_name not in self.indexes:
            self.indexes.append(index_name)
            self.indexes.sort()
        self.mark_clean(index_name)
        return

    def remove_index(self, index_name: str) -> None:
        """Remove an index from state.

        :param index_name: Index name.
        :return: None.
        """
        if index_name in self.indexes:
            self.indexes.remove(index_name)
        self.mark_clean(index_name)
        return

