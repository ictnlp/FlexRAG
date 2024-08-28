import os
import json
from hashlib import blake2b
from typing import Any, Optional


class Fingerprint:
    def __init__(
        self, path: Optional[str] = None, features: Optional[Any] = None
    ) -> None:
        # prepare fingerprint path
        if path is not None:
            self.path = path
        else:
            assert features is not None
            features = blake2b(json.dumps(features).encode("utf-8")).hexdigest()
            self.path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "librarian",
                "fingerprints",
                features,
            )

        # initialize fingerprint
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._state = int(f.read())
        else:
            self.clean()
        return

    def update(self, texts: list[str | bytes] | str | bytes) -> None:
        if not isinstance(texts, list):
            texts = [texts]
        for text in texts:
            if isinstance(text, str):
                self._state ^= int(blake2b(text.encode("utf-8")).hexdigest(), 16)
            else:
                self._state ^= int(blake2b(text).hexdigest(), 16)
        with open(self.path, "w") as f:
            f.write(str(self._state))
        return

    def hexdigest(self) -> str:
        return hex(self._state)[2:]

    def digest(self) -> bytes:
        return self._state.to_bytes(64)

    def clean(self) -> None:
        self._state = 0
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write(str(self._state))
        return
