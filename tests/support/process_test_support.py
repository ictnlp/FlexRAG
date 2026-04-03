import time
from dataclasses import field

import numpy as np

from flexrag.common import configure
from flexrag.models.encoders.encoder_base import EncoderBase
from flexrag.models.encoders.local_process_encoder_base import LocalProcessEncoderBase
from flexrag.models.scorers.local_process_scorer_base import LocalProcessScorerBase
from flexrag.models.scorers.scorer_base import PairScorerBase


@configure
class FakeLocalTextEncoderConfig:
    device_id: list[int] = field(default_factory=list)
    delay_s: float = 0.0
    error_on: str | None = None
    embedding_dim: int = 3


class FakeLocalTextEncoderImpl(EncoderBase):
    def __init__(self, config: FakeLocalTextEncoderConfig) -> None:
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        self._embedding_dim = config.embedding_dim
        return

    def encode(self, texts: list[str] | str) -> np.ndarray:
        texts = texts if isinstance(texts, list) else [texts]
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.error_on is not None and any(self.error_on in text for text in texts):
            raise ValueError(f"boom: {self.error_on}")
        embeddings = []
        for text in texts:
            checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 997
            vector = [
                float(len(text)),
                float(sum(ord(ch) for ch in text) % 97),
                float(checksum),
            ]
            embeddings.append(vector[: self._embedding_dim])
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_size(self) -> int:
        return self._embedding_dim


class FakeLocalTextEncoder(LocalProcessEncoderBase):
    impl_cls = FakeLocalTextEncoderImpl


@configure
class FakeLocalPairScorerConfig:
    device_id: list[int] = field(default_factory=list)
    delay_s: float = 0.0
    error_on: str | None = None


class FakeLocalPairScorerImpl(PairScorerBase):
    def __init__(self, config: FakeLocalPairScorerConfig) -> None:
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        return

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.error_on is not None and any(
            self.error_on in query or self.error_on in candidate
            for query, candidate in pairs
        ):
            raise ValueError(f"boom: {self.error_on}")

        scores = []
        for query, candidate in pairs:
            checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(query + candidate))
            scores.append(
                float((len(query) * 3 + len(candidate) * 5 + checksum) % 997) / 997.0
            )
        return np.array(scores, dtype=np.float32)


class FakeLocalPairScorer(LocalProcessScorerBase):
    impl_cls = FakeLocalPairScorerImpl
