import time
from dataclasses import field

import numpy as np

from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models.encoders import LocalEncoderBase
from flexrag.models.generators import GenerationConfig, LocalGeneratorBase
from flexrag.models.scorers import LocalPairScorerBase
from flexrag.resources.runtime_adapters import (
    ProcessEncoderAdapter,
    ProcessGeneratorAdapter,
    ProcessScorerAdapter,
)


@configure
class FakeLocalTextEncoderConfig:
    device_id: list[int] = field(default_factory=list)
    batch_size: int = 32
    delay_s: float = 0.0
    error_on: str | None = None
    embedding_dim: int = 3


class FakeLocalTextEncoderImpl(LocalEncoderBase):
    def __init__(self, config: FakeLocalTextEncoderConfig) -> None:
        super().__init__(batch_size=config.batch_size)
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        self._embedding_dim = config.embedding_dim
        return

    def _encode_batch(self, inputs: list[str]) -> np.ndarray:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.error_on is not None and any(self.error_on in text for text in inputs):
            raise ValueError(f"boom: {self.error_on}")
        embeddings = []
        for text in inputs:
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


class FakeLocalTextEncoder(ProcessEncoderAdapter):
    impl_cls = FakeLocalTextEncoderImpl

    def __init__(
        self,
        config: FakeLocalTextEncoderConfig,
        *,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            config,
            input_format="text",
            batch_size=batch_size,
        )
        return


@configure
class FakeLocalPairScorerConfig:
    device_id: list[int] = field(default_factory=list)
    batch_size: int = 32
    delay_s: float = 0.0
    error_on: str | None = None


class FakeLocalPairScorerImpl(LocalPairScorerBase):
    def __init__(self, config: FakeLocalPairScorerConfig) -> None:
        super().__init__(batch_size=config.batch_size)
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        return

    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
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


class FakeLocalPairScorer(ProcessScorerAdapter):
    impl_cls = FakeLocalPairScorerImpl

    def __init__(
        self,
        config: FakeLocalPairScorerConfig,
        *,
        batch_size: int = 32,
    ) -> None:
        super().__init__(config, batch_size=batch_size)
        return


@configure
class FakeLocalGeneratorConfig:
    device_id: list[int] = field(default_factory=list)
    batch_size: int = 1
    delay_s: float = 0.0
    error_on: str | None = None


class FakeLocalGeneratorImpl(LocalGeneratorBase):
    def __init__(self, config: FakeLocalGeneratorConfig) -> None:
        super().__init__(batch_size=config.batch_size)
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        return

    def _generate_batch(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.error_on is not None and any(
            self.error_on in prefix for prefix in prefixes
        ):
            raise ValueError(f"boom: {self.error_on}")

        sample_num = (
            generation_config.sample_num if generation_config is not None else 1
        )
        return [
            [f"{prefix} -> sample {i}" for i in range(sample_num)]
            for prefix in prefixes
        ]

    def _chat_batch(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        if not all(isinstance(message, ChatMessages) for message in messages):
            raise TypeError(
                "FakeLocalGeneratorImpl.chat expects normalized ChatMessages batches."
            )

        if self.delay_s > 0:
            time.sleep(self.delay_s)

        prompts = []
        for message in messages:
            prompt_text = " ".join(turn.text_content for turn in message)
            prompts.append(prompt_text)
        if self.error_on is not None and any(
            self.error_on in prompt for prompt in prompts
        ):
            raise ValueError(f"boom: {self.error_on}")

        sample_num = (
            generation_config.sample_num if generation_config is not None else 1
        )
        return [
            [
                ChatTurn(role="assistant", content=f"{prompt} -> reply {i}")
                for i in range(sample_num)
            ]
            for prompt in prompts
        ]


class FakeLocalGenerator(ProcessGeneratorAdapter):
    impl_cls = FakeLocalGeneratorImpl

    def __init__(
        self,
        config: FakeLocalGeneratorConfig,
        *,
        batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_size=batch_size)
        return
