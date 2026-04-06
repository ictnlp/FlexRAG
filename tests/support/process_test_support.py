import time
from dataclasses import field

import numpy as np

from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models.encoders.encoder_base import (
    EncoderInput,
    extract_text_encoder_inputs,
)
from flexrag.models.encoders.local_process_encoder_base import LocalProcessEncoderBase
from flexrag.models.generators.generator_base import GenerationConfig
from flexrag.models.generators.local_process_generator_base import (
    LocalProcessGeneratorBase,
)
from flexrag.models.scorers.local_process_scorer_base import LocalProcessScorerBase


@configure
class FakeLocalTextEncoderConfig:
    device_id: list[int] = field(default_factory=list)
    delay_s: float = 0.0
    error_on: str | None = None
    embedding_dim: int = 3


class FakeLocalTextEncoderImpl:
    def __init__(self, config: FakeLocalTextEncoderConfig) -> None:
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        self._embedding_dim = config.embedding_dim
        return

    def encode(self, inputs: EncoderInput | list[EncoderInput]) -> np.ndarray:
        texts = extract_text_encoder_inputs(inputs, encoder_name="FakeLocalTextEncoder")
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


class FakeLocalPairScorerImpl:
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


@configure
class FakeLocalGeneratorConfig:
    device_id: list[int] = field(default_factory=list)
    delay_s: float = 0.0
    error_on: str | None = None


class FakeLocalGeneratorImpl:
    def __init__(self, config: FakeLocalGeneratorConfig) -> None:
        self.delay_s = config.delay_s
        self.error_on = config.error_on
        return

    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
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

    def chat(
        self,
        messages: list[ChatMessages] | ChatMessages,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        if isinstance(messages, ChatMessages):
            messages = [messages]
        elif not all(isinstance(message, ChatMessages) for message in messages):
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


class FakeLocalGenerator(LocalProcessGeneratorBase):
    impl_cls = FakeLocalGeneratorImpl
