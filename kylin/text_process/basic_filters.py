from dataclasses import dataclass, field
from typing import Optional

from .processor import PROCESSORS, Processor, TextUnit
from .utils import UnifiedTokenizer, UTokenizerConfig


@PROCESSORS("exact_deduplicate")
class ExactDeduplicate(Processor):
    def __init__(self) -> None:
        self.seen = set()
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        if input_text.content in self.seen:
            input_text.reserved = False
        self.seen.add(input_text.content)
        return input_text


@dataclass
class LengthFilterConfig:
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    max_chars: Optional[int] = None
    min_chars: Optional[int] = None
    max_bytes: Optional[int] = None
    min_bytes: Optional[int] = None
    tokenizer_config: UTokenizerConfig = field(default_factory=UTokenizerConfig)


@PROCESSORS("length_filter", config_class=LengthFilterConfig)
class LengthFilter(Processor):
    def __init__(self, cfg: LengthFilterConfig) -> None:
        super().__init__()
        self.max_tokens = cfg.max_tokens
        self.min_tokens = cfg.min_tokens
        self.max_chars = cfg.max_chars
        self.min_chars = cfg.min_chars
        self.max_bytes = cfg.max_bytes
        self.min_bytes = cfg.min_bytes
        if self.max_tokens is not None or self.min_tokens is not None:
            self.tokenizer = UnifiedTokenizer(cfg.tokenizer_config)
        else:
            self.tokenizer = None
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(input_text.content)
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            input_text.reserved = False
        if self.min_tokens is not None and len(tokens) < self.min_tokens:
            input_text.reserved = False
        if self.max_chars is not None and len(input_text.content) > self.max_chars:
            input_text.reserved = False
        if self.min_chars is not None and len(input_text.content) < self.min_chars:
            input_text.reserved = False
        if (
            self.max_bytes is not None
            and len(input_text.content.encode("utf-8")) > self.max_bytes
        ):
            input_text.reserved = False
        if (
            self.min_bytes is not None
            and len(input_text.content.encode("utf-8")) < self.min_bytes
        ):
            input_text.reserved = False
        return input_text
