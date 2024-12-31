from dataclasses import dataclass
from typing import Optional

from flexrag.utils import Choices


@dataclass
class UTokenizerConfig:
    tokenizer_type: Choices(["hf", "tiktoken", "moses"]) = "moses"  # type: ignore
    hf_tokenizer_path: Optional[str] = None
    tiktok_tokenizer_name: Optional[str] = None
    lang: Optional[str] = None


class UnifiedTokenizer:
    def __init__(self, cfg: UTokenizerConfig) -> None:
        self.tokenizer_type = cfg.tokenizer_type
        match self.tokenizer_type:
            case "hf":
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_tokenizer_path)
            case "tiktoken":
                import tiktoken

                self.tokenizer = tiktoken.get_encoding(cfg.tiktok_tokenizer_name)
            case "moses":
                from sacremoses import MosesDetokenizer, MosesTokenizer

                self.tokenizer = MosesTokenizer(cfg.lang)
                self.detokenizer = MosesDetokenizer(cfg.lang)
            case _:
                raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")
        return

    def tokenize(self, texts: str) -> list[str | int]:
        match self.tokenizer_type:
            case "hf":
                tokens = self.tokenizer.encode(texts)
            case "tiktoken":
                tokens = self.tokenizer.encode(texts)
            case "moses":
                tokens = self.tokenizer.tokenize(texts)
        return tokens

    def detokenize(self, tokens: list[str | int]) -> str:
        match self.tokenizer_type:
            case "hf":
                texts = self.tokenizer.decode(tokens)
            case "tiktoken":
                texts = self.tokenizer.decode(tokens)
            case "moses":
                texts = self.detokenizer.detokenize(tokens)
        return texts
