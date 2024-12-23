from dataclasses import dataclass
from functools import partial
from typing import Optional

from librarian.utils import LOGGER_MANAGER

from .chunker_base import ChunkerBase, CHUNKERS
from ..text_process.utils import UnifiedTokenizer, UTokenizerConfig


logger = LOGGER_MANAGER.get_logger("librarian.data.chunking.basic_chunkers")


@dataclass
class CharChunkerConfig:
    chunk_size: int = 2048
    overlap: int = 0


@CHUNKERS("char", config_class=CharChunkerConfig)
class CharChunker(ChunkerBase):
    def __init__(self, cfg: CharChunkerConfig) -> None:
        self.chunk_size = cfg.chunk_size
        self.overlap = cfg.overlap
        return

    def chunk(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks


@dataclass
class TokenChunkerConfig(UTokenizerConfig):
    chunk_size: int = 512
    overlap: int = 0


@CHUNKERS("token", config_class=TokenChunkerConfig)
class TokenChunker(ChunkerBase):
    def __init__(self, cfg: TokenChunkerConfig) -> None:
        self.chunk_size = cfg.chunk_size
        self.overlap = cfg.overlap
        self.tokenizer = UnifiedTokenizer(cfg)
        return

    def chunk(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunks.append(self.tokenizer.detokenize(tokens[i : i + self.chunk_size]))
        return chunks


@dataclass
class SentenceChunkerConfig(UTokenizerConfig):
    max_sentences: Optional[int] = None
    max_tokens: Optional[int] = None
    max_chars: Optional[int] = None
    overlap: int = 0
    language: str = "english"


@CHUNKERS("sentence", config_class=SentenceChunkerConfig)
class SentenceChunker(ChunkerBase):
    def __init__(self, cfg: SentenceChunkerConfig) -> None:
        # set arguments
        self.max_sents = cfg.max_sentences
        self.max_tokens = cfg.max_tokens
        self.max_chars = cfg.max_chars
        assert not all(
            i is None for i in [self.max_sents, self.max_tokens, self.max_chars]
        ), "At least one of max_sentences, max_tokens, max_chars should be set."
        self.overlap = cfg.overlap

        # prepare spliter
        self.tokenizer = UnifiedTokenizer(cfg)
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError("nltk is required for SentenceChunker")
        self.sent_tokenize = partial(sent_tokenize, language=cfg.language)

        self.long_sentence_counter = 0
        return

    def chunk(self, text: str) -> list[str]:
        max_sents = self.max_sents if self.max_sents is not None else float("inf")
        max_tokens = self.max_tokens if self.max_tokens is not None else float("inf")
        max_chars = self.max_chars if self.max_chars is not None else float("inf")
        sentences = self.sent_tokenize(text)
        token_counts = [
            len(self.tokenizer.tokenize(s)) if self.max_tokens is not None else 0
            for s in sentences
        ]
        char_counts = [len(s) if self.max_chars is not None else 0 for s in sentences]

        chunks = []
        start_pointer = 0
        end_pointer = 0
        while end_pointer < len(sentences):
            while end_pointer < len(sentences) and (
                ((end_pointer - start_pointer) < max_sents)
                and (sum(token_counts[start_pointer : end_pointer + 1]) <= max_tokens)
                and (sum(char_counts[start_pointer : end_pointer + 1]) <= max_chars)
            ):
                end_pointer += 1

            if end_pointer == start_pointer:
                end_pointer += 1
                self.long_sentence_counter += 1
                if self.long_sentence_counter == 100:
                    logger.warning(
                        "There are 100 sentences have more than `max_tokens` tokens or `max_chars` characters. "
                        "Please check the configuration of SentenceChunker."
                    )
            chunks.append(" ".join(sentences[start_pointer:end_pointer]))
            start_pointer = max(end_pointer - self.overlap, start_pointer + 1)
            end_pointer = start_pointer
        return chunks
