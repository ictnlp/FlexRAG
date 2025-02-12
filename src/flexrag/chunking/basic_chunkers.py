from dataclasses import dataclass, field
from typing import Optional

from flexrag.utils import LOGGER_MANAGER
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig

from .chunker_base import ChunkerBase, CHUNKERS
from .sentence_splitter import (
    SENTENCE_SPLITTERS,
    SentenceSplitterConfig,
    RegexSplitterConfig,
    RegexSplitter,
    PREDEFINED_SPLIT_PATTERNS,
)


logger = LOGGER_MANAGER.get_logger("flexrag.chunking.basic_chunkers")


@dataclass
class CharChunkerConfig:
    """Configuration for CharChunker.

    :param max_chars: The number of characters in each chunk. Default is 2048.
    :type max_chars: int
    :param overlap: The number of characters to overlap between chunks. Default is 0.
    :type overlap: int

    For example, to chunk a text into chunks with 1024 characters with 128 characters overlap:

    .. code-block:: python

        from flexrag.chunking import CharChunkerConfig, CharChunker

        cfg = CharChunkerConfig(max_chars=1024, overlap=128)
        chunker = CharChunker(cfg)
    """

    max_chars: int = 2048
    overlap: int = 0


@CHUNKERS("char", config_class=CharChunkerConfig)
class CharChunker(ChunkerBase):
    """CharChunker splits text into chunks with fixed length of characters."""

    def __init__(self, cfg: CharChunkerConfig) -> None:
        self.chunk_size = cfg.max_chars
        self.overlap = cfg.overlap
        return

    def chunk(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks


@dataclass
class TokenChunkerConfig(TokenizerConfig):
    """Configuration for TokenChunker.

    :param max_tokens: The number of tokens in each chunk. Default is 512.
    :type max_tokens: int
    :param overlap: The number of tokens to overlap between chunks. Default is 0.
    :type overlap: int

    For example, to chunk a text into chunks with 256 tokens with 128 tokens overlap:

    .. code-block:: python

        from flexrag.chunking import TokenChunkerConfig, TokenChunker
        from flexrag.models.tokenizer import TikTokenTokenizerConfig

        cfg = TokenChunkerConfig(
            max_tokens=256,
            overlap=128,
            tokenizer_type="tiktoken",
            tiktoken_config=TikTokenTokenizerConfig(model_name="gpt-4o"),
        )
        chunker = TokenChunker(cfg)

    Note that the ``TokenChunker`` relies on the ``tokenize`` and ``detokenize`` methods of the tokenizer to split the text.
    Thus the space between may be lost if the tokenizer is not reversible.
    """

    max_tokens: int = 512
    overlap: int = 0


@CHUNKERS("token", config_class=TokenChunkerConfig)
class TokenChunker(ChunkerBase):
    """TokenChunker splits text into chunks with fixed number of tokens."""

    def __init__(self, cfg: TokenChunkerConfig) -> None:
        self.chunk_size = cfg.max_tokens
        self.overlap = cfg.overlap
        self.tokenizer = TOKENIZERS.load(cfg)
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {cfg.tokenizer_type} is not reversible. "
                "Some characters may be lost during detokenization."
            )
        return

    def chunk(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunks.append(self.tokenizer.detokenize(tokens[i : i + self.chunk_size]))
        return chunks


@dataclass
class RecursiveChunkerConfig(TokenizerConfig):
    """Configuration for RecursiveChunker.

    :param max_tokens: The maximum number of tokens in each chunk. Default is 512.
    :type max_tokens: int
    :param seperators: The seperators used to split text recursively.
        The order of the seperators matters. Default is ``PREDEFINED_SPLIT_PATTERNS["en"]``.
    :type seperators: dict[str, str]

    For example, to split a text recursively with 256 tokens in each chunk:

    .. code-block:: python

        from flexrag.chunking import RecursiveChunkerConfig, RecursiveChunker

        cfg = RecursiveChunkerConfig(max_tokens=256)
        chunker = RecursiveChunker(cfg)

    You can also specify your own seperator list:

    .. code-block:: python

        from flexrag.chunking import RecursiveChunkerConfig, RecursiveChunker

        cfg = RecursiveChunkerConfig(
            max_tokens=256,
            split_pattern={"level1": "pattern1", "level2": "pattern2"},
        )
        chunker = RecursiveChunker(cfg)

    Note that the ``RecursiveChunker`` relies on the regex pattern to split the text,
    thus you need to make sure your pattern will not consume the splitter.
    A good practice is to use the lookbehind and lookahead assertion to avoid consuming the splitter.
    """

    max_tokens: int = 512
    split_pattern: dict[str, str] = field(
        default_factory=lambda: PREDEFINED_SPLIT_PATTERNS["en"]
    )


@CHUNKERS("recursive", config_class=RecursiveChunkerConfig)
class RecursiveChunker(ChunkerBase):
    """RecursiveChunker splits text into chunks recursively using the specified seperators.

    The order of the seperators matters. The text will be split recursively based on the seperators in the order of the list.
    The default seperators are defined in ``PREDEFINED_SPLIT_PATTERNS``.

    If the text is still too long after splitting with the last level seperators, the text will be split into tokens.
    """

    def __init__(self, cfg: RecursiveChunkerConfig) -> None:
        self.splitter = [
            RegexSplitter(RegexSplitterConfig(pattern=p))
            for p in cfg.split_pattern.values()
        ]
        self.chunk_size = cfg.max_tokens
        self.tokenizer = TOKENIZERS.load(cfg)
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {cfg.tokenizer_type} is not reversible. "
                "Some characters may be lost during detokenization."
            )
        return

    def chunk(self, text: str) -> list[str]:
        return self._recursive_chunk(text, 0)

    def _recursive_chunk(self, text: str, level: int) -> list[str]:
        if level == len(self.splitter):
            tokens = self.tokenizer.tokenize(text)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunks.append(
                    self.tokenizer.detokenize(tokens[i : i + self.chunk_size])
                )
            return chunks
        else:
            chunks = self.splitter[level].split(text)
            new_chunks = []
            chunk = ""
            for chunk_ in chunks:
                token_count_ = len(self.tokenizer.tokenize(chunk_))
                merged_count = len(self.tokenizer.tokenize(chunk + chunk_))
                if merged_count <= self.chunk_size:
                    chunk += chunk_
                elif token_count_ <= self.chunk_size:
                    if chunk:
                        new_chunks.append(chunk)
                    chunk = chunk_
                else:
                    if chunk:
                        new_chunks.append(chunk)
                    new_chunks.extend(self._recursive_chunk(chunk_, level + 1))
                    chunk = ""
            if chunk:
                new_chunks.append(chunk)
            return new_chunks


@dataclass
class SentenceChunkerConfig(TokenizerConfig, SentenceSplitterConfig):
    """Configuration for SentenceChunker.

    :param max_sents: The maximum number of sentences in each chunk. Default is None.
    :type max_sents: Optional[int]
    :param max_tokens: The maximum number of tokens in each chunk. Default is None.
    :type max_tokens: Optional[int]
    :param max_chars: The maximum number of characters in each chunk. Default is None.
    :type max_chars: Optional[int]
    :param overlap: The number of sentences to overlap between chunks. Default is 0.
    :type overlap: int

    For example, to chunk a text into chunks with 10 sentences in each chunk:

    .. code-block:: python

        from flexrag.chunking import SentenceChunkerConfig, SentenceChunker

        cfg = SentenceChunkerConfig(max_sents=10)
        chunker = SentenceChunker(cfg)

    Note that the ``SentenceChunker`` relies on the sentence splitter to split the text,
    thus the space between may be lost if the sentence splitter is not reversible.
    """

    max_sents: Optional[int] = None
    max_tokens: Optional[int] = None
    max_chars: Optional[int] = None
    overlap: int = 0


@CHUNKERS("sentence", config_class=SentenceChunkerConfig)
class SentenceChunker(ChunkerBase):
    """SentenceChunker first splits text into sentences using the specified sentence splitter,
    then merges the sentences into chunks based on the specified constraints.
    """

    def __init__(self, cfg: SentenceChunkerConfig) -> None:
        # set arguments
        assert not all(
            i is None for i in [cfg.max_sents, cfg.max_tokens, cfg.max_chars]
        ), "At least one of max_sentences, max_tokens, max_chars should be set."
        self.max_sents = cfg.max_sents if cfg.max_sents else float("inf")
        self.max_tokens = cfg.max_tokens if cfg.max_tokens else float("inf")
        self.max_chars = cfg.max_chars if cfg.max_chars else float("inf")
        self.overlap = cfg.overlap
        self.tokenizer = TOKENIZERS.load(cfg)
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {cfg.tokenizer_type} is not reversible. "
                "Some characters may be lost during detokenization."
            )

        # load splitter
        self.splitter = SENTENCE_SPLITTERS.load(cfg)

        self.long_sentence_counter = 0
        return

    def chunk(self, text: str) -> list[str]:
        sentences = self.splitter.split(text)
        if self.max_tokens != float("inf"):
            token_counts = [len(self.tokenizer.tokenize(s)) for s in sentences]
        else:
            token_counts = [0] * len(sentences)
        char_counts = [len(s) for s in sentences]

        chunks = []
        start_pointer = 0
        end_pointer = 0
        while end_pointer < len(sentences):
            while end_pointer < len(sentences) and (
                ((end_pointer - start_pointer) < self.max_sents)
                and (
                    sum(token_counts[start_pointer : end_pointer + 1])
                    <= self.max_tokens
                )
                and (
                    sum(char_counts[start_pointer : end_pointer + 1]) <= self.max_chars
                )
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
            chunks.append("".join(sentences[start_pointer:end_pointer]))
            start_pointer = max(end_pointer - self.overlap, start_pointer + 1)
            end_pointer = start_pointer
        return chunks
