from dataclasses import field
from typing import Optional

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig

from .chunker_base import CHUNKERS, Chunk, ChunkerBase
from .sentence_splitter import (
    PREDEFINED_SPLIT_PATTERNS,
    SENTENCE_SPLITTERS,
    RegexSplitter,
    RegexSplitterConfig,
    SentenceSplitterConfig,
)

logger = LOGGER_MANAGER.get_logger("flexrag.chunking.basic_chunkers")


@configure
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


@CHUNKERS("char_chunker", config_class=CharChunkerConfig)
class CharChunker(ChunkerBase):
    """CharChunker splits text into chunks with fixed length of characters."""

    def __init__(self, cfg: CharChunkerConfig) -> None:
        self.chunk_size = cfg.max_chars
        self.overlap = cfg.overlap
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(
                Chunk(
                    text=text[i : i + self.chunk_size],
                    start=1,
                    end=min(len(text), i + self.chunk_size),
                )
            )
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks


@configure
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


@CHUNKERS("token_chunker", config_class=TokenChunkerConfig)
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

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        tokens = self.tokenizer.tokenize(text)
        chunks: list[Chunk] = []
        current_index = 0
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_text = self.tokenizer.detokenize(tokens[i : i + self.chunk_size])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=current_index,
                    end=current_index + len(chunk_text),
                )
            )
            overlap_text = self.tokenizer.detokenize(
                tokens[i + self.chunk_size - self.overlap : i + self.chunk_size]
            )
            current_index += len(chunk_text) - len(overlap_text)
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks


@configure
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


@CHUNKERS("recursive_chunker", config_class=RecursiveChunkerConfig)
class RecursiveChunker(ChunkerBase):
    """RecursiveChunker splits text into chunks recursively using the specified seperators.

    The order of the seperators matters.
    The text will be split recursively based on the seperators in the order of the list.
    The default seperators are defined in ``PREDEFINED_SPLIT_PATTERNS``.

    If the text is still too long after splitting with the last level seperators,
    the text will be split into tokens.
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

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        texts = self._recursive_chunk(text, 0)
        chunks = []
        current_index = 0
        for text in texts:
            chunks.append(
                Chunk(
                    text=text,
                    start=current_index,
                    end=current_index + len(text),
                )
            )
            current_index += len(text)
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks

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
            chunks = [s["text"] for s in self.splitter[level].split(text)]
            new_chunks = []
            chunk = ""
            chunk_token_count = 0
            for chunk_ in chunks:
                token_count_ = len(self.tokenizer.tokenize(chunk_))
                if chunk_token_count + token_count_ <= self.chunk_size:
                    chunk += chunk_
                    chunk_token_count += token_count_
                elif token_count_ <= self.chunk_size:
                    if chunk:
                        new_chunks.append(chunk)
                    chunk = chunk_
                    chunk_token_count = token_count_
                else:
                    if chunk:
                        new_chunks.append(chunk)
                    new_chunks.extend(self._recursive_chunk(chunk_, level + 1))
                    chunk = ""
                    chunk_token_count = 0
            if chunk:
                new_chunks.append(chunk)
            return new_chunks


@configure
class SentenceChunkerConfig(TokenizerConfig, SentenceSplitterConfig):
    """Configuration for SentenceChunker.

    :param max_sents: The maximum number of sentences in each chunk. Default is None.
    :type max_sents: Optional[int]
    :param max_tokens: The maximum number of tokens in each chunk. Default is None.
    :type max_tokens: Optional[int]
    :param overlap: The number of sentences to overlap between chunks. Default is 0.
    :type overlap: int

    For example, to chunk a text into chunks with 10 sentences in each chunk:

    .. code-block:: python

        from flexrag.chunking import SentenceChunkerConfig, SentenceChunker

        cfg = SentenceChunkerConfig(max_sents=10)
        chunker = SentenceChunker(cfg)

    Note that sentences longer than ``max_tokens`` will be further split into smaller
    chunks.
    """

    max_sents: Optional[int] = None
    max_tokens: Optional[int] = None
    overlap: int = 0


@CHUNKERS("sentence_chunker", config_class=SentenceChunkerConfig)
class SentenceChunker(ChunkerBase):
    """SentenceChunker first splits text into sentences using the specified sentence
    splitter, then merges the sentences into chunks based on the specified constraints.
    """

    def __init__(self, cfg: SentenceChunkerConfig) -> None:
        # set arguments
        assert not all(
            i is None for i in [cfg.max_sents, cfg.max_tokens]
        ), "At least one of max_sentences, max_tokens should be set."
        self.max_sents = cfg.max_sents if cfg.max_sents else float("inf")
        self.max_tokens = cfg.max_tokens if cfg.max_tokens else float("inf")
        self.overlap = cfg.overlap
        self.tokenizer = TOKENIZERS.load(cfg)
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {cfg.tokenizer_type} is not reversible. "
                "Some characters may be lost during detokenization."
            )

        # load splitter
        self.splitter = SENTENCE_SPLITTERS.load(cfg)
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        # split document into sentences
        sentences_ = self.splitter.split(text)

        # make sure all sentences lengths are less than max_tokens
        sentences = []
        for sent in sentences_:
            if (self.max_tokens == float("inf")) or (
                len(self.tokenizer.tokenize(sent["text"])) <= self.max_tokens
            ):
                sentences.append(sent)
                continue

            # split the long sentence
            char_start, _ = sent["char_span"]
            tokens = self.tokenizer.tokenize(sent["text"])
            curr_pos = 0
            for i in range(0, len(tokens), self.max_tokens):
                sub_text = self.tokenizer.detokenize(tokens[i : i + self.max_tokens])
                span = (
                    (char_start + curr_pos, char_start + curr_pos + len(sub_text))
                    if char_start != -1
                    else (-1, -1)
                )
                sentences.append({"text": sub_text, "char_span": span})
                curr_pos += len(sub_text)

        if self.max_tokens != float("inf"):
            token_counts = [len(self.tokenizer.tokenize(s["text"])) for s in sentences]
        else:
            token_counts = [0] * len(sentences)

        # merge sentences into chunks
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
            ):
                end_pointer += 1

            if end_pointer == start_pointer:
                end_pointer += 1
            try:
                char_start = sentences[start_pointer]["char_span"][0]
                char_end = sentences[end_pointer - 1]["char_span"][1]
                assert char_start != -1 and char_end != -1
                chunk_text = text[char_start:char_end]
            except AssertionError:
                char_start, char_end = None, None
                chunk_text = " ".join(
                    s["text"] for s in sentences[start_pointer:end_pointer]
                )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=sentences[start_pointer]["char_span"][0],
                    end=sentences[end_pointer - 1]["char_span"][1],
                )
            )
            new_start = max(end_pointer - self.overlap, start_pointer + 1)
            start_pointer = new_start
            end_pointer = start_pointer
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks
