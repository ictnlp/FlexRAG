from dataclasses import field
from typing import Optional

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models.tokenizer import TokenizerProtocol

from .chunker_base import CHUNKERS, Chunk, ChunkerProtocol
from .sentence_splitter import (
    PREDEFINED_SPLIT_PATTERNS,
    RegexSplitter,
    RegexSplitterConfig,
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
class CharChunker:
    """CharChunker splits text into chunks with fixed length of characters."""

    def __init__(self, cfg: CharChunkerConfig) -> None:
        if cfg.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0.")
        if cfg.overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0.")
        if cfg.overlap >= cfg.max_chars:
            raise ValueError("overlap must be smaller than max_chars.")
        self.chunk_size = cfg.max_chars
        self.overlap = cfg.overlap
        return

    def chunk(self, text: str) -> list[Chunk]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(
                Chunk(
                    text=text[i : i + self.chunk_size],
                    start=i,
                    end=min(len(text), i + self.chunk_size),
                )
            )
        return chunks


@configure
class TokenChunkerConfig:
    """Configuration for TokenChunker.

    :param max_tokens: The number of tokens in each chunk. Default is 512.
    :param overlap: The number of tokens to overlap between chunks. Default is 0.

    For example, to chunk a text into chunks with 256 tokens with 128 tokens overlap:

    .. code-block:: python

        from flexrag.chunking import TokenChunkerConfig, TokenChunker
        from flexrag.models.tokenizer import SpaceTokenizer

        tokenizer = SpaceTokenizer()
        cfg = TokenChunkerConfig(max_tokens=256, overlap=128)
        chunker = TokenChunker(cfg, tokenizer=tokenizer)

    Note that the ``TokenChunker`` relies on the ``tokenize`` and ``detokenize`` methods of the tokenizer to split the text.
    Thus the space between may be lost if the tokenizer is not reversible.
    """

    max_tokens: int = 512
    overlap: int = 0


@CHUNKERS("token_chunker", config_class=TokenChunkerConfig)
class TokenChunker:
    """TokenChunker splits text into chunks with fixed number of tokens."""

    def __init__(self, cfg: TokenChunkerConfig, tokenizer: TokenizerProtocol) -> None:
        self.chunk_size = cfg.max_tokens
        self.overlap = cfg.overlap
        self.tokenizer = tokenizer
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {type(tokenizer).__name__} is not reversible. "
                "Some characters may be lost during detokenization."
            )
        return

    def chunk(self, text: str) -> list[Chunk]:
        # employ encode and decode for faster processing
        if self.tokenizer.vocab_size > 0:
            encode_fn = self.tokenizer.encode
            decode_fn = self.tokenizer.decode
        else:
            encode_fn = self.tokenizer.tokenize
            decode_fn = self.tokenizer.detokenize
        tokens = encode_fn(text)
        chunks: list[Chunk] = []
        current_index = 0
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_text = decode_fn(tokens[i : i + self.chunk_size])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=current_index,
                    end=current_index + len(chunk_text),
                )
            )
            overlap_text = decode_fn(
                tokens[i + self.chunk_size - self.overlap : i + self.chunk_size]
            )
            current_index += len(chunk_text) - len(overlap_text)
        return chunks


@configure
class RecursiveChunkerConfig:
    """Configuration for RecursiveChunker.

    :param max_tokens: The maximum number of tokens in each chunk. Default is 512.
    :param separators: The separators used to split text recursively.
        The order of the separators matters. Default is ``PREDEFINED_SPLIT_PATTERNS["en"]``.

    For example, to split a text recursively with 256 tokens in each chunk:

    .. code-block:: python

        from flexrag.chunking import RecursiveChunkerConfig, RecursiveChunker
        from flexrag.models.tokenizer import SpaceTokenizer

        tokenizer = SpaceTokenizer()
        cfg = RecursiveChunkerConfig(max_tokens=256)
        chunker = RecursiveChunker(cfg, tokenizer=tokenizer)

    You can also specify your own seperator list:

    .. code-block:: python

        from flexrag.chunking import RecursiveChunkerConfig, RecursiveChunker
        from flexrag.models.tokenizer import SpaceTokenizer

        tokenizer = SpaceTokenizer()
        cfg = RecursiveChunkerConfig(
            max_tokens=256,
            split_pattern={"level1": "pattern1", "level2": "pattern2"},
        )
        chunker = RecursiveChunker(cfg, tokenizer=tokenizer)

    """

    max_tokens: int = 512
    split_pattern: dict[str, str] = field(
        default_factory=lambda: PREDEFINED_SPLIT_PATTERNS["en"]
    )


@CHUNKERS("recursive_chunker", config_class=RecursiveChunkerConfig)
class RecursiveChunker:
    """RecursiveChunker splits text into chunks recursively using the specified separators.

    The order of the separators matters.
    The text will be split recursively based on the separators in the order of the list.
    The default separators are defined in ``PREDEFINED_SPLIT_PATTERNS``.

    If the text is still too long after splitting with the last level separators,
    the text will be split into tokens.
    """

    def __init__(
        self, cfg: RecursiveChunkerConfig, tokenizer: TokenizerProtocol
    ) -> None:
        self.splitter = [
            RegexSplitter(RegexSplitterConfig(pattern=p))
            for p in cfg.split_pattern.values()
        ]
        self.chunk_size = cfg.max_tokens
        self.tokenizer = tokenizer
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {type(tokenizer).__name__} is not reversible. "
                "Some characters may be lost during detokenization."
            )
        return

    def chunk(self, text: str) -> list[Chunk]:
        return self._recursive_chunk(text, 0, (0, len(text)))

    def _recursive_chunk(
        self,
        text: str,
        level: int,
        span: tuple[Optional[int], Optional[int]],
    ) -> list[Chunk]:
        if self.tokenizer.vocab_size > 0:
            encode_fn = self.tokenizer.encode
            decode_fn = self.tokenizer.decode
        else:
            encode_fn = self.tokenizer.tokenize
            decode_fn = self.tokenizer.detokenize
        if level == len(self.splitter):
            tokens = encode_fn(text)
            chunks = []
            current_index = span[0]
            # Warning: token chunking loses exact character span information
            for i in range(0, len(tokens), self.chunk_size):
                chunk_text = decode_fn(tokens[i : i + self.chunk_size])
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start=current_index,
                        end=(
                            current_index + len(chunk_text)
                            if current_index is not None
                            else None
                        ),
                        metadata={"split_level": level},
                    )
                )
                if current_index is not None:
                    current_index += len(chunk_text)
            return chunks
        else:
            sub_chunks = self.splitter[level].chunk(text)
            new_chunks = []

            # temporary storage for the current chunk
            current_sub_chunks = []
            current_tokens = 0

            for sub_chunk in sub_chunks:
                text_ = sub_chunk.text
                local_span = (sub_chunk.start, sub_chunk.end)

                # fix span to global
                if (
                    span[0] is not None
                    and local_span[0] is not None
                    and local_span[1] is not None
                ):
                    global_span = (span[0] + local_span[0], span[0] + local_span[1])
                else:
                    global_span = (None, None)

                tokens_count = len(encode_fn(text_))

                if current_tokens + tokens_count <= self.chunk_size:
                    current_sub_chunks.append((text_, local_span, global_span))
                    current_tokens += tokens_count

                elif tokens_count <= self.chunk_size:
                    # Flush current
                    if current_sub_chunks:
                        # try to retrieve text from original text
                        if (
                            current_sub_chunks[0][1][0] is not None
                            and current_sub_chunks[-1][1][1] is not None
                        ):
                            chunk_text = text[
                                current_sub_chunks[0][1][0] : current_sub_chunks[-1][1][
                                    1
                                ]
                            ]
                        else:
                            chunk_text = "".join([c[0] for c in current_sub_chunks])

                        new_chunks.append(
                            Chunk(
                                text=chunk_text,
                                start=current_sub_chunks[0][2][0],
                                end=current_sub_chunks[-1][2][1],
                                metadata={"split_level": level},
                            )
                        )
                        current_sub_chunks = []
                        current_tokens = 0

                    # Add new
                    current_sub_chunks.append((text_, local_span, global_span))
                    current_tokens = tokens_count

                else:
                    # Flush current
                    if current_sub_chunks:
                        if (
                            current_sub_chunks[0][1][0] is not None
                            and current_sub_chunks[-1][1][1] is not None
                        ):
                            chunk_text = text[
                                current_sub_chunks[0][1][0] : current_sub_chunks[-1][1][
                                    1
                                ]
                            ]
                        else:
                            chunk_text = "".join([c[0] for c in current_sub_chunks])

                        new_chunks.append(
                            Chunk(
                                text=chunk_text,
                                start=current_sub_chunks[0][2][0],
                                end=current_sub_chunks[-1][2][1],
                                metadata={"split_level": level},
                            )
                        )
                        current_sub_chunks = []
                        current_tokens = 0

                    # Recurse
                    new_chunks.extend(
                        self._recursive_chunk(text_, level + 1, global_span)
                    )

            # Final flush
            if current_sub_chunks:
                if (
                    current_sub_chunks[0][1][0] is not None
                    and current_sub_chunks[-1][1][1] is not None
                ):
                    chunk_text = text[
                        current_sub_chunks[0][1][0] : current_sub_chunks[-1][1][1]
                    ]
                else:
                    chunk_text = "".join([c[0] for c in current_sub_chunks])

                new_chunks.append(
                    Chunk(
                        text=chunk_text,
                        start=current_sub_chunks[0][2][0],
                        end=current_sub_chunks[-1][2][1],
                        metadata={"split_level": level},
                    )
                )
            return new_chunks


@configure
class SentenceChunkerConfig:
    """Configuration for SentenceChunker.

    :param max_sents: The maximum number of sentences in each chunk. Default is None.
    :param max_tokens: The maximum number of tokens in each chunk. Default is None.
    :param overlap: The number of sentences to overlap between chunks. Default is 0.

    For example, to chunk a text into chunks with 10 sentences in each chunk:

    .. code-block:: python

        from flexrag.processors.chunkers import (
            RegexSplitter,
            RegexSplitterConfig,
            SentenceChunker,
            SentenceChunkerConfig,
        )
        from flexrag.models.tokenizer import SpaceTokenizer

        tokenizer = SpaceTokenizer()
        cfg = SentenceChunkerConfig(max_sents=10)
        splitter = RegexSplitter(RegexSplitterConfig())
        chunker = SentenceChunker(
            cfg,
            tokenizer=tokenizer,
            splitter=splitter,
        )

    Note that sentences longer than ``max_tokens`` will be further split into smaller
    chunks.
    """

    max_sents: Optional[int] = None
    max_tokens: Optional[int] = None
    overlap: int = 0


@CHUNKERS("sentence_chunker", config_class=SentenceChunkerConfig)
class SentenceChunker:
    """SentenceChunker first splits text into sentences using the specified sentence
    splitter, then merges the sentences into chunks based on the specified constraints.
    """

    def __init__(
        self,
        cfg: SentenceChunkerConfig,
        *,
        tokenizer: TokenizerProtocol,
        splitter: ChunkerProtocol,
    ) -> None:
        # set arguments
        assert not all(i is None for i in [cfg.max_sents, cfg.max_tokens]), (
            "At least one of max_sentences, max_tokens should be set."
        )
        self.max_sents = cfg.max_sents if cfg.max_sents else float("inf")
        self.max_tokens = cfg.max_tokens if cfg.max_tokens else float("inf")
        self.overlap = cfg.overlap
        self.tokenizer = tokenizer
        self.splitter = splitter
        if not self.tokenizer.reversible:
            logger.warning(
                f"Tokenizer {type(tokenizer).__name__} is not reversible. "
                "Some characters may be lost during detokenization."
            )
        return

    def chunk(self, text: str) -> list[Chunk]:
        # split document into sentences
        sentences_ = self.splitter.chunk(text)

        # make sure all sentences lengths are less than max_tokens
        sentences = []
        for sent in sentences_:
            if (self.max_tokens == float("inf")) or (
                len(self.tokenizer.tokenize(sent.text)) <= self.max_tokens
            ):
                sentences.append(sent)
                continue

            # split the long sentence
            tokens = self.tokenizer.tokenize(sent.text)
            curr_pos = 0
            for i in range(0, len(tokens), self.max_tokens):
                sub_text = self.tokenizer.detokenize(tokens[i : i + self.max_tokens])
                start = sent.start + curr_pos if sent.start is not None else None
                end = start + len(sub_text) if start is not None else None
                sentences.append(Chunk(text=sub_text, start=start, end=end))
                curr_pos += len(sub_text)

        if self.max_tokens != float("inf"):
            token_counts = [len(self.tokenizer.tokenize(s.text)) for s in sentences]
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
            char_start = sentences[start_pointer].start
            char_end = sentences[end_pointer - 1].end
            if char_start is not None and char_end is not None:
                chunk_text = text[char_start:char_end]
            else:
                char_start, char_end = None, None
                chunk_text = " ".join(
                    s.text for s in sentences[start_pointer:end_pointer]
                )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=char_start,
                    end=char_end,
                )
            )
            new_start = max(end_pointer - self.overlap, start_pointer + 1)
            start_pointer = new_start
            end_pointer = start_pointer
        return chunks
