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
        chunks = self._recursive_chunk(text, 0, (0, len(text)))
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks

    def _recursive_chunk(
        self, text: str, level: int, span: tuple[int, int]
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
                        end=current_index + len(chunk_text),
                        meta_data={"split_level": level},
                    )
                )
                current_index += len(chunk_text)
            return chunks
        else:
            sub_chunks = self.splitter[level].split(text)
            new_chunks = []

            # temporary storage for the current chunk
            current_sub_chunks = []
            current_tokens = 0

            for sub_chunk in sub_chunks:
                text_ = sub_chunk["text"]
                local_span = sub_chunk["char_span"]

                # fix span to global
                if span[0] != -1 and local_span[0] != -1:
                    global_span = (span[0] + local_span[0], span[0] + local_span[1])
                else:
                    global_span = (-1, -1)

                tokens_count = len(encode_fn(text_))

                if current_tokens + tokens_count <= self.chunk_size:
                    current_sub_chunks.append((text_, local_span, global_span))
                    current_tokens += tokens_count

                elif tokens_count <= self.chunk_size:
                    # Flush current
                    if current_sub_chunks:
                        # try to retrieve text from original text
                        if (
                            current_sub_chunks[0][1][0] != -1
                            and current_sub_chunks[-1][1][1] != -1
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
                                meta_data={"split_level": level},
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
                            current_sub_chunks[0][1][0] != -1
                            and current_sub_chunks[-1][1][1] != -1
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
                                meta_data={"split_level": level},
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
                    current_sub_chunks[0][1][0] != -1
                    and current_sub_chunks[-1][1][1] != -1
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
                        meta_data={"split_level": level},
                    )
                )
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
