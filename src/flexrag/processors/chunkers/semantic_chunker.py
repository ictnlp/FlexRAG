from dataclasses import field
from typing import Annotated, Optional

import numpy as np

from flexrag.common import LOGGER_MANAGER, Choices, configure
from flexrag.models.encoders import EncoderProtocol
from flexrag.models.tokenizer import TokenizerProtocol

from .basic_chunkers import SentenceChunker, SentenceChunkerConfig
from .chunker_base import CHUNKERS, Chunk, ChunkerBase

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.semantic_chunker")


@configure
class SemanticChunkerConfig:
    """Configuration for SemanticChunker.

    :param max_tokens: The maximum number of tokens in each chunk. Default is None.
    :param threshold: The threshold for semantic similarity. Default is None.
        If provided, the `threshold_percentile` and `max_tokens` will be ignored.
    :param threshold_percentile: The ratio of the threshold for semantic similarity. Default is None.
        Should be a value between 0 and 100. Higher values will result in more chunks. 5 is a good starting point.
        If provided, the `max_tokens` will be ignored.
    :param similarity_window: The window size for calculating semantic similarity. Default is None.
    :param similarity_function: The similarity function to use. Default is "COS".
        Available choices are "L2" for the reciprocal of euclidean distance, "IP" for inner product, and "COS" for cosine similarity.
    :param pre_chunk_config: The configuration for the pre-chunker used to split the text into sentences.
        Default is SentenceChunkerConfig with max_sents=1, max_tokens=128, and overlap=0.

    The similarity higher than the threshold will be considered as coherent, and the chunks will be split at the points where the similarity is below the threshold.
    Thus, at least one of `max_tokens`, `threshold`, or `threshold_percentile` should be provided.
    If `threshold` is provided, the chunks will be split directly based on the threshold.
    If `threshold_percentile` is provided, the threshold will be calculated automatically based on the similarity distribution.
    If `max_tokens` is provided, the threshold will be calculated to ensure the chunks are within the token limit.

    For example, to split the text into chunks with a maximum of 512 tokens, you can use the following configuration:

        >>> from flexrag.chunking import SemanticChunker, SemanticChunkerConfig
        >>> from flexrag.models import HFEncoder, HFEncoderConfig
        >>> from flexrag.models.tokenizer import SpaceTokenizer
        >>> tokenizer = SpaceTokenizer()
        >>> encoder = HFEncoder(HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"))
        >>> config = SemanticChunkerConfig(
        ...     max_tokens=512,
        ... )
        >>> chunker = SemanticChunker(config, encoder=encoder, tokenizer=tokenizer)

    To split the text into chunks with a threshold_percentile of 5%, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold_percentile=5,
        ... )
        >>> chunker = SemanticChunker(config, encoder=encoder, tokenizer=tokenizer)

    To split the text into chunks with a given threshold, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold=0.8,
        ... )
        >>> chunker = SemanticChunker(config, encoder=encoder, tokenizer=tokenizer)
    """

    max_tokens: Optional[int] = None
    threshold: Optional[float] = None
    threshold_percentile: Optional[float] = None
    similarity_window: int = 1
    similarity_function: Annotated[str, Choices("L2", "IP", "COS")] = "COS"
    pre_chunk_config: SentenceChunkerConfig = field(
        default_factory=lambda: SentenceChunkerConfig(
            max_sents=1, max_tokens=128, overlap=0
        )
    )


@CHUNKERS("semantic_chunker", config_class=SemanticChunkerConfig)
class SemanticChunker(ChunkerBase):
    """SemanticChunker splits text into sentences and then groups them into chunks based on semantic similarity.
    This chunker is inspired by the Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """

    def __init__(
        self,
        cfg: SemanticChunkerConfig,
        encoder: EncoderProtocol,
        tokenizer: TokenizerProtocol,
    ) -> None:
        # set the basic configurations
        self.max_tokens = cfg.max_tokens if cfg.max_tokens is not None else float("inf")
        self.threshold = cfg.threshold
        self.similarity_window = cfg.similarity_window
        self.threshold_percentile = cfg.threshold_percentile
        self.similarity_function = cfg.similarity_function

        # load the sentence splitter
        if cfg.pre_chunk_config.max_tokens is not None:
            if (
                cfg.max_tokens is not None
                and cfg.pre_chunk_config.max_tokens > cfg.max_tokens
            ):
                cfg.pre_chunk_config.max_tokens = cfg.max_tokens
                logger.warning(
                    f"pre_chunk_config.max_tokens is greater than max_tokens, "
                    f"setting pre_chunk_config.max_tokens to {cfg.max_tokens}"
                )
        else:
            if cfg.max_tokens is not None:
                cfg.pre_chunk_config.max_tokens = cfg.max_tokens
                logger.warning(
                    f"pre_chunk_config.max_tokens is not set, "
                    f"setting pre_chunk_config.max_tokens to {cfg.max_tokens}"
                )
        assert cfg.pre_chunk_config.overlap == 0, "pre_chunk_config.overlap must be 0"
        self.prechunker = SentenceChunker(cfg.pre_chunk_config, tokenizer=tokenizer)

        self.encoder = encoder
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        # prepare length function
        if self.prechunker.tokenizer.vocab_size > 0:
            length_fn = lambda x: len(self.prechunker.tokenizer.encode(x))
        else:
            length_fn = lambda x: len(self.prechunker.tokenizer.tokenize(x))

        # split the text into base chunks
        base_chunks = self.prechunker.chunk(text)
        sentences = [s.text for s in base_chunks]
        if len(sentences) == 1:
            chunks = base_chunks
            if return_str:
                return [chunk.text for chunk in chunks]
            return chunks

        # combine the sentences to calculate the embeddings
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentences.append(
                " ".join(
                    sentences[
                        max(0, i - self.similarity_window) : i
                        + self.similarity_window
                        + 1
                    ]
                )
            )
        embeddings = self.encoder.encode(combined_sentences)

        # calculate the similarity between the combined sentences
        emb1 = embeddings[1:]
        emb2 = embeddings[:-1]
        match self.similarity_function:
            case "L2":
                similarity = 1 / np.linalg.norm(emb1 - emb2, axis=1)
            case "IP":
                similarity = np.einsum("ij,ij->i", emb1, emb2)
            case "COS":
                similarity = np.einsum("ij,ij->i", emb1, emb2) / (
                    np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
                )
            case _:
                raise ValueError(
                    f"Unknown similarity function: {self.similarity_function}"
                )

        # calculate the threshold
        if self.threshold is not None:
            threshold = self.threshold
        elif self.threshold_percentile is not None:
            threshold = np.percentile(similarity, self.threshold_percentile)
        else:
            assert self.max_tokens is not None, (
                "At least one of max_tokens, threshold, or threshold_percentile should be provided."
            )
            threshold = None

        # group the sentences into chunks based on the threshold
        if threshold is not None:
            chunks = self._group_chunks(base_chunks, similarity, threshold, text)
        else:
            # try to find the threshold that best fits the max_tokens
            base_chunk_lens = [length_fn(s) for s in sentences]
            thresholds = np.sort(similarity)
            left_pointer = 0
            right_threshold = len(thresholds) - 1
            while True:
                mid_pointer = (left_pointer + right_threshold) // 2
                threshold = thresholds[mid_pointer] + 1e-6
                max_tokens = self._get_max_tokens(
                    base_chunk_lens, similarity, threshold
                )
                if left_pointer >= right_threshold:
                    break
                if max_tokens > self.max_tokens:
                    left_pointer = mid_pointer + 1
                else:
                    right_threshold = mid_pointer

            # use the last threshold that fits the max_tokens
            if max_tokens > self.max_tokens:
                if mid_pointer + 1 < len(thresholds):
                    threshold = thresholds[mid_pointer + 1] + 1e-6
                else:
                    logger.warning("Cannot find a suitable threshold.")
                    threshold = thresholds[mid_pointer] + 1e-6
            else:
                threshold = thresholds[mid_pointer] + 1e-6
            chunks = self._group_chunks(base_chunks, similarity, threshold, text)
        if return_str:
            return [chunk.text for chunk in chunks]
        return chunks

    def _get_max_tokens(
        self,
        base_chunk_lens: list[int],
        similarity: np.ndarray,
        threshold: float,
    ) -> int:
        max_tokens = 0
        current_len = base_chunk_lens[0]
        for i in range(1, len(base_chunk_lens)):
            if similarity[i - 1] < threshold:
                if current_len > max_tokens:
                    max_tokens = current_len
                current_len = base_chunk_lens[i]
            else:
                current_len += base_chunk_lens[i]
        if current_len > max_tokens:
            max_tokens = current_len
        return max_tokens

    def _group_chunks(
        self,
        base_chunks: list[Chunk],
        similarity: np.ndarray,
        threshold: float,
        text: str,
    ) -> list[Chunk]:
        chunks = []
        if not base_chunks:
            return chunks

        start_index = 0
        for i in range(1, len(base_chunks)):
            if similarity[i - 1] < threshold:
                start_chunk = base_chunks[start_index]
                end_chunk = base_chunks[i - 1]
                if start_chunk.start is not None and end_chunk.end is not None:
                    chunk_text = text[start_chunk.start : end_chunk.end]
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start=start_chunk.start,
                            end=end_chunk.end,
                        )
                    )
                else:
                    chunk_text = " ".join([c.text for c in base_chunks[start_index:i]])
                    chunks.append(Chunk(text=chunk_text))
                start_index = i

        # Last chunk
        start_chunk = base_chunks[start_index]
        end_chunk = base_chunks[-1]
        if start_chunk.start is not None and end_chunk.end is not None:
            chunk_text = text[start_chunk.start : end_chunk.end]
            chunks.append(
                Chunk(text=chunk_text, start=start_chunk.start, end=end_chunk.end)
            )
        else:
            chunk_text = " ".join([c.text for c in base_chunks[start_index:]])
            chunks.append(Chunk(text=chunk_text))
        return chunks
