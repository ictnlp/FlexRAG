from typing import Annotated, Optional

import numpy as np

from flexrag.common import LOGGER_MANAGER, Choices, configure
from flexrag.models.encoders import EncoderProtocol
from flexrag.models.tokenizer import TokenizerProtocol

from .chunker_base import CHUNKERS, Chunk, ChunkerBase, ChunkerProtocol

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.semantic_chunker")


@configure
class SemanticChunkerConfig:
    """Configuration for SemanticChunker.

    :param threshold: The threshold for semantic similarity. Default is None.
    :param threshold_percentile: The ratio of the threshold for semantic similarity. Default is None.
        Should be a value between 0 and 100. Higher values will result in more chunks. 5 is a good starting point.
    :param target_max_tokens: The best-effort target for the maximum number of
        tokens in each chunk. A base chunk larger than this target cannot be
        split by SemanticChunker. Default is None.
    :param similarity_window: The window size for calculating semantic similarity. Default is 1.
    :param similarity_function: The similarity function to use. Default is "COS".
        Available choices are "L2" for the reciprocal of euclidean distance, "IP" for inner product, and "COS" for cosine similarity.

    The similarity higher than the threshold will be considered as coherent, and the chunks will be split at the points where the similarity is below the threshold.
    Exactly one of ``threshold``, ``threshold_percentile``, or
    ``target_max_tokens`` must be provided.
    If `threshold` is provided, the chunks will be split directly based on the threshold.
    If `threshold_percentile` is provided, the threshold will be calculated automatically based on the similarity distribution.
    If `target_max_tokens` is provided, the threshold will be calculated to
    best fit the token target at the granularity provided by the base chunker.

    For example, to split the text into chunks with a maximum of 512 tokens, you can use the following configuration:

        >>> from flexrag.processors.chunkers import (
        ...     RegexSplitter,
        ...     RegexSplitterConfig,
        ...     SemanticChunker,
        ...     SemanticChunkerConfig,
        ... )
        >>> from flexrag.models import HFEncoder, HFEncoderConfig
        >>> from flexrag.models.tokenizer import SpaceTokenizer
        >>> tokenizer = SpaceTokenizer()
        >>> encoder = HFEncoder(HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"))
        >>> base_chunker = RegexSplitter(RegexSplitterConfig())
        >>> config = SemanticChunkerConfig(target_max_tokens=512)
        >>> chunker = SemanticChunker(
        ...     config,
        ...     encoder=encoder,
        ...     base_chunker=base_chunker,
        ...     tokenizer=tokenizer,
        ... )

    To split the text into chunks with a threshold_percentile of 5%, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold_percentile=5,
        ... )
        >>> chunker = SemanticChunker(
        ...     config,
        ...     encoder=encoder,
        ...     base_chunker=base_chunker,
        ... )

    To split the text into chunks with a given threshold, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold=0.8,
        ... )
        >>> chunker = SemanticChunker(
        ...     config,
        ...     encoder=encoder,
        ...     base_chunker=base_chunker,
        ... )
    """

    threshold: Optional[float] = None
    threshold_percentile: Optional[float] = None
    target_max_tokens: Optional[int] = None
    similarity_window: int = 1
    similarity_function: Annotated[str, Choices("L2", "IP", "COS")] = "COS"


@CHUNKERS("semantic_chunker", config_class=SemanticChunkerConfig)
class SemanticChunker(ChunkerBase):
    """Group base chunks into larger chunks based on semantic similarity.

    This chunker is inspired by the Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """

    def __init__(
        self,
        cfg: SemanticChunkerConfig,
        *,
        encoder: EncoderProtocol,
        base_chunker: ChunkerProtocol,
        tokenizer: Optional[TokenizerProtocol] = None,
    ) -> None:
        strategies = (
            cfg.threshold,
            cfg.threshold_percentile,
            cfg.target_max_tokens,
        )
        if sum(strategy is not None for strategy in strategies) != 1:
            raise ValueError(
                "Exactly one of threshold, threshold_percentile, or "
                "target_max_tokens must be provided."
            )
        if cfg.target_max_tokens is not None and tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when target_max_tokens is set."
            )

        self.threshold = cfg.threshold
        self.threshold_percentile = cfg.threshold_percentile
        self.target_max_tokens = cfg.target_max_tokens
        self.similarity_window = cfg.similarity_window
        self.similarity_function = cfg.similarity_function
        self.encoder = encoder
        self.base_chunker = base_chunker
        self.tokenizer = tokenizer

    def chunk(self, text: str) -> list[Chunk]:
        # split the text into base chunks
        base_chunks = self.base_chunker.chunk(text)
        if len(base_chunks) <= 1:
            return base_chunks

        sentences = [s.text for s in base_chunks]

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
            threshold = None

        # group the sentences into chunks based on the threshold
        if threshold is not None:
            chunks = self._group_chunks(base_chunks, similarity, threshold, text)
        else:
            # Try to find the threshold that best fits the token target.
            assert self.tokenizer is not None
            assert self.target_max_tokens is not None
            base_chunk_lens = [
                len(self.tokenizer.tokenize(sentence)) for sentence in sentences
            ]
            thresholds = np.sort(similarity)
            left_pointer = 0
            right_threshold = len(thresholds) - 1
            while True:
                mid_pointer = (left_pointer + right_threshold) // 2
                threshold = thresholds[mid_pointer] + 1e-6
                largest_chunk_tokens = self._get_largest_chunk_tokens(
                    base_chunk_lens, similarity, threshold
                )
                if left_pointer >= right_threshold:
                    break
                if largest_chunk_tokens > self.target_max_tokens:
                    left_pointer = mid_pointer + 1
                else:
                    right_threshold = mid_pointer

            # Use the last threshold that fits the target.
            if largest_chunk_tokens > self.target_max_tokens:
                if mid_pointer + 1 < len(thresholds):
                    threshold = thresholds[mid_pointer + 1] + 1e-6
                else:
                    logger.warning("Cannot find a suitable threshold.")
                    threshold = thresholds[mid_pointer] + 1e-6
            else:
                threshold = thresholds[mid_pointer] + 1e-6
            chunks = self._group_chunks(base_chunks, similarity, threshold, text)
        return chunks

    def _get_largest_chunk_tokens(
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
