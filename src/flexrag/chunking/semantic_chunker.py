from dataclasses import dataclass
from typing import Optional

import numpy as np

from flexrag.models import ENCODERS, EncoderConfig
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig
from flexrag.utils import Choices, LOGGER_MANAGER

from .chunker_base import CHUNKERS, ChunkerBase
from .sentence_splitter import SENTENCE_SPLITTERS, SentenceSplitterConfig


logger = LOGGER_MANAGER.get_logger("flexrag.chunking.semantic_chunker")


@dataclass
class SemanticChunkerConfig(SentenceSplitterConfig, EncoderConfig, TokenizerConfig):
    """Configuration for SemanticChunker.

    :param max_tokens: The maximum number of tokens in each chunk. Default is None.
    :type max_tokens: Optional[int]
    :param threshold: The threshold for semantic similarity. Default is None.
        If provided, the `threshold_percentile` and `max_tokens` will be ignored.
    :type threshold: Optional[float]
    :param threshold_percentile: The ratio of the threshold for semantic similarity. Default is None.
        Should be a value between 0 and 100. Higher values will result in more chunks. 5 is a good starting point.
        If provided, the `max_tokens` will be ignored.
    :type threshold_percentile: Optional[float]
    :param similarity_window: The window size for calculating semantic similarity. Default is None.
    :type similarity_window: Optional[int]
    :param similarity_function: The similarity function to use. Default is "COS".
        Available choices are "L2" for the reciprocal of euclidean distance, "IP" for inner product, and "COS" for cosine similarity.
    :type similarity_function: str

    The similarity higher than the threshold will be considered as coherent, and the chunks will be split at the points where the similarity is below the threshold.
    Thus, at least one of `max_tokens`, `threshold`, or `threshold_percentile` should be provided.
    If `threshold` is provided, the chunks will be split directly based on the threshold.
    If `threshold_percentile` is provided, the threshold will be calculated automatically based on the similarity distribution.
    If `max_tokens` is provided, the threshold will be calculated to ensure the chunks are within the token limit.


    For example, to split the text into chunks with a maximum of 512 tokens, you can use the following configuration:

        >>> from flexrag.chunking import SemanticChunker, SemanticChunkerConfig
        >>> from flexrag.models import HFEncoderConfig
        >>> config = SemanticChunkerConfig(
        ...     max_tokens=512,
        ...     encoder_type="hf",
        ...     hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
        ... )
        >>> chunker = SemanticChunker(config)

    To split the text into chunks with a threshold_percentile of 5%, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold_percentile=5,
        ...     encoder_type="hf",
        ...     hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
        ... )
        >>> chunker = SemanticChunker(config)

    To split the text into chunks with a given threshold, you can use the following configuration:

        >>> config = SemanticChunkerConfig(
        ...     threshold=0.8,
        ...     encoder_type="hf",
        ...     hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
        ... )
        >>> chunker = SemanticChunker(config)
    """

    max_tokens: Optional[int] = None
    max_tokens_per_sentence: Optional[int] = None
    threshold: Optional[float] = None
    threshold_percentile: Optional[float] = None
    similarity_window: int = 1
    similarity_function: Choices(["L2", "IP", "COS"]) = "COS"  # type: ignore


@CHUNKERS("semantic", config_class=SemanticChunkerConfig)
class SemanticChunker(ChunkerBase):
    """SemanticChunker splits text into sentences and then groups them into chunks based on semantic similarity.
    This chunker is inspired by the Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """

    def __init__(self, cfg: SemanticChunkerConfig) -> None:
        # set the basic configurations
        self.max_tokens = cfg.max_tokens if cfg.max_tokens is not None else float("inf")
        self.threshold = cfg.threshold
        self.similarity_window = cfg.similarity_window
        self.threshold_percentile = cfg.threshold_percentile
        self.similarity_function = cfg.similarity_function
        if cfg.max_tokens_per_sentence is not None:
            self.max_tokens_per_sentence = cfg.max_tokens_per_sentence
        elif self.max_tokens is not None:
            self.max_tokens_per_sentence = cfg.max_tokens
        else:
            self.max_tokens_per_sentence = None

        # load the sentence splitter
        self.splitter = SENTENCE_SPLITTERS.load(cfg)

        # load the encoder
        self.encoder = ENCODERS.load(cfg)

        # load the tokenizer
        self.tokenizer = TOKENIZERS.load(cfg)
        return

    def chunk(self, text: str) -> list[str]:
        # split the text into sentences
        sentences = self._split_sentences(text)
        if len(sentences) == 1:
            return sentences

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
            assert (
                self.max_tokens is not None
            ), "At least one of max_tokens, threshold, or threshold_percentile should be provided."
            threshold = None

        # group the sentences into chunks based on the threshold
        if threshold is not None:
            chunks = self._group_sentences(sentences, similarity, threshold)
        else:
            # check if the max_tokens is feasible
            max_tokens = max(len(self.tokenizer.tokenize(sent)) for sent in sentences)
            if max_tokens == self.max_tokens:
                chunks = sentences
                return chunks
            elif max_tokens > self.max_tokens:
                logger.warning(
                    f"The maximum number of tokens in a sentence is {max_tokens}, "
                    f"which is greater than the specified max_tokens {self.max_tokens}."
                )
                chunks = sentences
                return chunks

            # try to find the threshold that best fits the max_tokens
            thresholds = np.sort(similarity)
            left_pointer = 0
            right_threshold = len(thresholds) - 1
            while True:
                mid_pointer = (left_pointer + right_threshold) // 2
                threshold = thresholds[mid_pointer] + 1e-6
                chunks_ = self._group_sentences(sentences, similarity, threshold)
                max_tokens = max(
                    len(self.tokenizer.tokenize(chunk)) for chunk in chunks_
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
            chunks = self._group_sentences(sentences, similarity, threshold)
        return chunks

    def _group_sentences(
        self, sentences: list[str], similarity: np.ndarray, threshold: float
    ) -> list[str]:
        chunks = []
        chunk = sentences[0]
        for i in range(1, len(sentences)):
            if similarity[i - 1] < threshold:
                chunks.append(chunk)
                chunk = sentences[i]
            else:
                if self.splitter.reversible:
                    chunk += sentences[i]
                else:
                    chunk += " " + sentences[i]
        chunks.append(chunk)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split the text into sentences."""
        sents = self.splitter.split(text)
        if self.max_tokens_per_sentence is None:
            return sents
        new_sents = []
        for sent in sents:
            tokens = self.tokenizer.tokenize(sent)
            if len(tokens) <= self.max_tokens_per_sentence:
                new_sents.append(sent)
                continue
            splitted_sents = []
            for i in range(0, len(tokens), self.max_tokens_per_sentence):
                splitted_sents.append(
                    self.tokenizer.detokenize(
                        tokens[i : i + self.max_tokens_per_sentence]
                    )
                )
            new_sents.extend(splitted_sents)
        return new_sents
