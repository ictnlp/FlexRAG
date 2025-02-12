from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Generic, TypeVar

from omegaconf import MISSING

from flexrag.utils import Register


TokenType = TypeVar("TokenType")


class TokenizerBase(ABC, Generic[TokenType]):
    """TokenizerBase is an abstract class that defines the interface for all tokenizers.
    These tokenizers are useful in the `text_processing` module and the `chunking` module.

    The subclasses should implement the `tokenize` and `detokenize` methods to convert text to tokens and vice versa.
    The `reversible` property should return True if the tokenizer can detokenize the tokens back to the original text.
    """

    @abstractmethod
    def tokenize(self, texts: str) -> list[TokenType]:
        """Tokenize the given text into tokens.

        :param texts: The text to tokenize.
        :type texts: str
        :return: The tokens of the text.
        :rtype: list[TokenType]
        """
        return

    @abstractmethod
    def detokenize(self, tokens: list[TokenType]) -> str:
        """Detokenize the tokens back to text.

        :param tokens: The tokens to detokenize.
        :type tokens: list[TokenType]
        :return: The detokenized text.
        :rtype: str
        """
        return

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """Return True if the tokenizer can detokenize the tokens back to the original text."""
        return


TOKENIZERS = Register[TokenizerBase]("tokenizer")


@dataclass
class HuggingFaceTokenizerConfig:
    """Configuration for HuggingFaceTokenizer.

    :param tokenizer_path: The path to the HuggingFace tokenizer.
    :type tokenizer_path: str
    """

    tokenizer_path: str = MISSING


@TOKENIZERS("hf", config_class=HuggingFaceTokenizerConfig)
class HuggingFaceTokenizer(TokenizerBase[int]):
    """A wrapper for HuggingFace tokenizers."""

    def __init__(self, cfg: HuggingFaceTokenizerConfig) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        return

    def tokenize(self, texts: str) -> list[int]:
        return self.tokenizer.encode(texts)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def reversible(self) -> bool:
        """Most HuggingFace tokenizers that employs BPE/SPM model are reversible."""
        return True


@dataclass
class TikTokenTokenizerConfig:
    """Configuration for TikTokenTokenizer.

    :param tokenizer_name: Load the tokenizer by the name. Default is None.
    :type tokenizer_name: Optional[str]
    :param model_name: Load the tokenizer by the corresponding OpenAI's model. Default is "gpt-4o".
    :type model_name: Optional[str]

    At least one of tokenizer_name or model_name must be provided.
    """

    tokenizer_name: Optional[str] = None
    model_name: Optional[str] = "gpt-4o"


@TOKENIZERS("tiktoken", config_class=TikTokenTokenizerConfig)
class TikTokenTokenizer(TokenizerBase[int]):
    """A wrapper for TikToken tokenizers."""

    def __init__(self, cfg: TikTokenTokenizerConfig) -> None:
        import tiktoken

        if cfg.tokenizer_name is not None:
            self.tokenizer = tiktoken.get_encoding(cfg.tokenizer_name)
        elif cfg.model_name is not None:
            self.tokenizer = tiktoken.encoding_for_model(cfg.model_name)
        else:
            raise ValueError("Either tokenizer_name or model_name must be provided.")
        return

    def tokenize(self, texts: str) -> list[int]:
        return self.tokenizer.encode(texts)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def reversible(self) -> bool:
        """TikTokenTokenizer is reversible."""
        return True


@dataclass
class MosesTokenizerConfig:
    """Configuration for MosesTokenizer.

    :param lang: The language code for the tokenizer. Default is "en".
    :type lang: str
    """

    lang: str = "en"


@TOKENIZERS("moses", config_class=MosesTokenizerConfig)
class MosesTokenizer(TokenizerBase[str]):
    """A wrapper for SacreMoses tokenizers."""

    def __init__(self, cfg: MosesTokenizerConfig) -> None:
        from sacremoses import MosesDetokenizer, MosesTokenizer

        self.tokenizer = MosesTokenizer(cfg.lang)
        self.detokenizer = MosesDetokenizer(cfg.lang)
        return

    def tokenize(self, texts: str) -> list[str]:
        return self.tokenizer.tokenize(texts)

    def detokenize(self, tokens: list[str]) -> str:
        return self.detokenizer.detokenize(tokens)

    @property
    def reversible(self) -> bool:
        """MosesTokenizer is not reversible as it may lose sapces and punctuations."""
        return False


@dataclass
class NLTKTokenizerConfig:
    """Configuration for NLTKTokenizer.

    :param lang: The language to use for the tokenizer. Default is "english".
    :type lang: str
    """

    lang: str = "english"


@TOKENIZERS("nltk_tokenizer", config_class=NLTKTokenizerConfig)
class NLTKTokenizer(TokenizerBase[str]):
    """A wrapper for NLTK tokenizers."""

    def __init__(self, cfg: NLTKTokenizerConfig) -> None:
        from nltk.tokenize import word_tokenize

        self.lang = cfg.lang
        self.tokenize_func = partial(word_tokenize, language=cfg.lang)
        return

    def tokenize(self, texts: str) -> list[str]:
        return self.tokenize_func(texts)

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    @property
    def reversible(self) -> bool:
        """NLTKTokenizer is not reversible as it may lose sapces."""
        return False


@dataclass
class JiebaTokenizerConfig:
    """Configuration for JiebaTokenizer.

    :param enable_hmm: Whether to use the Hidden Markov Model. Default is True.
    :type enable_hmm: bool
    :param cut_all: Whether to use the full mode. Default is False.
    :type cut_all: bool
    """

    enable_hmm: bool = True
    cut_all: bool = False


@TOKENIZERS("jieba", config_class=JiebaTokenizerConfig)
class JiebaTokenizer(TokenizerBase[str]):
    """A wrapper for Jieba tokenizers."""

    def __init__(self, cfg: JiebaTokenizerConfig) -> None:
        import jieba

        jieba.disable_parallel()
        self.tokenize_func = partial(jieba.cut, HMM=cfg.enable_hmm, cut_all=cfg.cut_all)
        return

    def tokenize(self, texts: str) -> list[str]:
        return list(self.tokenize_func(texts))

    def detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)

    @property
    def reversible(self) -> bool:
        """JiebaTokenizer is reversible."""
        return True


TokenizerConfig = TOKENIZERS.make_config(default="tiktoken")
