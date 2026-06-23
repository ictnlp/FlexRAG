from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Optional

from flexrag.common import Register, configure


class TokenizerBase(ABC):
    """A tokenizer is a component that converts raw natural language text into discrete tokens
    (such as words, subwords, or symbols) for subsequent modeling and processing.
    These tokenizers are useful in the `text_processing` module and the `chunking` module.

    TokenizerBase is an abstract class that defines the interface for all tokenizers.

    The subclasses should implement the following methods:

        >>> def tokenize(self, text: str) -> list[str]:
        >>>     # Tokenize the given text into tokens
        >>>     ...

        >>> def detokenize(self, tokens: list[str]) -> str:
        >>>     # Detokenize the tokens back to text
        >>>     ...

        >>> def encode(self, text: str) -> list[int]:
        >>>     # Encode the given text into token ids
        >>>     ...

        >>> def decode(self, tokens: list[int]) -> str:
        >>>     # Decode the token ids back to text
        >>>     ...

        >>> @property
        >>> def reversible(self) -> bool:
        >>>     # Return True if the tokenizer can decode the tokens back to the original text strictly
        >>>     ...

        >>> @property
        >>> def vocab_size(self) -> int:
        >>>     # Return the size of the tokenizer vocabulary
        >>>     ...

    The `reversible` property should return True if the tokenizer can decode the tokens back to the original text.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize the given text into tokens.

        :param text: The text to tokenize.
        :type text: str
        :return: The tokens of the text.
        :rtype: list[str]
        """
        return

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Detokenize the tokens back to text.

        :param tokens: The tokens to detokenize.
        :type tokens: list[str]
        :return: The detokenized text.
        :rtype: str
        """
        return

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode the given text into token ids.

        :param text: The text to tokenize.
        :type text: str
        :return: The tokens of the text.
        :rtype: list[int]
        """
        return

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode the token ids back to text.

        :param tokens: The tokens to decode.
        :type tokens: list[int]
        :return: The decoded text.
        :rtype: str
        """
        return

    @abstractmethod
    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert tokens to token ids.

        :param tokens: The tokens to convert.
        :type tokens: list[str]
        :return: The token ids.
        :rtype: list[int]
        """
        return

    @abstractmethod
    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert token ids to tokens.

        :param token_ids: The token ids to convert.
        :type token_ids: list[int]
        :return: The tokens.
        :rtype: list[str]
        """
        return

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """Return True if the tokenizer can decode the tokens back to the original text strictly."""
        return

    @property
    def vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary. If the tokenizer
        does not support `encode` and `decode` methods, it should return 0.
        """
        return 0


TOKENIZERS = Register[TokenizerBase]("tokenizer")


@configure
class SpaceTokenizerConfig:
    """Configuration for :class:`SpaceTokenizer`."""


@TOKENIZERS("space", config_class=SpaceTokenizerConfig)
class SpaceTokenizer(TokenizerBase):
    """A simple tokenizer that splits text by spaces."""

    def __init__(self, cfg: SpaceTokenizerConfig | None = None) -> None:
        return

    def tokenize(self, texts: str) -> list[str]:
        return texts.split()

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("SpaceTokenizer does not support `encode` method.")

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("SpaceTokenizer does not support `decode` method.")

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError(
            "SpaceTokenizer does not support `tokens_to_ids` method."
        )

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        raise NotImplementedError(
            "SpaceTokenizer does not support `ids_to_tokens` method."
        )

    @property
    def reversible(self) -> bool:
        """SpaceTokenizer is not reversible as it may lose spaces."""
        return False


@configure
class MosesTokenizerConfig:
    """Configuration for MosesTokenizer.

    :param lang: The language code for the tokenizer. Default is "en".
    :type lang: str
    """

    lang: str = "en"


@TOKENIZERS("moses", config_class=MosesTokenizerConfig)
class MosesTokenizer(TokenizerBase):
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

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("MosesTokenizer does not support `encode` method.")

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("MosesTokenizer does not support `decode` method.")

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError(
            "MosesTokenizer does not support `tokens_to_ids` method."
        )

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        raise NotImplementedError(
            "MosesTokenizer does not support `ids_to_tokens` method."
        )

    @property
    def reversible(self) -> bool:
        """MosesTokenizer is not reversible as it may lose sapces and punctuations."""
        return False


@configure
class NLTKTokenizerConfig:
    """Configuration for NLTKTokenizer.

    :param lang: The language to use for the tokenizer. Default is "english".
    :type lang: str
    """

    lang: str = "english"


@TOKENIZERS("nltk", config_class=NLTKTokenizerConfig)
class NLTKTokenizer(TokenizerBase):
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

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("NLTKTokenizer does not support `encode` method.")

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("NLTKTokenizer does not support `decode` method.")

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError(
            "NLTKTokenizer does not support `tokens_to_ids` method."
        )

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        raise NotImplementedError(
            "NLTKTokenizer does not support `ids_to_tokens` method."
        )

    @property
    def reversible(self) -> bool:
        """NLTKTokenizer is not reversible as it may lose spaces."""
        return False


@configure
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
class JiebaTokenizer(TokenizerBase):
    """A wrapper for Jieba tokenizers.
    Jieba keeps all characters including spaces and punctuations during tokenization,
    making it reversible.
    """

    def __init__(self, cfg: JiebaTokenizerConfig) -> None:
        import jieba

        jieba.disable_parallel()
        self.tokenize_func = partial(jieba.cut, HMM=cfg.enable_hmm, cut_all=cfg.cut_all)
        return

    def tokenize(self, texts: str) -> list[str]:
        return list(self.tokenize_func(texts))

    def detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("JiebaTokenizer does not support `encode` method.")

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("JiebaTokenizer does not support `decode` method.")

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError(
            "JiebaTokenizer does not support `tokens_to_ids` method."
        )

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        raise NotImplementedError(
            "JiebaTokenizer does not support `ids_to_tokens` method."
        )

    @property
    def reversible(self) -> bool:
        """JiebaTokenizer is reversible."""
        return True


@configure
class HuggingFaceTokenizerConfig:
    """Configuration for HuggingFaceTokenizer.

    :param tokenizer_path: The path to the HuggingFace tokenizer.
    :type tokenizer_path: str
    """

    tokenizer_path: Optional[str] = None


@TOKENIZERS("hf", config_class=HuggingFaceTokenizerConfig)
class HuggingFaceTokenizer(TokenizerBase):
    """A wrapper for HuggingFace tokenizers."""

    def __init__(self, cfg: HuggingFaceTokenizerConfig) -> None:
        from transformers import AutoTokenizer

        assert cfg.tokenizer_path is not None, "`tokenizer_path` must be provided"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        return

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def reversible(self) -> bool:
        """Most HuggingFace tokenizers that employs BPE/SPM model are reversible."""
        return True


@configure
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
class TikTokenTokenizer(TokenizerBase):
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

    def tokenize(self, texts: str) -> list[str]:
        # tiktoken works on bytes; here we expose string tokens for consistency.
        token_ids = self.tokenizer.encode(texts)
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def detokenize(self, tokens: list[str]) -> str:
        # Re-tokenize each token string and concatenate; this may not be strictly reversible
        # for arbitrary manipulations but is consistent with `encode`/`decode` behavior.
        token_ids = list(
            chain.from_iterable(self.tokenizer.encode(token) for token in tokens)
        )
        return self.tokenizer.decode(token_ids)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return list(
            chain.from_iterable(self.tokenizer.encode(token) for token in tokens)
        )

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    @property
    def vocab_size(self) -> int:
        # tiktoken encoders expose number of tokens via `n_vocab` attribute
        return int(getattr(self.tokenizer, "n_vocab", 0))

    @property
    def reversible(self) -> bool:
        """TikTokenTokenizer is reversible."""
        return True


TokenizerConfig = TOKENIZERS.make_config(default="tiktoken")
