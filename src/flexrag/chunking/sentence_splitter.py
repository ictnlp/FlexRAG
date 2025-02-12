from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from flexrag.utils import Register


class SentenceSplitterBase(ABC):
    """Sentence splitter that splits text into sentences.
    This is an abstract class that defines the interface for all sentence splitters.
    The subclasses should implement the `split` method to split the text.
    The `reversible` property should return True if the splitted sentences can be concatenate back to the original text.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split the given text into sentences.

        :param text: The text to split.
        :type text: str
        :return: The sentences of the text.
        :rtype: list[str]
        """
        return

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """return True if the splitted sentences can be concatenate back to the original text."""
        return


SENTENCE_SPLITTERS = Register[SentenceSplitterBase]("sentence_splitter")


@dataclass
class NLTKSentenceSplitterConfig:
    """Configuration for NLTKSentenceSplitter.

    :param language: The language to use for the sentence splitter. Default is "english".
    :type language: str
    """

    language: str = "english"


@SENTENCE_SPLITTERS("nltk_splitter", config_class=NLTKSentenceSplitterConfig)
class NLTKSentenceSplitter(SentenceSplitterBase):
    """NLTKSentenceSplitter splits text into sentences using NLTK's PunktSentenceTokenizer.
    For more information, see https://www.nltk.org/api/nltk.tokenize.punkt.html#module-nltk.tokenize.punkt.
    """

    def __init__(self, cfg: NLTKSentenceSplitterConfig) -> None:
        try:
            import nltk
        except ImportError:
            raise ImportError("NLTK is required for NLTKSentenceSplitter.")
        self.splitter = partial(nltk.sent_tokenize, language=cfg.language)
        return

    def split(self, text: str) -> list[str]:
        return self.splitter(text)

    @property
    def reversible(self) -> bool:
        """NLTKSentenceSplitter is not reversible as it may lose spaces between sentences."""
        return False


PREDEFINED_SPLIT_PATTERNS = {
    "en": {
        "big_paragraph": r"(?<=\R{2,})",
        "paragraph": r"(?<=\R)",
        "sentence": r"(?<=[.?!])",
        "subsentence": r"(?<=[,;\"'{}<>\[\]`~])",
        "word": r"(?<=\s)",
    },
    "zh": {
        "big_paragraph": r"(?<=\R{2,})",
        "paragraph": r"(?<=\R)",
        "setence": r"(?<=[。！？])",
        "subsentence": r"(?<=[，；：“”‘’《》【】、])",
    },
}


@dataclass
class RegexSplitterConfig:
    """Configuration for RegexSentenceSplitter.

    :param pattern: The regular expression pattern to split the text.
        Default is ``PREDEFINED_SPLIT_PATTERNS["en"]["sentence"]``
    :type pattern: str

    Note that some patterns may lose the seperators between sentences.
    A good practice is to use the lookbehind and lookahead assertion to avoid consuming the splitter.
    """

    pattern: str = PREDEFINED_SPLIT_PATTERNS["en"]["sentence"]


@SENTENCE_SPLITTERS("regex", config_class=RegexSplitterConfig)
class RegexSplitter(SentenceSplitterBase):
    """RegexSentenceSplitter splits text into sentences using a regular expression pattern.

    Note that this splitter uses the `regex` module, which might be slightly different from the built-in `re` module.
    """

    def __init__(self, cfg: RegexSplitterConfig) -> None:
        import regex

        self.pattern = regex.compile(cfg.pattern)
        return

    def split(self, text: str) -> list[str]:
        return self.pattern.split(text)

    @property
    def reversible(self) -> bool:
        """The default RegexSplitter is reversible. However, the reversibility depends on the pattern used."""
        return True


@dataclass
class SpacySentenceSplitterConfig:
    """Configuration for SpacySentenceSplitter.

    :param model: The spaCy model to use for sentence splitting. Default is "en_core_web_sm".
    :type model: str
    """

    model: str = "en_core_web_sm"


@SENTENCE_SPLITTERS("spacy", config_class=SpacySentenceSplitterConfig)
class SpacySentenceSplitter(SentenceSplitterBase):
    """SpacySentenceSplitter splits text into sentences using spaCy's sentence splitter."""

    def __init__(self, cfg: SpacySentenceSplitterConfig) -> None:
        try:
            import spacy
        except ImportError:
            raise ImportError("spaCy is required for SpacySentenceSplitter.")
        self.nlp = spacy.load(cfg.model)
        return

    def split(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]

    @property
    def reversible(self) -> bool:
        """SpacySentenceSplitter is not reversible as it may lose spaces between sentences."""
        return False


SentenceSplitterConfig = SENTENCE_SPLITTERS.make_config(
    default="nltk_splitter", config_name="SentenceSplitterConfig"
)
