from abc import ABC, abstractmethod
from functools import partial

from flexrag.common import Register, configure


class SentenceSplitterBase(ABC):
    """Sentence splitter that splits text into sentences.
    This is an abstract class that defines the interface for all sentence splitters.
    The subclasses should implement the `split` method to split the text.
    """

    @abstractmethod
    def split(self, text: str) -> list[dict[str, str | tuple[int, int]]]:
        """Split the given text into sentences.

        :param text: The text to split.
        :type text: str
        :return: The splitted sentences with their spans.
            Each item is a dictionary with two fields:

            * ``text`` — the sentence string.
            * ``char_span`` — a tuple ``(start, end)`` indicating the
              character span of the sentence within the original text.

            The span follows a half-open interval ``[start, end)``,
            where ``start`` is inclusive and ``end`` marks the position
            immediately after the last character of the span.

            Example:
                ``[{"text": "sentence1", "char_span": (0, 9)},`
                `{"text": "sentence2", "char_span": (10, 20)}]``

            If the character span is not available, ``char_span`` is set to ``(-1, -1)``.
        :rtype: list[dict[str, str | tuple[int, int]]]
        """
        return


SENTENCE_SPLITTERS = Register[SentenceSplitterBase]("sentence_splitter")


@configure
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

    def split(self, text: str) -> list[dict[str, str | tuple[int, int]]]:
        sents = [s for s in self.splitter(text)]
        spans = []
        start = 0
        for sent in sents:
            try:
                start = text.index(sent, start)
            except ValueError:
                spans.append((-1, -1))
                continue
            end = start + len(sent)
            spans.append((start, end))
            start = end
        return [{"text": sent, "char_span": span} for sent, span in zip(sents, spans)]


# \R only supported in regex module, not re module
PREDEFINED_SPLIT_PATTERNS = {
    "en": {
        "big_paragraph": r"\R{2,}",
        "paragraph": r"\R",
        "sentence": r"(?<=[.?!]+[\"')\]]*)\s+",
        "subsentence": r"(?<=[,;\"'{}<>\[\]`~])\s*",
        "word": r"\s+",
    },
    "zh": {
        "big_paragraph": r"\R{2,}",
        "paragraph": r"\R",
        "sentence": r"(?<=[。！？])",
        "subsentence": r"(?<=[，；：“”‘’《》【】、])",
    },
}


@configure
class RegexSplitterConfig:
    """Configuration for RegexSentenceSplitter.

    :param pattern: The regular expression pattern to split the text.
        Default is ``PREDEFINED_SPLIT_PATTERNS["en"]["sentence"]``
    :type pattern: str

    Note that some patterns may lose the separators between sentences.
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

    def split(self, text: str) -> list[dict[str, str | tuple[int, int]]]:
        # Use regex split while keeping track of character spans.
        # Note: this assumes the pattern is a zero-width assertion (e.g. lookbehind),
        # which is the recommended usage in PREDEFINED_SPLIT_PATTERNS.
        sents: list[str] = []
        spans: list[tuple[int, int]] = []

        last_idx = 0
        for matched in self.pattern.finditer(text):
            start, end = matched.start(), matched.end()
            if start > last_idx:
                sent = text[last_idx:start]
                sents.append(sent)
                spans.append((last_idx, start))
            last_idx = end

        # Tail after the last match
        if last_idx < len(text):
            sent = text[last_idx:]
            sents.append(sent)
            spans.append((last_idx, len(text)))

        return [{"text": sent, "char_span": span} for sent, span in zip(sents, spans)]


@configure
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

        # load the spacy model with parser / sentencizer enabled
        self.nlp = spacy.load(cfg.model)
        all_pipes = set(self.nlp.pipe_names)
        required_pipes = []
        if "parser" in self.nlp.pipe_names:
            for pipe_name in self.nlp.pipe_names:
                if pipe_name == "tagger":
                    continue
                required_pipes.append(pipe_name)
                if pipe_name == "parser":
                    break
        elif "senter" in all_pipes:
            for pipe_name in self.nlp.pipe_names:
                required_pipes.append(pipe_name)
                if pipe_name == "senter":
                    break
        elif "sentencizer" in all_pipes:
            for pipe_name in self.nlp.pipe_names:
                required_pipes.append(pipe_name)
                if pipe_name == "sentencizer":
                    break
        else:
            raise ValueError(
                f"The spaCy model '{cfg.model}' does not have a sentence boundary detector."
            )
        self.nlp.select_pipes(enable=required_pipes)
        return

    def split(self, text: str) -> list[dict[str, str | tuple[int, int]]]:
        return [
            {"text": sent.text, "char_span": (sent.start_char, sent.end_char)}
            for sent in self.nlp(text).sents
        ]


SentenceSplitterConfig = SENTENCE_SPLITTERS.make_config(
    default="regex", config_name="SentenceSplitterConfig"
)
