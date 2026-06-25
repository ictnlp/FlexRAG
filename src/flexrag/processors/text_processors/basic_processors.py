import re
from dataclasses import field
from typing import Optional

import regex

from flexrag.common import configure

from .processor import PROCESSORS, Processor, TextUnit
from .utils import UnifiedTokenizer, UTokenizerConfig


@configure
class TokenNormalizerConfig:
    lang: str = "en"
    penn: bool = True
    norm_quote_commas: bool = True
    norm_numbers: bool = True
    pre_replace_unicode_punct: bool = False
    post_remove_control_chars: bool = False
    perl_parity: bool = False


@PROCESSORS("token_normalize", config_class=TokenNormalizerConfig)
class TokenNormalizer(Processor):
    def __init__(self, cfg: TokenNormalizerConfig) -> None:
        from sacremoses import MosesPunctNormalizer

        self.normalizer = MosesPunctNormalizer(
            lang=cfg.lang,
            penn=cfg.penn,
            norm_quote_commas=cfg.norm_quote_commas,
            norm_numbers=cfg.norm_numbers,
            pre_replace_unicode_punct=cfg.pre_replace_unicode_punct,
            post_remove_control_chars=cfg.post_remove_control_chars,
            perl_parity=cfg.perl_parity,
        )
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        input_text.content = self.normalizer.normalize(input_text.content)
        return input_text


@PROCESSORS("chinese_simplify")
class ChineseSimplifier(Processor):
    def __init__(self):
        import opencc

        self.converter = opencc.OpenCC()
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        input_text.content = self.converter.convert(input_text.content)
        return input_text


@PROCESSORS("lowercase")
class Lowercase(Processor):
    def process(self, input_text: TextUnit) -> TextUnit:
        input_text.content = input_text.content.lower()
        return input_text


@PROCESSORS("unify")
class Unifier(Processor):
    def __init__(self) -> None:
        from unidecode import unidecode

        self.unidecode = unidecode
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        input_text.content = self.unidecode(input_text.content)
        return input_text


@configure
class TruncatorConfig:
    max_chars: Optional[int] = None
    max_bytes: Optional[int] = None
    max_tokens: Optional[int] = None
    tokenizer_config: UTokenizerConfig = field(default_factory=UTokenizerConfig)


@PROCESSORS("truncate", config_class=TruncatorConfig)
class Truncator(Processor):
    def __init__(self, cfg: TruncatorConfig) -> None:
        self.max_chars = cfg.max_chars
        self.max_bytes = cfg.max_bytes
        self.max_tokens = cfg.max_tokens
        if self.max_tokens is not None:
            self.tokenizer = UnifiedTokenizer(cfg=cfg.tokenizer_config)
        if (
            self.max_chars is None
            and self.max_bytes is None
            and self.max_tokens is None
        ):
            raise ValueError(
                "At least one of max_chars, max_bytes, or max_tokens should be set."
            )
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        if self.max_tokens is not None:
            tokens = self.tokenizer.tokenize(input_text.content)
            input_text.content = self.tokenizer.detokenize(tokens[: self.max_tokens])
        if self.max_chars is not None:
            input_text.content = input_text.content[: self.max_chars]
        if self.max_bytes is not None:
            input_bytes = input_text.content.encode("utf-8")
            if len(input_bytes) > self.max_bytes:
                trunc_point = self.max_bytes

                # do not truncate in the middle of a character
                while trunc_point > 0 and (input_bytes[trunc_point] & 0xC0) == 0x80:
                    trunc_point -= 1

                # ensure utf-8 encoding
                while trunc_point > 0:
                    try:
                        _ = input_bytes[:trunc_point].decode("utf-8")
                        break
                    except:
                        trunc_point -= 1
                input_text.content = input_bytes[:trunc_point].decode("utf-8")
        return input_text


@PROCESSORS("simplify_answer")
class AnswerSimplifier(Processor):
    article_pattern = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
    punctuation_pattern = regex.compile(r"\p{P}+", re.UNICODE)
    whitespace_pattern = re.compile(r"\s+", re.UNICODE)

    def process(self, input_text: TextUnit) -> TextUnit:
        # lower case
        text = input_text.content.lower()
        # remove_articles
        text = self.article_pattern.sub(" ", text)
        # remove punctuation
        text = self.punctuation_pattern.sub(" ", text)
        # unify white space
        text = self.whitespace_pattern.sub(" ", text).strip()
        input_text.content = text
        return input_text
