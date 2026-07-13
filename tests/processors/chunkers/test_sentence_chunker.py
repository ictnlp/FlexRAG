import re

import pytest

from flexrag.models.tokenizer import TikTokenTokenizer, TikTokenTokenizerConfig
from flexrag.processors.chunkers import (
    NLTKSentenceSplitter,
    NLTKSentenceSplitterConfig,
    RegexSplitter,
    RegexSplitterConfig,
    SentenceChunker,
    SentenceChunkerConfig,
)

pytestmark = pytest.mark.integration


class TestSentenceChunker:
    docs = [
        "This is the first sentence. This is the second sentence. This is the third sentence.",
        "This is a paragraph without any punctuation This is the second sentence This is the third sentence",
        (
            "This is the title.\n"
            "This is a document with a lot of paragraphs. "
            "This is the first paragraph. "
            "The first paragraph has some sentences. "
            "This is the forth sentence in the first paragraph. "
            "This is the last sentence in the first paragraph.\n\n"
            "This is the second paragraph. "
            "The second paragraph also has some sentences. "
            "This is the third sentence in the second paragraph. "
            "The second paragraph has more sentences than the first paragraph. "
            "This is the last sentence in the second paragraph."
        ),
    ]

    def chunks_test(self, chunks: list[str], doc: str, strict: bool = True):
        if strict:
            assert "".join(chunks) == doc
        else:
            assert re.sub(r"\s", "", "".join(chunks)) == re.sub(r"\s", "", doc)
        return

    def test_sentence_chunker(self):
        tokenizer = TikTokenTokenizer(TikTokenTokenizerConfig())

        # test nltk sentence splitter
        try:
            splitter = NLTKSentenceSplitter(NLTKSentenceSplitterConfig())
            chunker = SentenceChunker(
                SentenceChunkerConfig(max_sents=2),
                tokenizer=tokenizer,
                splitter=splitter,
            )
            for doc in self.docs:
                chunks = [chunk.text for chunk in chunker.chunk(doc)]
                self.chunks_test(chunks, doc, strict=False)
        except LookupError:
            # NLTK punkt data is optional in the local smoke environment.
            pass

        # test regex sentence splitter
        splitter = RegexSplitter(RegexSplitterConfig())
        chunker = SentenceChunker(
            SentenceChunkerConfig(max_sents=2),
            tokenizer=tokenizer,
            splitter=splitter,
        )
        for doc in self.docs:
            chunks = [chunk.text for chunk in chunker.chunk(doc)]
            self.chunks_test(chunks, doc, strict=False)
        return
