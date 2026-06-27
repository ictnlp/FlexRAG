import re

import numpy as np

from flexrag.models.tokenizer import TikTokenTokenizer, TikTokenTokenizerConfig
from flexrag.processors.chunkers import (
    CharChunker,
    CharChunkerConfig,
    RecursiveChunker,
    RecursiveChunkerConfig,
    SemanticChunker,
    SemanticChunkerConfig,
    SentenceChunker,
    SentenceChunkerConfig,
    TokenChunker,
    TokenChunkerConfig,
)


class _FakeEncoder:
    @property
    def embedding_size(self) -> int:
        return 3

    def encode(self, inputs: list[str]) -> np.ndarray:
        return np.array(
            [
                [float(idx + 1), float(len(text) + 1), 1.0]
                for idx, text in enumerate(inputs)
            ],
            dtype=np.float32,
        )


class TestChunker:
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

    def test_char_chunker(self):
        # chunk without overlap
        chunker = CharChunker(CharChunkerConfig(max_chars=10, overlap=0))
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            for chunk in chunks:
                assert len(chunk) <= 10
            self.chunks_test(chunks, doc)

        # chunk with overlap
        chunker = CharChunker(CharChunkerConfig(max_chars=10, overlap=3))
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            for chunk in chunks:
                assert len(chunk) <= 10
        return

    def test_token_chunker(self):
        tokenizer = TikTokenTokenizer(TikTokenTokenizerConfig())

        # chunk without overlap
        chunker = TokenChunker(
            TokenChunkerConfig(max_tokens=5, overlap=0),
            tokenizer=tokenizer,
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 5
            self.chunks_test(chunks, doc)

        # chunk with overlap
        chunker = TokenChunker(
            TokenChunkerConfig(max_tokens=5, overlap=1),
            tokenizer=tokenizer,
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 5
        return

    def test_recursive_chunker(self):
        tokenizer = TikTokenTokenizer(TikTokenTokenizerConfig())
        chunker = RecursiveChunker(
            RecursiveChunkerConfig(max_tokens=10),
            tokenizer=tokenizer,
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 10
            self.chunks_test(chunks, doc, strict=False)
        return

    def test_sentence_chunker(self):
        tokenizer = TikTokenTokenizer(TikTokenTokenizerConfig())

        # test nltk sentence splitter
        try:
            chunker = SentenceChunker(
                SentenceChunkerConfig(
                    max_sents=2, sentence_splitter_type="nltk_splitter"
                ),
                tokenizer=tokenizer,
            )
            for doc in self.docs:
                chunks = chunker.chunk(doc, return_str=True)
                self.chunks_test(chunks, doc, strict=False)
        except LookupError:
            # NLTK punkt data is optional in the local smoke environment.
            pass

        # test regex sentence splitter
        chunker = SentenceChunker(
            SentenceChunkerConfig(max_sents=2, sentence_splitter_type="regex"),
            tokenizer=tokenizer,
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc, return_str=True)
            self.chunks_test(chunks, doc, strict=False)
        return

    def test_sementic_chunker(self):
        encoder = _FakeEncoder()
        tokenizer = TikTokenTokenizer(TikTokenTokenizerConfig())
        for doc in self.docs:
            # chunk with threshold_percentile
            cfg = SemanticChunkerConfig(threshold_percentile=50)
            chunker = SemanticChunker(cfg, encoder=encoder, tokenizer=tokenizer)
            chunks = chunker.chunk(doc, return_str=True)
            self.chunks_test(chunks, doc, strict=False)

            # chunk with threshold
            cfg = SemanticChunkerConfig(threshold=0.95)
            chunker = SemanticChunker(cfg, encoder=encoder, tokenizer=tokenizer)
            chunks = chunker.chunk(doc, return_str=True)
            self.chunks_test(chunks, doc, strict=False)

            # chunk with max_tokens
            cfg = SemanticChunkerConfig(max_tokens=30)
            chunker = SemanticChunker(cfg, encoder=encoder, tokenizer=tokenizer)
            chunks = chunker.chunk(doc, return_str=True)
            self.chunks_test(chunks, doc, strict=False)
        return
