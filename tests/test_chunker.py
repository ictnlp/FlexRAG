import re

from flexrag.chunking import (
    CharChunker,
    CharChunkerConfig,
    RecursiveChunker,
    RecursiveChunkerConfig,
    SentenceChunker,
    SentenceChunkerConfig,
    TokenChunker,
    TokenChunkerConfig,
    SemanticChunker,
    SemanticChunkerConfig,
)
from flexrag.models import HFEncoderConfig
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig


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
            assert re.sub("\s", "", "".join(chunks)) == re.sub("\s", "", doc)
        return

    def test_char_chunker(self):
        # chunk without overlap
        chunker = CharChunker(CharChunkerConfig(max_chars=10, overlap=0))
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                assert len(chunk) <= 10
            self.chunks_test(chunks, doc)

        # chunk with overlap
        chunker = CharChunker(CharChunkerConfig(max_chars=10, overlap=3))
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                assert len(chunk) <= 10
        return

    def test_token_chunker(self):
        tokenizer = TOKENIZERS.load(TokenizerConfig())

        # chunk without overlap
        chunker = TokenChunker(TokenChunkerConfig(max_tokens=5, overlap=0))
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 5
            self.chunks_test(chunks, doc)

        # chunk with overlap
        chunker = TokenChunker(TokenChunkerConfig(max_tokens=5, overlap=1))
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 5
        return

    def test_recursive_chunker(self):
        tokenizer = TOKENIZERS.load(TokenizerConfig())
        chunker = RecursiveChunker(RecursiveChunkerConfig(max_tokens=10))
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                assert len(tokenizer.tokenize(chunk)) <= 10
            self.chunks_test(chunks, doc)
        return

    def test_sentence_chunker(self):
        # test nltk sentence splitter
        chunker = SentenceChunker(
            SentenceChunkerConfig(max_sents=2, sentence_splitter_type="nltk_splitter")
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            self.chunks_test(chunks, doc, strict=False)

        # test regex sentence splitter
        chunker = SentenceChunker(
            SentenceChunkerConfig(max_sents=2, sentence_splitter_type="regex")
        )
        for doc in self.docs:
            chunks = chunker.chunk(doc)
            self.chunks_test(chunks, doc)
        return

    def test_sementic_chunker(self):
        for doc in self.docs:
            # chunk with threshold_percentile
            cfg = SemanticChunkerConfig(
                threshold_percentile=50,
                encoder_type="hf",
                hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
            )
            chunker = SemanticChunker(cfg)
            chunks = chunker.chunk(doc)
            self.chunks_test(chunks, doc, strict=False)

            # chunk with threshold
            cfg = SemanticChunkerConfig(
                threshold=0.95,
                encoder_type="hf",
                hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
            )
            chunker = SemanticChunker(cfg)
            chunks = chunker.chunk(doc)
            self.chunks_test(chunks, doc, strict=False)

            # chunk with max_tokens
            cfg = SemanticChunkerConfig(
                max_tokens=30,
                encoder_type="hf",
                hf_config=HFEncoderConfig(model_path="BAAI/bge-small-en-v1.5"),
            )
            chunker = SemanticChunker(cfg)
            chunks = chunker.chunk(doc)
            self.chunks_test(chunks, doc, strict=False)
        return
