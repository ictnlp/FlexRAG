from __future__ import annotations

from .base import TypedHandle


class TokenizerHandle(TypedHandle):
    """Typed proxy for tokenizer resources.

    The handle forwards the formal tokenizer contract and does not own
    tokenizer lifecycle.
    """

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into token strings."""
        return self._target.call("tokenize", text)

    def detokenize(self, tokens: list[str]) -> str:
        """Convert token strings back to text."""
        return self._target.call("detokenize", tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._target.call("encode", text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids into text."""
        return self._target.call("decode", tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert token strings to token ids."""
        return self._target.call("tokens_to_ids", tokens)

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert token ids to token strings."""
        return self._target.call("ids_to_tokens", token_ids)

    @property
    def reversible(self) -> bool:
        """Return whether tokenization is reversible."""
        return bool(self._target.getattr("reversible"))

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return int(self._target.getattr("vocab_size"))
