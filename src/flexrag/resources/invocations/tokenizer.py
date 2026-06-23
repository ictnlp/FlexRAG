from typing import Any


class TokenizerInvocation:
    """Invocation semantics for managed tokenizers.

    The invocation is intentionally thin: tokenizer inputs are already simple
    scalar values, so it only proxies tokenizer methods and properties through
    the selected runtime adapter.
    """

    def __init__(self, runtime: Any) -> None:
        """Create a tokenizer invocation.

        :param runtime: Runtime adapter used to execute tokenizer calls.
        """
        self.runtime = runtime
        return

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into string tokens.

        :param text: Text to tokenize.
        :return: Token strings.
        """
        return self.runtime.call("tokenize", text)

    def detokenize(self, tokens: list[str]) -> str:
        """Convert string tokens back to text.

        :param tokens: Token strings to detokenize.
        :return: Detokenized text.
        """
        return self.runtime.call("detokenize", tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids.

        :param text: Text to encode.
        :return: Token ids.
        """
        return self.runtime.call("encode", text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids into text.

        :param tokens: Token ids to decode.
        :return: Decoded text.
        """
        return self.runtime.call("decode", tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert string tokens to token ids.

        :param tokens: Token strings to convert.
        :return: Token ids.
        """
        return self.runtime.call("tokens_to_ids", tokens)

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert token ids to string tokens.

        :param token_ids: Token ids to convert.
        :return: Token strings.
        """
        return self.runtime.call("ids_to_tokens", token_ids)

    @property
    def reversible(self) -> bool:
        """Return whether tokenization is strictly reversible."""
        return self.runtime.getattr("reversible")

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.runtime.getattr("vocab_size")
