Tokenizer
=========
This module is a simple wrapper around other tokenizers. 
It provides a simple and consistent interface for tokenizing a text into tokens (maybe string or int).

The Tokenizer Interface
-----------------------
``TokenizerBase`` is the base class for all tokenizers.

.. autoclass:: flexrag.models.tokenizer.TokenizerBase
    :members:
    :inherited-members:


Tokenizers
----------
The wrapped tokenizers.

.. autoclass:: flexrag.models.tokenizer.HuggingFaceTokenizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.tokenizer.HuggingFaceTokenizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.models.tokenizer.TikTokenTokenizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.tokenizer.TikTokenTokenizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.models.tokenizer.MosesTokenizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.tokenizer.MosesTokenizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.models.tokenizer.NLTKTokenizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.tokenizer.NLTKTokenizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.models.tokenizer.JiebaTokenizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.tokenizer.JiebaTokenizer
    :members:
    :show-inheritance: