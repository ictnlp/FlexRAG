Tokenizer
=========
This module is a simple wrapper around other tokenizers. 
It provides a simple and consistent interface for tokenizing a text into tokens (maybe string or int).

The Tokenizer Interface
-----------------------
``TokenizerProtocol`` defines the structural interface shared by all tokenizers.

.. autoclass:: flexrag.models.tokenizer.TokenizerProtocol
    :members:


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
