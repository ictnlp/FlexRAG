Chunking
========
This module provides a set of classes for chunking a long text into smaller chunks.


The Chunker Interface
---------------------
``ChunkerProtocol`` defines the structural interface shared by all chunkers.
The chunking process is controlled by a configuration object that is passed to
the chunker's constructor.

.. autoclass:: flexrag.processors.chunkers.ChunkerProtocol
    :members:


Chunkers
--------

.. autoclass:: flexrag.processors.chunkers.CharChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.chunkers.CharChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.TokenChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.TokenChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.RecursiveChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.RecursiveChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.SentenceChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.SentenceChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.SemanticChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.SemanticChunker
    :members:
    :show-inheritance:


Sentence Splitters
------------------
This submodule provides sentence-oriented implementations of ``ChunkerProtocol``.

.. autoclass:: flexrag.processors.chunkers.NLTKSentenceSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.chunkers.NLTKSentenceSplitter
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.RegexSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.chunkers.RegexSplitter
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.chunkers.SpacySentenceSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.chunkers.SpacySentenceSplitter
    :members:
    :show-inheritance:

.. autoattribute:: flexrag.processors.chunkers.PREDEFINED_SPLIT_PATTERNS

    A dictionary of predefined sentence splitting patterns.
    The keys are the names of the patterns, and the values are the corresponding regular expressions.
    Currently, ``FlexRAG`` provides 2 sets of predefined patterns: "en" for English and "zh" for Chinese.
    Please refer to the source code for more details.

General Configuration
---------------------
The configuration provides a general interface for loading and configurate the chunker or the sentence splitter.

.. autoclass:: flexrag.processors.chunkers.ChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.chunkers.SentenceSplitterConfig
    :members:
    :inherited-members:
