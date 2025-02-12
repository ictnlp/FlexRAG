Chunking
========
This module provides a set of classes for chunking a long text into smaller chunks.


The Chunker Interface
---------------------
ChunkerBase is the base class for all chunkers. 
It provides a simple interface for chunking a text into smaller chunks. 
The chunking process is controlled by a configuration object that is passed to the chunker's constructor.

.. autoclass:: flexrag.chunking.ChunkerBase
    :members:
    :inherited-members:


Chunkers
--------

.. autoclass:: flexrag.chunking.CharChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.CharChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.TokenChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.TokenChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.RecursiveChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.RecursiveChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.SentenceChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.SentenceChunker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.SemanticChunkerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.SemanticChunker
    :members:
    :show-inheritance:


Sentence Splitters
------------------
This submodule provides a set of useful tools for splitting a text into sentences.

.. autoclass:: flexrag.chunking.sentence_splitter.SentenceSplitterBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.sentence_splitter.NLTKSentenceSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.sentence_splitter.NLTKSentenceSplitter
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.sentence_splitter.RegexSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.sentence_splitter.RegexSplitter
    :members:
    :show-inheritance:

.. autoclass:: flexrag.chunking.sentence_splitter.SpacySentenceSplitterConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.sentence_splitter.SpacySentenceSplitter
    :members:
    :show-inheritance:

.. autoattribute:: flexrag.chunking.sentence_splitter.PREDEFINED_SPLIT_PATTERNS

    A dictionary of predefined sentence splitting patterns. 
    The keys are the names of the patterns, and the values are the corresponding regular expressions.
    Currently, ``FlexRAG`` provides 2 sets of predefined patterns: "en" for English and "zh" for Chinese.
    Please refer to the source code for more details.

General Configuration
---------------------
The configuration provides a general interface for loading and configurate the chunker or the sentence splitter.

.. autoclass:: flexrag.chunking.ChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.chunking.sentence_splitter.SentenceSplitterConfig
    :members:
    :inherited-members:
