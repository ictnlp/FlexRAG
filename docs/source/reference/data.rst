Data
====
Data module provides a set of classes and functions for data manipulation and processing.

Chunking
--------
This submodule provides a set of classes and functions for chunking a long text into smaller chunks.

.. autoclass:: flexrag.data.chunking.ChunkerBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.chunking.CharChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.chunking.CharChunker
    :members:
    :show-inheritance:
    :exclude-members: chunk

.. autoclass:: flexrag.data.chunking.TokenChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.chunking.TokenChunker
    :members:
    :show-inheritance:
    :exclude-members: chunk

.. autoclass:: flexrag.data.chunking.SentenceChunkerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.chunking.SentenceChunker
    :members:
    :show-inheritance:
    :exclude-members: chunk

Document Parser
---------------
This submodule provides a set of classes and functions for parsing a formated document (such as PDF, Word, etc.) into a structured format.

.. autoclass:: flexrag.data.document_parser.Document
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.document_parser.DocumentParserBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.document_parser.DoclingConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.document_parser.DoclingParser
    :members:
    :show-inheritance:
    :exclude-members: parse

.. autoclass:: flexrag.data.document_parser.MarkItDownParser
    :members:
    :show-inheritance:
    :exclude-members: parse


Text Processing
---------------
This submodule provides a set of classes and functions for preprocessing and filtering text data, including normalization, length filtering, etc.

.. autoclass:: flexrag.data.text_process.TextUnit
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.text_process.Processor
    :members:
    :inherited-members:
    :special-members: __call__

.. autoclass:: flexrag.data.text_process.TextProcessPipelineConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.text_process.TextProcessPipeline
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.text_process.TokenNormalizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.text_process.TokenNormalizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.ChineseSimplifier
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.Lowercase
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.Unifier
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.TruncatorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.text_process.Truncator
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.AnswerSimplifier
    :members:
    :show-inheritance:

.. autoclass:: flexrag.data.text_process.ExactDeduplicate
    :members:
    :show-inheritance:


Datasets
--------
This submodule provides a set of classes and functions for loading and processing datasets, including the `RAGTestIterableDataset` class for loading Knowledge Intensive datasets for RAG tasks.

.. autoclass:: flexrag.data.Dataset
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.data.ConcateDataset
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.data.RAGTestData
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.RAGTestIterableDataset
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.data.RetrievalTestData
    :members:
    :inherited-members:

.. autoclass:: flexrag.data.RetrievalTestIterableDataset
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.data.LineDelimitedDataset
    :members:
    :inherited-members:
    :show-inheritance: