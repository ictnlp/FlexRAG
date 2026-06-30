Retrievers
==========
Retrievers are used to manage and search collection-like knowledge bases.


The Retriever Interface
-----------------------
``RetrieverBase`` is the base class for all retrievers.


.. autoclass:: flexrag.retriever.RetrieverBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrieverBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrieverConfig
    :members:
    :inherited-members:

RetrieverConfig is the legacy general configuration for registered retrievers.
``FlexRetriever`` collections are opened from their collection path directly:

.. code-block:: python

    from flexrag.retriever import FlexRetriever, FlexRetrieverConfig

    retriever = FlexRetriever(FlexRetrieverConfig(), '<path_to_retriever>')

.. ElasticSearch Retriever
.. autoclass:: flexrag.retriever.ElasticRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.ElasticRetriever
    :members:
    :show-inheritance:

.. FlexRetriever
.. autoclass:: flexrag.retriever.FlexRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.IndexFieldsConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.FlexRetriever
    :members:
    :show-inheritance:

Retriever Index
---------------
``RetrieverIndex`` is used in ``FlexRetriever`` to store and retrieve dense embeddings.

.. RetrieverIndex Interface
.. autoclass:: flexrag.retriever.index.RetrieverIndexBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.RetrieverIndexBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.RetrieverIndexConfig
    :members:
    :inherited-members:

RetrieverConfig is the general configuration for all registered RetrieverIndexes.
You can load any RetrieverIndex by specifying the ``index_type`` in the configuration.
For example, to load the ``BM25Index``, you can use the following configuration:

.. code-block:: python

    from flexrag.retriever.index import RetrieverIndexConfig, RETRIEVER_INDEX, BM25IndexConfig

    config = RetrieverIndexConfig(index_type='bm25', bm25_config=BM25IndexConfig())
    index = RETRIEVER_INDEX.load(config)

.. Faiss Index
.. autoclass:: flexrag.retriever.index.FaissIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.FaissIndex
    :members:
    :show-inheritance:

.. ScaNN Index
.. autoclass:: flexrag.retriever.index.ScaNNIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.ScaNNIndex
    :members:
    :show-inheritance:

.. BM25 Index
.. autoclass:: flexrag.retriever.index.BM25IndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.BM25Index
    :members:
    :show-inheritance:
