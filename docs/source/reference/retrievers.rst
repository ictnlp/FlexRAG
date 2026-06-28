Retrievers
==========
Retrievers are used to retrieve data from the local knowledge base or the web.


The Retriever Interface
-----------------------
``RetrieverBase`` is the base class for all retrievers,
including the subclasses of ``EditableRetriever``.


.. autoclass:: flexrag.retriever.RetrieverBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrieverBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrieverConfig
    :members:
    :inherited-members:

RetrieverConfig is the general configuration for all registered retrievers.
You can load any retriever by specifying the retriever name in the configuration.
For example, to load the pre-built ``FlexRetriever`` retriever,
you can use the following configuration:

.. code-block:: python

    from flexrag.retriever import LocalRetriever

    retriever = LocalRetriever.load_from_local('<path_to_retriever>')

Editable Retriever
------------------
.. autoclass:: flexrag.retriever.EditableRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.EditableRetriever
    :members:
    :show-inheritance:

.. ElasticSearch Retriever
.. autoclass:: flexrag.retriever.ElasticRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.ElasticRetriever
    :members:
    :show-inheritance:

.. Typesense Retriever
.. autoclass:: flexrag.retriever.TypesenseRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.TypesenseRetriever
    :members:
    :show-inheritance:

.. LocalRetriever
.. autoclass:: flexrag.retriever.LocalRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.LocalRetriever
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

.. Hyde Retriever
.. autoclass:: flexrag.retriever.HydeRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.HydeRetriever
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

    config = RetrieverIndexConfig(
        index_type='bm25',
        bm25_config=BM25IndexConfig(
            index_path='<path_to_index>',
        )
    )
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
