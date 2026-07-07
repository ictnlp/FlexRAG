Retrievers
==========
Retrievers coordinate context stores and collection backends.


Core API
--------

.. autoclass:: flexrag.retrievers.FlexRetrieverConfig
    :members:

.. autoclass:: flexrag.retrievers.FlexRetriever
    :members:

.. autoclass:: flexrag.retrievers.RetrievalView
    :members:

.. autoclass:: flexrag.retrievers.Hit
    :members:


Context Stores
--------------

.. autoclass:: flexrag.retrievers.ContextStoreProtocol
    :members:

.. autoclass:: flexrag.retrievers.LMDBContextStoreConfig
    :members:

.. autoclass:: flexrag.retrievers.LMDBContextStore
    :members:

.. autoclass:: flexrag.retrievers.SQLiteContextStoreConfig
    :members:

.. autoclass:: flexrag.retrievers.SQLiteContextStore
    :members:


Collection Backends
-------------------

.. autoclass:: flexrag.retrievers.CollectionBackend
    :members:

.. autoclass:: flexrag.retrievers.BM25SBackendConfig
    :members:

.. autoclass:: flexrag.retrievers.BM25SBackend
    :members:

.. autoclass:: flexrag.retrievers.FaissBackendConfig
    :members:

.. autoclass:: flexrag.retrievers.FaissBackend
    :members:

.. autoclass:: flexrag.retrievers.ElasticBackendConfig
    :members:

.. autoclass:: flexrag.retrievers.ElasticBackend
    :members:

.. autoclass:: flexrag.retrievers.LanceBackendConfig
    :members:

.. autoclass:: flexrag.retrievers.LanceBackend
    :members:
