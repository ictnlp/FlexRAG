Retrievers
==========
Retrievers are used to retrieve data from the local knowledge base or the web.


The Retriever Interface
-----------------------
``RetrieverBase`` is the base class for all retrievers, including the subclasses of ``EditableRetriever`` and ``WebRetrieverBase``.


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
For example, to load the ``BM25S`` retriever, you can use the following configuration:

.. code-block:: python

    from flexrag.retriever import RetrieverConfig, RETRIEVERS, BM25SRetrieverConfig

    config = RetrieverConfig(
        retriever_type='bm25s',
        bm25s_config=BM25SRetrieverConfig(
            database_path='<path_to_database>',
        )
    )
    retriever = RETRIEVERS.load(config)

Editable Retrievers
-------------------
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


.. BM25S Retriever
.. autoclass:: flexrag.retriever.BM25SRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.BM25SRetriever
    :members:
    :show-inheritance:

.. Dense Retriever
.. autoclass:: flexrag.retriever.DenseRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.DenseRetriever
    :members:
    :show-inheritance:

.. Hyde Retriever
.. autoclass:: flexrag.retriever.HydeRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.HydeRetriever
    :members:
    :show-inheritance:

Dense Index
-----------
``DenseIndex`` is used in ``DenseRetriever`` to store and retrieve dense embeddings.

.. autoclass:: flexrag.retriever.index.DenseIndexBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.DenseIndexBaseConfig
    :members:
    :inherited-members:

.. Annoy Index
.. autoclass:: flexrag.retriever.index.AnnoyIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.AnnoyIndex
    :members:
    :show-inheritance:


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

Web Retrievers
--------------
``WebRetriever`` is used to retrieve data from the web. Different from the ``EditableRetriever``, web retrievers can be used without building a knowledge base, as they retrieve data using web search engines.

.. autoclass:: flexrag.retriever.web_retrievers.WebRetrieverBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.WebRetrieverBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.WebResource
    :members:
    :inherited-members:

FlexRAG provides two simple web retrievers, ``SimpleWebRetriever`` and ``WikipediaRetriever``.

.. autoclass:: flexrag.retriever.web_retrievers.SimpleWebRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SimpleWebRetriever
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.WikipediaRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.WikipediaRetriever
    :members:
    :show-inheritance:


Web Seekers
-----------
``WebSeeker`` is used to search the resources from the web for the given query.
The web resources could be sought by walking through a set of given web pages, by using a search engine, etc.
FlexRAG provides several web seekers using existing search engines.


.. Web Search Engines
.. autoclass:: flexrag.retriever.web_retrievers.BingEngineConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.BingEngine
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.DuckDuckGoEngineConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.DuckDuckGoEngine
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.GoogleEngineConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.GoogleEngine
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.SerpApiConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SerpApi
    :members:
    :show-inheritance:


Web Downloader
--------------
Web downloader is used to download data from the web.

.. autoclass:: flexrag.retriever.web_retrievers.WebDownloaderBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.WebDownloaderBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SimpleWebDownloaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SimpleWebDownloader
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.PlaywrightWebDownloaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.PlaywrightWebDownloader
    :members:
    :show-inheritance:


Web Reader
----------
Web reader is used to convert web data into LLM friendly format.

.. autoclass:: flexrag.retriever.web_retrievers.WebReaderBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReader
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderLMConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderLM
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.ScreenshotWebReader
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.web_retrievers.ScreenshotWebReaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SnippetWebReader
    :members:
    :inherited-members:
