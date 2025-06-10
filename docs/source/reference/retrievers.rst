Retrievers
==========
Retrievers are used to retrieve data from the local knowledge base or the web.


The Retriever Interface
-----------------------
``RetrieverBase`` is the base class for all retrievers,
including the subclasses of ``EditableRetriever`` and ``WebRetrieverBase``.


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

    from flexrag.retriever import RetrieverConfig, RETRIEVERS, FlexRetrieverConfig

    config = RetrieverConfig(
        retriever_type='flex',
        flex_config=FlexRetrieverConfig(
            retriever_path='<path_to_retriever>',
        )
    )
    retriever = RETRIEVERS.load(config)

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

.. MultiFieldIndex
.. autoclass:: flexrag.retriever.index.MultiFieldIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.MultiFieldIndex
    :members:
    :show-inheritance:

Web Retriever
-------------
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

.. autoclass:: flexrag.retriever.SimpleWebRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.SimpleWebRetriever
    :members:
    :show-inheritance:

.. autoclass:: flexrag.retriever.WikipediaRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.WikipediaRetriever
    :members:
    :show-inheritance:


Web Seeker
----------
``WebSeeker`` is used to search the resources from the web for the given query.
The web resources could be sought by walking through a set of given web pages, by using a search engine, etc.
FlexRAG provides several web seekers using existing search engines.

.. Base Web Seeker
.. autoclass:: flexrag.retriever.web_retrievers.WebSeekerBase
    :members:
    :inherited-members:

.. General Configuration
.. autoclass:: flexrag.retriever.web_retrievers.WebSeekerConfig
    :members:
    :inherited-members:

WebSeekerConfig is the general configuration for all registered WebSeekers.
You can load any WebSeekers by specifying the ``web_seeker_type`` in the configuration.
For example, to load the ``DuckDuckGoEngine``, you can use the following configuration:

.. code-block:: python

    from flexrag.retriever.web_retrievers import WebSeekerConfig, WEB_SEEKERS

    config = WebSeekerConfig(
        web_seeker_type='ddg',
    )
    seeker = WEB_SEEKERS.load(config)


.. General Configuration
.. autoclass:: flexrag.retriever.web_retrievers.SearchEngineConfig
    :members:
    :inherited-members:

SearchEngine is a type of WebSeeker that searches for web resources by leveraging existing search engines.
SearchEngineConfig is the general configuration for all registered SearchEngines.
You can load any SearchEngines by specifying the ``search_engine_type`` in the configuration.
For example, to load the ``DuckDuckGoEngine``, you can use the following configuration:

.. code-block:: python

    from flexrag.retriever.web_retrievers import SearchEngineConfig, SEARCH_ENGINES

    config = SearchEngineConfig(
        search_engine_type='ddg',
    )
    seeker = SEARCH_ENGINES.load(config)


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

.. autoclass:: flexrag.retriever.web_retrievers.WebReaderConfig
    :members:
    :inherited-members:

WebReaderConfig is the general configuration for all registered WebReaders.
You can load any WebReader by specifying the ``web_reader_type`` in the configuration.
For example, to load the ``JinaReader``, you can use the following configuration:

.. code-block:: python

    from flexrag.retriever.web_retrievers import WebReaderConfig, WEB_READERS

    config = WebReaderConfig(
        web_reader_type='jina_reader',
    )
    seeker = WEB_READERS.load(config)


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
