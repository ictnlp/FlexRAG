Retrievers
==========

.. autoclass:: flexrag.retriever.RetrieverBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrieverBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.RetrievedContext
    :members:
    :inherited-members:

Local Retrievers
----------------
Local retrievers are used to retrieve data from the local knowledge base.

.. autoclass:: flexrag.retriever.LocalRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.LocalRetriever
    :members:
    :show-inheritance:
    :exclude-members: search


.. BM25S Retriever
.. autoclass:: flexrag.retriever.BM25SRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.BM25SRetriever
    :members:
    :show-inheritance:
    :exclude-members: add_passages, clean, search_batch, fields


.. Dense Retriever
.. autoclass:: flexrag.retriever.DenseRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.DenseRetriever
    :members:
    :show-inheritance:
    :exclude-members: add_passages, clean, search_batch, fields


.. Elastic Search Retriever
.. autoclass:: flexrag.retriever.ElasticRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.ElasticRetriever
    :members:
    :show-inheritance:
    :exclude-members: add_passages, clean, search_batch, fields


.. Hyde Retriever
.. autoclass:: flexrag.retriever.HydeRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.HydeRetriever
    :members:
    :show-inheritance:
    :exclude-members: search_batch

Dense Index
-----------
Dense Index is used in DenseRetriever to store and retrieve dense embeddings.

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
    :exclude-members: build_index, clean, deserialize, embedding_size, is_trained, serialize


.. Faiss Index
.. autoclass:: flexrag.retriever.index.FaissIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.FaissIndex
    :members:
    :show-inheritance:
    :exclude-members: build_index, clean, deserialize, embedding_size, is_trained, serialize


.. ScaNN Index
.. autoclass:: flexrag.retriever.index.ScaNNIndexConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.index.ScaNNIndex
    :members:
    :show-inheritance:
    :exclude-members: build_index, clean, deserialize, embedding_size, is_trained, serialize

Web Retrievers
--------------
Web retrievers are used to retrieve data from the web.

.. autoclass:: flexrag.retriever.web_retrievers.WebRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.WebRetrieverBase
    :members:
    :inherited-members:


.. Web Search Engines
.. autoclass:: flexrag.retriever.BingRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.BingRetriever
    :members:
    :show-inheritance:
    :exclude-members: search_item

.. autoclass:: flexrag.retriever.DuckDuckGoRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.DuckDuckGoRetriever
    :members:
    :show-inheritance:
    :exclude-members: search_item

.. autoclass:: flexrag.retriever.GoogleRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.GoogleRetriever
    :members:
    :show-inheritance:
    :exclude-members: search_item

.. autoclass:: flexrag.retriever.SerpApiRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.SerpApiRetriever
    :members:
    :show-inheritance:
    :exclude-members: search_item

.. autoclass:: flexrag.retriever.WikipediaRetrieverConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.WikipediaRetriever
    :members:
    :show-inheritance:
    :exclude-members: search, fields


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
    :exclude-members: download_page

.. autoclass:: flexrag.retriever.web_retrievers.PuppeteerWebDownloaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.PuppeteerWebDownloader
    :members:
    :show-inheritance:
    :exclude-members: download_page, async_download


Web Reader
----------
Web reader is used to convert web data into LLM friendly format.

.. autoclass:: flexrag.retriever.web_retrievers.WebReaderBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.WebRetrievedContext
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReader
    :members:
    :show-inheritance:
    :exclude-members: fields, read

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderLMConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.JinaReaderLM
    :members:
    :show-inheritance:
    :exclude-members: fields, read

.. autoclass:: flexrag.retriever.web_retrievers.ScreenshotWebReader
    :members:
    :show-inheritance:
    :exclude-members: fields, read

.. autoclass:: flexrag.retriever.web_retrievers.ScreenshotWebReaderConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.retriever.web_retrievers.SnippetWebReader
    :members:
    :inherited-members:
    :exclude-members: fields, read
