Context Refiner
===============
The context refiner is responsible for refining the contexts retrieved by the retriever.
It can be used to rearrange the contexts, summarize them, or extract the most relevant information from them.

The Context Refiner Interface
-----------------------------
The `RefinerProtocol` defines the synchronous and asynchronous interfaces for
refining contexts retrieved by the retriever.

.. autoclass:: flexrag.processors.refiners.RefinerProtocol
    :members:
    :inherited-members:

Refiners
--------
FlexRAG provides several refiners that can be used to refine the contexts retrieved by the retriever.

.. autoclass:: flexrag.processors.refiners.ContextArrangerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.refiners.ContextArranger
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.refiners.AbstractiveSummarizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.refiners.AbstractiveSummarizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.refiners.RecompExtractiveSummarizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.refiners.RecompExtractiveSummarizer
    :members:
    :show-inheritance:
