Context Refiner
===============
The context refiner is responsible for refining the contexts retrieved by the retriever.
It can be used to rearrange the contexts, summarize them, or extract the most relevant information from them.

The Context Refiner Interface
-----------------------------
The `RefinerBase` is the base class for all refiners.
It provides the basic interface for refining the contexts retrieved by the retriever.

.. autoclass:: flexrag.context_refine.RefinerBase
    :members:
    :inherited-members:

Refiners
--------
FlexRAG provides several refiners that can be used to refine the contexts retrieved by the retriever.

.. autoclass:: flexrag.context_refine.ContextArrangerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.context_refine.ContextArranger
    :members:
    :show-inheritance:

.. autoclass:: flexrag.context_refine.AbstractiveSummarizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.context_refine.AbstractiveSummarizer
    :members:
    :show-inheritance:

.. autoclass:: flexrag.context_refine.RecompExtractiveSummarizerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.context_refine.RecompExtractiveSummarizer
    :members:
    :show-inheritance:
