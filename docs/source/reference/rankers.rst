Rankers
=======

The ranker is the component that determines the order of the results returned by the retriever. FlexRAG provides several rankers that can be used to sort the results based on various criteria.

``RankerProtocol`` is the public structural interface accepted by ranker users.
``RankerBase`` remains available for implementations that reuse its configuration
and asynchronous fallback behavior.

.. autoclass:: flexrag.processors.rankers.RankerProtocol
    :members:

.. autoclass:: flexrag.processors.rankers.RankerBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.rankers.RankerBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.rankers.RankingResult
    :members:
    :inherited-members:

Ranker Implementations
----------------------

.. autoclass:: flexrag.processors.rankers.HFRankerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.rankers.HFRanker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.rankers.RankGPTRankerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.rankers.RankGPTRanker
    :members:
    :show-inheritance:

.. autoclass:: flexrag.processors.rankers.LiteLLMRankerConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.processors.rankers.LiteLLMRanker
    :members:
    :show-inheritance:
