Metrics
=======

This module contains functions for evaluating the performance of a RAG assistant or a retriever.

.. autoclass:: flexrag.metrics.MetricsBase
    :members:
    :inherited-members:


Helper Class
------------
The RAGEvaluator takes a list of metrics and evaluates the performance of a RAG assistant or a retriever.

.. autoclass:: flexrag.metrics.EvaluatorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.Evaluator
    :members:
    :show-inheritance:


RAG Generation Metrics
----------------------

.. autoclass:: flexrag.metrics.BLEUConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.BLEU
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.Rouge
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.chrFConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.chrF
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.F1
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.Accuracy
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.ExactMatch
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.Precision
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.Recall
    :members:
    :show-inheritance:

Information Retrieval Metrics
-----------------------------

.. autoclass:: flexrag.metrics.SuccessRateConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.SuccessRate
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.RetrievalRecallConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.RetrievalRecall
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.RetrievalPrecisionConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.RetrievalPrecision
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.RetrievalMAPConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.RetrievalMAP
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.RetrievalNDCGConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.RetrievalNDCG
    :members:
    :show-inheritance:
    :exclude-members: compute
