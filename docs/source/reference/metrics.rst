Metrics
=======

This module contains functions for evaluating the performance of a RAG assistant or a retriever.

.. autoclass:: flexrag.metrics.MetricsBase
    :members:
    :inherited-members:


Helper Class
------------
The RAGEvaluator takes a list of metrics and evaluates the performance of a RAG assistant or a retriever.

.. autoclass:: flexrag.metrics.RAGEvaluatorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.RAGEvaluator
    :members:
    :show-inheritance:


Generation Metrics
------------------

.. autoclass:: flexrag.metrics.BLEUConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.BLEU
    :members:
    :show-inheritance:
    :exclude-members: compute

.. autoclass:: flexrag.metrics.Rouge1
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.Rouge2
    :members:
    :show-inheritance:

.. autoclass:: flexrag.metrics.RougeL
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

Retrieval Metrics
-----------------

.. autoclass:: flexrag.metrics.SuccessRateConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.metrics.SuccessRate
    :members:
    :show-inheritance:
    :exclude-members: compute
