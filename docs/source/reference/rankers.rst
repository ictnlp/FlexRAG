Rankers
=======

The ranker is the component that determines the order of the results returned by the retriever. FlexRAG provides several rankers that can be used to sort the results based on various criteria.

.. autoclass:: flexrag.ranker.RankerBaseConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.ranker.RankerBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.ranker.RankingResult
    :members:
    :inherited-members:

.. autoclass:: flexrag.ranker.RankerConfig
    :members:
    :inherited-members:


Local Ranker
------------
.. HF Cross Encoder Ranker
.. autoclass:: flexrag.ranker.HFCrossEncoderRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.HFCrossEncoderRanker
    :members:
    :show-inheritance:


.. HF Cross Seq2Seq Ranker
.. autoclass:: flexrag.ranker.HFSeq2SeqRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.HFSeq2SeqRanker
    :members:
    :show-inheritance:


.. HF Cross ColBERT Ranker
.. autoclass:: flexrag.ranker.HFColBertRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.HFColBertRanker
    :members:
    :show-inheritance:


.. RankGPT Ranker
.. autoclass:: flexrag.ranker.RankGPTRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.RankGPTRanker
    :members:
    :show-inheritance:


Oneline Ranker
--------------
.. Cohere Ranker
.. autoclass:: flexrag.ranker.CohereRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.CohereRanker
    :members:
    :show-inheritance:


.. Jina Ranker
.. autoclass:: flexrag.ranker.JinaRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.JinaRanker
    :members:
    :show-inheritance:


.. Mixedbread Ranker
.. autoclass:: flexrag.ranker.MixedbreadRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.MixedbreadRanker
    :members:
    :show-inheritance:


.. Voyage Ranker
.. autoclass:: flexrag.ranker.VoyageRankerConfig
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: flexrag.ranker.VoyageRanker
    :members:
    :show-inheritance:
