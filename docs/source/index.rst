|

.. image:: ../../assets/flexrag-wide.png
   :alt: FlexRAG
   :align: center

|

Welecome to FlexRAG Documentation
=================================

RAGFrame is a highly reproducible, easy-to-use, and high-performance RAG framework designed for both research and application scenarios. It supports **text**, **multimodal**, and **web-based** RAG, providing a **complete RAG pipelin**e and evaluation process. With built-in **asynchronous** processing and **persistent caching**, it ensures efficiency and scalability. Easily load retrievers from Hugging Face and quickly build powerful RAG solutions out of the box.

FlexRAG is under active development and is currently in the alpha stage. We welcome contributions from the community and are open to feedback and suggestions.

.. Check out the :doc:`tutorial` section for further information, including how to
.. :ref:`install <installation>` the project.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   getting_started/installation
   getting_started/quickstart1
   getting_started/quickstart2

.. toctree::
   :maxdepth: 1
   :caption: Tutorial:

   tutorial/preparing_retriever
   tutorial/building_assistant
   tutorial/entrypoints
   tutorial/using_register
   tutorial/preparing_web_retriever

.. toctree::
   :maxdepth: 1
   :caption: API Reference Manual:

   reference/assistant
   reference/chunking
   reference/common_dataclass
   reference/refiner
   reference/datasets
   reference/document_parser
   reference/encoders
   reference/generators
   reference/metrics
   reference/prompt
   reference/retrievers
   reference/rankers
   reference/tokenizers
   reference/text_process
   reference/utils
