Generators
==========

Local Generators
----------------

.. autoclass:: flexrag.models.GeneratorBase
    :members:
    :inherited-members:


.. Hugging Face Generators
.. autoclass:: flexrag.models.HFGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.HFGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate


.. Llamacpp Generators
.. autoclass:: flexrag.models.LlamacppGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.LlamacppGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate


.. Ollama Generators
.. autoclass:: flexrag.models.OllamaGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.OllamaGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate

.. VLLM Generators
.. autoclass:: flexrag.models.VLLMGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.VLLMGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate


Online Generators
-----------------

.. Anthropic Generators
.. autoclass:: flexrag.models.AnthropicGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.AnthropicGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate

.. OpenAI Generators
.. autoclass:: flexrag.models.OpenAIGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.OpenAIGenerator
    :members:
    :show-inheritance:
    :exclude-members: async_chat, async_generate, chat, generate


Visual Language Model Generators
--------------------------------

.. autoclass:: flexrag.models.VLMGeneratorBase
    :members:
    :inherited-members:
    :show-inheritance:

.. HF VLM Generators
.. autoclass:: flexrag.models.HFVLMGeneratorConfig
    :members:
    :inherited-members:

.. autoclass:: flexrag.models.HFVLMGenerator
    :members:
    :show-inheritance:
    :exclude-members: chat, generate