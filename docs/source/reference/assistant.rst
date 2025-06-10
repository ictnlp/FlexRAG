Assistant
=========

The ``Assistant`` class serves as an abstraction for Retrieval-Augmented Generation (RAG) behavior. It takes the user's query as input and returns an appropriate response. This class provides a flexible interface for defining how the assistant handles queries, including whether a retrieval step is required, how the retrieval should be conducted, and how the assistant generates the response based on the retrieved information.

The Assistant Interface
-----------------------
``AssistantBase`` is the base class for all assistants. It provides a simple interface for answering a user query. The answering process is controlled by a configuration object that is passed to the assistant's constructor.

.. autoclass:: flexrag.assistant.AssistantBase
    :members:
    :inherited-members:

.. autoclass:: flexrag.assistant.AssistantConfig
    :members:
    :show-inheritance:

AssistantConfig is the general configuration for all registered Assistant.
You can load any Assistant by specifying the ``assistant_type`` in the configuration.
For example, to load the ``BasicAssistant``, you can use the following configuration:

.. code-block:: python

    from flexrag.assistant import AssistantConfig, ASSISTANTS, BasicAssistantConfig
    from flexrag.models import OpenAIGeneratorConfig

    config = AssistantConfig(
        assistant_type="basic",
        basic_config=BasicAssistantConfig(
            generator_type="openai",
            openai_config=OpenAIGeneratorConfig(
                model_name="Qwen2-7B-Instruct",
                base_url="http://127.0.1:8000/v1",
            ),
        ),
    )
    assistant = ASSISTANTS.load(config)


FlexRAG Assistants
------------------
FlexRAG provides several assistant implementations that can be used out of the box. These implementations are designed to be flexible and extensible, allowing users to customize the assistant's behavior by providing their own retrieval and generation components.

.. autoclass:: flexrag.assistant.BasicAssistantConfig
    :members:
    :show-inheritance:

.. autoclass:: flexrag.assistant.BasicAssistant
    :members:
    :show-inheritance:
    :exclude-members: answer

.. autoclass:: flexrag.assistant.ModularAssistantConfig
    :members:
    :show-inheritance:

.. autoclass:: flexrag.assistant.ModularAssistant
    :members:
    :show-inheritance:
    :exclude-members: answer, search, answer_with_contexts

.. autoclass:: flexrag.assistant.ChatQAAssistant
    :members:
    :show-inheritance:
    :exclude-members: answer, get_formatted_input, answer_with_contexts
