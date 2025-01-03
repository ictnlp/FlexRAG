Prompt
======
This module provides two classes namely `ChatPrompt` and `ChatTemplate`. The `ChatPrompt` is used to store the system prompt, chat history, and demonstrations used to interact with the `Generator`. The `ChatTemplate` is used to convert the `ChatPrompt` into a string or a list of tokens that can be used by the model.

Prompt
------

.. autoclass:: flexrag.prompt.ChatTurn
    :members:
    :inherited-members:

.. autoclass:: flexrag.prompt.ChatPrompt
    :members:
    :inherited-members:

.. autoclass:: flexrag.prompt.MultiModelChatTurn
    :members:
    :inherited-members:

.. autoclass:: flexrag.prompt.MultiModelChatPrompt
    :members:
    :inherited-members:


Template
--------

.. autoclass:: flexrag.prompt.ChatTemplate
    :members:
    :inherited-members:

.. autoclass:: flexrag.prompt.HFTemplate
    :members:
    :show-inheritance:

.. autofunction:: flexrag.prompt.load_template
