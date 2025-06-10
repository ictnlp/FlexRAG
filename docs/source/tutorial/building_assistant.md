# Building Your Own RAG Assistant
FlexRAG provides a flexible and modularized components for building RAG assistants. You can build your own RAG assistant by defining your own `Assistant` class and registering it with the `ASSISTANTS` decorator.

## Define the Assistant Class
To build your RAG assistant, you can create a Python script file and import the necessary FlexRAG modules. Below is an example of how to construct a RAG assistant. In this example, we define a RAG assistant named `SimpleAssistant` by inheriting from the {class}`~flexrag.assistant.AssistantBase` class. This assistant includes a retriever ({class}`~flexrag.retriever.FlexRetriever`) and a generator ({class}`~flexrag.models.OpenAIGenerator`). Whenever a user asks a question, `SimpleAssistant` uses {class}`~flexrag.retriever.FlexRetriever` to retrieve relevant documents from the database, then concatenates these documents into the prompt and utilizes {class}`~flexrag.models.OpenAIGenerator` to generate the final response.

```python
from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS, AssistantBase
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import FlexRetriever, FlexRetrieverConfig


@dataclass
class SimpleAssistantConfig(FlexRetrieverConfig, OpenAIGeneratorConfig): ...


@ASSISTANTS("simple", config_class=SimpleAssistantConfig)
class SimpleAssistant(AssistantBase):
    def __init__(self, config: SimpleAssistantConfig):
        self.retriever = FlexRetriever(config)
        self.generator = OpenAIGenerator(config)
        return

    def answer(self, question: str) -> str:
        prompt = ChatPrompt()
        context = self.retriever.search(question)[0]
        prompt_str = "Please answer the following question based on the given text.\n\n"
        prompt_str += f"Question: {question}\n\n"
        for n, ctx in enumerate(context):
            prompt_str += f"Context {n}: {ctx.data['text']}\n"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = self.generator.chat([prompt])[0][0]
        prompt.update(ChatTurn(role="assistant", content=response))
        return response
```


### Evaluating your own RAG Application
After defining the `SimpleAssistant` class and registering it with the `ASSISTANTS` decorator, you can evaluate your assistant using FlexRAG's entrypoints by adding the `user_module=<your_module_path>` argument to the command.

For example, you can evaluate your assistant on the *Natural Questions* dataset using the following command:

```bash
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>
MODULE_PATH=<path_to_simple_assistant_module>

python -m flexrag.entrypoints.eval_assistant \
    user_module=${MODULE_PATH} \
    name=nq \
    split=test \
    assistant_type=simple \
    simple_config.model_name='gpt-4o-mini' \
    simple_config.api_key=${OPENAI_KEY} \
    simple_config.retriever_path=${DB_PATH} \
    simple_config.used_indexes=[contriever] \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    log_interval=10
```

In [FlexRAG_Examples](https://github.com/ictnlp/FlexRAG_Examples) repository, we provide several detailed examples of how to build a RAG assistant.
